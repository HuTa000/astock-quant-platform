"""数据更新编排服务。

职责（P1-S6）：
    1. 按配置的主/兜底数据源，拉取自选股（缺省时用内置演示清单）的日线并落库
       （R13 起同时落未复权 ``""`` 与前复权 ``"qfq"`` 两套口径）；
    2. 支持 1 分钟线更新（走配置的 ``data.source.minute``，即 pytdx），以及
       5/15/30/60 分钟线批量更新（走主/兜底日线源，供模拟训练按周期回放）；
    3. **增量更新**：已落库者从「最大 dt 的次日」开始；首次用默认起始；
    4. **重试 + 兜底**：主源失败重试 1 次 → 仍失败切兜底源；单只失败不中断整体；
    5. 全程通过 :mod:`app.core.event_bus` 广播进度事件（驱动 UI）；
    6. 结束后把 ``meta['last_update_time']`` 写为当前时间（ISO）。

设计约定（《开发手册与交接文档》§4）：
    * Bar 契约列为 ``[datetime, open, high, low, close, volume, amount]``，``volume`` 单位「手」；
    * 所有对外方法**不向 UI 抛异常**：内部消化为事件 / :class:`UpdateReport`；
    * 日志统一使用 ``logging``。

事件与 payload（《细化设计》§1.3）：
    * ``DATA_UPDATE_STARTED``   -> ``{"source": str, "scope": str}``
    * ``DATA_UPDATE_PROGRESS``  -> ``{"symbol": str, "current": int, "total": int}``
    * ``DATA_UPDATE_FAILED``    -> ``{"symbol": str, "error": str}``
    * ``DATA_UPDATE_COMPLETED`` -> ``{"symbols_count": int, "bars_written": int, "failed_count": int}``
      （名录更新 :meth:`UpdateService.update_symbols` 的 COMPLETED 会额外带 ``"scope": "symbols"``）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from app.core.config import get_config
from app.core.event_bus import get_bus
from app.data.datasource.base import (
    DataSource,
    DataSourceError,
    get_source,
    is_index_symbol,
)
from app.data.store.db import Database, get_db
from app.data.universe import DEFAULT_INDEX_CODE, quarter_ends, to_date_str

logger = logging.getLogger(__name__)

# ---- 事件名 ---- #
EVT_STARTED: str = "DATA_UPDATE_STARTED"
EVT_PROGRESS: str = "DATA_UPDATE_PROGRESS"
EVT_FAILED: str = "DATA_UPDATE_FAILED"
EVT_COMPLETED: str = "DATA_UPDATE_COMPLETED"

# meta 键：最后更新时间。
META_LAST_UPDATE: str = "last_update_time"

# meta 键：股票名录最后更新时间（与 K 线更新时间相互独立）。
META_STOCK_LIST_UPDATE: str = "stock_list_update_time"

# 首次更新（无历史）默认起始日期；可用配置键 ``update.default_start`` 覆盖。
DEFAULT_START: str = "2015-01-01"

# 自选股为空时的内置演示清单。
DEMO_SYMBOLS: tuple[str, ...] = ("000001", "600000", "600519")

# 日线每次落库的复权口径集合（R13：**同时**落未复权 ``""`` 与前复权 ``"qfq"`` 两套，
# 保证涨跌停/缺口判定、指标编辑器、选股引擎各自能取到所需口径，缺一不可）。
_DAILY_ADJUSTS: tuple[str, ...] = ("", "qfq")

# 批量更新的分钟周期（P6 优化：模拟训练可选周期，baostock/akshare 均直接提供 4 个周期）。
MINUTE_PERIODS: tuple[str, ...] = ("5m", "15m", "30m", "60m")

# 成分股快照自动采样的默认起始日（P7-S4：与阶段 1~3 的研究起点对齐）。
INDEX_SNAPSHOT_START: str = "2015-03-31"


@dataclass
class UpdateReport:
    """一次更新任务的结果报告。

    Attributes:
        symbols_count: 本次处理的标的数（symbol 级，不随复权口径翻倍）。
        bars_written: 落库行数，为**未复权 + 前复权两套口径写入行数之和**。
        failed: 失败的 ``(symbol, reason)`` 清单；某 symbol 任一套口径失败即计入一次。
        ok: 是否全部成功（``failed`` 为空即 ``True``）。
    """

    symbols_count: int = 0
    bars_written: int = 0
    failed: List[tuple[str, str]] = field(default_factory=list)
    ok: bool = True

    def __post_init__(self) -> None:
        """根据失败清单推导 ``ok``（无失败即视为成功）。"""
        self.ok = len(self.failed) == 0


def _fmt_date(value: Any) -> str:
    """把日期值规范为 ``YYYY-MM-DD`` 文本。"""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)


# 「原始异常特征 -> 面向用户的人话」映射表（**按顺序匹配，先到先得**）。
# 面向 UI 只讲人话；原始异常详情仍由调用处写日志，便于诊断。
_FRIENDLY_ERROR_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("unable to connect to proxy", "proxyerror"), "网络代理不可用"),
    (
        ("remote end closed connection", "connection aborted", "connectionreseterror"),
        "连接被中断（服务端或中间设备重置）",
    ),
    (
        ("getaddrinfo failed", "gaierror", "name or service not known", "nodename nor servname"),
        "域名解析失败",
    ),
    (("timed out", "timeout", "readtimeout", "connecttimeout"), "连接超时"),
    (("max retries exceeded",), "连接失败（重试耗尽）"),
)

# 未命中映射时的人话截断长度。
_FRIENDLY_ERROR_MAX_LEN: int = 80


def _root_cause(exc: object) -> object:
    """取出异常链**最内层的原始异常**。

    ``DataSourceError`` 会把源异常包在 ``.original`` 里，而面向用户的关键词只存在于最内层：
    直接对聚合异常取 ``str()`` 只会得到 ``"[源名] 描述"``（描述里已不含原始关键词），
    结果就是人话映射不命中、把一长串内部文案原样甩给用户。

    Args:
        exc: 任意异常（或非异常对象）。

    Returns:
        最内层原始异常；入参无 ``original`` 链时返回入参本身。带环保护。
    """
    seen: set[int] = set()
    current: object = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        original = getattr(current, "original", None)
        if original is None:
            break
        current = original
    return current


def friendly_error(exc: object) -> str:
    """把数据源异常转成面向用户的一句话人话（**不丢原始信息**，原详情仍应写日志）。

    会先经 :func:`_root_cause` 解包到最内层原始异常再匹配，因此既可直接传
    ``DataSourceError``，也可传其 ``.original``。

    Args:
        exc: 异常对象或字符串（``DataSourceError`` 的 ``str()`` 形如
            ``"[akshare] HTTPSConnectionPool(...)"``，匹配需能穿透此前缀）。

    Returns:
        命中的中文短语；未命中则返回截断后的原文（超出部分以 ``…`` 结尾）。
    """
    exc = _root_cause(exc)
    text = "" if exc is None else str(exc)
    lowered = text.lower()
    for keywords, message in _FRIENDLY_ERROR_PATTERNS:
        if any(keyword in lowered for keyword in keywords):
            return message
    collapsed = " ".join(text.split())
    if not collapsed:
        return "未知错误"
    if len(collapsed) > _FRIENDLY_ERROR_MAX_LEN:
        return collapsed[:_FRIENDLY_ERROR_MAX_LEN] + "…"
    return collapsed


class UpdateService:
    """数据更新编排服务。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        primary_name: Optional[str] = None,
        fallback_name: Optional[str] = None,
        sources: Optional[Dict[str, DataSource]] = None,
    ) -> None:
        """初始化更新服务。

        Args:
            db: 存储实例；默认 :func:`get_db`（连带建库建表）。
            primary_name: 主数据源名；默认取配置 ``data.source.primary``。
            fallback_name: 兜底数据源名；默认取配置 ``data.source.fallback``。
            sources: 可注入的 ``{name: DataSource}``（测试用）；提供时**仅**从中取源。
        """
        cfg = get_config()
        self._db: Database = db if db is not None else get_db()
        self._sources: Optional[Dict[str, DataSource]] = sources
        self._primary_name: str = primary_name or str(
            cfg.get("data.source.primary", "baostock")
        )
        self._fallback_name: str = fallback_name or str(
            cfg.get("data.source.fallback", "akshare")
        )
        self._minute_name: str = str(cfg.get("data.source.minute", "pytdx"))
        self._batch_size: int = int(cfg.get("update.batch_size", 50) or 50)
        self._retries: int = 1  # 主源失败后额外重试次数
        self._default_start: str = str(
            cfg.get("update.default_start", DEFAULT_START) or DEFAULT_START
        )

    # ------------------------------------------------------------------ #
    # 数据源解析
    # ------------------------------------------------------------------ #
    def _resolve_source(self, name: str) -> Optional[DataSource]:
        """按名称解析数据源；注入 ``sources`` 时仅从注入表取。"""
        if self._sources is not None:
            return self._sources.get(name)
        try:
            return get_source(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("数据源 %r 不可用：%s", name, exc)
            return None

    def _daily_sources(self) -> List[tuple[str, DataSource]]:
        """返回按优先级排序的（去重）日线数据源列表：主源 → 兜底源。"""
        ordered: List[tuple[str, DataSource]] = []
        seen: set[int] = set()
        for name in (self._primary_name, self._fallback_name):
            if not name:
                continue
            src = self._resolve_source(name)
            if src is None or id(src) in seen:
                continue
            seen.add(id(src))
            ordered.append((name, src))
        return ordered

    # ------------------------------------------------------------------ #
    # 增量起始
    # ------------------------------------------------------------------ #
    def _resolve_start(self, symbol: str, adjust: str, explicit: Optional[str]) -> str:
        """计算该 symbol 的起始日期：显式优先；否则已有数据取次日，无数据取默认。"""
        if explicit is not None:
            return _fmt_date(explicit)
        existing = self._db.get_bars(symbol, "d", adjust=adjust or "")
        if existing is None or existing.empty:
            return self._default_start
        max_dt = pd.to_datetime(existing["datetime"], errors="coerce").max()
        if pd.isna(max_dt):
            return self._default_start
        nxt = (pd.Timestamp(max_dt) + pd.Timedelta(days=1)).normalize()
        return nxt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ #
    # 取数（重试 + 兜底）
    # ------------------------------------------------------------------ #
    def _fetch_with_fallback(
        self, symbol: str, period: str, start: str, end: str, adjust: str
    ) -> tuple[pd.DataFrame, str]:
        """按「主源重试 → 兜底源」顺序取数。

        Returns:
            ``(bars_df, source_name)``。

        Raises:
            DataSourceError: 所有数据源均失败。
        """
        last_exc: Optional[Exception] = None
        primary_exc: Optional[Exception] = None
        for name, src in self._daily_sources():
            attempts = 1 + (self._retries if name == self._primary_name else 0)
            for attempt in range(max(1, attempts)):
                try:
                    frame = src.get_kline(symbol, period, start, end, adjust)
                    return frame, name
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if name == self._primary_name:
                        primary_exc = exc
                    logger.warning(
                        "数据源 %s 拉取 %s(%s) 失败（第 %d 次）：%s",
                        name, symbol, period, attempt + 1, exc,
                    )
        # 面向用户的报错优先反映**主源**的真实原因（而非最后一个兜底源的），
        # 否则当主源是「起止日期无效」这类参数错误、兜底源是网络错误时，
        # 用户会看到兜底源的「连接被中断」而误判根因。
        report_exc = primary_exc or last_exc or RuntimeError("无可用数据源")
        raise DataSourceError(
            self._primary_name or "update",
            report_exc,
            f"所有数据源拉取 {symbol} 失败：{friendly_error(report_exc)}",
        )

    # ------------------------------------------------------------------ #
    # 落库
    # ------------------------------------------------------------------ #
    def _persist(
        self, symbol: str, period: str, frame: Optional[pd.DataFrame], adjust: str
    ) -> int:
        """把 Bar 契约 DataFrame 落库（按 ``update.batch_size`` 分批写入）。

        Returns:
            写入行数。
        """
        if frame is None or frame.empty:
            return 0
        data = frame.copy()
        data["symbol"] = symbol
        data["period"] = period
        data["adjust"] = adjust or ""
        written = 0
        batch = max(1, self._batch_size)
        for begin in range(0, len(data), batch):
            chunk = data.iloc[begin : begin + batch]
            written += self._db.upsert_bars(chunk)
        return written

    # ------------------------------------------------------------------ #
    # 符号解析
    # ------------------------------------------------------------------ #
    def _resolve_symbols(self, symbols: Optional[Sequence[str]]) -> List[str]:
        """确定本次要更新的 symbol 列表（显式 > 自选股去重 > 内置演示清单）。"""
        if symbols is not None:
            result: List[str] = []
            seen: set[str] = set()
            for s in symbols:
                key = str(s)
                if key and key not in seen:
                    seen.add(key)
                    result.append(key)
            return result
        items = self._db.list_watch_items()
        result = []
        seen = set()
        for item in items:
            if item.symbol not in seen:
                seen.add(item.symbol)
                result.append(item.symbol)
        if not result:
            logger.info("自选股为空，使用内置演示清单：%s", list(DEMO_SYMBOLS))
            result = list(DEMO_SYMBOLS)
        return result

    # ------------------------------------------------------------------ #
    # 更新入口
    # ------------------------------------------------------------------ #
    def update_daily(
        self,
        symbols: Optional[List[str]] = None,
        adjust: str = "qfq",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> UpdateReport:
        """更新日线并落库（R13：**同时落未复权 ``""`` 与前复权 ``"qfq"`` 两套**）。

        Args:
            symbols: 标的列表；``None`` 取自定义股（空则用演示清单）。
            adjust: 兼容保留参数（历史调用仍可传入），**不再改变落库口径**：本方法
                始终遍历 :data:`_DAILY_ADJUSTS`（``""``/``"qfq"``）各落一套。
            start: 显式起始日期；``None`` 时按增量规则计算（**每套口径独立计算**，
                各自从各自口径库内 ``max(dt)+1`` 起始）。
            end: 结束日期；``None`` 时取今天。

        Returns:
            :class:`UpdateReport`（不抛异常，失败计入 ``failed``；``bars_written``
            为两套口径写入行数之和）。

        失败语义（``(symbol, adjust)`` 粒度）：
            * 每套 adjust 独立 try/except：某套拉取/落库失败**只跳过该套、继续该 symbol
              的另一套**，不互相影响；
            * 只要该 symbol 存在任一套失败，即记入 ``failed``（reason 注明失败的复权口径
              与人话原因）；两套都成功才不记 ``failed``；
            * ``EVT_FAILED`` 事件按 symbol 级发出一次（即使两套都失败也只发一次）。
        """
        bus = get_bus()
        try:
            resolved = self._resolve_symbols(symbols)
        except Exception as exc:  # noqa: BLE001 - 不向 UI 抛
            logger.exception("解析更新标的失败")
            resolved = []
        total = len(resolved)
        end_date = _fmt_date(end) if end is not None else date.today().strftime("%Y-%m-%d")
        bus.publish(EVT_STARTED, {"source": self._primary_name, "scope": "daily"})

        bars_written = 0
        failed: List[tuple[str, str]] = []
        current = 0
        try:
            for symbol in resolved:
                current += 1
                bus.publish(
                    EVT_PROGRESS, {"symbol": symbol, "current": current, "total": total}
                )
                failed_adjusts: List[str] = []
                # 指数（如 sh.000300）无除权概念，只落未复权一套，避免重复请求与重复行。
                adjusts = ("",) if is_index_symbol(symbol) else _DAILY_ADJUSTS
                for adjust_name in adjusts:
                    try:
                        eff_start = self._resolve_start(symbol, adjust_name, start)
                        # 增量起始已越过结束日期 ⇒ 该套库内数据已是最新（如本机已含今日），
                        # 无需再请求，否则 baostock 会直接报「起始日期大于终止日期」。
                        if eff_start > end_date:
                            logger.info(
                                "日线已是最新（口径=%s 起始=%s 晚于结束=%s），跳过 %s",
                                adjust_name, eff_start, end_date, symbol,
                            )
                            continue
                        frame, used = self._fetch_with_fallback(
                            symbol, "d", eff_start, end_date, adjust_name
                        )
                        written = self._persist(symbol, "d", frame, adjust_name)
                        bars_written += written
                        logger.info(
                            "日线更新完成：%s 口径=%s 起始=%s 行数=%d 源=%s",
                            symbol, adjust_name, eff_start, written, used,
                        )
                    except Exception as exc:  # noqa: BLE001 - 单套失败不中断另一套
                        label = "不复权" if adjust_name == "" else adjust_name
                        failed_adjusts.append(f"{label}:{friendly_error(exc)}")
                        logger.warning(
                            "日线更新失败：%s 口径=%s -> %s", symbol, adjust_name, exc,
                        )
                if failed_adjusts:
                    reason = "；".join(failed_adjusts)
                    failed.append((symbol, reason))
                    bus.publish(EVT_FAILED, {"symbol": symbol, "error": reason})
        finally:
            self._touch_last_update()
            bus.publish(
                EVT_COMPLETED,
                {
                    "symbols_count": total,
                    "bars_written": bars_written,
                    "failed_count": len(failed),
                },
            )
        return UpdateReport(
            symbols_count=total, bars_written=bars_written, failed=failed
        )

    def update_minutes(
        self,
        symbols: Optional[Sequence[str]] = None,
        periods: Sequence[str] = MINUTE_PERIODS,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> UpdateReport:
        """更新分钟线（默认 5/15/30/60m **一次拉全**）并落库（走主/兜底源的 ``get_kline``）。

        背景（R12/R3 缓解）：pytdx / akshare 的 1 分钟链路在当前环境不可用，
        而 baostock 的 5/15/30/60 分钟接口实测可用。P6 优化后模拟训练支持周期选择，
        故一次把 4 个分钟周期拉到本地，回放时直接取库内数据（缺失时 UI 再本地派生）。

        Args:
            symbols: 标的列表；``None`` 取自选股（空则演示清单）。
            periods: 目标分钟周期集合；默认 :data:`MINUTE_PERIODS`，非分钟周期自动忽略。
            start: 起始日期 ``YYYY-MM-DD``；``None`` 默认近 10 个自然日（供回放足够）。
            end: 结束日期；``None`` 取今天。

        Returns:
            :class:`UpdateReport`（不抛异常；某标的**任一周期**失败即计入 ``failed``，
            ``bars_written`` 为各周期写入行数之和）。
        """
        bus = get_bus()
        try:
            resolved = self._resolve_symbols(symbols)
        except Exception as exc:  # noqa: BLE001 - 不向 UI 抛
            logger.exception("解析分钟更新标的失败")
            resolved = []
        wanted = [p for p in periods if p in MINUTE_PERIODS]
        total = len(resolved)
        end_date = _fmt_date(end) if end is not None else date.today().strftime("%Y-%m-%d")
        start_date = _fmt_date(start) if start is not None else self._minute_default_start(end_date)
        bus.publish(EVT_STARTED, {"source": self._primary_name, "scope": "minutes"})

        bars_written = 0
        failed: List[tuple[str, str]] = []
        current = 0
        for symbol in resolved:
            current += 1
            bus.publish(EVT_PROGRESS, {"symbol": symbol, "current": current, "total": total})
            failed_periods: List[str] = []
            for period in wanted:
                try:
                    frame, used = self._fetch_with_fallback(
                        symbol, period, start_date, end_date, ""
                    )
                    written = self._persist(symbol, period, frame, "")
                    bars_written += written
                    logger.info(
                        "%s 分钟线更新完成：%s 起始=%s 行数=%d 源=%s",
                        period, symbol, start_date, written, used,
                    )
                except Exception as exc:  # noqa: BLE001 - 单周期失败不中断其余周期
                    failed_periods.append(f"{period}:{friendly_error(exc)}")
                    logger.warning("%s 分钟线更新失败：%s -> %s", period, symbol, exc)
            if failed_periods:
                reason = "；".join(failed_periods)
                failed.append((symbol, reason))
                bus.publish(EVT_FAILED, {"symbol": symbol, "error": reason})
        self._touch_last_update()
        bus.publish(
            EVT_COMPLETED,
            {"symbols_count": total, "bars_written": bars_written, "failed_count": len(failed)},
        )
        return UpdateReport(symbols_count=total, bars_written=bars_written, failed=failed)

    @staticmethod
    def _minute_default_start(end_date: str) -> str:
        """分钟数据的默认起始日：结束日前 10 个自然日（回放够用且请求量小）。"""
        try:
            end_ts = pd.Timestamp(end_date)
        except Exception:  # noqa: BLE001 - 非法日期回退
            return "2026-01-01"
        return (end_ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    def update_minute(self, symbol: str, date_: str) -> UpdateReport:
        """更新某标的一交易日的 1 分钟线并落库（走 ``data.source.minute``）。

        Args:
            symbol: 标的代码。
            date_: 交易日 ``YYYY-MM-DD``。

        Returns:
            :class:`UpdateReport`（不抛异常）。
        """
        bus = get_bus()
        bus.publish(EVT_STARTED, {"source": self._minute_name, "scope": "minute"})
        bars_written = 0
        failed: List[tuple[str, str]] = []
        bus.publish(EVT_PROGRESS, {"symbol": symbol, "current": 1, "total": 1})
        try:
            source = self._resolve_source(self._minute_name)
            if source is None:
                raise DataSourceError(self._minute_name, RuntimeError("不可用"), "分钟数据源不可用")
            last_exc: Optional[Exception] = None
            frame: Optional[pd.DataFrame] = None
            for attempt in range(1 + self._retries):
                try:
                    frame = source.get_minute(symbol, date_)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning("分钟数据源拉取 %s(%s) 失败（第 %d 次）：%s",
                                   symbol, date_, attempt + 1, exc)
            if frame is None:
                raise DataSourceError(
                    self._minute_name, last_exc or RuntimeError("失败"),
                    f"分钟数据拉取 {symbol} 失败",
                )
            bars_written = self._persist(symbol, "1m", frame, "")
            logger.info("分钟线更新完成：%s %s 行数=%d", symbol, date_, bars_written)
        except Exception as exc:  # noqa: BLE001 - 不向 UI 抛
            friendly = friendly_error(exc)
            failed.append((symbol, friendly))
            bus.publish(EVT_FAILED, {"symbol": symbol, "error": friendly})
            logger.warning("分钟线更新失败：%s -> %s", symbol, exc)
        finally:
            self._touch_last_update()
            bus.publish(
                EVT_COMPLETED,
                {"symbols_count": 1, "bars_written": bars_written, "failed_count": len(failed)},
            )
        return UpdateReport(symbols_count=1, bars_written=bars_written, failed=failed)

    # ------------------------------------------------------------------ #
    # 名录更新
    # ------------------------------------------------------------------ #
    def update_symbols(self) -> UpdateReport:
        """更新全 A 股代码与名称名录（P6 优化）。

        按 :meth:`_daily_sources` 顺序尝试各数据源的 :meth:`DataSource.list_stocks`，
        第一个成功（返回非空表）即落库并更新 ``meta['stock_list_update_time']``。
        **不调用** :meth:`_touch_last_update`（后者语义为 K 线更新时间）。

        Returns:
            :class:`UpdateReport`：成功时 ``symbols_count == bars_written == 名录行数``；
            全部数据源失败时 ``failed`` 含 ``("", 人话原因)``、``ok`` 为 ``False``。
        """
        bus = get_bus()
        try:
            last_exc: Optional[Exception] = None
            for name, src in self._daily_sources():
                try:
                    frame = src.list_stocks()
                except Exception as exc:  # noqa: BLE001 - 单个源失败继续兜底
                    last_exc = exc
                    logger.warning("数据源 %s 拉取名录失败：%s", name, exc)
                    continue
                if frame is None or frame.empty:
                    last_exc = last_exc or RuntimeError("名录为空")
                    logger.warning("数据源 %s 名录为空", name)
                    continue
                count = self._db.upsert_stock_list(frame)
                self._db.set_meta(
                    META_STOCK_LIST_UPDATE, datetime.now().isoformat(timespec="seconds")
                )
                bus.publish(EVT_STARTED, {"source": name, "scope": "symbols"})
                bus.publish(
                    EVT_COMPLETED,
                    {
                        "symbols_count": count,
                        "bars_written": count,
                        "failed_count": 0,
                        "scope": "symbols",
                    },
                )
                logger.info("名录更新完成：源=%s 行数=%d", name, count)
                return UpdateReport(symbols_count=count, bars_written=count)
            # 全部数据源失败 / 空表。
            reason = friendly_error(last_exc) if last_exc is not None else "无可用数据源"
            failed: List[tuple[str, str]] = [("", reason)]
            bus.publish(EVT_FAILED, {"symbol": "", "error": reason})
            bus.publish(
                EVT_COMPLETED,
                {
                    "symbols_count": 0,
                    "bars_written": 0,
                    "failed_count": 1,
                    "scope": "symbols",
                },
            )
            logger.warning("名录更新失败：%s", reason)
            return UpdateReport(symbols_count=0, bars_written=0, failed=failed)
        except Exception as exc:  # noqa: BLE001 - 绝不向 UI 抛
            friendly = friendly_error(exc)
            logger.exception("名录更新异常")
            bus.publish(EVT_FAILED, {"symbol": "", "error": friendly})
            bus.publish(
                EVT_COMPLETED,
                {
                    "symbols_count": 0,
                    "bars_written": 0,
                    "failed_count": 1,
                    "scope": "symbols",
                },
            )
            return UpdateReport(symbols_count=0, bars_written=0, failed=[("", friendly)])

    # ------------------------------------------------------------------ #
    # 指数成分股（P7-S4）
    # ------------------------------------------------------------------ #
    def _fetch_constituents(self, index_code: str, day: str) -> Optional[pd.DataFrame]:
        """按「主源 → 兜底源」顺序拉取一份成分股名单快照；全部失败返回 ``None``。

        数据源若无 ``get_index_constituents``（如 pytdx）自动跳过。
        """
        for name, src in self._daily_sources():
            getter = getattr(src, "get_index_constituents", None)
            if getter is None:
                continue
            try:
                frame = getter(index_code=index_code, date_=day)
            except Exception as exc:  # noqa: BLE001 - 单源失败继续兜底
                logger.warning(
                    "数据源 %s 拉取 %s 成分股(%s) 失败：%s", name, index_code, day, exc
                )
                continue
            if frame is not None and not frame.empty:
                return frame
        return None

    def update_index_constituents(
        self,
        index_code: str = DEFAULT_INDEX_CODE,
        dates: Optional[Sequence[Any]] = None,
        start: str = INDEX_SNAPSHOT_START,
        end: Optional[str] = None,
    ) -> UpdateReport:
        """拉取并落库指数成分股名单**快照**（P7-S4）。

        为什么按「快照」而不是「一份最新名单」：指数每半年调仓，只有拿到**历史多份**
        名单才能做 PIT 回溯（某交易日只用当时生效的那份），进而量化幸存者偏差。

        Args:
            index_code: 指数代码，默认 ``sh.000300``（沪深300）。
            dates: 显式的快照查询日列表；``None`` 时按季末自动生成 ``start``~``end``。
            start: 自动生成查询日的起始（默认 :data:`INDEX_SNAPSHOT_START`）。
            end: 自动生成查询日的结束；``None`` 取今天。

        Returns:
            :class:`UpdateReport`：``symbols_count`` = 实际查询的快照份数；
            ``bars_written`` = 落库行数（数据源会把同一份名单在多个查询日重复返回，
            已按 ``effective_date`` 去重，故不等于「份数 × 300」）；失败计入
            ``failed``，**不抛异常**。
        """
        bus = get_bus()
        query_dates = (
            [to_date_str(value) for value in dates]
            if dates is not None
            else quarter_ends(start, end)
        )
        query_dates = [value for value in query_dates if value]
        total = len(query_dates)
        bus.publish(
            EVT_STARTED, {"source": self._primary_name, "scope": "index_constituent"}
        )

        written = 0
        failed: List[tuple[str, str]] = []
        seen_effective: set[str] = set()
        try:
            for position, day in enumerate(query_dates, start=1):
                label = f"{index_code}@{day}"
                bus.publish(
                    EVT_PROGRESS, {"symbol": label, "current": position, "total": total}
                )
                frame = self._fetch_constituents(index_code, day)
                if frame is None or frame.empty:
                    failed.append((label, "未取到成分股名单"))
                    bus.publish(EVT_FAILED, {"symbol": label, "error": "未取到成分股名单"})
                    continue
                effective_dates = {
                    str(value) for value in frame["effective_date"].tolist()
                }
                if effective_dates and effective_dates.issubset(seen_effective):
                    # 数据源返回的还是已落库的那份名单（查询日未跨越下一次调仓），跳过。
                    continue
                try:
                    written += self._db.upsert_index_constituents(frame)
                except Exception as exc:  # noqa: BLE001 - 单份失败不中断整体
                    reason = friendly_error(exc)
                    failed.append((label, reason))
                    bus.publish(EVT_FAILED, {"symbol": label, "error": reason})
                    logger.warning("成分股落库失败：%s -> %s", label, exc)
                    continue
                seen_effective |= effective_dates
        finally:
            bus.publish(
                EVT_COMPLETED,
                {
                    "symbols_count": total,
                    "bars_written": written,
                    "failed_count": len(failed),
                    "scope": "index_constituent",
                },
            )
        logger.info(
            "成分股快照更新完成：%s 查询 %d 份 / 落库 %d 行 / 快照 %d 份 / 失败 %d",
            index_code, total, written, len(seen_effective), len(failed),
        )
        return UpdateReport(symbols_count=total, bars_written=written, failed=failed)

    # ------------------------------------------------------------------ #
    # 元数据
    # ------------------------------------------------------------------ #
    def _touch_last_update(self) -> None:
        """把当前时间写入 ``meta['last_update_time']``（ISO）。"""
        try:
            stamp = datetime.now().isoformat(timespec="seconds")
            self._db.set_meta(META_LAST_UPDATE, stamp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入 last_update_time 失败：%s", exc)

    def last_update_time(self) -> Optional[str]:
        """读取 ``meta['last_update_time']``；不存在返回 ``None``。"""
        try:
            return self._db.get_meta(META_LAST_UPDATE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 last_update_time 失败：%s", exc)
            return None
