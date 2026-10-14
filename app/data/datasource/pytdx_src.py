"""pytdx 数据源实现（专责 1 分钟历史 K 线）。

选型与版本
----------
* 采用 **pytdx 1.72**（dist 版本号；``importlib.metadata.version('pytdx')`` 报 ``1.72``）。
  经 venv 内自省，``pytdx`` 在 **Python 3.13.14** 下 ``import`` 正常
  （``from pytdx.hq import TdxHq_API`` 成功），故 **无需改用 mootdx**。
* 常量位于 ``pytdx.params.TDXParams``（**注意：并无 ``pytdx.consts`` 模块**）：
  ``KLINE_TYPE_1MIN = 8``（另有 ``KLINE_TYPE_EXHQ_1MIN = 7`` 为扩展时段 1 分钟，
  本实现使用标准 1 分钟 **8**）；``MARKET_SZ = 0`` / ``MARKET_SH = 1``；
  ``MAX_KLINE_COUNT = 800``（单次请求上限，即分页页大小）。
* 服务器列表取自 ``pytdx.config.hosts.hq_hosts``（共 104 台，元素为 ``(name, ip, port)``，
  端口 7709）。

能力边界
--------
* **仅提供 1 分钟历史 K 线**（``get_kline(symbol, '1m', ...)`` 与 ``get_minute``）。
  其余周期（d/w/m/5m…）返回 Bar 契约空表并提示请用 akshare/baostock；
  ``list_stocks`` / ``get_fundamental`` / ``get_spot`` / ``get_adj_factor`` 同理不是本源的职责，
  返回空结果并记录日志。
* 网络不可用时支持 **本地通达信 ``.lc1`` 解析兜底**：``read_local_minute(symbol, date)``，
  从 ``<tdx_dir>/vipdoc/{sh,sz}/minline/{sh,sz}<code>.lc1`` 读取；
  ``tdx_dir`` 取自配置键 ``data.tdx_dir``（缺省则不读，直接返回空表）。

⚠️ 成交量单位（跨源一致性，遵循《开发手册》§4.2「统一为『手』」）
    * **网络** ``get_security_bars`` 的 ``vol``：A 股单位为「手」，**无需换算**即符合项目口径
      （依据：pytdx 分笔示例 ``vol`` 注释「单位是『手』」；通达信↔内部单位对照表亦标注成交量
      「TDX 返回=手、无需转换」）。
    * **本地 ``.lc1``**：pytdx ``TdxMinBarReader`` 源码明确注释「注意成交量单位不是手」，
      且字节布局注释为「成交量（股）」，故本实现 **÷100 转为「手」**。
      （本地无真实样本，此换算依据 pytdx 源码注释，标注「待真机验证」。）

错误处理：所有对外方法内部异常统一包装为 :class:`DataSourceError`（保留异常链）。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from app.data.datasource.base import (
    BAR_COLUMNS,
    DataSourceError,
    derive_market,
    normalize_bars,
    register_source,
)

logger = logging.getLogger(__name__)

# ---- 导入 pytdx（失败时降级为可读错误，不影响本模块被 import） ---- #
try:  # pragma: no cover - 环境相关
    import pytdx  # noqa: F401
    from pytdx.config.hosts import hq_hosts as _HQ_HOSTS
    from pytdx.hq import TdxHq_API
    from pytdx.params import TDXParams
    from pytdx.reader import TdxMinBarReader

    _PYTDX_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover - 环境相关
    TdxHq_API = None  # type: ignore[assignment]
    TDXParams = None  # type: ignore[assignment]
    TdxMinBarReader = None  # type: ignore[assignment]
    _HQ_HOSTS = []  # type: ignore[assignment]
    _PYTDX_IMPORT_ERROR = _exc

try:  # pragma: no cover - 元数据相关
    from importlib.metadata import version as _pkg_version

    PYTDX_VERSION: str = _pkg_version("pytdx")
except Exception:  # pragma: no cover - 环境相关
    PYTDX_VERSION = "unknown"

# K 线类别 / 市场常量：优先取 pytdx 内置常量，缺失时用文档化的字面值兜底。
_KLINE_1MIN: int = getattr(TDXParams, "KLINE_TYPE_1MIN", 8)
_MARKET_SZ: int = getattr(TDXParams, "MARKET_SZ", 0)
_MARKET_SH: int = getattr(TDXParams, "MARKET_SH", 1)
_MAX_PER_PAGE: int = getattr(TDXParams, "MAX_KLINE_COUNT", 800)

# 每交易日 1 分钟 K 线上限（4 小时 × 60）。
_BARS_PER_DAY: int = 240

# 分页安全上限（防止向极早日期无限翻页）：500 页 × 800 根 ≈ 400k 根。
_MAX_PAGES_DEFAULT: int = 500

# Fundamental 全部字段（仅用于「非本源职责」时返回结构稳定的空骨架）。
_FUNDAMENTAL_KEYS: tuple[str, ...] = (
    "symbol",
    "name",
    "industry",
    "list_date",
    "total_market_cap",
    "float_market_cap",
    "pe",
    "pb",
    "roe",
    "revenue_yoy",
    "report_period",
    "summary",
)


def _to_tdx_code(symbol: str) -> tuple[int, str]:
    """把 ``000001`` / ``sh600000`` 转为 ``(market_int, code6)``。

    Args:
        symbol: 标的代码。

    Returns:
        ``(market, code)``，其中 ``market`` 为 ``MARKET_SH``/``MARKET_SZ``；
        ``code`` 为 6 位数字。

    Notes:
        pytdx 协议仅定义沪深两市常量（``MARKET_SZ=0`` / ``MARKET_SH=1``）。
        北交所（``bj``）无独立常量，暂归入 ``MARKET_SZ``，已在日志层面可辨。
    """
    market_name = derive_market(symbol)
    match = re.search(r"(\d{6})", str(symbol))
    code = match.group(1) if match else str(symbol)
    if market_name == "sh":
        return _MARKET_SH, code
    return _MARKET_SZ, code


def _day_start(value: Any) -> pd.Timestamp:
    """把日期/时间值归到当日 ``00:00:00``。"""
    return pd.Timestamp(pd.to_datetime(value, errors="coerce")).normalize()


def _day_end(value: Any) -> pd.Timestamp:
    """把日期/时间值归到当日 ``23:59:59``，用于包含整日范围。"""
    return _day_start(value) + pd.Timedelta(hours=23, minutes=59, seconds=59)


class PytdxSource:
    """基于 pytdx 的 1 分钟历史数据源实现（网络分页 + 本地 .lc1 兜底）。"""

    name: str = "pytdx"

    def __init__(
        self,
        hosts: Optional[Sequence[Sequence[Any]]] = None,
        timeout: float = 5.0,
        page_size: int = _MAX_PER_PAGE,
        max_pages: int = _MAX_PAGES_DEFAULT,
    ) -> None:
        """初始化（懒连接，首次取数时才尝试连接服务器）。

        Args:
            hosts: 服务器列表 ``[(name, ip, port), ...]``；默认取 ``hq_hosts``。
                依次尝试，失败自动切下一台（failover）。
            timeout: 单台连接超时（秒）。
            page_size: 分页页大小，默认 ``MAX_KLINE_COUNT``（800）。
            max_pages: 分页安全上限。
        """
        self._hosts: list[Sequence[Any]] = (
            list(hosts) if hosts is not None else list(_HQ_HOSTS)  # type: ignore[arg-type]
        )
        self._timeout: float = timeout
        self._page_size: int = int(page_size)
        self._max_pages: int = int(max_pages)
        self._api: Any = None
        self._connected: bool = False
        self._version: str = PYTDX_VERSION

    # ------------------------------------------------------------------ #
    # 连接管理（懒连接 + 复用 + failover）
    # ------------------------------------------------------------------ #
    def _ensure_connected(self) -> Any:
        """确保已连接，返回可用的 pytdx API 实例。

        Returns:
            已连接的 ``TdxHq_API`` 实例。

        Raises:
            DataSourceError: pytdx 不可用，或所有服务器均连接失败。
        """
        if self._connected and self._api is not None:
            return self._api
        if TdxHq_API is None:  # pragma: no cover - 环境相关
            raise DataSourceError(
                self.name,
                _PYTDX_IMPORT_ERROR or RuntimeError("pytdx 不可用"),
                "pytdx 未能导入，请检查安装",
            )
        last_exc: Optional[Exception] = None
        for entry in self._hosts:
            try:
                name, ip, port = entry[0], entry[1], entry[2]
            except (IndexError, TypeError):  # pragma: no cover - 防御
                continue
            api = TdxHq_API()
            try:
                if not api.connect(ip, port, time_out=self._timeout):
                    logger.debug("pytdx 连接失败：%s:%s(%s)", ip, port, name)
                    continue
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.debug("pytdx 连接异常 %s:%s(%s) -> %s", ip, port, name, exc)
                try:
                    api.disconnect()
                except Exception:  # noqa: BLE001
                    pass
                continue
            self._api = api
            self._connected = True
            logger.info("pytdx 已连接：%s:%s(%s)", ip, port, name)
            return api
        raise DataSourceError(
            self.name,
            last_exc or RuntimeError("无可用通达信服务器"),
            "所有通达信服务器均连接失败",
        )

    def close(self) -> None:
        """断开连接（幂等）。"""
        if self._api is not None:
            try:
                self._api.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.warning("pytdx 断开异常：%s", exc)
            finally:
                self._api = None
                self._connected = False

    def __enter__(self) -> "PytdxSource":
        """支持 ``with`` 用法（自动连接）。"""
        self._ensure_connected()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """退出 ``with`` 时断开。"""
        self.close()

    # ------------------------------------------------------------------ #
    # 分页内核（get_kline('1m') 与 get_minute 复用）
    # ------------------------------------------------------------------ #
    def _paginate(
        self,
        api: Any,
        category: int,
        market: int,
        code: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
    ) -> pd.DataFrame:
        """按页（每页 ``page_size``）向更早翻，累积 [start_dt, end_dt] 区间分钟线。

        通达信 ``start=0`` 为**最近**一根，``start += page_size`` 向更早翻页。
        当某页最旧时间已 ≤ ``start_dt``、或返回不足一页、或无数据时停止。

        Returns:
            Bar 契约 DataFrame（``datetime`` 升序、去重）。
        """
        collected: list[dict] = []
        offset = 0
        for _ in range(self._max_pages):
            page = api.get_security_bars(category, market, code, offset, self._page_size)
            if not page:
                break
            collected.extend(page)
            page_times = pd.to_datetime(
                pd.Series([b.get("datetime") for b in page]), errors="coerce"
            ).dropna()
            if not page_times.empty and page_times.min() <= start_dt:
                break
            if len(page) < self._page_size:
                break
            offset += self._page_size

        rows: list[dict] = []
        for bar in collected:
            dt = pd.to_datetime(bar.get("datetime"), errors="coerce")
            if pd.isna(dt) or dt < start_dt or dt > end_dt:
                continue
            rows.append(
                {
                    "datetime": dt,
                    "open": bar.get("open"),
                    "high": bar.get("high"),
                    "low": bar.get("low"),
                    "close": bar.get("close"),
                    # 网络 vol 单位已是「手」，无需换算。
                    "volume": bar.get("vol"),
                    "amount": bar.get("amount"),
                }
            )
        if not rows:
            return pd.DataFrame(columns=BAR_COLUMNS)
        return normalize_bars(pd.DataFrame(rows))

    # ------------------------------------------------------------------ #
    # K 线
    # ------------------------------------------------------------------ #
    def get_kline(
        self,
        symbol: str,
        period: str,
        start: Any,
        end: Any,
        adjust: str = "",
    ) -> pd.DataFrame:
        """获取 K 线。pytdx **仅支持 1 分钟**（``period='1m'``），跨日分页累积。

        Args:
            symbol: 标的代码。
            period: 周期；``'1m'`` 走网络分页，其余周期返回空表并提示。
            start: 起始日期。
            end: 结束日期。
            adjust: 复权口径（pytdx 不做复权，忽略，仅记录）。

        Returns:
            Bar 契约 DataFrame（1 分钟为真实数据，其余为空表）。
        """
        try:
            if period != "1m":
                logger.warning(
                    "pytdx 仅提供 1 分钟数据；周期 %r 请使用 akshare/baostock。", period
                )
                return pd.DataFrame(columns=BAR_COLUMNS)
            if adjust:
                logger.warning("pytdx 不支持复权（adjust=%r），返回未复权数据。", adjust)
            api = self._ensure_connected()
            market, code = _to_tdx_code(symbol)
            return self._paginate(
                api, _KLINE_1MIN, market, code, _day_start(start), _day_end(end)
            )
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("pytdx.get_kline(%s, %s) 失败", symbol, period)
            raise DataSourceError(self.name, exc) from exc

    def get_minute(self, symbol: str, date: Any) -> pd.DataFrame:
        """获取某交易日 1 分钟 K 线（\u2264 240 根，升序）。

        Args:
            symbol: 标的代码。
            date: 交易日。

        Returns:
            Bar 契约 DataFrame。
        """
        try:
            api = self._ensure_connected()
            market, code = _to_tdx_code(symbol)
            frame = self._paginate(
                api, _KLINE_1MIN, market, code, _day_start(date), _day_end(date)
            )
            if len(frame) > _BARS_PER_DAY:
                frame = frame.iloc[-_BARS_PER_DAY:].reset_index(drop=True)
            return frame
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("pytdx.get_minute(%s, %s) 失败", symbol, date)
            raise DataSourceError(self.name, exc) from exc

    # ------------------------------------------------------------------ #
    # 本地 .lc1 兜底
    # ------------------------------------------------------------------ #
    @staticmethod
    def _configured_tdx_dir() -> Optional[str]:
        """读取配置键 ``data.tdx_dir``；未配置返回 ``None``。"""
        try:
            from app.core.config import get_config

            value = get_config().get("data.tdx_dir")
            return str(value) if value else None
        except Exception:  # noqa: BLE001  # pragma: no cover - 防御
            return None

    def read_local_minute(
        self, symbol: str, date: Any, tdx_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """从本地通达信 ``.lc1`` 读取某交易日 1 分钟线（网络不可用时的兜底）。

        Args:
            symbol: 标的代码。
            date: 交易日。
            tdx_dir: 通达信安装目录；``None`` 时取配置 ``data.tdx_dir``（未配置则跳过）。

        Returns:
            Bar 契约 DataFrame（volume 已由「股」÷100 转为「手」）；无文件或无配置时为空表。
        """
        try:
            directory = tdx_dir if tdx_dir is not None else self._configured_tdx_dir()
            if not directory:
                logger.warning("未配置通达信目录（data.tdx_dir），跳过本地 .lc1 读取。")
                return pd.DataFrame(columns=BAR_COLUMNS)
            market, code = _to_tdx_code(symbol)
            prefix = "sh" if market == _MARKET_SH else "sz"
            path = Path(directory) / "vipdoc" / prefix / "minline" / f"{prefix}{code}.lc1"
            if not path.is_file():
                logger.warning("本地 .lc1 不存在：%s", path)
                return pd.DataFrame(columns=BAR_COLUMNS)
            reader = TdxMinBarReader()  # 模块级引用，便于测试替换
            raw = reader.get_df(str(path))
            if raw is None or raw.empty:
                return pd.DataFrame(columns=BAR_COLUMNS)
            frame = raw.reset_index()
            first_col = frame.columns[0]
            frame = frame.rename(columns={first_col: "datetime"})
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
            frame = frame[["datetime", "open", "high", "low", "close", "amount", "volume"]].copy()
            # 本地 .lc1 volume 单位为「股」→ ÷100 转「手」（待真机验证）。
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") / 100.0
            start_dt, end_dt = _day_start(date), _day_end(date)
            frame = frame[(frame["datetime"] >= start_dt) & (frame["datetime"] <= end_dt)]
            return normalize_bars(frame)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("pytdx.read_local_minute(%s, %s) 失败", symbol, date)
            raise DataSourceError(self.name, exc) from exc

    # ------------------------------------------------------------------ #
    # 非本源职责的能力
    # ------------------------------------------------------------------ #
    def list_stocks(self) -> pd.DataFrame:
        """pytdx 非股票列表提供方，返回空表。"""
        logger.warning("pytdx 不提供股票列表（list_stocks），请使用 akshare/baostock。")
        return pd.DataFrame(columns=["code", "name", "market"])

    def get_spot(self) -> pd.DataFrame:
        """pytdx 非实时快照提供方，返回空表。"""
        logger.warning("pytdx 不提供实时 spot（get_spot），请使用 akshare。")
        return pd.DataFrame(columns=["code", "name", "price", "pct_change"])

    def get_adj_factor(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        """pytdx 非复权因子提供方，返回空表。"""
        logger.warning("pytdx 不提供复权因子（get_adj_factor），请使用 akshare/baostock。")
        return pd.DataFrame(columns=["datetime", "factor"])

    def get_fundamental(self, symbol: str) -> dict:
        """pytdx 非基本面提供方，返回结构稳定的空骨架 dict（值均为 ``None``/``''``）。"""
        logger.warning("pytdx 不提供基本面（get_fundamental），请使用 akshare/baostock。")
        result: dict = {key: None for key in _FUNDAMENTAL_KEYS}
        result["symbol"] = str(symbol)
        result["industry"] = ""
        result["list_date"] = ""
        result["report_period"] = ""
        result["summary"] = ""
        return result


@register_source("pytdx")
def _create_pytdx_source() -> PytdxSource:
    """pytdx 数据源工厂（供注册表使用）。"""
    return PytdxSource()
