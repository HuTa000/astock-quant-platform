"""AkShare 数据源实现。

本实现基于 **akshare 1.18.96**（经 venv 内自省确认以下接口均存在）：
    * ``stock_info_a_code_name()``                —— 全量 A 股 code/name
    * ``stock_zh_a_hist(symbol, period, start_date, end_date, adjust, timeout)``
      —— 日/周/月线（``period ∈ {'daily','weekly','monthly'}``，``adjust ∈ {'','qfq','hfq'}``）
    * ``stock_zh_a_hist_min_em(symbol, start_date, end_date, period, adjust)``
      —— 分钟线（``period ∈ {'1','5','15','30','60'}``，时间为 ``'YYYY-MM-DD HH:MM:SS'``）
    * ``stock_zh_a_spot_em()``                    —— 实时/盘后 spot 快照
    * ``stock_individual_info_em(symbol, timeout)`` —— 个股基本信息（item/value 两列）
    * ``stock_zh_a_daily(symbol, start_date, end_date, adjust)`` —— 新浪行情，
      其中 ``adjust='qfq-factor'`` 返回 ``[date, qfq_factor]``、``'hfq-factor'`` 返回 ``[date, hfq_factor]``
    * ``stock_financial_analysis_indicator(symbol)`` —— 财务分析指标（含 ROE、营收增长率）

约定：
    * 所有 K 线返回前经 :func:`normalize_bars` 归一化为全局 Bar 契约；
    * ``get_minute``（1 分钟长历史）AkShare 不具备，返回 Bar 契约空表并记录提示（1 分钟由 P1-S5 pytdx 负责）；
    * **基本面 ``roe`` / ``revenue_yoy`` 统一为百分比数值**（如 ``4.69`` 表示 4.69%）；
      仅取 **最新报告期** 的值，最新期为空/占位符（``''``/``'-'``/``'--'``/``None``）时返回 ``None``，
      不回退到历史旧值（避免把陈旧数值误报为当前值）；
    * 所有对外方法内部异常统一包装为 :class:`DataSourceError`。
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from app.data.datasource.base import (
    BAR_COLUMNS,
    DataSourceError,
    derive_market,
    normalize_bars,
    register_source,
)

logger = logging.getLogger(__name__)

# ---- 导入 akshare（失败时给出可读错误，不影响本模块被 import） ---- #
try:  # pragma: no cover - 环境相关
    import akshare as ak

    AKSHARE_VERSION: str = getattr(ak, "__version__", "unknown")
    _AK_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover - 环境相关
    ak = None  # type: ignore[assignment]
    AKSHARE_VERSION = "unavailable"
    _AK_IMPORT_ERROR = _exc

# 周期编码 -> AkShare 参数
_DAILY_PERIOD_MAP: dict[str, str] = {"d": "daily", "w": "weekly", "m": "monthly"}
_MINUTE_PERIOD_MAP: dict[str, str] = {"5m": "5", "15m": "15", "30m": "30", "60m": "60"}

# spot 中文列 -> 统一英文列
_SPOT_RENAME: dict[str, str] = {
    "代码": "code",
    "名称": "name",
    "最新价": "price",
    "涨跌幅": "pct_change",
}

# 个股基本信息 item -> Fundamental 字段
_INFO_ITEM_MAP: dict[str, str] = {
    "股票简称": "name",
    "行业": "industry",
    "上市时间": "list_date",
    "总市值": "total_market_cap",
    "流通市值": "float_market_cap",
}

# Fundamental 全部字段（保证 get_fundamental 返回结构稳定）
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


def _parse_date(value: Any) -> datetime:
    """把多种日期表示（str/int/datetime/date）解析为 :class:`datetime`。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    if not text:
        raise ValueError("日期为空")
    text = text.replace("/", "-")
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(text)


def _to_compact(value: Any) -> str:
    """格式化为 AkShare 日线所需的 ``YYYYMMDD``。"""
    return _parse_date(value).strftime("%Y%m%d")


def _to_min_range_start(value: Any) -> str:
    """格式化为分钟线起始时间 ``YYYY-MM-DD 09:00:00``。"""
    return f"{_parse_date(value):%Y-%m-%d} 09:00:00"


def _to_min_range_end(value: Any) -> str:
    """格式化为分钟线结束时间 ``YYYY-MM-DD 15:30:00``。"""
    return f"{_parse_date(value):%Y-%m-%d} 15:30:00"


def _prefix_symbol(symbol: str) -> str:
    """把 ``000001`` 转为新浪格式 ``sz000001``。"""
    market = derive_market(symbol)
    match = re.search(r"(\d{6})", str(symbol))
    core = match.group(1) if match else str(symbol)
    return f"{market}{core}"


# 常见「空值 / 占位符」文本（AkShare 财务表可能出现），一律视为无效值。
_NULL_TOKENS: frozenset[str] = frozenset(
    {"", "-", "--", "—", "－", "None", "none", "NaN", "nan", "null", "NULL", "N/A", "n/a"}
)


def _parse_num(value: Any) -> Optional[float]:
    """把财务单元格转为 float；空值/占位符（``''``、``'-'``、``'--'``、``None``）返回 ``None``。

    Args:
        value: 原始单元格值（可能为 str / int / float / None / NaN）。

    Returns:
        有效数值；无效或占位符返回 ``None``（**不伪造数值**）。
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text in _NULL_TOKENS:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


class AkShareSource:
    """基于 AkShare 的数据源实现。"""

    name: str = "akshare"

    def __init__(self) -> None:
        """初始化。若 akshare 未成功导入，后续调用将抛 :class:`DataSourceError`。"""
        self._ak = ak
        self._version = AKSHARE_VERSION

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _require_ak(self) -> Any:
        """确保 akshare 可用。

        Raises:
            DataSourceError: akshare 未成功导入。
        """
        if self._ak is None:
            raise DataSourceError(
                self.name,
                _AK_IMPORT_ERROR or RuntimeError("akshare 不可用"),
                "akshare 未能导入，请检查安装与依赖（pandas/numpy）兼容性",
            )
        return self._ak

    # ------------------------------------------------------------------ #
    # 股票列表
    # ------------------------------------------------------------------ #
    def list_stocks(self) -> pd.DataFrame:
        """全量 A 股列表，返回列 ``[code, name, market]``。"""
        try:
            frame = self._require_ak().stock_info_a_code_name()
            result = pd.DataFrame(
                {
                    "code": frame["code"].astype(str).str.zfill(6),
                    "name": frame["name"].astype(str),
                }
            )
            result["market"] = result["code"].map(derive_market)
            return result[["code", "name", "market"]].reset_index(drop=True)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("akshare.list_stocks 失败")
            raise DataSourceError(self.name, exc) from exc

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
        """获取日/周/月/5m/15m/30m/60m K 线，返回 Bar 契约。

        Args:
            symbol: 标的代码，如 ``"000001"``。
            period: ``'d'|'w'|'m'|'5m'|'15m'|'30m'|'60m'``。
            start: 起始日期（``YYYYMMDD`` / ``YYYY-MM-DD`` / datetime）。
            end: 结束日期。
            adjust: ``''|'qfq'|'hfq'``。

        Returns:
            Bar 契约 DataFrame。

        Raises:
            DataSourceError: 周期不支持或接口调用失败。
        """
        try:
            akapi = self._require_ak()
            adjust = adjust or ""
            if period in _DAILY_PERIOD_MAP:
                raw = akapi.stock_zh_a_hist(
                    symbol=symbol,
                    period=_DAILY_PERIOD_MAP[period],
                    start_date=_to_compact(start),
                    end_date=_to_compact(end),
                    adjust=adjust,
                )
            elif period in _MINUTE_PERIOD_MAP:
                raw = akapi.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    start_date=_to_min_range_start(start),
                    end_date=_to_min_range_end(end),
                    period=_MINUTE_PERIOD_MAP[period],
                    adjust=adjust,
                )
            else:
                raise ValueError(f"不支持的周期：{period!r}")
            return normalize_bars(raw)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("akshare.get_kline(%s, %s) 失败", symbol, period)
            raise DataSourceError(self.name, exc) from exc

    # ------------------------------------------------------------------ #
    # 1 分钟（AkShare 不支持长历史）
    # ------------------------------------------------------------------ #
    def get_minute(self, symbol: str, date: Any) -> pd.DataFrame:
        """AkShare 不提供 1 分钟长历史，按约定返回 Bar 契约空表。

        Returns:
            空的 Bar 契约 DataFrame。
        """
        logger.warning(
            "akshare 不支持 1 分钟长历史（仅近 ~5 日且不复权）：symbol=%s date=%s；"
            "请使用 pytdx 数据源（P1-S5）。",
            symbol,
            date,
        )
        return pd.DataFrame(columns=BAR_COLUMNS)

    # ------------------------------------------------------------------ #
    # 基本面
    # ------------------------------------------------------------------ #
    def get_fundamental(self, symbol: str) -> dict:
        """拼装个股基本面摘要（缺字段以 ``None``/``''`` 填充，不抛异常）。

        Returns:
            键与 :class:`app.data.models.Fundamental` 一致的 dict。
        """
        try:
            akapi = self._require_ak()
            result: dict = {key: None for key in _FUNDAMENTAL_KEYS}
            result["symbol"] = str(symbol)
            result["industry"] = ""
            result["list_date"] = ""
            result["report_period"] = ""
            result["summary"] = ""

            # 1) 基本信息（东财）：item/value 两列
            try:
                info = akapi.stock_individual_info_em(symbol=symbol)
                info_map = self._info_to_dict(info)
                for item_key, field in _INFO_ITEM_MAP.items():
                    if item_key in info_map:
                        result[field] = info_map[item_key]
            except Exception as exc:  # noqa: BLE001
                logger.warning("stock_individual_info_em 失败：%s", exc)

            # 2) 估值（来自 spot 行，尽力而为）
            try:
                spot = self.get_spot()
                row = spot[spot["code"].astype(str).str.zfill(6) == str(symbol).zfill(6)]
                if not row.empty:
                    result["name"] = result["name"] or row.iloc[0].get("name")
                    if "市盈率-动态" in row.columns:
                        result["pe"] = row.iloc[0].get("市盈率-动态")
                    if "市净率" in row.columns:
                        result["pb"] = row.iloc[0].get("市净率")
            except Exception as exc:  # noqa: BLE001
                logger.warning("spot 估值字段获取失败：%s", exc)

            # 3) 财务指标（ROE / 营收增长率，尽力而为）
            try:
                fin = akapi.stock_financial_analysis_indicator(symbol=symbol)
                roe, yoy, period = self._extract_financial(fin)
                result["roe"] = roe if roe is not None else result["roe"]
                result["revenue_yoy"] = yoy if yoy is not None else result["revenue_yoy"]
                result["report_period"] = period or result["report_period"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("stock_financial_analysis_indicator 失败：%s", exc)

            result["summary"] = self._build_summary(result)
            return result
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("akshare.get_fundamental(%s) 失败", symbol)
            raise DataSourceError(self.name, exc) from exc

    @staticmethod
    def _info_to_dict(info: pd.DataFrame) -> dict:
        """把 ``stock_individual_info_em`` 的 item/value 两列转为 dict。"""
        if info is None or info.empty:
            return {}
        if "item" in info.columns and "value" in info.columns:
            return {str(r["item"]): r["value"] for _, r in info.iterrows()}
        # 兜底：两列时首列为键
        if info.shape[1] >= 2:
            return {str(r.iloc[0]): r.iloc[1] for _, r in info.iterrows()}
        return {}

    @staticmethod
    def _extract_financial(fin: pd.DataFrame) -> tuple[Optional[float], Optional[float], str]:
        """从财务指标表提取 (roe, revenue_yoy, report_period)。

        AkShare 该接口按时间升序返回多期数据；这里取 **最新报告期（末行）** 的值，
        且当最新期该字段为空/占位符时返回 ``None``——**不得回退到历史旧值**，
        否则会把陈旧数值（如某历史期的 ``-100.0``）误报为当前值。

        Args:
            fin: ``stock_financial_analysis_indicator`` 返回的原始表。

        Returns:
            ``(roe, revenue_yoy, report_period)``，缺失项为 ``None`` / ``""``。
        """
        if fin is None or fin.empty:
            return None, None, ""

        frame = fin.copy()
        date_col: Optional[str] = None
        for col in frame.columns:
            if "日期" in str(col):
                date_col = col
                break
        if date_col is not None:
            frame["_dt"] = pd.to_datetime(frame[date_col], errors="coerce")
            frame = frame.sort_values("_dt", kind="stable", na_position="first")
        if frame.empty:
            return None, None, ""

        latest = frame.iloc[-1]  # 最新报告期

        def _latest_of(keywords: tuple[str, ...]) -> Optional[float]:
            """在最新报告期行内，按关键字顺序返回首个有效数值；均无效返回 ``None``。"""
            for keyword in keywords:
                for col in frame.columns:
                    if keyword in str(col):
                        parsed = _parse_num(latest.get(col))
                        if parsed is not None:
                            return parsed
            return None

        period = ""
        if date_col is not None and pd.notna(latest.get(date_col)):
            period = str(latest.get(date_col))

        roe = _latest_of(("净资产收益率",))
        # 营收同比：优先「主营业务收入增长率」，回退「营业收入增长率」；均为空则 None。
        yoy = _latest_of(("主营业务收入增长率", "营业收入增长率"))
        return roe, yoy, period

    @staticmethod
    def _build_summary(result: dict) -> str:
        """用已获取字段拼一段可读摘要。"""
        parts: list[str] = []
        if result.get("name"):
            parts.append(str(result["name"]))
        if result.get("industry"):
            parts.append(f"行业：{result['industry']}")
        if result.get("list_date"):
            parts.append(f"上市：{result['list_date']}")
        if result.get("pe") is not None:
            parts.append(f"PE：{result['pe']}")
        if result.get("pb") is not None:
            parts.append(f"PB：{result['pb']}")
        if result.get("roe") is not None:
            parts.append(f"ROE：{result['roe']}%")
        if result.get("revenue_yoy") is not None:
            parts.append(f"营收同比：{result['revenue_yoy']}%")
        return "；".join(parts)

    # ------------------------------------------------------------------ #
    # 实时快照
    # ------------------------------------------------------------------ #
    def get_spot(self) -> pd.DataFrame:
        """实时/盘后 spot 快照。

        Returns:
            DataFrame，前四列固定为 ``[code, name, price, pct_change]``，其余列保留。
        """
        try:
            raw = self._require_ak().stock_zh_a_spot_em()
            frame = raw.rename(columns=_SPOT_RENAME)
            front = [c for c in ("code", "name", "price", "pct_change") if c in frame.columns]
            rest = [c for c in frame.columns if c not in front]
            return frame[front + rest].reset_index(drop=True)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("akshare.get_spot 失败")
            raise DataSourceError(self.name, exc) from exc

    # ------------------------------------------------------------------ #
    # 复权因子
    # ------------------------------------------------------------------ #
    def get_adj_factor(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        """复权因子序列（默认取前复权因子 qfq_factor），返回 ``[datetime, factor]``。

        Notes:
            经自省，akshare 1.18.96 无独立复权因子函数；使用
            ``stock_zh_a_daily(symbol, adjust='qfq-factor')``（新浪），
            其返回列 ``[date, qfq_factor]``。
        """
        try:
            akapi = self._require_ak()
            raw = akapi.stock_zh_a_daily(symbol=_prefix_symbol(symbol), adjust="qfq-factor")
            if raw is None or raw.empty:
                return pd.DataFrame(columns=["datetime", "factor"])
            frame = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(raw["date"], errors="coerce"),
                    "factor": pd.to_numeric(raw["qfq_factor"], errors="coerce"),
                }
            ).dropna(subset=["datetime", "factor"])
            frame = frame.sort_values("datetime", kind="stable")
            if start is not None:
                frame = frame[frame["datetime"] >= pd.Timestamp(_parse_date(start))]
            if end is not None:
                frame = frame[frame["datetime"] <= pd.Timestamp(_parse_date(end))]
            return frame.reset_index(drop=True)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("akshare.get_adj_factor(%s) 失败", symbol)
            raise DataSourceError(self.name, exc) from exc


@register_source("akshare")
def _create_akshare_source() -> AkShareSource:
    """AkShare 数据源工厂（供注册表使用）。"""
    return AkShareSource()
