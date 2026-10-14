"""统一 ``DataSource`` 接口与通用工具。

设计目标（《细化设计》§1.1 / 《开发手册》§4.3）：
    * 所有数据源实现同一协议，返回统一的 DataFrame 契约；
    * UI / 引擎只认接口，不感知具体数据源；
    * 提供 ``normalize_bars`` 保证所有 K 线输出符合全局 Bar 契约
      ``[datetime, open, high, low, close, volume, amount]``（datetime64 升序去重）。

数据源注册表供后续 P1-S4（baostock）/ P1-S5（pytdx）注册自己的实现。
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Dict, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

# 全局 Bar 契约列（《开发手册》§4.2）。
BAR_COLUMNS: list[str] = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

# 常见列名别名（含 akshare 的中文列），统一映射到 Bar 契约。
_COLUMN_ALIASES: dict[str, str] = {
    # 时间
    "datetime": "datetime",
    "date": "datetime",
    "time": "datetime",
    "dt": "datetime",
    "日期": "datetime",
    "时间": "datetime",
    "时间戳": "datetime",
    # 开盘
    "open": "open",
    "开盘": "open",
    "开盘价": "open",
    # 最高
    "high": "high",
    "最高": "high",
    "最高价": "high",
    # 最低
    "low": "low",
    "最低": "low",
    "最低价": "low",
    # 收盘
    "close": "close",
    "收盘": "close",
    "收盘价": "close",
    # 成交量
    "volume": "volume",
    "vol": "volume",
    "成交量": "volume",
    # 成交额
    "amount": "amount",
    "成交额": "amount",
    "成交金额": "amount",
}


class DataSourceError(Exception):
    """数据源异常（对外统一出口，由上层决定重试 / 切兜底）。

    Attributes:
        source: 数据源名称，如 ``"akshare"``。
        original: 触发该错误的原始异常。
        message: 可读描述。
    """

    def __init__(self, source: str, original: Exception, message: str = "") -> None:
        """构造数据源异常。

        Args:
            source: 数据源名称。
            original: 原始异常对象。
            message: 补充描述；为空时取 ``str(original)``。
        """
        self.source = source
        self.original = original
        self.message = message or str(original)
        super().__init__(f"[{source}] {self.message}")


@runtime_checkable
class DataSource(Protocol):
    """数据源统一协议（结构化子类型，无需显式继承）。"""

    name: str

    def list_stocks(self) -> pd.DataFrame:
        """全量 A 股列表。返回列 ``[code, name, market]``。"""
        ...

    def get_kline(
        self, symbol: str, period: str, start: str, end: str, adjust: str = ""
    ) -> pd.DataFrame:
        """日/周/月/5m/15m/30m/60m K 线，返回 Bar 契约。"""
        ...

    def get_minute(self, symbol: str, date: str) -> pd.DataFrame:
        """单交易日 1 分钟 K 线；不支持则返回 Bar 契约空表。"""
        ...

    def get_fundamental(self, symbol: str) -> dict:
        """公司基本信息 + 核心财务指标摘要（dict）。"""
        ...

    def get_spot(self) -> pd.DataFrame:
        """实时/盘后 spot 快照。返回列 ``[code, name, price, pct_change, ...]``。"""
        ...

    def get_adj_factor(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """复权因子序列。返回列 ``[datetime, factor]``。"""
        ...


def derive_market(code: str) -> str:
    """按代码推导交易市场。

    Args:
        code: 股票代码，可为 ``"600000"`` / ``"SH600000"`` / ``"600000.SH"`` 等形式。

    Returns:
        ``"sh" | "sz" | "bj"``；无法识别时返回 ``"unknown"``。
    """
    raw = str(code).strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if raw.startswith(prefix):
            return prefix
    match = re.search(r"(\d{6})", raw)
    core = match.group(1) if match else raw
    if core.startswith(("60", "68", "9")):
        return "sh"
    if core.startswith(("00", "30", "2")):
        return "sz"
    if core.startswith(("8", "4")):
        return "bj"
    return "unknown"


# 指数代码形态：**必须带显式市场前缀**（如 ``sh.000300``），避免 ``000300``
# 被 :func:`derive_market` 误判成深市股票（000 开头）；常见指数号段：
# 沪市 000xxx（上证/中证系列）、深市 399xxx（深证系列）、950xxx（中证细分）。
_INDEX_SYMBOL_PATTERN = re.compile(r"^(sh|sz|bj)\.(\d{6})$")
_INDEX_CODE_PREFIXES: tuple[str, ...] = ("000", "399", "950")


def is_index_symbol(code: str) -> bool:
    """判断代码是否为**指数**（如 ``sh.000300`` 沪深300）。

    仅在「显式市场前缀 + 6 位数字 + 常见指数号段」同时满足时判真；裸
    ``000300`` 不判为指数（可能是深市股票代码写法，交由调用方显式补前缀）。

    Args:
        code: 标的代码，如 ``"sh.000300"``。

    Returns:
        是指数返回 ``True``，否则 ``False``（含无法识别的情况）。
    """
    match = _INDEX_SYMBOL_PATTERN.match(str(code).strip().lower())
    if not match:
        return False
    return match.group(2).startswith(_INDEX_CODE_PREFIXES)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """把任意数据源返回的 K 线归一化为全局 Bar 契约。

    Args:
        df: 原始 DataFrame，列名可为英文或 akshare 中文列名
            （如 ``日期/开盘/收盘/最高/最低/成交量/成交额``）。

    Returns:
        列严格为 ``[datetime, open, high, low, close, volume, amount]`` 的
        DataFrame，``datetime`` 为 ``datetime64``、升序、按 datetime 去重。

    Raises:
        TypeError: 入参不是 DataFrame。
        ValueError: 缺少必需的 OHLC 列。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"normalize_bars 需要 DataFrame，实际为 {type(df)!r}")
    if df.empty and not set(_COLUMN_ALIASES).intersection(df.columns):
        # 空表：直接返回契约空表，避免误判缺列。
        return pd.DataFrame(columns=BAR_COLUMNS)

    rename_map = {col: _COLUMN_ALIASES[col] for col in df.columns if col in _COLUMN_ALIASES}
    frame = df.rename(columns=rename_map)

    required = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"K 线缺少必需列：{missing}；实际列：{list(df.columns)}")

    frame = frame[required].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # 丢弃时间无效的行，按时间升序，并按 datetime 去重（保留最后一条）。
    frame = frame.dropna(subset=["datetime"])
    frame = frame.sort_values("datetime", kind="stable")
    frame = frame.drop_duplicates(subset=["datetime"], keep="last")
    return frame[BAR_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 数据源注册表
# --------------------------------------------------------------------------- #
SOURCES: Dict[str, Callable[[], DataSource]] = {}


def register_source(name: str) -> Callable[[Callable[[], DataSource]], Callable[[], DataSource]]:
    """数据源工厂注册装饰器。

    Args:
        name: 数据源名称，如 ``"akshare"``。

    Returns:
        装饰器，注册后原样返回被装饰的工厂函数。
    """

    def decorator(factory: Callable[[], DataSource]) -> Callable[[], DataSource]:
        SOURCES[name] = factory
        logger.debug("已注册数据源：%s", name)
        return factory

    return decorator


def get_source(name: str) -> DataSource:
    """按名称获取数据源实例。

    Args:
        name: 数据源名称。

    Returns:
        该数据源的实例。

    Raises:
        KeyError: 名称未注册。错误信息中列出可用数据源。
    """
    if name not in SOURCES:
        raise KeyError(f"未知数据源 {name!r}，可用数据源：{sorted(SOURCES)}")
    return SOURCES[name]()
