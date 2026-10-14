"""内置指标函数封装（统一签名 + 单列取值）。

把 pandas-ta 常用指标包装成「统一签名」：**第一参数为 bars（Bar 契约 DataFrame）**，
其余为指标参数；返回值统一为**与 bars 等长**的 :class:`pd.Series`（多列指标默认取主列，
可用 ``column`` 参数切换）。对于只需 close 的指标（MA/EMA/MACD/RSI/BOLL），
第一参数也允许直接传 close 的 Series（会按其自身 index 对齐）。

多列指标的默认主列（「calc 返回单个 Series」场景的取舍约定）：

    * ``MACD`` → ``MACD`` 列（DIF 快慢线差值）；``column`` 可选 ``"MACDh"``(柱) / ``"MACDs"``(信号线)；
    * ``KDJ``  → ``K`` 列；``column`` 可选 ``"D"`` / ``"J"``；
    * ``BOLL`` → ``BBM``（中轨）；``column`` 可选 ``"BBU"``(上轨) / ``"BBL"``(下轨)
      以及 ``"BBB"``(带宽) / ``"BBP"``(百分比)。

``MA`` 与 ``EMA`` 的单列 Series 名分别沿用 pandas-ta 的 ``SMA_{length}`` / ``EMA_{length}``。
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd

from app.engine.indicator.talib_adapter import ta, pick_column

__all__ = ["MA", "EMA", "MACD", "RSI", "KDJ", "BOLL"]

# 用户友好列名 → pandas-ta 实际列前缀的映射。
_BOLL_COLUMN_ALIASES: dict[str, str] = {
    "LOWER": "BBL",
    "MID": "BBM",
    "MIDDLE": "BBM",
    "UPPER": "BBU",
    "BANDWIDTH": "BBB",
    "PERCENT": "BBP",
}
_MACD_COLUMN_ALIASES: dict[str, str] = {
    "DIF": "MACD",
    "HIST": "MACDh",
    "HISTOGRAM": "MACDh",
    "DEA": "MACDs",
    "SIGNAL": "MACDs",
}

# 可用于 close 类指标的输入（DataFrame 或 Series）。
SeriesLike = Union[pd.DataFrame, pd.Series]


def _close(source: SeriesLike) -> pd.Series:
    """从 bars DataFrame 或 close Series 中提取数值化 close 序列。"""
    if isinstance(source, pd.DataFrame):
        if "close" not in source.columns:
            raise ValueError(f"bars 缺少列：close（现有列：{list(source.columns)}）")
        return pd.to_numeric(source["close"], errors="coerce")
    if isinstance(source, pd.Series):
        return pd.to_numeric(source, errors="coerce")
    raise TypeError(f"不支持的输入类型：{type(source).__name__}（应为 DataFrame 或 Series）")


def _ohlc(source: SeriesLike, columns: tuple[str, ...]) -> tuple[pd.Series, ...]:
    """从 bars DataFrame 中提取多列数值序列（用于 KDJ 等需要 high/low/close 的指标）。"""
    if not isinstance(source, pd.DataFrame):
        raise TypeError(
            f"{columns} 类指标需要 bars DataFrame（含 {list(columns)} 列），"
            f"实际收到 {type(source).__name__}"
        )
    missing = [c for c in columns if c not in source.columns]
    if missing:
        raise ValueError(f"bars 缺少列：{missing}（现有列：{list(source.columns)}）")
    return tuple(pd.to_numeric(source[c], errors="coerce") for c in columns)


def _align(result: pd.Series, source: SeriesLike) -> pd.Series:
    """把结果 reindex 到输入 index，保证「长度与输入一致」。

    pandas-ta 的 rolling 指标会保留原始 index（前 N-1 个 NaN），此处的 ``reindex``
    确保即使传入的 Series/DataFrame 带非默认 index，也能对齐。
    """
    idx = getattr(source, "index", None)
    if idx is None:
        return result
    return result.reindex(idx)


def _empty_series(source: SeriesLike) -> pd.Series:
    """数据不足（行数 < 指标窗口）时返回与输入等长的全 NaN Series（降级不抛）。

    pandas-ta 在输入长度不足时会返回 ``None``（如 ``ta.sma(5 根 close, length=20)``），
    本函数把该情形收敛为「全 NaN」，与图表「数据未攒够前不画线」的语义一致。
    """
    idx = getattr(source, "index", None)
    if idx is None:
        return pd.Series(dtype="float64")
    return pd.Series(np.nan, index=idx, dtype="float64")


def _finish(result: Any, source: SeriesLike, prefix: str | None = None) -> pd.Series:
    """把 pandas-ta 结果统一收口为与输入等长的单列 Series。

    Args:
        result: pandas-ta 返回值（Series / DataFrame / 数据不足时的 ``None``）。
        source: 原始输入（DataFrame 或 Series），用于 index 对齐。
        prefix: 多列指标的取值列前缀；单列指标传 ``None``。

    Returns:
        与输入等长的 Series；``result`` 为 ``None`` 时返回全 NaN Series。
    """
    if result is None:
        return _empty_series(source)
    series = pick_column(result, prefix) if prefix is not None else result
    return _align(series, source)


def MA(bars: SeriesLike, length: int = 20) -> pd.Series:
    """简单移动平均（SMA）。返回与输入等长 Series，前 ``length-1`` 个为 NaN。

    Args:
        bars: Bar 契约 DataFrame，或 close Series。
        length: 窗口长度（默认 20）。
    """
    close = _close(bars)
    result = ta.sma(close, length=int(length))
    return _finish(result, bars)


def EMA(bars: SeriesLike, length: int = 20) -> pd.Series:
    """指数移动平均（EMA）。返回与输入等长 Series，前 ``length-1`` 个为 NaN。

    Args:
        bars: Bar 契约 DataFrame，或 close Series。
        length: 窗口长度（默认 20）。
    """
    close = _close(bars)
    result = ta.ema(close, length=int(length))
    return _finish(result, bars)


def MACD(
    bars: SeriesLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "MACD",
) -> pd.Series:
    """MACD。默认返回 DIF（``MACD`` 列），可用 ``column`` 切换柱 / 信号线。

    Args:
        bars: Bar 契约 DataFrame，或 close Series。
        fast: 快线窗口（默认 12）。
        slow: 慢线窗口（默认 26）。
        signal: 信号线窗口（默认 9）。
        column: 取值列，``"MACD"``(DIF) / ``"MACDh"``(柱) / ``"MACDs"``(信号线)，
            也支持 ``"DIF"/"HIST"/"DEA"/"SIGNAL"`` 别名。

    Returns:
        与输入等长的单列 Series。
    """
    close = _close(bars)
    frame = ta.macd(close, fast=int(fast), slow=int(slow), signal=int(signal))
    prefix = _MACD_COLUMN_ALIASES.get(str(column).upper(), str(column).upper())
    return _finish(frame, bars, prefix=prefix)


def RSI(bars: SeriesLike, length: int = 14) -> pd.Series:
    """相对强弱指标（RSI）。返回与输入等长 Series，前 ``length`` 个为 NaN。

    Args:
        bars: Bar 契约 DataFrame，或 close Series。
        length: 窗口长度（默认 14）。
    """
    close = _close(bars)
    result = ta.rsi(close, length=int(length))
    return _finish(result, bars)


def KDJ(
    bars: pd.DataFrame,
    length: int = 9,
    signal: int = 3,
    column: str = "K",
) -> pd.Series:
    """KDJ 随机指标。默认返回 ``K`` 列，可用 ``column`` 切换 ``D`` / ``J``。

    Args:
        bars: Bar 契约 DataFrame（需含 ``high`` / ``low`` / ``close``）。
        length: K 值窗口（默认 9）。
        signal: 平滑周期（默认 3）。
        column: 取值列，``"K"`` / ``"D"`` / ``"J"``。

    Returns:
        与 bars 等长的单列 Series。
    """
    high, low, close = _ohlc(bars, ("high", "low", "close"))
    frame = ta.kdj(high=high, low=low, close=close, length=int(length), signal=int(signal))
    return _finish(frame, bars, prefix=str(column).upper())


def BOLL(
    bars: SeriesLike,
    length: int = 20,
    std: float = 2.0,
    column: str = "MID",
) -> pd.Series:
    """布林带（BOLL）。默认返回中轨（``BBM``），可用 ``column`` 切换其他轨道。

    Args:
        bars: Bar 契约 DataFrame，或 close Series。
        length: 窗口长度（默认 20）。
        std: 标准差倍数（默认 2.0）。
        column: 取值列，``"MID"``(中轨) / ``"UPPER"``(上轨) / ``"LOWER"``(下轨)
            / ``"BBB"``(带宽) / ``"BBP"``(百分比)，也支持 ``"BBM"/"BBU"/"BBL"`` 等前缀。

    Returns:
        与输入等长的单列 Series。
    """
    close = _close(bars)
    frame = ta.bbands(close, length=int(length), std=float(std))
    prefix = _BOLL_COLUMN_ALIASES.get(str(column).upper(), str(column).upper())
    return _finish(frame, bars, prefix=prefix)
