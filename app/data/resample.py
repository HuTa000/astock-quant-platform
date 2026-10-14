"""周线 / 月线 / 大周期分钟线本地派生（纯函数，**无 Qt、无 DB 连接依赖**）。

背景：本版本「更新数据」仅拉取**日线**（``period='d'``），库里 ``w``/``m``/分钟级
周期均无数据。为让用户能看到周线 / 月线，改为**从本地日线重采样派生**（不联网、不落库）。
P6 优化（2026-10-01）：模拟训练支持周期选择后，15/30/60 分钟线在本地缺失时
**从本地 5 分钟线派生**（同样不落库），见 :func:`derive_minute_bars`。

聚合口径（A股惯例，与东财 / BaoStock 的周月线一致）：
    * ``open``  = 周期内**第一根**日线的 ``open``；
    * ``high``  = 周期内 ``high`` 的**最大值**；
    * ``low``   = 周期内 ``low`` 的**最小值**；
    * ``close`` = 周期内**最后一根**日线的 ``close``；
    * ``volume``= 周期内 ``volume`` 的**求和**（单位仍「手」）；
    * ``amount``= 周期内 ``amount`` 的**求和**（单位仍「元」）；
    * ``datetime`` = 周期内**最后一根**日线的 ``datetime``（东财 / BaoStock 周月线时间戳
      惯例为周期内最后一个交易日，**不是首日**）。

分组方式：``pd.to_datetime(...).dt.to_period(rule)`` + ``groupby(..., sort=True)``，
``rule`` 取 ``{"w": "W", "m": "M"}``（``"W"`` 即 W-SUN，A股周一至周五落同一周）。
**不使用 ``df.resample(...)``**：pandas 3.0 已弃用 ``"M"``（改 ``"ME"``），``to_period`` 无此问题。

**降级约定**：收益 / 空表 / ``None`` / 缺列 / ``period`` 不在 :data:`RESAMPLABLE` 时，
返回**空 Bar 契约表**并记 ``logger.warning``，**绝不抛异常**（与项目「降级不抛」风格一致）。
"""
from __future__ import annotations

import logging

import pandas as pd

from app.data.store.db import BAR_COLUMNS

logger = logging.getLogger(__name__)

# 可由日线派生的周期（周 / 月）。分钟级与日线不派生。
RESAMPLABLE: tuple[str, ...] = ("w", "m")

# 周期编码 → pandas Period 频率别名（"W" == W-SUN；A股周一至周五同组，正确）。
_PERIOD_RULE: dict[str, str] = {"w": "W", "m": "M"}

# 参与聚合的数值列。
_VALUE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")

# ---- 分钟周期派生（5m → 15/30/60m，P6 优化） ---- #
# 分钟周期编码 → 单根分钟数。
_MINUTE_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60}
# 目标分钟周期 ← 可派生的来源周期（来源分钟数整除且倍数 ≥2）。
MINUTE_RESAMPLABLE_FROM: dict[str, tuple[str, ...]] = {
    "15m": ("5m", "1m"),
    "30m": ("15m", "5m", "1m"),
    "60m": ("30m", "15m", "5m", "1m"),
}


def can_derive(period: str) -> bool:
    """判断某周期是否可由日线派生。

    Args:
        period: 周期编码（如 ``'w'`` / ``'m'`` / ``'d'`` / ``'5m'``）。

    Returns:
        可由日线派生返回 ``True``（``period`` ∈ :data:`RESAMPLABLE`），否则 ``False``。
    """
    return str(period) in RESAMPLABLE


def _empty_bars() -> pd.DataFrame:
    """返回空 Bar 契约表（列序 == :data:`BAR_COLUMNS`）。"""
    return pd.DataFrame(columns=BAR_COLUMNS)


def derive_bars(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    """把日线 Bar 契约 DataFrame 重采样为周线 / 月线。

    聚合口径见模块 docstring。结果列序等于 :data:`BAR_COLUMNS`，且按周期时间**升序**。

    Args:
        daily: 日线 Bar 契约 DataFrame（列 ``[datetime, open, high, low, close, volume, amount]``，
            ``datetime`` 升序；``volume`` 单位「手」、``amount`` 单位「元」）。
        period: 目标周期；仅 ``'w'`` / ``'m'`` 可派生。

    Returns:
        派生后的 Bar 契约 DataFrame；**收益 / 空表 / ``None`` / 缺列 / 不可派生周期**
        返回**空 Bar 契约表**（不抛异常，记 ``logger.warning``）。
    """
    if not can_derive(period):
        logger.warning("derive_bars 收到不可派生周期：%r（仅支持 %s）", period, RESAMPLABLE)
        return _empty_bars()
    if daily is None or not isinstance(daily, pd.DataFrame) or daily.empty:
        logger.warning("derive_bars 收到空数据（period=%s）", period)
        return _empty_bars()
    missing = [c for c in BAR_COLUMNS if c not in daily.columns]
    if missing:
        logger.warning("derive_bars 缺少列：%s（period=%s）", missing, period)
        return _empty_bars()

    frame = daily[BAR_COLUMNS].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for col in _VALUE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["datetime"])
    if frame.empty:
        logger.warning("derive_bars 全部 datetime 非法（period=%s）", period)
        return _empty_bars()

    # 稳定排序后再分组，保证 first/last 对应周期内首/末交易日。
    frame = frame.sort_values("datetime", kind="stable").reset_index(drop=True)
    keys = frame["datetime"].dt.to_period(_PERIOD_RULE[period])
    grouped = frame.groupby(keys, sort=True)

    result = pd.DataFrame(
        {
            "datetime": grouped["datetime"].last(),
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "amount": grouped["amount"].sum(),
        }
    ).reset_index(drop=True)
    result = result.sort_values("datetime", kind="stable").reset_index(drop=True)
    return result[BAR_COLUMNS]


# --------------------------------------------------------------------------- #
# 分钟周期派生（P6 优化：模拟训练多周期，5m → 15/30/60m）
# --------------------------------------------------------------------------- #
def can_derive_minute(source_period: str, target_period: str) -> bool:
    """判断目标分钟周期能否由来源分钟周期派生（如 ``5m`` → ``15m``）。

    Args:
        source_period: 来源分钟周期编码（``'1m'|'5m'|'15m'|'30m'|'60m'``）。
        target_period: 目标分钟周期编码。

    Returns:
        可派生返回 ``True``（见 :data:`MINUTE_RESAMPLABLE_FROM`），否则 ``False``。
    """
    sources = MINUTE_RESAMPLABLE_FROM.get(str(target_period), ())
    return str(source_period) in sources


def derive_minute_bars(
    bars: pd.DataFrame, source_period: str, target_period: str
) -> pd.DataFrame:
    """把较小周期分钟线聚合为较大周期（如 5m → 15/30/60m），**不落库**。

    聚合口径与日线派生一致：``open`` 取桶内首根、``high`` 取最大、``low`` 取最小、
    ``close`` 取末根、``volume`` / ``amount`` 求和；``datetime`` 取桶内**最后一根**的
    时间戳（与 BaoStock / 东财分钟线惯例一致：时间戳 = 该根 bar 的**结束时刻**）。

    分组方式：**每个交易日 × 上午/下午两个时段**内按固定根数分桶
    （``k = 目标分钟 / 来源分钟``，如 5m→60m 为 12 根一桶）。按时段分桶可避免
    午休（11:30–13:00）造成的桶错位，且对个别缺根更稳健；派生结果**绝不落库**。

    Args:
        bars: 来源分钟 Bar 契约 DataFrame（列 ``[datetime, open, high, low, close,
            volume, amount]``；``volume`` 单位「手」、``amount`` 单位「元」）。
        source_period: 来源分钟周期编码。
        target_period: 目标分钟周期编码（分钟数须为来源的整数倍且 ≥2）。

    Returns:
        派生后的 Bar 契约 DataFrame（升序）；**收益 / 空表 / ``None`` / 缺列 /
        不可派生组合**返回**空 Bar 契约表**（不抛异常，记 ``logger.warning``）。
    """
    if not can_derive_minute(source_period, target_period):
        logger.warning(
            "derive_minute_bars 不支持 %s → %s（来源可选 %s）",
            source_period, target_period, MINUTE_RESAMPLABLE_FROM.get(str(target_period), ()),
        )
        return _empty_bars()
    if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
        logger.warning("derive_minute_bars 收到空数据（%s → %s）", source_period, target_period)
        return _empty_bars()
    missing = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing:
        logger.warning("derive_minute_bars 缺少列：%s", missing)
        return _empty_bars()

    k = _MINUTE_MINUTES[str(target_period)] // _MINUTE_MINUTES[str(source_period)]
    frame = bars[BAR_COLUMNS].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    for col in _VALUE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime", kind="stable")
    frame = frame.reset_index(drop=True)
    if frame.empty:
        logger.warning("derive_minute_bars 全部 datetime 非法（%s → %s）", source_period, target_period)
        return _empty_bars()

    # 桶键：(交易日, 上午/下午, 时段内第几桶)。
    day_key = frame["datetime"].dt.normalize()
    session_key = frame["datetime"].dt.hour >= 12
    order_in_session = frame.groupby([day_key, session_key]).cumcount()
    bucket_key = order_in_session // k
    grouped = frame.groupby([day_key, session_key, bucket_key], sort=True)

    result = pd.DataFrame(
        {
            "datetime": grouped["datetime"].last(),
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "amount": grouped["amount"].sum(),
        }
    ).reset_index(drop=True)
    result = result.sort_values("datetime", kind="stable").reset_index(drop=True)
    return result[BAR_COLUMNS]


__all__ = [
    "MINUTE_RESAMPLABLE_FROM",
    "RESAMPLABLE",
    "can_derive",
    "can_derive_minute",
    "derive_bars",
    "derive_minute_bars",
]
