"""K 线辅助标注的**纯函数**内核（涨跌停 / 跳空缺口 / 除权除息）。

本模块不依赖 Qt / PyQtGraph，纯数据计算，便于单测。
:class:`app.ui.widgets.kline_view.KLineView` 负责把这里返回的事件渲染到主图。

单位与契约（《开发手册》§4.2）：
    * 入参 ``df`` 为 Bar 契约 DataFrame（``datetime`` 升序，
      ``volume`` 单位「手」、``amount`` 单位「元」）；
    * 价格判定一律使用**同一口径**序列（调用方保证传入未复权序列，见
      :meth:`KLineView.load` 的说明：复权价在除权日不连续，直接用复权价判定涨跌停/跳空会误判）。

事件字典字段约定：
    * 涨跌停：``{"index", "datetime", "kind": "limit_up"|"limit_down"|"touch_up"|"touch_down", "price"}``；
    * 缺口：``{"index", "datetime", "kind": "gap_up"|"gap_down", "lower", "upper"}``；
    * 除权：``{"datetime", "index", "factor_from", "factor_to"}``。
"""
from __future__ import annotations

import logging
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 涨跌停/缺口颜色的语义常量由 themes 提供，但为保持本模块无 UI 依赖，这里只返回事件。
_CENT: Decimal = Decimal("0.01")

# 各板块涨跌停幅度。
_LIMIT_STAR: float = 0.20   # 科创板（688/689）、创业板（300/301）
_LIMIT_BJ: float = 0.30     # 北交所（8/4 开头）
_LIMIT_MAIN: float = 0.10   # 主板（600/601/603/605/000/001/002 等）


def _code6(symbol: str) -> str:
    """从 ``000001`` / ``sh600000`` / ``sz300750`` 中提取 6 位数字代码。

    Args:
        symbol: 标的代码（可能带 ``sh``/``sz``/``bj`` 前缀）。

    Returns:
        6 位数字代码；无法提取时返回去空格后的原串。
    """
    match = re.search(r"(\d{6})", str(symbol))
    return match.group(1) if match else str(symbol).strip()


def limit_pct(symbol: str) -> float:
    """返回该标的所在板块的**涨跌停幅度**（小数，如 ``0.10`` 表示 ±10%）。

    规则：
        * ``688`` / ``689`` 科创板 → ``0.20``；
        * ``300`` / ``301`` 创业板 → ``0.20``；
        * ``8`` / ``4`` 开头北交所 → ``0.30``；
        * 其余（主板 ``600``/``601``/``603``/``605``/``000``/``001``/``002`` 等）→ ``0.10``。

    Note:
        **ST 股的 ±5% 本步不支持**——判定 ST 需依赖股票名称（``*ST`` / ``ST`` 前缀），
        属后续「名称缓存」工作；接入后只需在此函数内按名称覆盖即可。

    Args:
        symbol: 标的代码（可带板块前缀）。

    Returns:
        涨跌停幅度（小数）。
    """
    code = _code6(symbol)
    if code.startswith(("688", "689")):
        return _LIMIT_STAR
    if code.startswith(("300", "301")):
        return _LIMIT_STAR
    if code.startswith(("8", "4")):
        return _LIMIT_BJ
    return _LIMIT_MAIN


def limit_prices(prev_close: float, pct: float) -> tuple[float, float]:
    """由昨收价与涨跌停幅度计算 ``(涨停价, 跌停价)``，**四舍五入到分（0.01 元）**。

    ⚠️ 必须使用 ``ROUND_HALF_UP``（四舍五入），**不能**用 Python 内置 :func:`round`
    （银行家舍入）。例：``prev=10.25, pct=0.10`` 时跌停价 ``9.225`` 应得 ``9.23``；
    内置 ``round(9.225, 2)`` 会得到 ``9.22``（错）。

    Args:
        prev_close: 昨收价（元）。
        pct: 涨跌停幅度（小数，如 ``0.10``）。

    Returns:
        ``(涨停价, 跌停价)``，均已按 ROUND_HALF_UP 精确到分。
    """
    base = Decimal(str(prev_close))
    ratio = Decimal(str(pct))
    up = (base * (Decimal(1) + ratio)).quantize(_CENT, rounding=ROUND_HALF_UP)
    down = (base * (Decimal(1) - ratio)).quantize(_CENT, rounding=ROUND_HALF_UP)
    return float(up), float(down)


def detect_limit_events(
    df: Optional[pd.DataFrame],
    symbol: str,
    tol: float = 0.005,
    include_touch: bool = False,
) -> list[dict]:
    """检测涨跌停（收盘价封板）事件；可选检测盘中触及涨跌停。

    以「该 bar 的**上一根 close**」为 ``prev_close`` 计算涨跌停价；
    该 bar ``close`` 与涨停价/跌停价之差 ``<= tol`` 即判定命中。首根无 ``prev_close``，跳过。

    Args:
        df: Bar 契约 DataFrame（``datetime`` 升序；判定应传**未复权**序列）。
        symbol: 标的代码（决定涨跌停幅度）。
        tol: 收盘价与涨跌停价的容差（元），默认 ``0.005``。
        include_touch: 为 ``True`` 时额外把 ``high >= 涨停价`` / ``low <= 跌停价`` 记为
            ``touch_up`` / ``touch_down``（默认关闭）。

    Returns:
        事件列表，每项 ``{"index", "datetime", "kind", "price"}``；
        ``price`` 为**真实涨跌停价**（涨停价或跌停价）。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if "close" not in df.columns:
        logger.warning("detect_limit_events 缺少 close 列")
        return []
    pct = limit_pct(symbol)
    closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    highs = (
        pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
        if "high" in df.columns
        else closes
    )
    lows = (
        pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
        if "low" in df.columns
        else closes
    )
    dts = pd.to_datetime(df["datetime"], errors="coerce") if "datetime" in df.columns else None

    events: list[dict] = []
    for i in range(1, len(df)):
        prev_close = closes[i - 1]
        close = closes[i]
        if pd.isna(prev_close) or pd.isna(close):
            continue
        up, down = limit_prices(float(prev_close), pct)
        dt = pd.Timestamp(dts.iloc[i]) if dts is not None and not pd.isna(dts.iloc[i]) else None
        hit_up = abs(float(close) - up) <= tol
        hit_down = abs(float(close) - down) <= tol
        if hit_up:
            events.append({"index": i, "datetime": dt, "kind": "limit_up", "price": up})
        elif hit_down:
            events.append({"index": i, "datetime": dt, "kind": "limit_down", "price": down})
        if include_touch:
            high = highs[i]
            low = lows[i]
            if not hit_up and not pd.isna(high) and float(high) >= up - tol:
                events.append({"index": i, "datetime": dt, "kind": "touch_up", "price": up})
            if not hit_down and not pd.isna(low) and float(low) <= down + tol:
                events.append({"index": i, "datetime": dt, "kind": "touch_down", "price": down})
    return events


def detect_gaps(df: Optional[pd.DataFrame], min_pct: float = 0.005) -> list[dict]:
    """检测**跳空缺口**事件。

    判定：
        * 向上跳空：``low[i] > high[i-1]``；
        * 向下跳空：``high[i] < low[i-1]``；
        * 缺口幅度 ``>= min_pct``（相对前一根的 ``high`` / ``low``）才计入。

    Args:
        df: Bar 契约 DataFrame（``datetime`` 升序；判定应传**未复权**序列）。
        min_pct: 缺口最小幅度（相对值），默认 ``0.005``（0.5%）。

    Returns:
        事件列表，每项 ``{"index", "datetime", "kind", "lower", "upper"}``；
        ``lower`` / ``upper`` 为缺口区间的 y 上下界（用于画框）。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if not {"high", "low"}.issubset(df.columns):
        logger.warning("detect_gaps 缺少 high/low 列")
        return []
    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    dts = pd.to_datetime(df["datetime"], errors="coerce") if "datetime" in df.columns else None

    events: list[dict] = []
    for i in range(1, len(df)):
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        low = lows[i]
        high = highs[i]
        if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(low) or pd.isna(high):
            continue
        dt = pd.Timestamp(dts.iloc[i]) if dts is not None and not pd.isna(dts.iloc[i]) else None
        if prev_high > 0 and low > prev_high:
            pct = (low - prev_high) / prev_high
            if pct >= min_pct:
                events.append(
                    {"index": i, "datetime": dt, "kind": "gap_up",
                     "lower": float(prev_high), "upper": float(low)}
                )
        elif prev_low > 0 and high < prev_low:
            pct = (prev_low - high) / prev_low
            if pct >= min_pct:
                events.append(
                    {"index": i, "datetime": dt, "kind": "gap_down",
                     "lower": float(high), "upper": float(prev_low)}
                )
    return events


def detect_ex_dividend(
    factors: Optional[pd.DataFrame],
    bars: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """检测**除权除息日**：复权因子相对前一交易日发生变化的日期。

    Args:
        factors: ``db.get_adj_factors`` 返回的 DataFrame（列 ``[datetime, factor]``，升序）。
        bars: 可选；提供且能**精确对齐**日期时，为事件补上其在该 ``bars`` 中的索引，
            否则 ``index`` 为 ``None``。

    Returns:
        事件列表，每项 ``{"datetime", "index", "factor_from", "factor_to"}``。
    """
    if factors is None or not isinstance(factors, pd.DataFrame) or factors.empty:
        return []
    if not {"datetime", "factor"}.issubset(factors.columns):
        logger.warning("detect_ex_dividend 缺少 datetime/factor 列")
        return []
    frame = factors.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime", kind="stable")

    dt_to_idx: dict = {}
    if bars is not None and isinstance(bars, pd.DataFrame) and not bars.empty and "datetime" in bars.columns:
        for i, value in enumerate(pd.to_datetime(bars["datetime"], errors="coerce")):
            if not pd.isna(value):
                dt_to_idx[pd.Timestamp(value)] = i

    events: list[dict] = []
    prev_factor: Optional[float] = None
    for _, row in frame.iterrows():
        factor = row["factor"]
        if pd.isna(factor):
            continue
        current = float(factor)
        if prev_factor is not None and abs(current - prev_factor) > 1e-9:
            dt = pd.Timestamp(row["datetime"])
            events.append(
                {
                    "datetime": dt,
                    "index": dt_to_idx.get(dt),
                    "factor_from": prev_factor,
                    "factor_to": current,
                }
            )
        prev_factor = current
    return events


__all__ = [
    "limit_pct",
    "limit_prices",
    "detect_limit_events",
    "detect_gaps",
    "detect_ex_dividend",
]
