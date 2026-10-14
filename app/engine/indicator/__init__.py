"""指标计算引擎（P3-S1）：统一入口 + pandas-ta 适配 + 内置指标。

对外 API::

    from app.engine.indicator import IndicatorEngine, IndicatorResult, get_engine, ta, MA, MACD

    engine = get_engine()
    result = engine.calc("close.rolling(20).mean()", bars)
    if result.ok:
        series = result.series      # 与 bars 等长的 Series
    else:
        print(result.error)

    ta.sma(bars["close"], 20)       # 直接使用 pandas-ta 命名空间
    MA(bars, 20)                    # 内置指标（统一签名，默认返回主列）
"""
from __future__ import annotations

from app.engine.indicator.builtins import BOLL, EMA, KDJ, MA, MACD, RSI
from app.engine.indicator.engine import (
    IndicatorEngine,
    IndicatorResult,
    get_engine,
)
from app.engine.indicator.talib_adapter import TalibAdapter, pick_column, ta

__all__ = [
    "IndicatorEngine",
    "IndicatorResult",
    "get_engine",
    "TalibAdapter",
    "pick_column",
    "ta",
    "MA",
    "EMA",
    "MACD",
    "RSI",
    "KDJ",
    "BOLL",
]
