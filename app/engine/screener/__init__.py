"""多条件组合选股引擎（P3-S4）。

对外 API::

    from app.engine.screener import Condition, ScreenResult, ScreenerEngine, get_screener

    engine = get_screener()
    result = engine.run(
        [
            Condition("close.rolling(20).mean()", ">", 10),
            {"field": "pe", "op": "<", "value": 30},
        ],
        symbols=["000001", "600519"],
        logic="AND",
    )
    print(result.hits)     # 命中标的列表
    print(result.failed)   # 被跳过标的（无K线 / 基本面抓取失败）
"""
from __future__ import annotations

from app.engine.screener.screener import (
    FUNDAMENTAL_FIELDS,
    Condition,
    ScreenResult,
    ScreenerEngine,
    get_screener,
)

__all__ = [
    "Condition",
    "ScreenResult",
    "ScreenerEngine",
    "get_screener",
    "FUNDAMENTAL_FIELDS",
]
