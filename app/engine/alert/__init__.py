"""技术指标信号预警服务（P4-S1）。

对外 API::

    from app.engine.alert import AlertService, get_alert_service, EVT_ALERT_TRIGGERED

    service = get_alert_service()
    rule_id = service.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})
    triggered = service.check_symbol("600519")   # -> [{rule_id, kind, symbol, value, message}]

规则类型（``kind``）：
    * ``ma_cross``     均线交叉（fast 均线上下穿 slow 均线，``direction`` 取
      ``cross_up`` 金叉 / ``cross_down`` 死叉，缺省金叉）；
    * ``macd_cross``   MACD 金叉（DIF 上穿 DEA）；
    * ``volume_surge`` 成交量放大（当前量 > 前 N 日均量 × 倍数）；
    * ``expr``         自定义表达式（末位布尔真值视为触发，走指标引擎沙箱）。

触发时经事件总线发布 ``ALERT_TRIGGERED``：
``{"symbol": str, "kind": str, "rule_id": int, "params": dict, "value": float, "time": str(ISO)}``。
"""
from __future__ import annotations

from app.engine.alert.alert_service import (
    EVT_ALERT_TRIGGERED,
    KIND_EXPR,
    KIND_MACD_CROSS,
    KIND_MA_CROSS,
    KIND_VOLUME_SURGE,
    SUPPORTED_KINDS,
    AlertService,
    get_alert_service,
)

__all__ = [
    "AlertService",
    "get_alert_service",
    "EVT_ALERT_TRIGGERED",
    "KIND_MA_CROSS",
    "KIND_MACD_CROSS",
    "KIND_VOLUME_SURGE",
    "KIND_EXPR",
    "SUPPORTED_KINDS",
]
