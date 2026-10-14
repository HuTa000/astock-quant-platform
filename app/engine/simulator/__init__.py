"""模拟训练引擎（P5）。

对外出口：
    * :class:`SimContext` / :class:`Position` / :class:`Trade`：
      单标的训练会话状态容器 + 持仓/成交记录（:mod:`app.engine.simulator.sim_state`）；
    * :class:`ReplayClock` / :func:`interval_for_speed` / :data:`SIM_TICK` /
      :data:`SIM_FINISHED`：1 分钟回放时钟（:mod:`app.engine.simulator.replay_clock`）；
    * :func:`validate_order` / :data:`BUY` / :data:`SELL` / :data:`LOT_SIZE`：
      交易规则校验（P5-S2，T+1 / 做多 / 手数 / 资金 / 持仓）；
    * :class:`SimEngine` / :class:`ConditionalOrder` / :func:`execute_order` /
      :func:`check_conditional_orders` / :data:`SIM_ORDER_FILLED` /
      :data:`SIM_STATE_CHANGED`：训练驱动（回放 + 手动下单 + 条件单，P5-S3）。

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）；回放时钟为
纯 Python 手动推进，UI 层负责用 ``QTimer`` 驱动（P5-S4）。
"""
from __future__ import annotations

from app.engine.simulator.engine import SimEngine
from app.engine.simulator.executor import (
    SIM_ORDER_FILLED,
    SIM_STATE_CHANGED,
    ConditionalOrder,
    check_conditional_orders,
    execute_order,
)
from app.engine.simulator.replay_clock import (
    SIM_FINISHED,
    SIM_TICK,
    ReplayClock,
    interval_for_speed,
)
from app.engine.simulator.sim_state import Position, SimContext, Trade
from app.engine.simulator.trading_rules import (
    BUY,
    LOT_SIZE,
    SELL,
    max_buy_qty,
    qty_for_amount,
    validate_order,
)

__all__ = [
    "Position",
    "SimContext",
    "Trade",
    "ReplayClock",
    "SimEngine",
    "ConditionalOrder",
    "execute_order",
    "check_conditional_orders",
    "interval_for_speed",
    "SIM_TICK",
    "SIM_FINISHED",
    "SIM_ORDER_FILLED",
    "SIM_STATE_CHANGED",
    "BUY",
    "SELL",
    "LOT_SIZE",
    "max_buy_qty",
    "qty_for_amount",
    "validate_order",
]