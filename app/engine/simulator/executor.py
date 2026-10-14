"""模拟训练：成交执行与条件单（P5-S3）。

职责（《细化设计》§3.3）：
    * :func:`execute_order` —— 手动/条件单经「校验（P5-S2）→ 成交执行」：
        资金扣减/入账、持仓增减、成本加权、``Trade`` 记录，发 ``SIM_ORDER_FILLED`` /
        ``SIM_STATE_CHANGED`` 事件；
    * :class:`ConditionalOrder` —— 条件单（价格 / 涨跌幅两类触发；一次性，触发后移除）；
    * :func:`check_conditional_orders` —— 每 tick 对未触发条件单做判定，满足即委托
      :func:`execute_order` 成交（以触发时的最新价 close 成交，``reason='condition'``）。

条件单触发规则（§3.3）：
    买入：``price <= trigger``（价格）；或 ``pct_change >= trigger``（涨跌幅，百分数值）；
    卖出：``price >= trigger``（价格）；或 ``pct_change <= -trigger``（涨跌幅）。
    涨跌幅基准：**当前游标所在交易日（首根 bar 的 close）**——训练日内的相对表现。

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）；本模块纯 Python。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import pandas as pd

from app.core.event_bus import get_bus
from app.engine.simulator.sim_state import Position, SimContext, Trade
from app.engine.simulator.trading_rules import BUY, SELL, validate_order

logger = logging.getLogger(__name__)

# 事件名（《细化设计》§1.3 事件清单）。
SIM_ORDER_FILLED: str = "SIM_ORDER_FILLED"
SIM_STATE_CHANGED: str = "SIM_STATE_CHANGED"

# 条件单触发类型。
TRIGGER_PRICE: str = "price"
TRIGGER_PCT: str = "pct"


@dataclass
class ConditionalOrder:
    """条件单（一次性，触发并成交后自动移除）。

    Attributes:
        side: 方向（``'BUY'`` / ``'SELL'``）。
        trigger_type: ``'price'``（绝对价）或 ``'pct'``（涨跌幅百分比值）。
        trigger: 触发阈值。
        qty: 股数（100 整数倍）。
        symbol: 监控标的；``None`` 时按「当前标的」（添加时的 ``ctx.symbol`` 亦可
            显式写入，多标的会话中推荐显式指定，切换标的后续继续监控原标的）。
    """

    side: str
    trigger_type: str
    trigger: float
    qty: int
    symbol: Optional[str] = None


def _make_position_or_update(ctx: SimContext, symbol: str, price: float, qty: int) -> None:
    """买入成交：更新该标的持仓（加权成本 + 更新 buy_date 为成交当日）。"""
    today = ctx.cursor_dt.date().isoformat()
    pos = ctx.positions.get(symbol)
    if pos is None:
        ctx.positions[symbol] = Position(
            symbol=symbol, qty=qty, cost=price, buy_date=today
        )
        return
    # 加权平均成本。
    total_qty = pos.qty + qty
    if total_qty > 0:
        pos.cost = (pos.cost * pos.qty + price * qty) / total_qty
    pos.qty = total_qty
    # T+1 从严：buy_date 记录**最近一次买入**日。
    pos.buy_date = today


def _reduce_position(ctx: SimContext, symbol: str, price: float, qty: int) -> None:
    """卖出成交：减少该标的持仓；归零则移除（T+1 已在 validate_order 校验）。"""
    pos = ctx.positions.get(symbol)
    if pos is None:
        return
    pos.qty -= qty
    if pos.qty <= 0:
        ctx.positions.pop(symbol, None)


def _emit_execution(ctx: SimContext, symbol: str, side: str, price: float, qty: int,
                    reason: str, emit_events: bool) -> None:
    """发成交 / 状态事件（订阅者异常隔离）。"""
    if not emit_events:
        return
    try:
        get_bus().publish(
            SIM_ORDER_FILLED,
            {
                "side": side,
                "symbol": symbol,
                "price": float(price),
                "qty": int(qty),
                "time": ctx.cursor_dt.isoformat() if ctx.cursor_dt else "",
                "reason": reason,
            },
        )
        get_bus().publish(
            SIM_STATE_CHANGED,
            {
                "cash": float(ctx.cash),
                "positions": {
                    sym: {"qty": p.qty, "cost": p.cost} for sym, p in ctx.positions.items()
                },
                "pnl": float(ctx.total_asset() - (ctx.initial_cash or ctx.cash)),
            },
        )
    except Exception as exc:  # noqa: BLE001 - 事件隔离，成交不受影响
        logger.exception("发布成交事件失败：%s", exc)


def execute_order(
    ctx: SimContext,
    side: str,
    price: float,
    qty: int,
    reason: str = "manual",
    emit_events: bool = True,
    symbol: Optional[str] = None,
) -> Tuple[bool, str]:
    """执行一笔订单：校验 → 成交（资金 / 持仓 / Trade / 事件）。

    Args:
        ctx: 训练会话状态。
        side: ``'BUY'`` / ``'SELL'``。
        price: 成交价（正数）。
        qty: 股数（100 整数倍）。
        reason: ``'manual'``（手动下单）或 ``'condition'``（条件单）。
        emit_events: 是否发 ``SIM_ORDER_FILLED`` / ``SIM_STATE_CHANGED``。
        symbol: 成交标的；``None`` 时取当前标的 ``ctx.symbol``（多标的会话中
            由调用方显式指定，资金/持仓按该标的分别记账）。

    Returns:
        ``(True, "")`` 成交成功；``(False, 中文拒绝原因)``。

    Notes:
        首次成交时按当时现金记 ``ctx.initial_cash``（期初资金快照，供 PnL）。
    """
    key = str(symbol) if symbol else ctx.symbol
    ok, reason_text = validate_order(ctx, side, price, qty, symbol=key)
    if not ok:
        return False, reason_text

    price = float(price)
    qty = int(qty)

    if ctx.initial_cash is None:
        ctx.initial_cash = ctx.cash

    if side == BUY:
        ctx.cash -= price * qty
        _make_position_or_update(ctx, key, price, qty)
    else:
        ctx.cash += price * qty
        _reduce_position(ctx, key, price, qty)

    ctx.orders.append(
        Trade(
            symbol=key,
            side=side,
            price=price,
            qty=qty,
            dt=ctx.cursor_dt,
            reason=str(reason),
        )
    )
    _emit_execution(ctx, key, side, price, qty, str(reason), emit_events)
    return True, ""


def day_open_price(ctx: SimContext, symbol: Optional[str] = None) -> Optional[float]:
    """某标的当前游标所在交易日的**开盘价基准**（当日首根 bar 的 close）。

    Returns:
        基准价；游标未启动 / 无数据返回 ``None``。
    """
    if ctx.cursor_dt is None:
        return None
    bars = ctx.bars_for(symbol)
    if bars is None or bars.empty:
        return None
    day = ctx.cursor_dt.date()
    frame = bars[bars["datetime"].dt.date == day]
    if frame.empty:
        return None
    return float(frame.iloc[0]["close"])


def check_conditional_orders(
    ctx: SimContext,
    orders: List[ConditionalOrder],
    on_filled: Optional[Callable[[ConditionalOrder, float], None]] = None,
    emit_events: bool = True,
) -> List[ConditionalOrder]:
    """每 tick 评估条件单，满足条件的按最新价成交并从列表移除。

    Args:
        ctx: 训练会话状态（取最新价 / 涨跌幅基准）。
        orders: 待评估的条件单列表（**就地修改**：已触发者被移除）。
        on_filled: 成交后的回调（收到 ``(order, price)``），可空。
        emit_events: 是否发成交事件。

    Returns:
        本次触发并成交的条件单列表。
    """
    if not orders:
        return []
    hit: List[ConditionalOrder] = []
    remaining: List[ConditionalOrder] = []

    for order in orders:
        # 逐单按各自标的取价（多标的会话：条件单绑定添加时的标的）。
        key = str(order.symbol) if order.symbol else ctx.symbol
        price = ctx.last_price(symbol=key)
        if price is None:
            remaining.append(order)
            continue
        base = day_open_price(ctx, key)
        triggered = False
        if order.trigger_type == TRIGGER_PRICE:
            if order.side == BUY and price <= float(order.trigger):
                triggered = True
            elif order.side == SELL and price >= float(order.trigger):
                triggered = True
        elif order.trigger_type == TRIGGER_PCT:
            if base is None or base == 0:
                remaining.append(order)
                continue
            pct = (price - base) / base * 100.0
            if order.side == BUY and pct >= float(order.trigger):
                triggered = True
            elif order.side == SELL and pct <= -float(order.trigger):
                triggered = True
        else:
            remaining.append(order)
            continue

        if not triggered:
            remaining.append(order)
            continue

        ok, _ = execute_order(
            ctx, order.side, price, order.qty, reason="condition",
            emit_events=emit_events, symbol=key,
        )
        if ok:
            hit.append(order)
            if on_filled is not None:
                try:
                    on_filled(order, price)
                except Exception as exc:  # noqa: BLE001 - 回调异常隔离
                    logger.exception("条件单成交回调异常：%s", exc)
        else:
            # 校验失败（如 T+1/现金不足）：保留原单，等下一 tick 再由引擎决定处理。
            remaining.append(order)

    orders[:] = remaining
    return hit


__all__ = [
    "ConditionalOrder",
    "SIM_ORDER_FILLED",
    "SIM_STATE_CHANGED",
    "TRIGGER_PCT",
    "TRIGGER_PRICE",
    "check_conditional_orders",
    "day_open_price",
    "execute_order",
]