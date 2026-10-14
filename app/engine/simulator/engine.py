"""模拟训练：训练引擎驱动（P5-S3）。

职责（《细化设计》§3.2 / §3.3）：
    * :class:`SimEngine` 组合 :class:`~app.engine.simulator.replay_clock.ReplayClock`（回放推进）
      与**交易执行**（手动下单 :meth:`SimEngine.place_order` + 条件单
      :meth:`SimEngine.add_conditional_order`）；
    * 每 tick 推进一根 1 分钟 bar 后**自动评估条件单**（满足 → 以最新价成交）；
    * 倍速 / 暂停状态管理：``speed`` 决定回调间隔（经 :func:`interval_for_speed`
      由 UI 定时器消费），``paused`` 由 UI 控制是否调用 :meth:`tick`——引擎层
      只暴露状态与推进，**不 import 任何 Qt 代码**。

驱动循环（UI 层 P5-S4 用法）::

    engine = SimEngine(ctx)
    def drive():
        if not engine.tick():
            timer.stop()
    timer = QTimer(); timer.setInterval(engine.interval_ms()); timer.timeout.connect(drive)

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

from app.engine.simulator.executor import (
    SIM_ORDER_FILLED,
    SIM_STATE_CHANGED,
    ConditionalOrder,
    check_conditional_orders,
    execute_order,
)
from app.engine.simulator.replay_clock import ReplayClock, interval_for_speed
from app.engine.simulator.sim_state import SimContext
from app.engine.simulator.trading_rules import BUY, SELL

logger = logging.getLogger(__name__)

# 回放起始速度兜底。
_PAUSED_SPEED: int = 0


class SimEngine:
    """单标的模拟训练引擎（回放 + 交易执行 + 条件单）。"""

    def __init__(
        self,
        ctx: SimContext,
        speed: Optional[int] = None,
        emit_events: bool = True,
        on_tick: Optional[Callable[[dict], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        on_fill: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """构造训练引擎。

        Args:
            ctx: 训练会话状态（含 bars / 资金 / 持仓）。
            speed: 初始倍速（``None`` → ``ctx.speed``）。
            emit_events: 是否发 ``SIM_TICK`` / ``SIM_ORDER_FILLED`` 等总线事件。
            on_tick: 每推进一根的回调（收到 ``{"cursor_dt", "price"}``）。
            on_finished: 回放结束回调。
            on_fill: 每次成交回调（收到成交 dict）。
        """
        self._ctx = ctx
        if speed is not None:
            ctx.speed = int(speed)
        self._emit_events = bool(emit_events)
        # 保存回调以便 :meth:`rebase` 重建时钟（多标的切换：换帧继续回放）。
        self._on_tick_cb = on_tick
        self._on_finished_cb = on_finished

        def _fill_cb(payload: dict) -> None:
            if on_fill is not None:
                try:
                    on_fill(payload)
                except Exception as exc:  # noqa: BLE001 - 回调异常隔离
                    logger.exception("成交回调异常：%s", exc)

        # 条件单在 tick 后由引擎自动评估；引擎不经总线监听（保持纯 Python 可测）。
        self._clock = ReplayClock(
            ctx,
            on_tick=on_tick,
            on_finished=on_finished,
            emit_events=emit_events,
        )
        self._conditional: List[ConditionalOrder] = []
        self._on_fill_cb = _fill_cb

    # ------------------------------------------------------------------ #
    # 回放控制
    # ------------------------------------------------------------------ #
    def rebase(self) -> None:
        """按 ``ctx`` 当前标的行情**重建回放时钟**（多标的切换后继续逐根回放）。

        时钟在构造时快照标的行情；:meth:`SimContext.set_symbol` 换帧后必须重建，
        否则会沿旧标的的时序推进。结束态一并复位（新标的可能仍有数据）。
        """
        self._clock = ReplayClock(
            self._ctx,
            on_tick=self._on_tick_cb,
            on_finished=self._on_finished_cb,
            emit_events=self._emit_events,
        )

    def tick(self) -> bool:
        """推进一根 1 分钟 bar，随后评估条件单（触发则成交）。

        Returns:
            是否发生了推进（False = 已结束 / 空数据）。
        """
        advanced = self._clock.tick()
        if advanced:
            self._evaluate_conditional()
        return advanced

    def set_speed(self, speed: int) -> None:
        """设置倍速（校验合法；非法抛 ValueError）。"""
        interval_for_speed(speed)  # 仅校验
        self._ctx.speed = int(speed)

    def speed(self) -> int:
        """当前倍速（``0`` 表示暂停）。"""
        return int(getattr(self._ctx, "speed", _PAUSED_SPEED))

    def interval_ms(self) -> int:
        """当前倍速对应的定时器间隔（毫秒）；暂停（speed=0）时返回 None 由 UI 停表。"""
        current = self.speed()
        if current == _PAUSED_SPEED:
            return 0
        return interval_for_speed(current)

    def pause(self) -> None:
        """暂停：把 speed 记为 0（UI 据此停止定时器）。"""
        self._ctx.speed = _PAUSED_SPEED

    def resume(self, speed: Optional[int] = None) -> None:
        """恢复：speed 归回非 0（默认 1），UI 据此重启定时器。"""
        self._ctx.speed = int(speed) if speed is not None else 1

    def is_done(self) -> bool:
        """回放是否结束。"""
        return self._clock.is_done()

    # ------------------------------------------------------------------ #
    # 交易
    # ------------------------------------------------------------------ #
    def place_order(
        self,
        side: str,
        qty: int,
        price: Optional[float] = None,
        reason: str = "manual",
        symbol: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """手动下单：可按当前最新价（``price=None`` → 当前 cursor 最新 close）成交。

        Args:
            side: ``'BUY'`` / ``'SELL'``。
            qty: 股数。
            price: 指定价；``None`` 用当前最新价（无可用价 → 拒）。
            reason: 成交原因（默认 ``manual``）。
            symbol: 下单标的；``None`` 取当前标的 ``ctx.symbol``（多标的会话中
                可为其它已加载标的挂单，账户与持仓共享）。

        Returns:
            ``(True, "")``；``(False, 中文原因)``。
        """
        key = str(symbol) if symbol else self._ctx.symbol
        fill_price = price
        if fill_price is None:
            fill_price = self._ctx.last_price(symbol=key)
            if fill_price is None:
                return False, "当前无最新价（回放未开始或数据为空）"
        ok, reason_text = execute_order(
            self._ctx, side, float(fill_price), int(qty),
            reason=reason, emit_events=self._emit_events, symbol=key,
        )
        if ok and self._on_fill_cb is not None:
            # 从 orders 取最新一笔作同步回调（execute_order 已发总线事件）。
            self._on_fill_cb(
                {
                    "side": side,
                    "symbol": key,
                    "price": float(fill_price),
                    "qty": int(qty),
                    "time": self._ctx.cursor_dt.isoformat() if self._ctx.cursor_dt else "",
                    "reason": reason,
                }
            )
        return ok, reason_text

    # ------------------------------------------------------------------ #
    # 条件单
    # ------------------------------------------------------------------ #
    def add_conditional_order(self, order: ConditionalOrder) -> None:
        """添加一个待触发的条件单（逐 tick 评估）。"""
        self._conditional.append(order)

    def conditional_orders(self) -> List[ConditionalOrder]:
        """当前未触发的条件单列表（副本）。"""
        return list(self._conditional)

    def conditional_order_count(self) -> int:
        """未触发条件单数量。"""
        return len(self._conditional)

    def clear_conditional_orders(self) -> None:
        """清空全部条件单。"""
        self._conditional = []

    def _evaluate_conditional(self) -> List[ConditionalOrder]:
        """评估一次条件单（在 tick 推进后调用），触发者成交并移除。"""
        hit = check_conditional_orders(
            self._ctx,
            self._conditional,
            on_filled=lambda order, price: None,
            emit_events=self._emit_events,
        )
        for order in hit:
            key = str(order.symbol) if order.symbol else self._ctx.symbol
            try:
                self._on_fill_cb(
                    {
                        "side": order.side,
                        "symbol": key,
                        "price": self._ctx.last_price(symbol=key),
                        "qty": int(order.qty),
                        "time": self._ctx.cursor_dt.isoformat() if self._ctx.cursor_dt else "",
                        "reason": "condition",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - 回调异常隔离
                logger.exception("条件单回调异常：%s", exc)
        return hit


__all__ = [
    "SimEngine",
    "SIM_ORDER_FILLED",
    "SIM_STATE_CHANGED",
    "BUY",
    "SELL",
]