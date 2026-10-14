"""P5-S3 模拟训练 —— SimEngine 驱动 + 成交执行 + 条件单。

覆盖（对齐《细化设计》§3.2/§3.3）：
    1. 手动下单：无价取当前最新价成交；指定价成交；资金/持仓/Trade 记录正确；
    2. 成交执行：买入（加权成本、buy_date=当日）、卖出（减仓、归零移除）、
       realized/unrealized PnL、initial_cash 首次记账；
    3. 事件：SIM_ORDER_FILLED / SIM_STATE_CHANGED 走总线（payload 契约）；
    4. 条件单：价格触发（BUY 低于、SELL 高于）、涨跌幅触发（>=/<=-）、一次性移除、
       成交价=触发时最新价、reason='condition'；
    5. 倍速/暂停：set_speed 校验、pause/resume、interval_ms 映射；
    6. 多标的（P6）：rebase 后游标不跳变并继续逐根、place_order(symbol=) 分别记账
       且现金共享、SELL 按目标标的持仓校验、条件单绑定添加时标的。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.core.event_bus import EventBus  # noqa: E402
from app.engine.simulator import (  # noqa: E402
    BUY,
    SELL,
    ConditionalOrder,
    SIM_ORDER_FILLED,
    SIM_STATE_CHANGED,
    Position,
    SimContext,
    SimEngine,
    interval_for_speed,
)


def _bars(n: int = 40, start: str = "2026-01-05") -> pd.DataFrame:
    """构造连续 1 分钟 bar（首日 09:30 起，价格线性上升）。"""
    base = pd.Timestamp(start) + pd.Timedelta(hours=9, minutes=30)
    times = [base + pd.Timedelta(minutes=i) for i in range(n)]
    closes = [10.0 + i * 0.05 for i in range(n)]
    return pd.DataFrame(
        {
            "datetime": times,
            "open": [c - 0.02 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "amount": [1_000_000.0] * n,
        }
    )


def _ctx(cash: float = 100_000.0, **kwargs) -> SimContext:
    return SimContext(symbol="000001", cash=cash, _bars=_bars(), **kwargs)


# --------------------------------------------------------------------------- #
# 1. 手动下单
# --------------------------------------------------------------------------- #
def test_manual_order_uses_cursor_price_when_none(qapp=None) -> None:
    ctx = _ctx()
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()  # 首根 close=10.0
    ok, reason = engine.place_order(BUY, 100)
    assert ok is True, reason
    pos = ctx.position("000001")
    assert pos is not None and pos.qty == 100
    # 成本=10.0（首根），现金减少 1000。
    assert ctx.cash == pytest.approx(100_000 - 1000.0)
    assert len(ctx.orders) == 1
    assert ctx.orders[0].reason == "manual"


def test_manual_order_explicit_price() -> None:
    ctx = _ctx()
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()
    engine.tick()  # 第二根 close=10.05
    ok, _ = engine.place_order(BUY, 100, price=10.0)
    assert ok is True
    assert ctx.cash == pytest.approx(100_000 - 1000.0)


def test_manual_sell_reduces_position_and_cash(qapp=None) -> None:
    ctx = _ctx()
    # 预置前一日买入持仓（次日方可卖）。
    ctx.positions["000001"] = Position("000001", 500, 10.0, "2026-01-04")
    ctx.cash = 95_000.0
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()  # 首根 close=10.0
    ok, reason = engine.place_order(SELL, 200)
    assert ok is True, reason
    pos = ctx.position("000001")
    assert pos is not None and pos.qty == 300
    assert ctx.cash == pytest.approx(95_000 + 200 * 10.0)


def test_manual_order_rejected_before_replay(qapp=None) -> None:
    ctx = _ctx()
    engine = SimEngine(ctx, emit_events=False)
    ok, reason = engine.place_order(BUY, 100)  # 未 tick，无最新价
    assert ok is False and "最新价" in reason


# --------------------------------------------------------------------------- #
# 2. 成交执行语义
# --------------------------------------------------------------------------- #
def test_buy_second_lot_averages_cost_and_refreshes_buy_date(qapp=None) -> None:
    ctx = _ctx()
    ctx.positions["000001"] = Position("000001", 100, 10.0, "2026-01-04")
    ctx.cash = 98_000.0
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()  # close=10.0
    engine.tick()  # 10.05
    ok, _ = engine.place_order(BUY, 100)  # 以 10.05 加仓
    assert ok is True
    pos = ctx.position("000001")
    assert pos.qty == 200
    assert pos.cost == pytest.approx((100 * 10.0 + 100 * 10.05) / 200)
    assert pos.buy_date == "2026-01-05"  # T+1 从严：最近买入日


def test_sell_all_removes_position(qapp=None) -> None:
    ctx = _ctx()
    ctx.positions["000001"] = Position("000001", 300, 10.0, "2026-01-04")
    ctx.cash = 97_000.0
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()
    ok, _ = engine.place_order(SELL, 300)
    assert ok is True
    assert ctx.position("000001") is None
    assert ctx.cash == pytest.approx(97_000 + 300 * 10.0)


def test_execution_disallowed_same_day_t1(qapp=None) -> None:
    ctx = _ctx()
    ctx.positions["000001"] = Position("000001", 300, 10.0, "2026-01-05")  # 当日买入
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()
    ok, reason = engine.place_order(SELL, 100)
    assert ok is False and "T+1" in reason


def test_initial_cash_snapshot_on_first_fill(qapp=None) -> None:
    ctx = _ctx(cash=50_000.0)
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()
    assert ctx.initial_cash is None
    engine.place_order(BUY, 100)
    assert ctx.initial_cash == ctx.cash + 1000.0  # 首笔成交前快照 = 原始现金


def test_realized_unrealized_pnl(qapp=None) -> None:
    ctx = _ctx(cash=100_000.0)
    ctx.positions["000001"] = Position("000001", 500, 10.0, "2026-01-04")
    ctx.cash = 95_000.0
    ctx.initial_cash = 100_000.0
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()  # close=10.0
    assert ctx.unrealized_pnl() == pytest.approx(0.0)  # 市价=成本
    engine.tick()  # close=10.05
    assert ctx.unrealized_pnl() == pytest.approx(500 * 0.05)
    ok, _ = engine.place_order(SELL, 500)
    assert ok is True
    assert ctx.realized_pnl() == pytest.approx(500 * 0.05)


# --------------------------------------------------------------------------- #
# 3. 事件
# --------------------------------------------------------------------------- #
def test_fill_and_state_events_published(monkeypatch, qapp=None) -> None:
    bus = EventBus()
    filled: list = []
    state: list = []
    bus.subscribe(SIM_ORDER_FILLED, lambda p: filled.append(dict(p)))
    bus.subscribe(SIM_STATE_CHANGED, lambda p: state.append(dict(p)))
    # 引擎/执行器内部经 get_bus() 发布；换成独立 bus 以隔离全局单例。
    monkeypatch.setattr("app.engine.simulator.executor.get_bus", lambda: bus)

    ctx = _ctx()
    ctx.positions["000001"] = Position("000001", 500, 10.0, "2026-01-04")
    ctx.cash = 95_000.0
    engine = SimEngine(ctx, emit_events=True)  # 走总线
    engine.tick()
    ok, _ = engine.place_order(SELL, 200)
    assert ok is True

    assert len(filled) == 1
    ev = filled[0]
    assert ev["side"] == "SELL" and ev["symbol"] == "000001"
    assert ev["qty"] == 200 and ev["reason"] == "manual" and "time" in ev
    assert len(state) >= 1
    assert state[0]["cash"] == pytest.approx(95_000 + 200 * 10.0)
    assert "positions" in state[0] and "pnl" in state[0]


# --------------------------------------------------------------------------- #
# 4. 条件单
# --------------------------------------------------------------------------- #
def test_condition_price_buy_fires_once(qapp=None) -> None:
    ctx = _ctx()
    engine = SimEngine(ctx, emit_events=False)
    engine.add_conditional_order(ConditionalOrder(BUY, "price", 10.0, 100))
    # 首根 close=10.0 恰好 <= 10.0 → 首 tick 即触发。
    engine.tick()
    assert ctx.position("000001") is not None and ctx.position("000001").qty == 100
    # 单次触发：条件单已移除，再推进不重复买。
    engine.tick()
    assert ctx.position("000001").qty == 100
    assert engine.conditional_order_count() == 0


def test_condition_price_buy_not_triggered_above(qapp=None) -> None:
    ctx = _ctx()
    engine = SimEngine(ctx, emit_events=False)
    engine.add_conditional_order(ConditionalOrder(BUY, "price", 9.0, 100))  # 目标 9.0
    engine.tick()  # close=10.0 > 9.0 不触发
    assert ctx.position("000001") is None
    # 价格持续上行，始终不触发。
    for _ in range(30):
        engine.tick()
    assert engine.conditional_order_count() == 1


def test_condition_price_sell_fires_on_rise(qapp=None) -> None:
    ctx = _ctx()
    ctx.positions["000001"] = Position("000001", 200, 10.0, "2026-01-04")
    engine = SimEngine(ctx, emit_events=False)
    engine.add_conditional_order(ConditionalOrder(SELL, "price", 10.5, 100))
    engine.tick()  # 10.0
    assert ctx.position("000001").qty == 200
    for _ in range(20):
        engine.tick()  # close 升至 11.0 → 越过 10.5 触发
    assert ctx.position("000001").qty == 100
    assert ctx.orders[-1].reason == "condition"


def test_condition_pct_buy_fires_on_gain(qapp=None) -> None:
    ctx = _ctx()
    engine = SimEngine(ctx, emit_events=False)
    engine.add_conditional_order(ConditionalOrder(BUY, "pct", 2.0, 100))  # 涨幅 ≥2%
    engine.tick()  # 10.0 → pct 0
    engine.tick()  # 10.05 → 0.5%
    for _ in range(8):
        engine.tick()  # 10.15 → 1.5% … 10.45 → 4.5%
    assert ctx.position("000001") is not None
    # 触发价 = 越过阈值那根的 close（约 10.2）。
    assert ctx.orders[0].price >= 10.2 * 0.999


def test_condition_pct_sell_fires_on_drop(qapp=None) -> None:
    fall = _bars()
    fall["close"] = [10.0 - i * 0.2 for i in range(40)]  # 首日下跌
    ctx = SimContext(symbol="000001", cash=100_000.0, _bars=fall)
    ctx.positions["000001"] = Position("000001", 200, 10.0, "2026-01-04")
    engine = SimEngine(ctx, emit_events=False)
    engine.add_conditional_order(ConditionalOrder(SELL, "pct", 3.0, 100))  # 跌幅 ≥3%
    for _ in range(10):
        engine.tick()  # 10.0 → 8.0
    # 跌幅 3% 出现在 close≈9.7（第 2 根 9.6 → -4%）
    assert ctx.position("000001").qty == 100


def test_condition_order_reason_condition_and_callback(qapp=None) -> None:
    ctx = _ctx()
    fills: list = []

    def on_fill(payload: dict) -> None:
        fills.append(dict(payload))

    engine = SimEngine(ctx, emit_events=False, on_fill=on_fill)
    engine.add_conditional_order(ConditionalOrder(BUY, "price", 10.0, 100))
    engine.tick()
    assert len(fills) == 1
    assert fills[0]["reason"] == "condition"


# --------------------------------------------------------------------------- #
# 5. 倍速 / 暂停
# --------------------------------------------------------------------------- #
def test_speed_and_interval() -> None:
    ctx = _ctx()
    engine = SimEngine(ctx)
    assert engine.speed() == 1
    assert engine.interval_ms() == interval_for_speed(1)
    engine.set_speed(5)
    assert engine.speed() == 5
    assert engine.interval_ms() == interval_for_speed(5)


def test_speed_invalid_raises() -> None:
    engine = SimEngine(_ctx())
    with pytest.raises(ValueError):
        engine.set_speed(3)  # 不在默认映射


def test_pause_resume() -> None:
    ctx = _ctx()
    engine = SimEngine(ctx)
    engine.pause()
    assert engine.speed() == 0
    assert engine.interval_ms() == 0
    engine.resume()
    assert engine.speed() == 1
    engine.resume(speed=10)
    assert engine.speed() == 10


def test_replay_completes(qapp=None) -> None:
    ctx = _ctx(40)
    engine = SimEngine(ctx, emit_events=False)
    count = 0
    while engine.tick():
        count += 1
    assert count == 40
    assert engine.is_done() is True


# --------------------------------------------------------------------------- #
# 6. 多标的（P6）：rebase 继续回放 + 指定标的成交 + 条件单绑定标的
# --------------------------------------------------------------------------- #
def test_rebase_keeps_cursor_and_continues_on_new_frame(qapp=None) -> None:
    """rebase 后游标不变，之后按新标的帧继续逐根推进。"""
    ctx = _ctx()  # A：000001，close 10.0 起递增 0.05
    engine = SimEngine(ctx, emit_events=False)
    engine.tick(); engine.tick(); engine.tick()
    cursor = ctx.cursor_dt
    assert cursor == pd.Timestamp("2026-01-05 09:32")

    bars_b = _bars()
    bars_b["close"] = [20.0 + i * 0.05 for i in range(len(bars_b))]
    ctx.set_symbol("600000", bars_b)
    engine.rebase()                                # 换帧重建时钟
    assert ctx.cursor_dt == cursor                 # 游标不跳变
    assert ctx.last_price() == pytest.approx(float(bars_b["close"].iloc[2]))

    assert engine.tick() is True
    assert ctx.cursor_dt == cursor + pd.Timedelta(minutes=1)   # 逐根推进
    assert ctx.last_price() == pytest.approx(float(bars_b["close"].iloc[3]))


def test_place_order_multi_symbol_shares_cash(qapp=None) -> None:
    """按 symbol 分别成交与记账；现金为账户级共享（买 A 后买 B 扣同一现金）。"""
    ctx = _ctx(cash=100_000.0)
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()                                  # 启动游标（A 首根）
    bars_b = _bars()
    bars_b["close"] = [20.0] * len(bars_b)
    ctx.set_symbol("600000", bars_b)
    engine.rebase()

    ok_a, reason_a = engine.place_order(BUY, 100, price=10.0, symbol="000001")
    assert ok_a is True, reason_a
    assert ctx.cash == pytest.approx(100_000.0 - 1000.0)
    assert ctx.position("000001").qty == 100

    ok_b, reason_b = engine.place_order(BUY, 100, price=20.0, symbol="600000")
    assert ok_b is True, reason_b
    # 同一账户现金继续扣减；两标的持仓各自记账。
    assert ctx.cash == pytest.approx(100_000.0 - 1000.0 - 2000.0)
    assert ctx.position("000001").qty == 100
    assert ctx.position("600000").qty == 100
    assert ctx.orders[-1].symbol == "600000"


def test_sell_validates_against_target_symbol_position(qapp=None) -> None:
    """卖出按目标标的的持仓校验：A 的持仓不能用于 B 的卖出。"""
    ctx = _ctx(cash=100_000.0)
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()
    bars_b = _bars()
    bars_b["close"] = [20.0] * len(bars_b)
    ctx.set_symbol("600000", bars_b)
    engine.rebase()

    # 仅持有 A（前一日买入），未持有 B → 卖 B 视为卖空。
    ctx.positions["000001"] = Position("000001", 1000, 10.0, "2026-01-04")
    ok, reason = engine.place_order(SELL, 100, price=20.0, symbol="600000")
    assert ok is False and "卖空" in reason
    assert ctx.position("000001").qty == 1000      # A 持仓未被误动

    # B 持仓 100，卖 200 → 持仓不足（即便 A 有 1000 股）。
    ctx.positions["600000"] = Position("600000", 100, 20.0, "2026-01-04")
    ok2, reason2 = engine.place_order(SELL, 200, price=20.0, symbol="600000")
    assert ok2 is False and "持仓不足" in reason2

    # 卖 B 100 → 成功，仅 B 归零。
    ok3, _ = engine.place_order(SELL, 100, price=20.0, symbol="600000")
    assert ok3 is True
    assert ctx.position("600000") is None
    assert ctx.position("000001").qty == 1000


def test_conditional_order_bound_symbol_survives_switch(qapp=None) -> None:
    """条件单绑定添加时的标的：切到 B 后仍按 A 的价格触发。"""
    ctx = _ctx(cash=100_000.0)                     # A：close 10.0 → 11.95
    ctx.positions["000001"] = Position("000001", 100, 10.0, "2026-01-04")
    engine = SimEngine(ctx, emit_events=False)
    engine.tick()                                  # 游标到 A 首根（close=10.0）
    engine.add_conditional_order(
        ConditionalOrder(SELL, "price", 10.5, 100, symbol="000001")
    )

    # 切到 B（低位横盘 5.0）：若条件单误用「当前标的」将永不触发。
    bars_b = _bars()
    bars_b["close"] = [5.0] * len(bars_b)
    ctx.set_symbol("600000", bars_b)
    engine.rebase()
    assert ctx.last_price() == pytest.approx(5.0)

    for _ in range(20):
        engine.tick()                              # A 第 11 根 close=10.5 → 触发
        if engine.conditional_order_count() == 0:
            break
    assert engine.conditional_order_count() == 0
    assert ctx.position("000001") is None          # A 已卖出
    assert ctx.orders[-1].symbol == "000001"
    assert ctx.orders[-1].reason == "condition"