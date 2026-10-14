"""P5-S2 模拟训练 —— 交易规则校验（T+1 / 做多 / 手数 / 资金 / 持仓）。

覆盖（对齐《细化设计》§3.4 + 本步边界约定）：
    1. 参数校验：方向非法 / 价格非法（0、负、NaN、非数）/ 数量非法（0、负、bool、非整）；
    2. 手数：非 100 倍数拒绝；
    3. 回放未开始（cursor 未启动）→ 拒绝（T+1 无「当日」可判）；
    4. BUY：资金不足拒绝；boundary（正好够）通过；
    5. SELL：无持仓 → 卖空拒绝；持仓不足拒绝；T+1（当日买入不可卖）拒绝；
       次日卖出通过；同日（cursor 与 buy_date 同日、但不同分钟）仍拒绝；
    6. 合法单（BUY 且钱够、SELL 且隔日）通过；
    7. 仓位换算纯函数 max_buy_qty / qty_for_amount：向下取整到整手、非法入参返回 0、
       自定义 lot、math.inf。
"""
from __future__ import annotations

import math  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.engine.simulator import (  # noqa: E402
    BUY,
    SELL,
    Position,
    SimContext,
    max_buy_qty,
    qty_for_amount,
    validate_order,
)
from app.engine.simulator.replay_clock import ReplayClock  # noqa: E402


def _bars(n: int = 5, start: str = "2026-01-05") -> pd.DataFrame:
    """构造连续 1 分钟 bar（start 日 09:30 起，每分钟一根）。"""
    base = pd.Timestamp(start) + pd.Timedelta(hours=9, minutes=30)
    times = [base + pd.Timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame(
        {
            "datetime": times,
            "open": [10.0] * n,
            "high": [10.5] * n,
            "low": [9.5] * n,
            "close": [10.0 + i * 0.1 for i in range(n)],
            "volume": [1000.0] * n,
            "amount": [1_000_000.0] * n,
        }
    )


def _ctx(cash: float = 100_000.0, **kwargs) -> SimContext:
    return SimContext(symbol="000001", cash=cash, _bars=_bars(), **kwargs)


def _start_clock(ctx: SimContext) -> ReplayClock:
    clock = ReplayClock(ctx, emit_events=False)
    return clock


# --------------------------------------------------------------------------- #
# 1. 参数校验
# --------------------------------------------------------------------------- #
def test_side_invalid() -> None:
    ctx = _ctx()
    ok, reason = validate_order(ctx, "HOLD", 10.0, 100)
    assert ok is False and "方向" in reason


def test_price_invalid() -> None:
    ctx = _ctx()
    assert validate_order(ctx, BUY, 0, 100)[0] is False
    assert validate_order(ctx, BUY, -5, 100)[0] is False
    assert validate_order(ctx, BUY, float("nan"), 100)[0] is False
    assert validate_order(ctx, BUY, "abc", 100)[0] is False


def test_qty_invalid() -> None:
    ctx = _ctx()
    assert validate_order(ctx, BUY, 10.0, 0)[0] is False       # 0
    assert validate_order(ctx, BUY, 10.0, -100)[0] is False    # 负
    assert validate_order(ctx, BUY, 10.0, True)[0] is False    # bool 当 1
    assert validate_order(ctx, BUY, 10.0, 100.5)[0] is False   # 非整数


# --------------------------------------------------------------------------- #
# 2. 手数
# --------------------------------------------------------------------------- #
def test_lot_size_requirement() -> None:
    ctx = _ctx()
    ok, reason = validate_order(ctx, BUY, 10.0, 150)
    assert ok is False and "100" in reason


def test_lot_size_ok_boundary() -> None:
    ctx = _started_ctx()
    assert validate_order(ctx, BUY, 10.0, 100)[0] is True


# --------------------------------------------------------------------------- #
# 3. 回放未开始
# --------------------------------------------------------------------------- #
def test_replay_not_started_rejects() -> None:
    ctx = _ctx()
    # 未推进 ReplayClock，cursor 为 None。
    assert validate_order(ctx, BUY, 10.0, 100)[0] is False
    ok, reason = validate_order(ctx, BUY, 10.0, 100)
    assert "回放尚未开始" in reason


# 需要游标已推进的辅助（cursor 落在首根 bar）。
def _started_ctx(**kwargs) -> SimContext:
    ctx = _ctx(**kwargs)
    _start_clock(ctx).tick()
    return ctx


# --------------------------------------------------------------------------- #
# 4. BUY：资金
# --------------------------------------------------------------------------- #
def test_buy_insufficient_cash() -> None:
    ctx = _started_ctx(cash=999.0)
    ok, reason = validate_order(ctx, BUY, 10.0, 100)   # 1000 > 999
    assert ok is False and "现金不足" in reason


def test_buy_exact_cash_ok() -> None:
    ctx = _started_ctx(cash=1000.0)
    assert validate_order(ctx, BUY, 10.0, 100)[0] is True  # 恰好够


# --------------------------------------------------------------------------- #
# 5. SELL：卖空 / 持仓 / T+1
# --------------------------------------------------------------------------- #
def test_sell_without_position_rejected() -> None:
    ctx = _started_ctx()
    ok, reason = validate_order(ctx, SELL, 10.0, 100)
    assert ok is False and "卖空" in reason


def test_sell_exceeds_position_rejected() -> None:
    ctx = _started_ctx()
    ctx.positions["000001"] = Position("000001", qty=100, cost=10.0, buy_date="2026-01-04")
    ok, reason = validate_order(ctx, SELL, 10.0, 200)
    assert ok is False and "持仓不足" in reason


def test_t1_same_day_rejected() -> None:
    """cursor 与 buy_date 同日：即使已过几根分钟 bar，仍不可卖。"""
    ctx = _started_ctx()
    ctx.positions["000001"] = Position("000001", qty=100, cost=10.0, buy_date="2026-01-05")
    ok, reason = validate_order(ctx, SELL, 10.0, 100)
    assert ok is False and "T+1" in reason and "当日" in reason


def test_t1_next_day_ok() -> None:
    ctx = _started_ctx()
    ctx.positions["000001"] = Position("000001", qty=100, cost=10.0, buy_date="2026-01-04")
    ok, reason = validate_order(ctx, SELL, 10.0, 100)
    assert ok is True, reason


def test_t1_older_buy_date_ok() -> None:
    ctx = _started_ctx()
    ctx.positions["000001"] = Position("000001", qty=1000, cost=9.0, buy_date="2025-12-31")
    assert validate_order(ctx, SELL, 10.0, 1000)[0] is True


# --------------------------------------------------------------------------- #
# 6. 正常交易路径
# --------------------------------------------------------------------------- #
def test_buy_then_sell_next_day_path() -> None:
    ctx = _started_ctx(cash=100_000.0)
    assert validate_order(ctx, BUY, 10.0, 1000)[0] is True

    # 同日立即卖 → T+1 拒（模拟当日买入）。
    ctx.positions["000001"] = Position("000001", qty=1000, cost=10.0, buy_date="2026-01-05")
    ok, reason = validate_order(ctx, SELL, 10.0, 1000)
    assert ok is False and "T+1" in reason

    # 次日卖 → 通过。
    ctx.positions["000001"] = Position("000001", qty=1000, cost=10.0, buy_date="2026-01-04")
    assert validate_order(ctx, SELL, 10.0, 1000)[0] is True


# --------------------------------------------------------------------------- #
# 7. 仓位换算纯函数（多标的改造新增）：max_buy_qty / qty_for_amount
# --------------------------------------------------------------------------- #
def test_max_buy_qty_floors_to_lot() -> None:
    """正常入参：向下取整到整手（100000 / 10.5 → 9500）。"""
    assert max_buy_qty(100_000.0, 10.5) == 9500
    assert max_buy_qty(100_000.0, 10.5) % 100 == 0


def test_qty_for_amount_floors_to_lot() -> None:
    """金额换算：向下取整到整手（5000 / 10 → 500；4500 / 10 → 400）。"""
    assert qty_for_amount(5_000.0, 10.0) == 500
    assert qty_for_amount(4_500.0, 10.0) == 400
    assert qty_for_amount(4_500.0, 10.0) % 100 == 0


def test_buy_qty_insufficient_for_one_lot_returns_zero() -> None:
    """不足一手 → 0。"""
    assert max_buy_qty(500.0, 10.5) == 0        # 500 < 10.5 * 100
    assert qty_for_amount(500.0, 10.0) == 0     # 500 < 10 * 100


def test_buy_qty_invalid_price_returns_zero() -> None:
    """价格 0 / 负数 / NaN / None / 字符串 → 0。"""
    for bad in (0, -5, float("nan"), None, "abc"):
        assert max_buy_qty(100_000.0, bad) == 0, bad
        assert qty_for_amount(100_000.0, bad) == 0, bad


def test_buy_qty_invalid_amount_returns_zero() -> None:
    """现金 / 金额 0 / 负数 / None / 字符串 → 0。"""
    for bad in (0, -1_000, None, "abc"):
        assert max_buy_qty(bad, 10.0) == 0, bad
        assert qty_for_amount(bad, 10.0) == 0, bad


def test_buy_qty_custom_lot() -> None:
    """自定义一手股数（lot=10）；非法 lot（<=0）→ 0。"""
    assert max_buy_qty(1_000.0, 10.0, lot=10) == 100
    assert qty_for_amount(1_000.0, 10.0, lot=10) == 100
    assert max_buy_qty(1_000.0, 10.0, lot=0) == 0
    assert qty_for_amount(1_000.0, 10.0, lot=-1) == 0


def test_buy_qty_inf_returns_zero() -> None:
    """math.inf 入参（现金 / 金额 / 价格）→ 0（非有限数）。"""
    assert max_buy_qty(math.inf, 10.0) == 0
    assert max_buy_qty(100_000.0, math.inf) == 0
    assert qty_for_amount(math.inf, 10.0) == 0
    assert qty_for_amount(100_000.0, math.inf) == 0