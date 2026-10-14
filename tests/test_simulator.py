"""P5-S1 模拟训练 —— SimContext 状态容器 + as_of 未来屏蔽门禁 + ReplayClock 回放时钟。

覆盖（对齐《细化设计》§3.1/§3.2/§3.5）：
    1. as_of 未来屏蔽门禁（⚠️ 最高优先级）：cursor 处后视 = as_of(now)，
       as_of(future) == as_of(now)（未来物理不可见）；cursor=None 时空表；
    2. last_price / is_finished / total_asset（持仓市值估值）；
    3. ReplayClock：逐根推进、交易日历推导、当前索引、done 语义、空数据边界、
       SIM_TICK/SIM_FINISHED 事件与回调、非法 bars 归一降级；
    4. interval_for_speed 纯函数：配置兜底 / 自定义映射 / 非法倍速抛 ValueError；
    5. 多标的（P6）：set_symbol 换帧与 _frames 播种、bars_for 默认/指定标的、
       total_asset / unrealized_pnl 各标的独立计价（无价回退成本）。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.core.event_bus import EventBus, get_bus  # noqa: E402
from app.engine.simulator import (  # noqa: E402
    SIM_FINISHED,
    SIM_TICK,
    Position,
    ReplayClock,
    SimContext,
    Trade,
    interval_for_speed,
)


# --------------------------------------------------------------------------- #
# 合成 1 分钟数据（Bar 契约，两日各 240 根）
# --------------------------------------------------------------------------- #
def _minute_bars(days: int = 2, per_day: int = 240, base: float = 10.0) -> pd.DataFrame:
    """构造升序 1 分钟 Bar 契约数据（09:30-11:30 + 13:00-15:00，忽略午休切分）。
    """
    import numpy as np

    times: list = []
    for day in range(days):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        for i in range(per_day):
            times.append(date + pd.Timedelta(minutes=9 * 60 + 30 + i))
    closes = base + np.linspace(0, days * per_day - 1, days * per_day) * 0.01
    df = pd.DataFrame(
        {
            "datetime": times,
            "open": closes - 0.05,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": np.full(len(times), 1000.0),
            "amount": np.full(len(times), 1_000_000.0),
        }
    )
    return df


def _make_ctx(bars: pd.DataFrame, **kwargs) -> SimContext:
    return SimContext(symbol="000001", start_date="2026-01-01", _bars=bars, **kwargs)


# --------------------------------------------------------------------------- #
# 1. as_of 未来屏蔽门禁（最高优先级）
# --------------------------------------------------------------------------- #
def test_as_of_returns_empty_before_start() -> None:
    ctx = _make_ctx(_minute_bars(1, 10))
    assert ctx.cursor_dt is None
    assert ctx.as_of().empty
    assert ctx.as_of(pd.Timestamp("2030-12-31")).empty  # 就算传未来也看不见


def test_as_of_visible_equals_cursor_past() -> None:
    bars = _minute_bars(1, 10)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    clock.tick()
    clock.tick()
    clock.tick()  # 游标在第 3 根
    now = ctx.cursor_dt
    assert now is not None

    visible = ctx.as_of()
    past = ctx.as_of(now - pd.Timedelta(minutes=1))   # 第 2 根（含）
    earlier = ctx.as_of(now - pd.Timedelta(minutes=5))  # 第 1 根前的时刻 → 0 根
    future = ctx.as_of(now + pd.Timedelta(hours=1))     # 未来请求

    # 现在可见 = 截至游标；过去可见 <= 现在可见；未来请求 == 现在可见（未来物理不可见）。
    assert len(visible) == 3
    assert len(past) == 2
    assert len(earlier) == 0
    assert len(future) == len(visible)
    pd.testing.assert_frame_equal(future, visible)


def test_as_of_gate_strictly_blocks_future_data() -> None:
    """门禁核心反证：游标停在 t3，任何 > t3 的请求都拿不到 t3 之后的数据。"""
    bars = _minute_bars(1, 5)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    # 推进到第 3 根。
    assert clock.tick()
    assert clock.tick()
    assert clock.tick()

    visible = ctx.as_of()
    assert len(visible) == 3
    assert visible["datetime"].iloc[-1] == bars["datetime"].iloc[2]

    far_future = ctx.as_of(bars["datetime"].iloc[-1] + pd.Timedelta(days=365))
    assert len(far_future) == 3  # 不泄露第 4/5 根
    pd.testing.assert_frame_equal(far_future, visible)


def test_as_of_returns_copy_not_reference() -> None:
    bars = _minute_bars(1, 3)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    clock.tick()
    snapshot = ctx.as_of()
    snapshot["close"] = 0.0  # 原地变异副本不得污染内部分析
    assert ctx.last_price() == pytest.approx(float(bars["close"].iloc[0]))


# --------------------------------------------------------------------------- #
# 2. 便捷访问器
# --------------------------------------------------------------------------- #
def test_last_price() -> None:
    bars = _minute_bars(1, 3)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    assert ctx.last_price() is None
    clock.tick()
    assert ctx.last_price() == pytest.approx(float(bars["close"].iloc[0]))
    clock.tick()
    assert ctx.last_price() == pytest.approx(float(bars["close"].iloc[1]))


def test_is_finished() -> None:
    bars = _minute_bars(1, 3)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    assert ctx.is_finished() is False
    clock.tick(); clock.tick(); clock.tick()
    assert ctx.is_finished() is True
    assert clock.is_done() is True


def test_total_asset_with_position() -> None:
    bars = _minute_bars(1, 3)
    ctx = _make_ctx(bars, cash=50_000.0)
    ctx.positions["000001"] = Position(symbol="000001", qty=1000, cost=10.0, buy_date="2026-01-01")
    clock = ReplayClock(ctx, emit_events=False)
    count = 0
    while clock.tick():
        count += 1
        # 总资产 = 现金 + 持股 × 最新价（随游标推进，最新价 = 当前游标的 close）。
        price = ctx.last_price()
        assert price is not None
        assert ctx.total_asset() == pytest.approx(50_000.0 + 1000.0 * price)
    assert count == 3


# --------------------------------------------------------------------------- #
# 3. ReplayClock
# --------------------------------------------------------------------------- #
def test_clock_advances_through_all_bars() -> None:
    bars = _minute_bars(2, 240)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    ticked = 0
    while clock.tick():
        ticked += 1
    assert ticked == 480
    assert clock.is_done() is True
    assert ctx.is_finished() is True
    assert clock.count() == 480
    assert ctx.cursor_dt == bars["datetime"].iloc[-1]


def test_clock_after_done_returns_false_no_tick() -> None:
    bars = _minute_bars(1, 5)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    ticked = 0
    while clock.tick():
        ticked += 1
    assert ticked == 5
    assert clock.tick() is False  # 已结束，不再推进
    assert ctx.cursor_dt == bars["datetime"].iloc[-1]


def test_clock_empty_bars_is_done() -> None:
    ctx = _make_ctx(None)
    clock = ReplayClock(ctx, emit_events=False)
    assert clock.has_data() is False
    assert clock.tick() is False
    assert clock.is_done() is True


def test_clock_missing_columns_degrades_to_empty() -> None:
    df = pd.DataFrame({"foo": [1, 2]})
    ctx = _make_ctx(df)
    clock = ReplayClock(ctx, emit_events=False)
    assert clock.has_data() is False
    assert clock.tick() is False


def test_clock_non_chronological_input_normalized() -> None:
    bars = _minute_bars(1, 5)
    shuffled = bars.sample(frac=1, random_state=42).reset_index(drop=True)  # 乱序
    ctx = _make_ctx(shuffled)
    clock = ReplayClock(ctx, emit_events=False)
    assert clock.trading_days() == ["2026-01-01"]
    assert clock.count() == 5
    first = clock.tick()
    assert first is True
    assert ctx.cursor_dt == sorted(bars["datetime"])[0]


def test_clock_trading_days() -> None:
    bars = _minute_bars(3, 10)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    assert clock.trading_days() == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_clock_current_index() -> None:
    bars = _minute_bars(1, 5)
    ctx = _make_ctx(bars)
    clock = ReplayClock(ctx, emit_events=False)
    assert clock.current_index() == -1
    clock.tick()
    assert clock.current_index() == 0
    clock.tick(); clock.tick()
    assert clock.current_index() == 2


def test_clock_publishes_sim_tick(monkeypatch) -> None:
    bus = EventBus()
    controller = get_bus()
    monkeypatch.setattr(controller, "publish", bus.publish)  # 隔离：换用独立 bus
    events: list = []
    bus.subscribe(SIM_TICK, lambda p: events.append(dict(p)))
    bus.subscribe(SIM_FINISHED, lambda p: events.append(("FIN", dict(p))))

    ctx = _make_ctx(_minute_bars(1, 3))
    clock = ReplayClock(ctx, emit_events=True)  # 走总线
    while clock.tick():
        pass
    assert len(events) == 3 + 1
    assert events[0]["cursor_dt"] is not None and "price" in events[0]
    assert events[-1] == ("FIN", {})


def test_clock_callback_on_tick_and_finished() -> None:
    ticks: list = []
    fin: list = []
    ctx = _make_ctx(_minute_bars(1, 3))
    clock = ReplayClock(ctx, on_tick=ticks.append, on_finished=lambda: fin.append(1),
                        emit_events=False)
    while clock.tick():
        pass
    assert len(ticks) == 3
    assert len(fin) == 1
    assert ticks[0]["price"] == pytest.approx(float(ctx._bars["close"].iloc[0]))


# --------------------------------------------------------------------------- #
# 4. interval_for_speed
# --------------------------------------------------------------------------- #
def test_interval_for_speed_default_map(monkeypatch) -> None:
    monkeypatch.delenv("ASTOCK_CONFIG", raising=False)
    # 兜底内置表（2026-10-01：1x=400ms，用户要求调慢到原来 1/2）。
    assert interval_for_speed(1, None) in (400,)
    assert interval_for_speed(10, None) in (40,)


def test_interval_for_speed_custom_map() -> None:
    assert interval_for_speed(1, {1: 500, 5: 60}) == 500
    assert interval_for_speed(5, {1: 500, 5: 60}) == 60


def test_interval_for_speed_invalid() -> None:
    with pytest.raises(ValueError):
        interval_for_speed(3)          # 不在默认映射
    with pytest.raises(ValueError):
        interval_for_speed("fast")
    with pytest.raises(ValueError):
        interval_for_speed(0, {1: 100})


# --------------------------------------------------------------------------- #
# 5. 数据模型形状
# --------------------------------------------------------------------------- #
def test_position_and_trade_shapes() -> None:
    pos = Position(symbol="000001", qty=200, cost=10.5, buy_date="2026-01-01")
    assert pos.qty == 200 and pos.cost == 10.5

    trade = Trade(symbol="000001", side="BUY", price=10.5, qty=200,
                  dt=pd.Timestamp("2026-01-01 09:30"), reason="manual")
    assert trade.side == "BUY" and trade.reason == "manual"

    ctx = _make_ctx(_minute_bars(1, 3))
    ctx.positions["000001"] = pos
    ctx.orders.append(trade)
    assert ctx.position("000001") is pos
    assert ctx.position("600000") is None
    assert len(ctx.orders) == 1


# --------------------------------------------------------------------------- #
# 6. 多标的（P6）：set_symbol 换帧 / bars_for / 各标的独立计价
# --------------------------------------------------------------------------- #
def test_set_symbol_switches_views_and_keeps_cursor_gate() -> None:
    """换帧后 as_of / last_price 取新标的，且仍受 cursor_dt 截断（无未来数据）。"""
    bars_a = _minute_bars(1, 10, base=10.0)
    bars_b = _minute_bars(1, 10, base=20.0)
    ctx = _make_ctx(bars_a)
    clock = ReplayClock(ctx, emit_events=False)
    clock.tick(); clock.tick(); clock.tick()      # 游标停在第 3 根
    cursor = ctx.cursor_dt
    assert cursor is not None

    ctx.set_symbol("600000", bars_b)
    assert ctx.symbol == "600000"
    # 新标的可见根数与 A 一致（同一时间轴，受同一 cursor 截断）。
    visible = ctx.as_of()
    assert len(visible) == 3
    assert visible["datetime"].iloc[-1] == cursor
    # 最新价改用 B 的 close（第 3 根）。
    assert ctx.last_price() == pytest.approx(float(bars_b["close"].iloc[2]))
    # 未来请求仍被物理屏蔽。
    pd.testing.assert_frame_equal(
        ctx.as_of(cursor + pd.Timedelta(days=365)), visible
    )


def test_set_symbol_seeds_initial_frame_when_frames_empty() -> None:
    """单标的起步：_frames 为空时换标的会播种原标的帧，切走后仍能取原标的价。"""
    bars_a = _minute_bars(1, 10, base=10.0)
    bars_b = _minute_bars(1, 10, base=20.0)
    ctx = _make_ctx(bars_a)
    assert ctx.frames() == {}                      # 起步：尚无多标的表
    clock = ReplayClock(ctx, emit_events=False)
    clock.tick(); clock.tick()
    cursor = ctx.cursor_dt

    ctx.set_symbol("600000", bars_b)
    # 原标的被播种进多标的表（关键补丁），否则切走后其持仓将无价可计。
    assert set(ctx.frames()) == {"000001", "600000"}
    assert not ctx.bars_for("000001").empty
    assert ctx.last_price(symbol="000001") == pytest.approx(
        float(bars_a["close"].iloc[1])
    )
    assert ctx.last_price(symbol="600000") == pytest.approx(
        float(bars_b["close"].iloc[1])
    )
    assert ctx.cursor_dt == cursor                 # 切换不改变游标


def test_bars_for_default_and_explicit_symbol() -> None:
    """bars_for() 默认取当前标的；显式 symbol 取指定标的；未知标的为空表。"""
    bars_a = _minute_bars(1, 10, base=10.0)
    bars_b = _minute_bars(1, 10, base=20.0)
    ctx = _make_ctx(bars_a)
    ctx.set_symbol("600000", bars_b)               # 播种 A，登记 B

    assert len(ctx.bars_for()) == 10               # 默认 = 当前标的 B
    assert float(ctx.bars_for()["close"].iloc[0]) == pytest.approx(20.0)
    assert float(ctx.bars_for("000001")["close"].iloc[0]) == pytest.approx(10.0)
    assert ctx.bars_for("999999").empty            # 未加载标的


def test_bars_for_single_symbol_session() -> None:
    """单标的（从未 set_symbol）：bars_for() 返回当前 _bars。"""
    bars_a = _minute_bars(1, 5)
    ctx = _make_ctx(bars_a)
    assert ctx.frames() == {}
    assert len(ctx.bars_for()) == 5
    assert len(ctx.bars_for("000001")) == 5


def test_total_asset_and_unrealized_pnl_multi_symbol() -> None:
    """多持仓标的按各自最新价计价；无可见价的标的回退成本价。"""
    bars_a = _minute_bars(1, 10, base=10.0)
    bars_b = _minute_bars(1, 10, base=20.0)
    ctx = _make_ctx(bars_a, cash=50_000.0)
    clock = ReplayClock(ctx, emit_events=False)
    clock.tick(); clock.tick()                     # 游标停在第 2 根
    ctx.set_symbol("600000", bars_b)

    ctx.positions["000001"] = Position("000001", 1000, 10.0, "2026-01-01")
    ctx.positions["600000"] = Position("600000", 500, 20.0, "2026-01-01")
    # 无行情标的：回退成本价（市值=成本，未实现盈亏贡献 0）。
    ctx.positions["300001"] = Position("300001", 100, 7.0, "2026-01-01")

    price_a = float(bars_a["close"].iloc[1])
    price_b = float(bars_b["close"].iloc[1])

    assert ctx.total_asset() == pytest.approx(
        50_000.0 + 1000 * price_a + 500 * price_b + 100 * 7.0
    )
    assert ctx.unrealized_pnl() == pytest.approx(
        1000 * (price_a - 10.0) + 500 * (price_b - 20.0)
    )