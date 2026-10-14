"""P5-S4 模拟训练窗口 —— UI 层测试（P6 优化：周期/起始日/固定显示区/按日回放/盈亏曲线）。

覆盖（对齐《细化设计》§3.5 / §4「P5-S4」验收 + 2026-10-01 用户优化项）：
    1. 空数据：库无分钟数据 → 开始按钮禁用、状态提示；
    2. 回放启动：加载数据 → 引擎建立、定时器启动、逐 tick 推进；
    3. **as_of 视图受限（look-ahead 落到 UI）**：主图只显示截至游标的 bar，绝不显示未来；
    4. **固定显示区**：主图 X 视窗固定为一个交易日的槽位数（逐根显现时宽度恒定）；
    5. **按日回放**：当日放完自动暂停 + 点亮「下一天」，点击后只显示下一交易日的 K 线；
    6. **周期 / 起始日**：1m 选项保留但禁用（含提示），起始日可自选且数据从该日切片；
       15/30/60m 本地缺失时从 5m 派生；
    7. **盈亏**：账户指标（已实现 / 浮动 / 总资产 / 收益率）与盈亏曲线逐根记录；
    8. 手动买入/卖出/平仓、条件单、倍速/暂停/重置；
    9. 菜单接线（MainWindow）：「训练(&T) → 模拟训练」存在且触发不抛。
"""
from __future__ import annotations

import os
from datetime import date

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QCoreApplication, QDate, QEvent, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.data.update_service import UpdateReport  # noqa: E402
from app.engine.simulator import BUY, SELL, max_buy_qty, qty_for_amount  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.widgets.sim_window import (  # noqa: E402
    SimWindow,
    is_rolling_window,
    window_slots,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _session_times(day: str, per_day: int) -> list:
    """按 A 股交易时段生成 5 分钟 bar 时间戳（时间戳 = bar 结束时刻）。

    上午 09:35–11:30 共 24 根、下午 13:05–15:00 共 24 根；``per_day`` 截断取前 N 根。
    """
    morning = [
        pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=35) + pd.Timedelta(minutes=5 * i)
        for i in range(24)
    ]
    afternoon = [
        pd.Timestamp(day) + pd.Timedelta(hours=13, minutes=5) + pd.Timedelta(minutes=5 * i)
        for i in range(24)
    ]
    return (morning + afternoon)[:per_day]


def _bars_5m(days: list, per_day: int = 48, base: float = 10.0) -> pd.DataFrame:
    """构造多日 5 分钟 Bar 契约数据（线性上升，便于断言）。"""
    rows = []
    step = 0
    for day in days:
        for ts in _session_times(day, per_day):
            close = base + 0.01 * step
            step += 1
            rows.append(
                {
                    "datetime": ts,
                    "open": close - 0.02,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 1000.0,
                    "amount": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _seed_minute(
    db: Database,
    symbol: str = "000001",
    days: list | None = None,
    per_day: int = 48,
    period: str = "5m",
    base: float = 10.0,
) -> None:
    """把 5 分钟数据写入库（默认单日 2026-01-05；``base`` 可抬高价格用于资金不足等场景）。"""
    frame = _bars_5m(days or ["2026-01-05"], per_day, base=base).copy()
    frame["symbol"] = symbol
    frame["period"] = period
    frame["adjust"] = ""
    db.upsert_bars(frame)


class _FakeSource:
    name = "fake"

    def get_kline(self, *a, **k):
        return None

    def get_minute(self, *a, **k):
        return None

    def list_stocks(self):
        return None

    def get_spot(self):
        return None

    def get_adj_factor(self, *a, **k):
        return None

    def get_fundamental(self, *a, **k):
        return None


def _make_window(qapp, db: Database) -> MainWindow:
    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    return MainWindow(db=db, service=service)


def _make_sim_window(qapp, db: Database, symbol: str = "000001") -> SimWindow:
    win = SimWindow(db=db, symbols=[symbol], default_symbol=symbol)
    win.show()
    qapp.processEvents()
    return win


# --------------------------------------------------------------------------- #
# 1. 纯函数：周期 → 固定显示槽位数
# --------------------------------------------------------------------------- #
def test_window_slots_period_constants() -> None:
    assert window_slots("5m") == 48   # 240 / 5
    assert window_slots("15m") == 16
    assert window_slots("30m") == 8
    assert window_slots("60m") == 4
    assert window_slots("d") == 60    # 日K 滚动窗口
    assert is_rolling_window("d") is True
    assert is_rolling_window("5m") is False


# --------------------------------------------------------------------------- #
# 2. 空数据策略
# --------------------------------------------------------------------------- #
def test_empty_data_disables_start(qapp) -> None:
    db = _mem_db()  # 无任何数据
    win = _make_sim_window(qapp, db, symbol="000001")
    try:
        win.start_replay()
        assert win._btn_start.isEnabled() is False
        assert "本地暂无分钟数据" in win._status_label.text()
    finally:
        win.close()


def test_no_symbol_prompts(qapp) -> None:
    db = _mem_db()
    win = SimWindow(db=db, symbols=None, default_symbol=None)
    win.show()
    try:
        win.start_replay()
        assert "先输入标的代码" in win._status_label.text()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 3. 回放启动与推进（as_of 受限 + 固定显示区）
# --------------------------------------------------------------------------- #
def test_start_creates_engine_and_timer(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001")
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        assert win._engine is not None
        assert win._timer.isActive() is True
        # 手动驱动一次（不进事件循环等待真实 timer）。
        win._on_timer()
        ctx = win._engine_ctx()
        assert ctx is not None and ctx.cursor_dt is not None
    finally:
        win.close()


def test_tick_advances_and_view_is_as_of_limited(qapp) -> None:
    """关键 look-ahead 用例：第 k 次 tick 后视图只显示 k 根（截至游标），无未来。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=48)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        # 手动推进 5 根。
        for _ in range(5):
            win._on_timer()
        ctx = win._engine_ctx()
        assert ctx is not None
        assert ctx.cursor_dt == ctx._bars["datetime"].iloc[4]  # 游标停在第 5 根
        assert win._chart.bar_count() == 5  # 视图只显示截至游标的 5 根
        # 视图数据的最后一根 == 游标时刻（绝不显示第 6 根）。
        last_dt = pd.to_datetime(win._chart._bars["datetime"]).max()
        assert last_dt == ctx.cursor_dt
        # 固定显示区 + 最新根居中：视窗宽度恒为 48 槽，最新根（索引 4）位于中心。
        lo, hi = win._chart._x_range
        assert hi - lo == pytest.approx(float(win.window_slots()))
        assert (lo + hi) / 2.0 == pytest.approx(4.0)
    finally:
        win.close()


def test_full_playback_reaches_finish(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        for _ in range(20):  # 够多
            win._on_timer()
        assert win._timer.isActive() is False  # 回放结束停表
        ctx = win._engine_ctx()
        assert ctx is not None and ctx.cursor_dt == ctx._bars["datetime"].iloc[-1]
        assert "回放完成" in win._status_label.text()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 4. 按日回放：「下一天」按钮
# --------------------------------------------------------------------------- #
def test_day_boundary_pauses_and_next_day_shows_only_new_day(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", days=["2026-01-05", "2026-01-06"], per_day=6)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        # 当日 6 根全部放完 → 自动暂停 + 点亮「下一天」。
        for _ in range(6):
            win._on_timer()
        assert win._timer.isActive() is False
        assert win._btn_next_day.isEnabled() is True
        assert "下一天" in win._status_label.text()
        assert win._chart.bar_count() == 6
        assert win._day == date(2026, 1, 5)

        # 点击「下一天」→ 只显示下一交易日的 K 线（第 1 根），计时器恢复。
        win.next_day()
        assert win._day == date(2026, 1, 6)
        assert win._chart.bar_count() == 1
        assert win._btn_next_day.isEnabled() is False
        assert win._timer.isActive() is True
        ctx = win._engine_ctx()
        assert ctx is not None
        assert ctx.cursor_dt == ctx._bars["datetime"].iloc[6]  # 第 7 根 = 次日首根
        # 图上只画当日（次日）的数据，不含 1/5 的根。
        shown = pd.to_datetime(win._chart._bars["datetime"]).dt.date.unique().tolist()
        assert shown == [date(2026, 1, 6)]
    finally:
        win.close()


def test_pause_toggle_hints_next_day_when_day_complete(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", days=["2026-01-05", "2026-01-06"], per_day=3)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        for _ in range(3):
            win._on_timer()
        assert win._btn_next_day.isEnabled() is True
        win.toggle_pause()  # 当日已放完 → 不重启定时器，只提示
        assert win._timer.isActive() is False
        assert "下一天" in win._status_label.text()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 5. 周期 / 起始日
# --------------------------------------------------------------------------- #
def test_period_choices_and_disabled_1m(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001")
    win = _make_sim_window(qapp, db, "000001")
    try:
        codes = [
            str(win._period_combo.itemData(i)) for i in range(win._period_combo.count())
        ]
        assert codes == ["1m", "5m", "15m", "30m", "60m", "d"]
        assert win.current_period() == "5m"  # 默认 5m
        # 1m 项禁用 + 悬停提示。
        model = win._period_combo.model()
        item_1m = model.item(0)
        assert bool(item_1m.flags() & Qt.ItemFlag.ItemIsEnabled) is False
        assert "1 分钟链路不可用" in str(
            win._period_combo.itemData(0, Qt.ItemDataRole.ToolTipRole)
        )
    finally:
        win.close()


def test_start_date_slices_session(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", days=["2026-01-05", "2026-01-06"], per_day=6)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win._date_edit.setDate(QDate(2026, 1, 6))  # 自选起始日 = 次日
        win.start_replay()
        assert win._day == date(2026, 1, 6)
        assert len(win._bars) == 6  # 只保留下单日（1/5 被切掉）
        first = pd.to_datetime(win._bars["datetime"].iloc[0]).date()
        assert first == date(2026, 1, 6)
    finally:
        win.close()


def test_derived_period_from_local_5m(qapp) -> None:
    """15/30/60m 本地缺失 → 从 5m 派生（不落库）。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=48)  # 一整天 48 根 5m
    win = _make_sim_window(qapp, db, "000001")
    try:
        idx = win._period_combo.findData("60m")
        win._period_combo.setCurrentIndex(idx)
        assert win.current_period() == "60m"
        win.start_replay()
        assert len(win._bars) == 4  # 240 / 60 = 4 根
        assert win.window_slots() == 4
        # 派生不落库：库里仍无 60m 数据。
        assert db.get_bars("000001", "60m").empty
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 6. 账户指标与盈亏曲线
# --------------------------------------------------------------------------- #
def test_account_metrics_and_equity_curve(qapp, monkeypatch) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        win._on_timer()
        win._qty_spin.setValue(100)
        win._on_place_order(BUY)
        # 账户指标：四个数值标签都有内容。
        assert "已实现盈亏：" in win._realized_label.text()
        assert "浮动盈亏：" in win._unrealized_label.text()
        assert "总资产：" in win._asset_label.text()
        assert "收益率：" in win._return_label.text()
        # 盈亏曲线：逐根记录（启动基线 + 每 tick + 每笔成交），曲线点数与记录一致。
        assert len(win._equity) >= 3
        xs, ys = win._equity_curve.getData()
        assert len(xs) == len(win._equity) == len(ys)
        fx, fy = win._float_curve.getData()
        assert len(fx) == len(win._equity) == len(fy)
        # 累计盈亏 = 总资产 - 期初资金（此时持仓成本=最新价，浮盈为 0）。
        ctx = win._engine_ctx()
        assert ctx is not None
        assert "累计盈亏" in win._eq_cum_label.text()
        assert win._equity[-1][1] == pytest.approx(ctx.total_asset())
    finally:
        win.close()


def test_equity_reset_on_stop(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        win._on_timer()
        assert win._equity != []
        win.stop_replay()
        assert win._equity == []
        assert win._equity_curve.getData()[0] is None or len(win._equity_curve.getData()[0]) == 0
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 7. 手动下单 / 平仓
# --------------------------------------------------------------------------- #
def test_manual_buy_updates_state(qapp, monkeypatch) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    # 隔离模态弹窗：QMessageBox.warning 在离屏下 exec 会阻塞测试，替换为 no-op。
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        win._on_timer()  # 首根 close=10.0
        win._qty_spin.setValue(100)
        win._on_place_order(BUY)
        ctx = win._engine_ctx()
        assert ctx is not None
        pos = ctx.position("000001")
        assert pos is not None and pos.qty == 100
        assert ctx.cash == pytest.approx(100_000 - 1000.0)
        assert "买入" in win._status_label.text()
        # 成交表 1 行。
        assert win._trades_table.rowCount() == 1
    finally:
        win.close()


def test_manual_sell_and_close(qapp, monkeypatch) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        win._on_timer()
        win._qty_spin.setValue(100)
        win._on_place_order(BUY)
        # T+1 当日不可卖（引擎校验）。
        win._on_place_order(SELL)
        ctx = win._engine_ctx()
        assert ctx.position("000001") is not None  # 卖出被拒，持仓仍在
        assert len(ctx.orders) == 1
        # 平仓同样受 T+1 挡住（当日买入）；次日才能平 → 直接推进到次日不足以（同日），
        # 我们验证「平仓被拒不崩溃、状态提示」。
        win._on_close_position()
        assert ctx.position("000001") is not None
    finally:
        win.close()


def test_close_position_when_absent_prompts(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        win._on_timer()
        win._on_close_position()
        assert "无持仓" in win._status_label.text()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 7b. ② 下单辅助：可买卖股数 / 全仓 / 1/2仓 / 1/4仓 / 按金额
# --------------------------------------------------------------------------- #
def test_avail_label_and_quick_buy_fraction(qapp, monkeypatch) -> None:
    """② 面板显示可买（现金整手取整）/ 可卖（T+1）股数；1/4 仓按钮直接成交。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)  # 首根 close = 10.0
    win = _make_sim_window(qapp, db, "000001")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        win._on_timer()  # 首根可见，最新价 = 10.0
        ctx = win._engine_ctx()
        price = ctx.last_price()
        assert price == pytest.approx(10.0)
        assert f"可买 {max_buy_qty(ctx.cash, price)} 股" in win._avail_label.text()
        assert "可卖 0 股" in win._avail_label.text()

        win._quick_buy(0.25)  # 1/4 仓
        expected = max_buy_qty(100_000 * 0.25, price)
        pos = ctx.position("000001")
        assert expected == 2500  # 25000 / (10.0×100) × 100
        assert pos is not None and pos.qty == expected
        assert win._qty_spin.value() == expected  # 回填数量框
        assert ctx.cash == pytest.approx(100_000 - expected * price)
        assert "买入" in win._status_label.text()
    finally:
        win.close()


def test_quick_buy_half_and_full(qapp, monkeypatch) -> None:
    """② 1/2 仓 / 全仓按钮按可用现金折算整手并直接买入（真实点击，覆盖 UI 接线）。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        win._on_timer()
        ctx = win._engine_ctx()
        price = ctx.last_price()
        buttons = {
            btn.text(): btn
            for btn in win.findChildren(type(win._btn_buy))
            if btn.text() in ("全仓", "1/2仓", "1/4仓")
        }
        assert set(buttons) == {"全仓", "1/2仓", "1/4仓"}
        buttons["1/2仓"].click()
        assert ctx.position("000001").qty == max_buy_qty(100_000 * 0.5, price) == 5000
        buttons["全仓"].click()
        # 全仓 = 剩余现金（50000）全部折算 → 追加 5000 股，现金清零。
        assert ctx.position("000001").qty == 10_000
        assert ctx.cash == pytest.approx(0.0)
    finally:
        win.close()


def test_buy_by_amount_dialog_amount(qapp, monkeypatch) -> None:
    """② 「按金额…」：弹窗金额折算整手股数并直接买入（取消则不下单）。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "PySide6.QtWidgets.QInputDialog.getDouble",
        staticmethod(lambda *a, **k: (5080.0, True)),
    )
    try:
        win.start_replay()
        win._on_timer()
        ctx = win._engine_ctx()
        price = ctx.last_price()
        win._btn_by_amount.click()
        expected = qty_for_amount(5080.0, price)
        assert expected == 500  # 5080 / (10×100) 向下取整
        assert ctx.position("000001").qty == expected

        # 取消 → 不下单。
        monkeypatch.setattr(
            "PySide6.QtWidgets.QInputDialog.getDouble",
            staticmethod(lambda *a, **k: (1000.0, False)),
        )
        win._btn_by_amount.click()
        assert ctx.position("000001").qty == expected
    finally:
        win.close()


def test_quick_buy_insufficient_funds_prompts(qapp, monkeypatch) -> None:
    """② 资金不足一手（高股价）→ 提示且不下单、不弹窗。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10, base=100.0)  # 一手 = 1 万元 > 初始资金 1000
    win = _make_sim_window(qapp, db, "000001")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win._cash_spin.setValue(1000)
        win.start_replay()
        win._on_timer()
        win._quick_buy(1.0)
        assert "资金不足一手" in win._status_label.text()
        ctx = win._engine_ctx()
        assert ctx.positions == {} and ctx.cash == pytest.approx(1000.0)
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 7c. ④ 回放中切换标的：账户 / 持仓不变，可同时买卖多只
# --------------------------------------------------------------------------- #
def _make_multi_window(qapp, db: Database) -> SimWindow:
    win = SimWindow(db=db, symbols=["000001", "600000"], default_symbol="000001")
    win.show()
    qapp.processEvents()
    return win


def test_switch_symbol_keeps_account_and_positions(qapp, monkeypatch) -> None:
    """④ 回放中换标的：账户与持仓不变；同一账户可分别持有两只股票。"""
    db = _mem_db()
    _seed_minute(db, "000001", days=["2026-01-05", "2026-01-06"], per_day=10)
    _seed_minute(db, "600000", days=["2026-01-05", "2026-01-06"], per_day=10, base=20.0)
    win = _make_multi_window(qapp, db)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        win._on_timer()
        win._qty_spin.setValue(100)
        win._on_place_order(BUY)  # 买入 000001 @10.0
        ctx = win._engine_ctx()
        cash_after_a = ctx.cash
        assert cash_after_a == pytest.approx(100_000 - 100 * 10.0)

        # 用户在下拉里选第二只 → 自动切换回放标的（账户 / 持仓不变）。
        win._symbol_combo.setCurrentIndex(win._symbol_combo.findData("600000"))
        qapp.processEvents()
        assert ctx.symbol == "600000"
        assert ctx.cash == pytest.approx(cash_after_a)  # 现金不动
        assert ctx.position("000001") is not None  # 原持仓保留
        assert win.current_symbol() == "600000"
        assert "600000" in win.windowTitle()

        # 同一账户买入第二只 → 两笔持仓并存、各自计价。
        win._qty_spin.setValue(200)
        win._on_place_order(BUY)  # 买入 600000 @20.0
        assert ctx.position("600000") is not None
        assert set(ctx.positions) == {"000001", "600000"}
        assert ctx.cash == pytest.approx(cash_after_a - 200 * 20.0)
        assert "000001" in win._all_pos_label.text()
        assert "600000" in win._all_pos_label.text()

        # 成交表：5 列（含「标的」列），两笔成交带各自代码。
        assert win._trades_table.columnCount() == 5
        assert win._trades_table.horizontalHeaderItem(1).text() == "标的"
        assert {
            win._trades_table.item(r, 1).text()
            for r in range(win._trades_table.rowCount())
        } == {"000001", "600000"}

        # 切回第一只：持仓 / 账户仍不变。
        win._symbol_combo.setCurrentIndex(win._symbol_combo.findData("000001"))
        qapp.processEvents()
        assert ctx.symbol == "000001"
        assert set(ctx.positions) == {"000001", "600000"}
    finally:
        win.close()


def test_switch_symbol_missing_data_reverts(qapp) -> None:
    """④ 切换到本地无数据的标的：回退原标的 + 提示，账户不受影响。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_multi_window(qapp, db)
    try:
        win.start_replay()
        win._on_timer()
        ctx = win._engine_ctx()
        win._on_symbol_text_changed("600519")  # 手动输入库里没有的代码
        assert ctx.symbol == "000001"  # 未切换
        assert win.current_symbol() == "000001"  # 组合框回退
        assert "本地暂无该周期数据" in win._status_label.text()
    finally:
        win.close()


def test_sell_uses_target_symbol_position(qapp, monkeypatch) -> None:
    """④ 卖出 / 平仓按**当前标的**持仓判定：A 的持仓不能用于 B 的卖出。"""
    db = _mem_db()
    _seed_minute(db, "000001", days=["2026-01-05", "2026-01-06"], per_day=10)
    _seed_minute(db, "600000", days=["2026-01-05", "2026-01-06"], per_day=10, base=20.0)
    win = _make_multi_window(qapp, db)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        # 第一天：买入 000001，然后推进到第二天（跨日 → 需「下一天」）。
        win._on_timer()
        win._qty_spin.setValue(100)
        win._on_place_order(BUY)
        for _ in range(9):
            win._on_timer()
        assert win._btn_next_day.isEnabled() is True
        win.next_day()

        # 切到 600000（无持仓）→ 卖出被拒（不能拿 A 的持仓卖 B）；平仓提示无持仓。
        win._symbol_combo.setCurrentIndex(win._symbol_combo.findData("600000"))
        qapp.processEvents()
        ctx = win._engine_ctx()
        assert ctx.symbol == "600000"
        win._qty_spin.setValue(100)
        win._on_place_order(SELL)
        assert "做多" in win._status_label.text() or "持仓不足" in win._status_label.text()
        assert len(ctx.orders) == 1  # 只有 A 的那笔买入
        win._on_close_position()
        assert "无持仓" in win._status_label.text()

        # 切回 000001：次日可卖（T+1 已过），平仓成交。
        win._symbol_combo.setCurrentIndex(win._symbol_combo.findData("000001"))
        qapp.processEvents()
        win._on_close_position()
        assert ctx.position("000001") is None
        assert len(ctx.orders) == 2
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 7d. 2026-10-07 用户优化：最新K线居中 / Y轴即时跟随 / 交易价滑钮
# --------------------------------------------------------------------------- #
def _seed_frame(db: Database, frame: pd.DataFrame, symbol: str = "000001",
                period: str = "5m") -> None:
    """把给定 Bar 帧补齐 ``symbol/period/adjust`` 契约列后写入库。"""
    frame = frame.copy()
    frame["symbol"] = symbol
    frame["period"] = period
    frame["adjust"] = ""
    db.upsert_bars(frame)


def test_latest_bar_centered_every_tick(qapp) -> None:
    """② 每显现一根 K 线，视窗即以最新根为中心（无需鼠标移入，宽度恒定）。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=48)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        for shown in range(1, 13):
            win._on_timer()
            lo, hi = win._chart._x_range
            assert hi - lo == pytest.approx(float(win.window_slots()))
            assert (lo + hi) / 2.0 == pytest.approx(float(shown - 1))
    finally:
        win.close()


def test_daily_rolling_window_centers_newest(qapp) -> None:
    """② 日K（滚动窗口）同样以最新根为中心。"""
    db = _mem_db()
    _seed_frame(db, _bars_5m(["2026-01-05"], per_day=20), period="d")
    win = _make_sim_window(qapp, db, "000001")
    try:
        combo = win._period_combo
        combo.setCurrentIndex(combo.findData("d"))
        win.start_replay()
        for _ in range(10):
            win._on_timer()
        lo, hi = win._chart._x_range
        assert hi - lo == pytest.approx(float(win.window_slots()))
        assert (lo + hi) / 2.0 == pytest.approx(9.0)
    finally:
        win.close()


def test_y_axis_follows_price_jump(qapp) -> None:
    """② 大幅波动（末根跳至 2 倍价）时 Y 轴随新根即时自适应，不跳出纵向视野。"""
    db = _mem_db()
    frame = _bars_5m(["2026-01-05"], per_day=10)
    for col, value in (("open", 19.9), ("high", 20.2), ("low", 19.8), ("close", 20.0)):
        frame.loc[frame.index[-1], col] = value
    _seed_frame(db, frame)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        for _ in range(9):
            win._on_timer()
        assert win._chart._main_plot.getViewBox().viewRange()[1][1] < 20.0
        win._on_timer()  # 第 10 根跳到 20 元
        upper = win._chart._main_plot.getViewBox().viewRange()[1][1]
        assert upper >= 20.2  # Y 轴已跟随（含该根 high）
        assert win._chart.bar_count() == 10
    finally:
        win.close()


def test_trade_price_slider_default_and_range(qapp) -> None:
    """③ 滑钮范围 = 当日（截至游标）震荡范围；默认 = 最新收盘，不随 tick 重置。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        assert win._price_slider.isEnabled() is False  # 未开始 → 禁用
        assert win.current_trade_price() is None
        win.start_replay()
        assert win._price_slider.isEnabled() is True
        win._on_timer()  # 首根 close=10.0 → low=9.9 / high=10.1
        assert win._price_slider.minimum() == 990
        assert win._price_slider.maximum() == 1010
        assert win._price_slider.value() == 1000
        assert win.current_trade_price() == pytest.approx(10.0)
        assert win._chart.trade_price_value() == pytest.approx(10.0)
        assert win._chart._trade_line.isVisible() is True
        win._on_timer()  # 第 2 根 close=10.01 → 范围右扩，值保持不变（未拖动）
        assert win._price_slider.maximum() == 1011
        assert win._price_slider.value() == 1000
    finally:
        win.close()


def test_trade_price_drag_and_order_uses_it(qapp, monkeypatch) -> None:
    """③ 拖动滑钮：成交按滑钮价（非收盘价），主图黄虚线同步；重置后禁用并隐藏。"""
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        staticmethod(lambda *a, **k: None),
    )
    try:
        win.start_replay()
        win._on_timer()
        win._price_slider.setValue(990)  # 拖到当日最低 9.90
        assert win.current_trade_price() == pytest.approx(9.9)
        assert win._chart._trade_line.value() == pytest.approx(9.9)
        assert "按交易价 9.90" in win._avail_label.text()

        win._qty_spin.setValue(100)
        win._on_place_order(BUY)
        ctx = win._engine_ctx()
        assert ctx.orders[-1].price == pytest.approx(9.9)  # 成交价 = 滑钮价
        assert ctx.cash == pytest.approx(100_000 - 100 * 9.9)
        assert win._trades_table.item(0, 3).text() == "9.90"

        win._on_timer()  # 拖动后固定，不随 tick 重置
        assert win.current_trade_price() == pytest.approx(9.9)

        win.stop_replay()
        assert win._price_slider.isEnabled() is False
        assert win.current_trade_price() is None
        assert win._chart._trade_line.isVisible() is False
    finally:
        win.close()


def test_trade_price_flat_bar_no_crash(qapp) -> None:
    """③ 一字板（高 == 低）时滑钮范围退化为单值，不崩且成交价 = 该价。"""
    db = _mem_db()
    frame = _bars_5m(["2026-01-05"], per_day=5)
    for col in ("open", "high", "low", "close"):
        frame[col] = 10.0
    _seed_frame(db, frame)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        win._on_timer()
        assert win._price_slider.minimum() == win._price_slider.maximum() == 1000
        assert win.current_trade_price() == pytest.approx(10.0)
        assert win._chart.trade_price_value() == pytest.approx(10.0)
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 8. 条件单
# --------------------------------------------------------------------------- #
def test_condition_order_added_and_fires(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        # 价格触发：首根 close=10.0，买入价 <= 10.0 → 首 tick 触发。
        win._cond_side.setCurrentIndex(0)  # 买入
        win._cond_trigger.setValue(10.0)
        win._cond_qty.setValue(100)
        win._btn_add_cond.click()  # 真实点击（UI 触达层）
        assert win._engine is not None
        assert win._engine.conditional_order_count() == 1
        win._on_timer()
        ctx = win._engine_ctx()
        assert ctx is not None and ctx.position("000001") is not None
        assert win._engine.conditional_order_count() == 0  # 一次性移除
        assert ctx.orders[-1].reason == "condition"
    finally:
        win.close()


def test_condition_order_kept_when_not_triggered(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        win._cond_trigger.setValue(9.0)  # 价格不断上行，永不触发
        win._btn_add_cond.click()
        for _ in range(12):
            win._on_timer()
        assert win._engine is not None
        assert win._engine.conditional_order_count() == 1
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 9. 倍速 / 暂停 / 重置
# --------------------------------------------------------------------------- #
def test_speed_change_updates_interval(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        from app.engine.simulator import interval_for_speed

        assert win._engine is not None
        assert win._timer.interval() == interval_for_speed(1)
        assert win._timer.interval() == 400  # 1x = 400ms（调慢到原来 1/2）
        win._speed_combo.setCurrentIndex(3)  # 10x
        win._speed_combo.currentIndexChanged.emit(3)
        # SimWindow 监听倍速变更 → 同步 set_speed。
        assert win._engine.speed() == 10
    finally:
        win.close()


def test_pause_toggle_stops_timer(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        assert win._timer.isActive() is True
        win.toggle_pause()
        assert win._timer.isActive() is False
        assert win._btn_pause.text() == "继续"
        win.toggle_pause()
        assert win._timer.isActive() is True
        assert win._btn_pause.text() == "暂停"
    finally:
        win.close()


def test_reset_stops_and_clears(qapp) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", per_day=10)
    win = _make_sim_window(qapp, db, "000001")
    try:
        win.start_replay()
        win._on_timer()
        win.stop_replay()
        assert win._engine is None
        assert win._timer.isActive() is False
        assert win._btn_start.isEnabled() is True
        assert win._btn_next_day.isEnabled() is False
        assert win._chart.is_placeholder_visible()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 10. ① 菜单接线（MainWindow）：非模态多窗口
# --------------------------------------------------------------------------- #
class _FakeSignal:
    """极简信号替身：记录槽并支持手动 ``emit``（供 FakeWindow 的 ``destroyed``）。"""

    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:  # noqa: ANN001 - 测试替身
        self.slots.append(slot)

    def emit(self, *args) -> None:
        for slot in list(self.slots):
            slot(*args)


def test_sim_menu_exists_and_opens_nonmodal(qapp, monkeypatch) -> None:
    """① 非模态多窗口：触发菜单 → 新窗口 ``show()``（不 exec、不阻塞主界面），可多开。"""
    db = _mem_db()
    win = _make_window(qapp, db)
    try:
        sim_menu = None
        for action in win.menuBar().actions():
            if "训练" in action.text():
                sim_menu = action.menu()
                break
        assert sim_menu is not None
        assert any("模拟训练" in a.text() for a in sim_menu.actions())

        captured: dict = {}

        class _FakeWindow:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                captured["created"] = captured.get("created", 0) + 1
                self.destroyed = _FakeSignal()

            def setAttribute(self, *_args):
                captured["attr"] = True

            def setWindowFlag(self, *_args):
                captured["flag"] = True

            def show(self):
                captured["show"] = True

            def raise_(self):
                captured["raise"] = True

            def activateWindow(self):
                captured["activate"] = True

        monkeypatch.setattr("app.ui.main_window.SimWindow", _FakeWindow)
        win._open_sim_window()
        # 非模态：show / raise / activate；无 exec、无 deleteLater（关闭由 WA_DeleteOnClose 接管）。
        assert captured.get("show") is True
        assert captured.get("raise") is True
        assert captured.get("activate") is True
        assert captured.get("attr") is True and captured.get("flag") is True
        assert captured["db"] is db and captured.get("parent") is None
        # 多窗口：引用列表持有实例（防 GC），再点一次 → 再开一个。
        assert len(win._sim_windows) == 1
        win._open_sim_window()
        assert len(win._sim_windows) == 2 and captured["created"] == 2
        # 窗口销毁 → destroyed 回调移除引用。
        win._sim_windows[0].destroyed.emit()
        assert len(win._sim_windows) == 1
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 11. ⑤ 股票代码与名称名录（MainWindow 接线）
# --------------------------------------------------------------------------- #
def test_data_menu_has_stock_list_action(qapp) -> None:
    """⑤ 「数据」菜单含「股票代码与名称…」入口。"""
    db = _mem_db()
    win = _make_window(qapp, db)
    try:
        texts = [a.text() for a in win._data_menu.actions()]
        assert any("股票代码与名称" in t for t in texts)
        assert win._stock_list_dialog is None
    finally:
        win.close()


def test_update_worker_symbols_kind_writes_list(qapp) -> None:
    """⑤ 后台 worker 的 ``kind='symbols'`` 分支：拉名录 → 落库（同步执行线程体）。"""
    from app.ui.main_window import _UpdateWorker

    db = _mem_db()

    class _ListSource(_FakeSource):
        def list_stocks(self):
            return pd.DataFrame(
                {"symbol": ["000001", "600000"], "name": ["平安银行", "浦发银行"]}
            )

    service = UpdateService(
        db=db, sources={"baostock": _ListSource(), "akshare": _ListSource()}
    )
    worker = _UpdateWorker(service, kind="symbols")
    worker.run()  # 同步跑线程体，不开线程
    assert db.count_stocks() == 2
    assert db.list_stocks("平安")[0].symbol == "000001"
    assert db.list_stocks("600000")[0].name == "浦发银行"


def test_on_update_symbols_starts_background_job_and_message(qapp, monkeypatch) -> None:
    """⑤ 名录更新经 MainWindow 后台线程编排；完成文案为名录专用文案。"""
    db = _mem_db()
    win = _make_window(qapp, db)
    captured: dict = {}

    class _FakeWorker:
        def __init__(self, service, parent=None, symbols=None, kind="daily"):
            captured["kind"] = kind

        def isRunning(self):
            return False

        def start(self):
            captured["started"] = True

        done = _FakeSignal()
        finished = _FakeSignal()

    monkeypatch.setattr("app.ui.main_window._UpdateWorker", _FakeWorker)
    try:
        win._on_update_symbols()
        assert captured["kind"] == "symbols" and captured["started"] is True
        assert win._update_kind == "symbols"
        assert "正在更新股票名录" in win.statusBar().currentMessage()
        # 完成回调：名录文案（n 只）而非 K 线文案（写入 n 行）。
        win._on_update_finished(UpdateReport(symbols_count=5000, bars_written=5000))
        assert "名录更新完成：共 5000 只" in win.statusBar().currentMessage()
        win._on_update_finished(UpdateReport(failed=[("", "网络不可用")]))
        assert "名录更新失败：网络不可用" in win.statusBar().currentMessage()
    finally:
        win.close()


def test_open_stock_list_reuses_dialog_and_loads_choice(qapp, monkeypatch) -> None:
    """⑤ 名录对话框：单实例复用；选中代码 → 主窗口加载该股（本地无数据则自动更新）。"""
    db = _mem_db()
    _seed_minute(db, "600000", per_day=5)  # 600000 本地已有数据
    win = _make_window(qapp, db)
    updates: list = []
    monkeypatch.setattr(win, "_on_update_daily", lambda symbols=None: updates.append(symbols))
    try:
        win._open_stock_list()
        dialog = win._stock_list_dialog
        assert dialog is not None and dialog.isVisible() is True
        # 单实例：再次打开复用同一对话框（不重复构造）。
        win._open_stock_list()
        assert win._stock_list_dialog is dialog

        # 选中已有数据的股票 → 主窗口加载 K 线，且不重复拉数据。
        dialog.symbolChosen.emit("600000")
        assert win.kline_view.current_symbol() == "600000"
        assert "已选择：600000" in win.statusBar().currentMessage()
        assert updates == []
        assert win.kline_view.bar_count() >= 0  # 加载路径不抛（本地无日线则为 0）

        # 选中本地无数据的股票 → 自动触发单股更新。
        dialog.symbolChosen.emit("600519")
        assert win.kline_view.current_symbol() == "600519"
        assert updates == [["600519"]]

        # 关闭 → 销毁（WA_DeleteOnClose）→ 引用自动置空。
        dialog.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        assert win._stock_list_dialog is None
    finally:
        win.close()