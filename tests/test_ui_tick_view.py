"""``TickView`` 分时图离屏测试（``QT_QPA_PLATFORM=offscreen``，全离线，合成 Bar 数据）。

覆盖 8 类断言：
    1. 纯函数 :func:`avg_price_series` —— **必须 ×100 手→股**（差 100 倍回归防护）；
    2. 纯函数 :func:`change_pct`（正常 / 无昨收 / 昨收为 0）；
    3. 纯函数 :func:`tick_color`（红涨绿跌 / 退回首笔开盘 / 双双缺失中性）；
    4. :meth:`set_minute` 渲染（价格线 + 均价线 + 量柱 + 时间轴）；
    5. 主图 / 副图 X 轴联动（``setXLink``）；
    6. 十字光标存在；
    7. 空数据 / 空库降级；
    8. 集成：选股 → 分时页加载 + 头部信息 + ``1m`` 周期自动切页。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QTabWidget  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.themes import colors  # noqa: E402
from app.ui.widgets.kline_view import CompactAxisItem  # noqa: E402
from app.ui.widgets.tick_view import (  # noqa: E402
    _NO_LOCAL_MINUTE_TEXT,
    TickView,
    avg_price_series,
    change_pct,
    tick_color,
)


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #
def _make_minute(n: int = 30, start: float = 10.0, end: float = 11.0) -> pd.DataFrame:
    """生成 n 根确定性 1 分钟 Bar 契约数据（含 volume「手」/ amount「元」）。"""
    dt = pd.date_range("2024-01-02 09:30", periods=n, freq="min")
    closes = np.linspace(start, end, n)
    volume = np.full(n, 100.0)  # 手
    amount = closes * volume * 100.0  # 元（= 价 × 股数）
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": closes,
            "high": closes + 0.05,
            "low": closes - 0.05,
            "close": closes,
            "volume": volume,
            "amount": amount,
        }
    )


def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


class _FakeSource:
    name = "fake"

    def get_kline(self, *a, **k):
        return pd.DataFrame()

    def get_minute(self, *a, **k):
        return pd.DataFrame()

    def list_stocks(self):
        return pd.DataFrame()

    def get_spot(self):
        return pd.DataFrame()

    def get_adj_factor(self, *a, **k):
        return pd.DataFrame()

    def get_fundamental(self, *a, **k):
        return {}


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def view(qapp):
    v = TickView(db=_mem_db())
    yield v
    v.close()


# --------------------------------------------------------------------------- #
# 1. 纯函数：均价线 VWAP（×100 手→股）
# --------------------------------------------------------------------------- #
def test_avg_price_series_uses_lot_to_share_factor() -> None:
    """VWAP = cumsum(amount) / (cumsum(volume) * 100)，分母须 ×100。"""
    df = pd.DataFrame(
        {
            "volume": [10.0, 20.0],   # 手
            "amount": [1000.0, 4200.0],  # 元
        }
    )
    series = avg_price_series(df)
    # cum_amt = [1000, 5200]；cum_vol*100 = [1000, 3000]
    assert abs(series.iloc[0] - 1.0) < 1e-9
    assert abs(series.iloc[1] - (5200.0 / 3000.0)) < 1e-9
    # 反例：若漏掉 ×100，则会是 100 / 210 这样偏大 100 倍的值。
    assert abs(series.iloc[1] - 173.333) > 1.0


def test_avg_price_series_empty_and_missing_columns() -> None:
    """空 / None / 缺列时返回空或全 NaN，且不抛异常。"""
    assert avg_price_series(None).empty
    assert avg_price_series(pd.DataFrame()).empty
    bad = avg_price_series(pd.DataFrame({"close": [1.0, 2.0]}))
    assert len(bad) == 2 and bad.isna().all()


def test_avg_price_series_zero_volume_is_nan() -> None:
    """累计成交量为 0 处返回 NaN（不做除零伪造）。"""
    df = pd.DataFrame({"volume": [0.0, 0.0, 5.0], "amount": [0.0, 0.0, 300.0]})
    series = avg_price_series(df)
    assert pd.isna(series.iloc[0]) and pd.isna(series.iloc[1])
    assert abs(series.iloc[2] - (300.0 / 500.0)) < 1e-9


# --------------------------------------------------------------------------- #
# 2. 纯函数：涨跌幅
# --------------------------------------------------------------------------- #
def test_change_pct_up_down_and_edges() -> None:
    assert abs(change_pct(11.0, 10.0) - 10.0) < 1e-9
    assert abs(change_pct(9.0, 10.0) + 10.0) < 1e-9
    assert change_pct(10.0, None) is None
    assert change_pct(None, 10.0) is None
    assert change_pct(10.0, 0.0) is None


# --------------------------------------------------------------------------- #
# 3. 纯函数：涨跌→颜色
# --------------------------------------------------------------------------- #
def test_tick_color_red_up_green_down() -> None:
    assert tick_color(11.0, 10.0, 10.0) == colors.UP_COLOR
    assert tick_color(9.0, 10.0, 10.0) == colors.DOWN_COLOR


def test_tick_color_flat_is_up() -> None:
    assert tick_color(10.0, 10.0, 10.0) == colors.UP_COLOR


def test_tick_color_fallback_to_first_open() -> None:
    """无昨收时以当日首笔开盘为参考。"""
    assert tick_color(11.0, None, 10.0) == colors.UP_COLOR
    assert tick_color(9.0, None, 10.0) == colors.DOWN_COLOR


def test_tick_color_neutral_when_no_reference() -> None:
    assert tick_color(None, 10.0, 10.0) == colors.TEXT_COLOR
    assert tick_color(10.0, None, None) == colors.TEXT_COLOR


# --------------------------------------------------------------------------- #
# 4. set_minute 渲染
# --------------------------------------------------------------------------- #
def test_set_minute_renders_price_avg_and_volume(view: TickView) -> None:
    df = _make_minute(30)
    view.set_minute(df, prev_close=10.0)
    assert view.bar_count() == 30
    assert not view.is_placeholder_visible()
    assert isinstance(view._volume_item, pg.BarGraphItem)
    assert view._volume_item.scene() is not None
    assert view._main_axis.label_count() == 30
    # 价格线与均价线均已挂载数据
    price_y = view._price_curve.getData()[1]
    assert price_y is not None and len(price_y) == 30
    avg_y = view._avg_curve.getData()[1]
    assert avg_y is not None and len(avg_y) == 30
    # 均价线末值等于纯函数结果
    expected = avg_price_series(df).iloc[-1]
    assert abs(float(avg_y[-1]) - float(expected)) < 1e-6


def test_set_minute_none_and_empty_no_raise(view: TickView) -> None:
    view.set_minute(pd.DataFrame())
    assert view.bar_count() == 0 and view.is_placeholder_visible()
    assert view.placeholder_text() == "无数据"
    view.set_minute(None)  # 不应抛异常
    assert view.bar_count() == 0 and view.is_placeholder_visible()


# --------------------------------------------------------------------------- #
# 5. 主/副图 X 轴联动
# --------------------------------------------------------------------------- #
def test_main_and_volume_x_linked(view: TickView) -> None:
    view.set_minute(_make_minute(30), prev_close=10.0)
    view._main_plot.setXRange(5, 15, padding=0.0)
    xr = view._volume_plot.getViewBox().viewRange()[0]
    assert abs(xr[0] - 5) < 0.5
    assert abs(xr[1] - 15) < 0.5


# --------------------------------------------------------------------------- #
# 6. 十字光标
# --------------------------------------------------------------------------- #
def test_crosshair_lines_exist(view: TickView) -> None:
    view.set_minute(_make_minute(20), prev_close=10.0)
    assert isinstance(view._vline, pg.InfiniteLine)
    assert isinstance(view._hline, pg.InfiniteLine)
    assert isinstance(view._vol_vline, pg.InfiniteLine)
    assert view._vline.scene() is not None
    assert view._hline.scene() is not None


def test_volume_axis_is_compact(view: TickView) -> None:
    """成交量副图左轴用紧凑单位（避免 1e+ 科学计数法）。"""
    assert isinstance(view._volume_plot.getAxis("left"), CompactAxisItem)


def test_cursor_label_volume_compact(qapp, view: TickView) -> None:
    """十字光标角标「量」按量级压缩（17079 手 → 17k）。"""
    frame = _make_minute(30)
    frame.loc[10, "volume"] = 17079.0
    view.resize(800, 600)
    view.show()
    qapp.processEvents()
    view.set_minute(frame, prev_close=10.0)
    qapp.processEvents()
    pos = view._main_plot.getViewBox().mapViewToScene(
        QPointF(10.0, float(view._bars.iloc[10]["close"]))
    )
    view._on_mouse_moved(pos)
    assert "量17k" in view._cursor_label.text()


# --------------------------------------------------------------------------- #
# 7. 空库 load 降级
# --------------------------------------------------------------------------- #
def test_load_empty_db_shows_hint(view: TickView) -> None:
    view.load("000001")
    assert view.bar_count() == 0
    assert view.is_placeholder_visible()
    # 1 分钟线本版本拿不到（「更新数据」仅拉日线）→ 文案明确提示，非旧「本地无数据」。
    assert view.placeholder_text() == _NO_LOCAL_MINUTE_TEXT
    assert "日线" in view.placeholder_text()
    assert view.current_symbol() == "000001"


# --------------------------------------------------------------------------- #
# 8. 集成：选股 → 分时页 + 头部信息 + 1m 切页
# --------------------------------------------------------------------------- #
def _seed(db: Database, symbol: str, n: int = 30) -> None:
    """向内存库写入 1 分钟线（adjust=''）与昨收日线（adjust='qfq'）。"""
    minute = _make_minute(n)
    minute["symbol"] = symbol
    minute["period"] = "1m"
    minute["adjust"] = ""
    db.upsert_bars(minute)

    daily = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-01")],
            "open": [9.8],
            "high": [10.1],
            "low": [9.7],
            "close": [10.0],  # 昨收
            "volume": [1000.0],
            "amount": [1_000_000.0],
            "symbol": [symbol],
            "period": ["d"],
            "adjust": ["qfq"],
        }
    )
    db.upsert_bars(daily)


def _select_first_symbol(win: MainWindow) -> None:
    tree = win.watchlist_panel.list_widget
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            child = top.child(j)
            if child.data(0, int(Qt.ItemDataRole.UserRole)) == "000001":
                tree.setCurrentItem(child)
                return
    raise AssertionError("未找到自选股条目")


def test_right_area_is_tabwidget_with_kline_and_tick(qapp) -> None:
    db = _mem_db()
    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)
    try:
        assert isinstance(win.right_tabs, QTabWidget)
        assert win.right_tabs.count() == 2
        assert win.right_tabs.tabText(0) == "K线"
        assert win.right_tabs.tabText(1) == "分时"
        assert win.right_tabs.widget(0) is win.kline_view
        assert win.right_tabs.widget(1) is win.tick_view
    finally:
        win.close()


def test_integration_select_symbol_loads_tick(qapp) -> None:
    db = _mem_db()
    gid = db.create_group("默认")
    db.add_watch_item(gid, "000001")
    _seed(db, "000001", n=30)

    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)
    try:
        _select_first_symbol(win)  # 触发 symbolSelected -> tick_view.load
        assert win.tick_view.current_symbol() == "000001"
        assert win.tick_view.current_date() == "2024-01-02"
        assert win.tick_view.bar_count() == 30
        assert not win.tick_view.is_placeholder_visible()
        # 头部：现价 + 涨跌幅 + 均价
        assert win.tick_view.current_price() == pytest.approx(11.0, abs=1e-6)
        assert win.tick_view.current_change_pct() == pytest.approx(10.0, abs=1e-6)
        assert "现价" in win.tick_view._price_label.text()
        assert "%" in win.tick_view._change_label.text()
        assert "均价" in win.tick_view._avg_label.text()
    finally:
        win.close()


def test_period_1m_switches_to_tick_tab(qapp) -> None:
    db = _mem_db()
    gid = db.create_group("默认")
    db.add_watch_item(gid, "000001")
    _seed(db, "000001", n=30)

    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)
    try:
        _select_first_symbol(win)
        assert win.right_tabs.currentIndex() == 0  # 默认 K线页
        win.toolbar.set_period("1m")  # -> periodChanged -> _on_period_changed
        assert win.right_tabs.currentIndex() == 1
        assert win.right_tabs.currentWidget() is win.tick_view
    finally:
        win.close()
