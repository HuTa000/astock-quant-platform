"""``KLineView`` 离屏测试（``QT_QPA_PLATFORM=offscreen``，全离线，合成 Bar 数据）。"""
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
from PySide6.QtCore import Qt, QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.themes import colors  # noqa: E402
from app.ui.widgets.kline_view import (  # noqa: E402
    _NO_LOCAL_MINUTE_TEXT,
    _NO_LOCAL_TEXT,
    CompactAxisItem,
    KLineView,
    build_hover_html,
    candle_color,
    format_compact_number,
    hover_pct_change,
    volume_color,
)


# --------------------------------------------------------------------------- #
# 合成数据
# --------------------------------------------------------------------------- #
def _make_bars(n: int = 60, symbol: str = "000001", period: str = "d") -> pd.DataFrame:
    """生成 n 根确定性 Bar 契约数据（含涨/跌两类）。"""
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(10.0, 20.0, n)
    opens = base
    closes = base + np.where(np.arange(n) % 2 == 0, 0.5, -0.5)  # 偶涨奇跌
    highs = np.maximum(opens, closes) + 0.3
    lows = np.minimum(opens, closes) - 0.3
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.linspace(1000, 5000, n),
            "amount": np.linspace(1e6, 5e6, n),
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
    v = KLineView(db=_mem_db())
    yield v
    v.close()


# --------------------------------------------------------------------------- #
# 1. 纯函数：涨跌→颜色
# --------------------------------------------------------------------------- #
def test_candle_color_up_red() -> None:
    assert candle_color(10.0, 11.0) == "#e74c3c"


def test_candle_color_down_green() -> None:
    assert candle_color(11.0, 10.0) == "#2ecc71"


def test_candle_color_flat_is_red() -> None:
    assert candle_color(10.0, 10.0) == "#e74c3c"


def test_volume_color_matches_direction() -> None:
    assert volume_color(10.0, 11.0) == colors.VOL_UP
    assert volume_color(11.0, 10.0) == colors.VOL_DOWN


# --------------------------------------------------------------------------- #
# 2. set_bars 渲染 + X 轴联动
# --------------------------------------------------------------------------- #
def test_set_bars_renders_candles_and_volume(view: KLineView) -> None:
    df = _make_bars(60)
    view.set_bars(df)
    assert view.bar_count() == 60
    assert not view.is_placeholder_visible()
    assert view._candle_item.data_count() == 60
    assert isinstance(view._volume_item, pg.BarGraphItem)
    assert view._candle_item.scene() is not None
    assert view._volume_item.scene() is not None
    assert view._main_axis.label_count() == 60


def test_main_and_volume_x_linked(view: KLineView) -> None:
    view.set_bars(_make_bars(30))
    view._main_plot.setXRange(5, 15, padding=0.0)
    xr = view._volume_plot.getViewBox().viewRange()[0]
    assert abs(xr[0] - 5) < 0.5
    assert abs(xr[1] - 15) < 0.5


# --------------------------------------------------------------------------- #
# 3. 十字光标
# --------------------------------------------------------------------------- #
def test_crosshair_lines_exist(view: KLineView) -> None:
    view.set_bars(_make_bars(20))
    assert isinstance(view._vline, pg.InfiniteLine)
    assert isinstance(view._hline, pg.InfiniteLine)
    assert view._vline.scene() is not None
    assert view._hline.scene() is not None


# --------------------------------------------------------------------------- #
# 4. 空数据降级
# --------------------------------------------------------------------------- #
def test_set_bars_empty_shows_placeholder(view: KLineView) -> None:
    view.set_bars(pd.DataFrame())
    assert view.bar_count() == 0
    assert view.is_placeholder_visible()
    assert view.placeholder_text() == "无数据"


def test_set_bars_none_no_raise(view: KLineView) -> None:
    view.set_bars(None)  # 不应抛异常
    assert view.bar_count() == 0
    assert view.is_placeholder_visible()


# --------------------------------------------------------------------------- #
# 5. 空库 load
# --------------------------------------------------------------------------- #
def test_load_empty_db_shows_hint(view: KLineView) -> None:
    view.load("000001", "d")
    assert view.bar_count() == 0
    assert view.is_placeholder_visible()
    assert "本地无数据" in view.placeholder_text()
    assert view.current_symbol() == "000001"
    assert view.current_period() == "d"


# --------------------------------------------------------------------------- #
# 6. 集成：选股 → 图表显示本地根数
# --------------------------------------------------------------------------- #
def test_integration_select_symbol_renders(qapp) -> None:
    db = _mem_db()
    gid = db.create_group("默认")
    db.add_watch_item(gid, "000001")

    n = 42
    bars = _make_bars(n, symbol="000001", period="d")
    seeded = bars.copy()
    seeded["symbol"] = "000001"
    seeded["period"] = "d"
    seeded["adjust"] = "qfq"
    db.upsert_bars(seeded)
    assert db.count_bars("000001", "d") == n

    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)
    try:
        tree = win.watchlist_panel.list_widget
        child = None
        for i in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(i)
            for j in range(top.childCount()):
                c = top.child(j)
                if c.data(0, int(Qt.ItemDataRole.UserRole)) == "000001":
                    child = c
                    break
            if child is not None:
                break
        assert child is not None, "未找到自选股条目"
        tree.setCurrentItem(child)  # 触发 symbolSelected -> kline_view.load
        assert win.kline_view.current_symbol() == "000001"
        assert win.kline_view.bar_count() == n
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 7. 周线 / 月线本地派生（文件库只放日线；派生结果**不落库**）
# --------------------------------------------------------------------------- #
_WEEKLY_DAILY_ROWS = [
    ("2024-01-25", 10.0, 10.5, 9.8, 10.2, 100.0, 1000.0),
    ("2024-01-26", 10.2, 10.6, 10.0, 10.4, 200.0, 2000.0),
    ("2024-01-29", 10.4, 11.0, 10.3, 10.8, 300.0, 3000.0),
    ("2024-01-30", 10.8, 11.2, 10.6, 10.9, 400.0, 4000.0),
    ("2024-01-31", 10.9, 11.5, 10.7, 11.3, 500.0, 5000.0),
    ("2024-02-01", 11.3, 11.6, 11.0, 11.1, 600.0, 6000.0),
    ("2024-02-02", 11.1, 11.4, 10.9, 11.0, 700.0, 7000.0),
    ("2024-02-05", 11.0, 11.8, 10.95, 11.7, 800.0, 8000.0),
    ("2024-02-06", 11.7, 12.0, 11.5, 11.9, 900.0, 9000.0),
]


def _file_db(path) -> Database:
    """打开文件库并建表。"""
    db = Database(str(path))
    db.connect()
    db.init_schema()
    return db


def _seed_daily_only(db: Database, symbol: str = "000001", adjust: str = "qfq") -> int:
    """只写日线（period='d'，指定 adjust）；返回行数。"""
    frame = pd.DataFrame(
        {
            "datetime": [pd.Timestamp(r[0]) for r in _WEEKLY_DAILY_ROWS],
            "open": [r[1] for r in _WEEKLY_DAILY_ROWS],
            "high": [r[2] for r in _WEEKLY_DAILY_ROWS],
            "low": [r[3] for r in _WEEKLY_DAILY_ROWS],
            "close": [r[4] for r in _WEEKLY_DAILY_ROWS],
            "volume": [r[5] for r in _WEEKLY_DAILY_ROWS],
            "amount": [r[6] for r in _WEEKLY_DAILY_ROWS],
        }
    )
    frame["symbol"] = symbol
    frame["period"] = "d"
    frame["adjust"] = adjust
    db.upsert_bars(frame)
    return len(frame)


def test_load_weekly_derives_from_daily(tmp_path, qapp) -> None:
    db = _file_db(tmp_path / "k_w.db")
    n = _seed_daily_only(db, "000001")
    assert n == 9
    assert db.get_bars("000001", "w").empty  # 库里本无周线
    view = KLineView(db=db)
    try:
        view.load("000001", "w")
        assert view.is_placeholder_visible() is False
        assert view.bar_count() == 3  # 3 个自然周（手算）
    finally:
        view.close()
        db.close()


def test_load_monthly_derives_from_daily(tmp_path, qapp) -> None:
    db = _file_db(tmp_path / "k_m.db")
    _seed_daily_only(db, "000001")
    view = KLineView(db=db)
    try:
        view.load("000001", "m")
        assert view.is_placeholder_visible() is False
        assert view.bar_count() == 2  # 2024-01 / 2024-02
    finally:
        view.close()
        db.close()


def test_derived_weekly_not_persisted(tmp_path, qapp) -> None:
    """派生结果**绝不落库**：load('w') 后库中仍无 period='w' 的行。"""
    db = _file_db(tmp_path / "k_nowrite.db")
    _seed_daily_only(db, "000001")
    view = KLineView(db=db)
    try:
        view.load("000001", "w")
        assert view.bar_count() == 3
        assert db.get_bars("000001", "w").empty
        assert db.count_bars("000001", "w") == 0
        # 该股仍只有日线
        assert db.count_bars("000001", "d") == 9
    finally:
        view.close()
        db.close()


def test_load_minute_shows_minute_hint(tmp_path, qapp) -> None:
    """分钟级无数据 → 占位且文案含「日线」（不再误导为可点更新获取）。"""
    db = _file_db(tmp_path / "k_min.db")
    _seed_daily_only(db, "000001")
    view = KLineView(db=db)
    try:
        view.load("000001", "5m")
        assert view.is_placeholder_visible()
        assert "日线" in view.placeholder_text()
        assert view.placeholder_text() == _NO_LOCAL_MINUTE_TEXT
    finally:
        view.close()
        db.close()


def test_load_weekly_empty_db_keeps_update_hint(tmp_path, qapp) -> None:
    """日线也为空 → 周线派生必空 → 占位文案仍是「更新数据」指引（对日线有效）。"""
    db = _file_db(tmp_path / "k_empty.db")  # 完全空库
    view = KLineView(db=db)
    try:
        view.load("000001", "w")
        assert view.is_placeholder_visible()
        assert view.placeholder_text() == _NO_LOCAL_TEXT
    finally:
        view.close()
        db.close()


def test_load_daily_count_unchanged(tmp_path, qapp) -> None:
    """回归：日线根数与改动前一致（不因派生逻辑影响日线路径）。"""
    db = _file_db(tmp_path / "k_d.db")
    _seed_daily_only(db, "000001")
    view = KLineView(db=db)
    try:
        view.load("000001", "d")
        assert view.is_placeholder_visible() is False
        assert view.bar_count() == 9
    finally:
        view.close()
        db.close()


def test_load_monthly_uses_derived_unadjusted_for_detection(tmp_path, qapp) -> None:
    """回归（P6-S4 缺陷修复）：切月线时，未复权判定序列从**未复权日线派生**，
    不再因库里无未复权月线而回退用前复权显示序列（避免除权日附近误标涨跌停）。
    """
    db = _file_db(tmp_path / "k_m_det.db")
    _seed_daily_only(db, "000001", adjust="")
    _seed_daily_only(db, "000001", adjust="qfq")
    view = KLineView(db=db)
    try:
        # 月线本地无未复权数据 → 修复前会回退显示序列并记警告；修复后从未复权日线派生。
        view.load("000001", "m")
        assert view.is_placeholder_visible() is False
        assert view.bar_count() == 2
        # 判定序列应来自「未复权日线派生」：标注事件正常计算（不再触发回退警告）。
        counts = view.annotation_counts()
        assert "limit_up" in counts and "gap_up" in counts
    finally:
        view.close()
        db.close()


# --------------------------------------------------------------------------- #
# 8. 悬停气泡（纯函数）
# --------------------------------------------------------------------------- #
def test_hover_pct_change_prev_none_is_zero() -> None:
    assert hover_pct_change(None, 10.0) == 0.0


def test_hover_pct_change_prev_zero_is_zero() -> None:
    assert hover_pct_change(0.0, 10.0) == 0.0


def test_hover_pct_change_prev_nan_is_zero() -> None:
    assert hover_pct_change(float("nan"), 10.0) == 0.0


def test_hover_pct_change_up_and_down() -> None:
    assert hover_pct_change(10.0, 11.0) == pytest.approx(10.0)
    assert hover_pct_change(10.0, 9.0) == pytest.approx(-10.0)


def test_build_hover_html_contains_fields() -> None:
    html = build_hover_html("2024-01-02", 10.0, 11.0, 10.0)
    assert "2024-01-02" in html
    assert "开盘 10.00" in html
    assert "收盘 11.00" in html
    assert "%" in html


def test_build_hover_html_up_is_red_with_plus() -> None:
    html = build_hover_html("2024-01-02", 10.0, 11.0, 1.23)
    assert colors.UP_COLOR in html
    assert "+1.23%" in html


def test_build_hover_html_down_is_green_with_minus() -> None:
    html = build_hover_html("2024-01-02", 11.0, 10.0, -1.23)
    assert colors.DOWN_COLOR in html
    assert "-1.23%" in html


def test_build_hover_html_flat_is_gray() -> None:
    html = build_hover_html("2024-01-02", 10.0, 10.0, 0.0)
    assert colors.SUBTEXT_COLOR in html


# --------------------------------------------------------------------------- #
# 8.1 紧凑数值格式化（k / m / b / t）
# --------------------------------------------------------------------------- #
def test_format_compact_number_user_examples() -> None:
    """用户示例：17079 → 17k；2076245988 → 2.07b（截断、不进位）。"""
    assert format_compact_number(17079) == "17k"
    assert format_compact_number(2076245988) == "2.07b"


def test_format_compact_number_magnitudes_and_sign() -> None:
    assert format_compact_number(0) == "0"
    assert format_compact_number(999) == "999"
    assert format_compact_number(1000) == "1k"
    assert format_compact_number(1500) == "1.5k"
    assert format_compact_number(999999) == "999k"  # 不出现 1000k
    assert format_compact_number(123456789) == "123m"
    assert format_compact_number(2.5e12) == "2.5t"
    assert format_compact_number(-17079) == "-17k"


def test_format_compact_number_invalid() -> None:
    assert format_compact_number(None) == "—"
    assert format_compact_number(float("nan")) == "—"


def test_compact_axis_item_tick_strings_no_scientific(qapp) -> None:
    """自定义轴刻度无 1e+ 科学计数法，且与格式化函数一致。"""
    axis = CompactAxisItem(orientation="left")
    strings = axis.tickStrings([0.0, 17079.0, 2076245988.0], 1.0, 1e6)
    assert strings == ["0", "17k", "2.07b"]
    assert all("e+" not in s and "e-" not in s for s in strings)


def test_volume_axis_uses_compact_item(qapp) -> None:
    view = KLineView(db=_mem_db())
    try:
        assert isinstance(view._volume_plot.getAxis("left"), CompactAxisItem)
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 9. 悬停气泡（离屏交互）
# --------------------------------------------------------------------------- #
def _hover_ready_view(qapp) -> KLineView:
    """构造已显示、带数据的视图（``show()`` 后 isVisible 才可信，离屏平台同样如此）。"""
    view = KLineView(db=_mem_db())
    view.resize(800, 600)
    view.show()
    qapp.processEvents()
    view.set_bars(_make_bars(30))
    qapp.processEvents()
    return view


def _scene_pos_at(view: KLineView, index: int) -> QPointF:
    """用 ViewBox 把 (序号, 收盘价) 映射为场景坐标（供模拟鼠标悬停）。"""
    price = float(view._bars.iloc[index]["close"])
    return view._main_plot.getViewBox().mapViewToScene(QPointF(float(index), price))


def test_hover_bubble_shows_on_hit(qapp) -> None:
    view = _hover_ready_view(qapp)
    try:
        view._on_mouse_moved(_scene_pos_at(view, 10))
        assert view._hover_label.isVisible()
        text = view._hover_label.text()
        assert "2024-01-11" in text  # 2024-01-01 起第 11 根
        assert "%" in text
    finally:
        view.close()


def test_hover_bubble_above_left_of_cursor(qapp) -> None:
    """气泡默认位于鼠标左上方（未越界时不翻转）。"""
    view = _hover_ready_view(qapp)
    try:
        pos = _scene_pos_at(view, 10)
        view._on_mouse_moved(pos)
        label = view._hover_label
        assert label.isVisible()
        viewport = view._main_plot.viewport()
        local = view._chart_page.mapFromGlobal(
            viewport.mapToGlobal(view._main_plot.mapFromScene(pos))
        )
        assert label.x() + label.width() <= local.x() + 1   # 整体在光标左侧
        assert label.y() + label.height() <= local.y() + 1  # 且在上方
    finally:
        view.close()


def test_hover_bubble_hides_outside(qapp) -> None:
    view = _hover_ready_view(qapp)
    try:
        view._on_mouse_moved(_scene_pos_at(view, 10))
        assert view._hover_label.isVisible()
        view._on_mouse_moved(QPointF(-1000.0, -1000.0))  # 主图外
        assert not view._hover_label.isVisible()
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 10. 交易价参考线（2026-10-07：模拟训练自定义成交价的黄色虚线）
# --------------------------------------------------------------------------- #
def test_trade_price_line_set_and_clear(qapp) -> None:
    """设置 / 清除交易价：黄色虚线位置与可见性同步，不影响坐标轴范围。"""
    view = _hover_ready_view(qapp)
    try:
        assert view.trade_price_value() is None
        assert view._trade_line.isVisible() is False

        view.set_trade_price(12.34)
        assert view.trade_price_value() == pytest.approx(12.34)
        assert view._trade_line.value() == pytest.approx(12.34)
        assert view._trade_line.isVisible() is True

        view.set_trade_price(None)
        assert view.trade_price_value() is None
        assert view._trade_line.isVisible() is False
    finally:
        view.close()


def test_trade_price_line_ignores_bad_values(qapp) -> None:
    """非法 / NaN / Inf 价格一律视为清除（不崩、不画线）。"""
    view = _hover_ready_view(qapp)
    try:
        for bad in ("abc", float("nan"), float("inf")):
            view.set_trade_price(bad)
            assert view.trade_price_value() is None
            assert view._trade_line.isVisible() is False
    finally:
        view.close()


def test_trade_price_line_hidden_on_placeholder(qapp) -> None:
    """占位页（无数据）隐藏参考线；数据恢复后重新显示。"""
    view = _hover_ready_view(qapp)
    try:
        view.set_trade_price(12.34)
        assert view._trade_line.isVisible() is True

        view.set_bars(None)
        assert view.is_placeholder_visible() is True
        assert view._trade_line.isVisible() is False

        view.set_bars(_make_bars(30))
        assert view._trade_line.isVisible() is True
        assert view.trade_price_value() == pytest.approx(12.34)
    finally:
        view.close()
