"""P4-S4 K 线高清图导出 —— UI 层离屏测试。

覆盖：
    1. ``KLineView.export_pixmap``：有数据返回非空 ``QPixmap``、尺寸符合 DPI 缩放；
       无数据（占位页）返回 ``None``；
    2. ``MainWindow`` 导出菜单：无 symbol → 提示「请先选中」；有 symbol → 调导出函数，
       且路径默认名含 symbol/period、DPI 取配置值。
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
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.export.image_exporter import target_size  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.widgets.kline_view import KLineView  # noqa: E402


# --------------------------------------------------------------------------- #
# 合成数据 / 假数据源
# --------------------------------------------------------------------------- #
def _make_bars(n: int = 30, symbol: str = "000001") -> pd.DataFrame:
    """生成 n 根确定性 Bar 契约数据。"""
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(10.0, 20.0, n)
    opens = base
    closes = base + np.where(np.arange(n) % 2 == 0, 0.5, -0.5)
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


def _seed_daily(db: Database, symbol: str = "000001", n: int = 30) -> int:
    """写入日线（period='d', adjust='qfq'），返回行数。"""
    seeded = _make_bars(n, symbol).copy()
    seeded["symbol"] = symbol
    seeded["period"] = "d"
    seeded["adjust"] = "qfq"
    db.upsert_bars(seeded)
    return n


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


def _make_window(qapp, db: Database) -> MainWindow:
    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    return MainWindow(db=db, service=service)


# --------------------------------------------------------------------------- #
# 1. KLineView.export_pixmap
# --------------------------------------------------------------------------- #
def test_export_pixmap_returns_scaled_pixmap_with_data(qapp) -> None:
    view = KLineView(db=_mem_db())
    try:
        view.set_bars(_make_bars(30))
        view.resize(800, 600)
        view.show()
        qapp.processEvents()

        pixmap = view.export_pixmap(300)
        assert pixmap is not None
        assert not pixmap.isNull()

        src_w = view._chart_page.width()
        src_h = view._chart_page.height()
        assert src_w > 0 and src_h > 0
        assert (pixmap.width(), pixmap.height()) == target_size(src_w, src_h, 300)
    finally:
        view.close()


def test_export_pixmap_returns_none_without_data(qapp) -> None:
    view = KLineView(db=_mem_db())
    try:
        view.show()
        qapp.processEvents()
        assert view.is_placeholder_visible()
        assert view.export_pixmap(300) is None
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 2. MainWindow 导出菜单
# --------------------------------------------------------------------------- #
def test_export_menu_no_symbol_prompts(qapp) -> None:
    db = _mem_db()
    win = _make_window(qapp, db)
    try:
        win._on_export_kline()
        assert "请先选中" in win.statusBar().currentMessage()
    finally:
        win.close()


def test_export_menu_with_symbol_calls_export(qapp, tmp_path, monkeypatch) -> None:
    db = _mem_db()
    assert _seed_daily(db, "000001", 30) == 30
    win = _make_window(qapp, db)
    try:
        win.kline_view.load("000001", "d")
        assert win.kline_view.current_symbol() == "000001"
        assert win.kline_view.bar_count() == 30

        # 隔离「抓图」与「写盘」：抓图用 dummy 非空 pixmap，写盘用 spy。
        monkeypatch.setattr(
            win.kline_view, "export_pixmap", lambda dpi=300: QPixmap(100, 100)
        )

        captured: dict = {}

        def fake_save(parent, caption, directory, filter_):
            captured["caption"] = caption
            captured["directory"] = directory
            return str(tmp_path / "out.png"), "PNG 图片 (*.png)"

        monkeypatch.setattr("app.ui.main_window.QFileDialog.getSaveFileName", fake_save)

        calls: list = []

        def spy_export(widget, path, dpi=300, scale=None):
            calls.append((widget, path, dpi, scale))
            return str(path)

        monkeypatch.setattr("app.ui.main_window.export_kline_image", spy_export)

        win._on_export_kline()

        assert len(calls) == 1
        widget, path, dpi, scale = calls[0]
        assert widget is win.kline_view
        assert path == str(tmp_path / "out.png")
        assert dpi == 300  # 配置 export.image_dpi 默认 300
        assert scale is None
        # 默认文件名含 symbol 与 period。
        assert "K线_000001_d_" in captured["directory"]
        assert captured["directory"].endswith(".png")
        assert "已导出到" in win.statusBar().currentMessage()
    finally:
        win.close()
