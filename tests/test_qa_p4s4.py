"""QA 终审验证 P4-S4（临时文件，跑完删除）。

独立证伪视角逐项验证：DPI 缩放 / 导出正确性 / 抓图目标一致性 / 无数据路径 /
写盘容错 / UI 触达 / 端到端真实导出。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402
from app.export.image_exporter import (  # noqa: E402
    ImageExportError,
    dpi_scale,
    ensure_png_path,
    export_kline_image,
    scale_pixmap,
    target_size,
)
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.widgets.kline_view import KLineView  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_bars(n: int = 30) -> pd.DataFrame:
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(10.0, 20.0, n)
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.2,
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
    seeded = _make_bars(n).copy()
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


def _make_window(qapp, db: Database) -> MainWindow:
    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    return MainWindow(db=db, service=service)


# --------------------------------------------------------------------------- #
# 1. DPI 缩放正确性
# --------------------------------------------------------------------------- #
def test_qa1_dpi_scale_values() -> None:
    assert dpi_scale(300) == 3.125
    assert dpi_scale(96) == 1.0
    assert dpi_scale(300) == pytest.approx(300 / 96.0)


def test_qa2_target_size() -> None:
    # 200 × 3.125 = 625 精确；100 × 3.125 = 312.5 → 银行家舍入 312
    assert target_size(200, 100, 300) == (625, 312)
    assert target_size(100, 100, 300) == (312, 312)
    assert target_size(800, 600, 300) == (2500, 1875)
    assert target_size(640, 480, 96) == (640, 480)
    # 逐维 min 1
    assert target_size(0, 0, 300) == (1, 1)
    assert target_size(0, 5, 300) == (1, 16)


def test_qa3_dpi_scale_invalid() -> None:
    for bad in (0, -1, -300, "abc", None, "1.5"):
        with pytest.raises(ImageExportError):
            dpi_scale(bad)


# --------------------------------------------------------------------------- #
# 2. 导出正确性 + SmoothTransformation
# --------------------------------------------------------------------------- #
def test_qa4_export_readback_size(qapp, tmp_path) -> None:
    view = KLineView(db=_mem_db())
    try:
        view.set_bars(_make_bars())
        view.resize(200, 100)
        view.show()
        qapp.processEvents()
        w, h = view.width(), view.height()
        assert w > 0 and h > 0
        saved = export_kline_image(view, tmp_path / "k.png", dpi=300)
        assert os.path.exists(saved)
        assert os.path.getsize(saved) > 0
        img = QImage(saved)
        assert not img.isNull()
        assert (img.width(), img.height()) == target_size(w, h, 300)
    finally:
        view.close()


def test_qa5_scale_pixmap_smooth_not_fast(qapp, monkeypatch) -> None:
    captured: dict = {}
    orig = QPixmap.scaled

    def fake_scaled(self, w, h, am, tm):
        captured["am"] = am
        captured["tm"] = tm
        return orig(self, w, h, am, tm)

    monkeypatch.setattr(QPixmap, "scaled", fake_scaled)
    pm = QPixmap(96, 96)
    pm.fill("#1e1e1e")
    out = scale_pixmap(pm, 300 / 96.0)
    assert out.width() == 300 and out.height() == 300
    assert captured["tm"] == Qt.TransformationMode.SmoothTransformation
    assert captured["tm"] != Qt.TransformationMode.FastTransformation


# --------------------------------------------------------------------------- #
# 3. 抓图目标一致性（重点疑点）
# --------------------------------------------------------------------------- #
def test_qa6_grab_target_consistency(qapp) -> None:
    view = KLineView(db=_mem_db())
    try:
        view.set_bars(_make_bars())
        view.resize(400, 300)
        view.show()
        qapp.processEvents()

        a = view._chart_page.grab()  # export_pixmap 内部抓的
        b = view.grab()  # export_kline_image(self.kline_view) 实际抓的

        assert (a.width(), a.height()) == (b.width(), b.height())
        ia = a.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        ib = b.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        diff = sum(
            1
            for y in range(ia.height())
            for x in range(ia.width())
            if ia.pixel(x, y) != ib.pixel(x, y)
        )
        assert diff == 0, f"chart_page.grab() 与 view.grab() 有 {diff} 像素不一致"
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 4. 无数据路径
# --------------------------------------------------------------------------- #
def test_qa7_no_symbol_no_dialog_no_export(qapp, monkeypatch) -> None:
    db = _mem_db()
    win = _make_window(qapp, db)
    try:
        assert win.kline_view.current_symbol() is None
        dialog_called: list = []
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (dialog_called.append(1), "x.png"),
        )
        export_called: list = []
        monkeypatch.setattr(
            "app.ui.main_window.export_kline_image",
            lambda *a, **k: export_called.append(1),
        )
        win._on_export_kline()
        assert "请先选中" in win.statusBar().currentMessage()
        assert not dialog_called
        assert not export_called
    finally:
        win.close()


def test_qa8_placeholder_no_data_no_dialog(qapp, monkeypatch) -> None:
    db = _mem_db()
    win = _make_window(qapp, db)
    try:
        win.kline_view.load("999999", "d")  # 有 symbol 但本地无数据 → 占位页
        assert win.kline_view.current_symbol() == "999999"
        assert win.kline_view.is_placeholder_visible()
        assert win.kline_view.export_pixmap(300) is None

        dialog_called: list = []
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (dialog_called.append(1), "x.png"),
        )
        win._on_export_kline()
        assert "请先选中" in win.statusBar().currentMessage()
        assert not dialog_called
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 5. 写盘容错
# --------------------------------------------------------------------------- #
def test_qa9_nonexist_dir_raises(qapp, tmp_path) -> None:
    view = KLineView(db=_mem_db())
    try:
        view.set_bars(_make_bars())
        view.resize(100, 80)
        with pytest.raises(ImageExportError):
            export_kline_image(view, tmp_path / "nope" / "x.png")
    finally:
        view.close()


def test_qa10_null_grab_raises(qapp, tmp_path) -> None:
    class _NullGrab:
        def grab(self):
            return QPixmap()  # null pixmap

    with pytest.raises(ImageExportError):
        export_kline_image(_NullGrab(), tmp_path / "x.png")


def test_qa11_append_png(tmp_path) -> None:
    assert ensure_png_path(tmp_path / "a") == tmp_path / "a.png"
    assert ensure_png_path(tmp_path / "a.PNG") == tmp_path / "a.PNG"
    assert ensure_png_path(tmp_path / "a.jpg") == tmp_path / "a.png"


def test_qa12_ui_surface_error(qapp, tmp_path, monkeypatch) -> None:
    db = _mem_db()
    _seed_daily(db, "000001", 30)
    win = _make_window(qapp, db)
    try:
        win.kline_view.load("000001", "d")
        monkeypatch.setattr(
            win.kline_view, "export_pixmap", lambda dpi=300: QPixmap(100, 100)
        )
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(tmp_path / "out.png"), "PNG"),
        )

        def boom(widget, path, dpi=300, scale=None):
            raise ImageExportError("测试失败原因")

        monkeypatch.setattr("app.ui.main_window.export_kline_image", boom)
        win._on_export_kline()
        msg = win.statusBar().currentMessage()
        assert "导出失败" in msg
        assert "测试失败原因" in msg
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 6. UI 触达
# --------------------------------------------------------------------------- #
def test_qa13_menu_exists(qapp) -> None:
    win = _make_window(qapp, _mem_db())
    try:
        export_menu = None
        for a in win.menuBar().actions():
            if "导出" in a.text():
                export_menu = a.menu()
                break
        assert export_menu is not None
        texts = [a.text() for a in export_menu.actions()]
        assert any("导出K线图" in t for t in texts)
    finally:
        win.close()


def test_qa14_with_symbol_calls_export(qapp, tmp_path, monkeypatch) -> None:
    db = _mem_db()
    _seed_daily(db, "000001", 30)
    win = _make_window(qapp, db)
    try:
        win.kline_view.load("000001", "d")
        monkeypatch.setattr(
            win.kline_view, "export_pixmap", lambda dpi=300: QPixmap(100, 100)
        )
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(tmp_path / "out.png"), "PNG"),
        )
        calls: list = []

        def spy(widget, path, dpi=300, scale=None):
            calls.append((widget, path, dpi, scale))
            return str(path)

        monkeypatch.setattr("app.ui.main_window.export_kline_image", spy)
        win._on_export_kline()
        assert len(calls) == 1
        widget, path, dpi, scale = calls[0]
        assert widget is win.kline_view
        assert path == str(tmp_path / "out.png")
        assert dpi == 300
        assert scale is None
        assert "已导出到" in win.statusBar().currentMessage()
    finally:
        win.close()


# --------------------------------------------------------------------------- #
# 7. 端到端真实导出（不 mock 导出函数）
# --------------------------------------------------------------------------- #
def test_qa15_e2e_real_export(qapp, tmp_path, monkeypatch) -> None:
    db = _mem_db()
    _seed_daily(db, "000001", 30)
    win = _make_window(qapp, db)
    try:
        win.resize(800, 600)
        win.show()
        qapp.processEvents()
        win.kline_view.load("000001", "d")
        qapp.processEvents()

        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(tmp_path / "out.png"), "PNG 图片 (*.png)"),
        )
        win._on_export_kline()

        out = tmp_path / "out.png"
        assert out.exists()
        assert out.stat().st_size > 0
        img = QImage(str(out))
        assert not img.isNull()
        vw, vh = win.kline_view.width(), win.kline_view.height()
        assert vw > 0 and vh > 0
        # 实际导出 = view.grab() 再按 DPI 缩放
        assert (img.width(), img.height()) == target_size(vw, vh, 300)
        assert "已导出到" in win.statusBar().currentMessage()
    finally:
        win.close()
