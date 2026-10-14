"""P4-S5 个股详情（备注）弹窗 —— 纯逻辑 + 对话框 + UI 触达层测试。

覆盖：
    1. 纯格式化函数：市值亿/元、百分比、数值、文本、缺键占位；
    2. ``format_fundamental``：字段映射与占位；
    3. ``NoteDetailDialog``：既有备注回填、保存 / 清除备注（含信号与 ★ 联动回调）、
       脏关闭守卫（保存 / 放弃 / 取消）、后台加载成功 / 失败降级、无 loader 直接占位；
    4. ``WatchlistPanel``：``set_note_mark`` 显式置 ★；右键「备注/详情…」菜单项
       存在且动作连线到 ``detailRequested``（半端到端：QMenu.exec 模态不可离屏点击，
       与 P2-S6 既有限制一致，验证菜单项→槽连线）；
    5. ``MainWindow._open_note_detail``：向弹窗传 symbol/db、★ 联动接线、关闭后析构。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402
from app.export.image_exporter import ImageExportError  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.widgets.note_editor import (  # noqa: E402
    NoteDetailDialog,
    format_fundamental,
    fmt_cap,
    fmt_num,
    fmt_pct,
    fmt_text,
)
from app.ui.widgets.watchlist_panel import WatchlistPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# 辅助：内存库
# --------------------------------------------------------------------------- #
def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _seed_watchlist(db: Database, group: str = "我的自选", symbol: str = "000001") -> int:
    """建组并加入一只自选股，返回 group_id。"""
    group_id = db.create_group(group)
    db.add_watch_item(group_id, symbol)
    return int(group_id)


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


def _full_fund() -> dict:
    """一份字段齐全的基本面数据（单位符合 §4.2 约定）。"""
    return {
        "symbol": "000001",
        "name": "平安银行",
        "industry": "银行",
        "list_date": "1991-04-03",
        "total_market_cap": 2_000_000_000.0,   # 元 → 20 亿
        "float_market_cap": 1_800_000_000.0,   # 18 亿
        "pe": 6.15,
        "pb": 0.87,
        "roe": 4.6746,
        "revenue_yoy": -3.2,
        "report_period": "2026Q2",
        "summary": "平安银行；行业：银行；上市：1991-04-03；ROE：4.67%",
    }


# --------------------------------------------------------------------------- #
# 1. 纯格式化函数
# --------------------------------------------------------------------------- #
def test_fmt_num() -> None:
    assert fmt_num(6.156) == "6.16"
    assert fmt_num(None) == "—"
    assert fmt_num("abc") == "—"


def test_fmt_cap() -> None:
    assert fmt_cap(2_000_000_000.0) == "20.00 亿"
    assert fmt_cap(99_000_000.0) == "99,000,000.00 元"
    assert fmt_cap(None) == "—"


def test_fmt_pct() -> None:
    assert fmt_pct(4.6746) == "4.67%"
    assert fmt_pct(-3.2) == "-3.20%"
    assert fmt_pct(None) == "—"


def test_fmt_text() -> None:
    assert fmt_text("  平安银行 ") == "平安银行"
    assert fmt_text("") == "—"
    assert fmt_text(None) == "—"


# --------------------------------------------------------------------------- #
# 2. format_fundamental
# --------------------------------------------------------------------------- #
def test_format_fundamental_full() -> None:
    out = format_fundamental(_full_fund())
    assert out["名称"] == "平安银行"
    assert out["行业"] == "银行"
    assert out["上市日期"] == "1991-04-03"
    assert out["总市值"] == "20.00 亿"
    assert out["流通市值"] == "18.00 亿"
    assert out["PE"] == "6.15"
    assert out["PB"] == "0.87"
    assert out["ROE"] == "4.67%"
    assert out["同比"] == "-3.20%"
    assert out["报告期"] == "2026Q2"


def test_format_fundamental_missing_keys() -> None:
    out = format_fundamental({})
    for label in ("名称", "行业", "总市值", "PE", "ROE", "报告期"):
        assert out[label] == "—"


def test_format_fundamental_none() -> None:
    out = format_fundamental(None)
    assert out["名称"] == "—"


# --------------------------------------------------------------------------- #
# 3. NoteDetailDialog
# --------------------------------------------------------------------------- #
def test_dialog_placeholder_without_loader(qapp) -> None:
    db = _mem_db()
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    try:
        assert dialog.fundamental_placeholder()["名称"] == "—"
        assert "未加载" in dialog.status_text()
    finally:
        dialog.close()


def test_dialog_loads_existing_note(qapp) -> None:
    db = _mem_db()
    db.set_note("000001", "  已有备注  ")
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    try:
        assert dialog.note_text() == "已有备注"
        assert not dialog.is_dirty()
    finally:
        dialog.close()


def test_save_note_writes_and_emits(qapp) -> None:
    db = _mem_db()
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    saved: list = []
    dialog.note_saved.connect(saved.append)
    try:
        dialog._note_edit.setPlainText("观察异动")
        assert dialog.save_note() is True
        note = db.get_note("000001")
        assert note is not None and note.text == "观察异动"
        assert saved == ["000001"]
        assert not dialog.is_dirty()
    finally:
        dialog.close()


def test_save_empty_text_delegates_to_clear(qapp) -> None:
    db = _mem_db()
    db.set_note("000001", "旧内容")
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    cleared: list = []
    saved: list = []
    dialog.note_cleared.connect(cleared.append)
    dialog.note_saved.connect(saved.append)
    try:
        dialog._note_edit.setPlainText("   ")
        assert dialog.save_note() is True
        assert db.get_note("000001") is None
        assert cleared == ["000001"]
        assert saved == []
    finally:
        dialog.close()


def test_clear_note_deletes_and_emits(qapp) -> None:
    db = _mem_db()
    db.set_note("000001", "要删除")
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    cleared: list = []
    dialog.note_cleared.connect(cleared.append)
    try:
        assert dialog.clear_note() is True
        assert db.get_note("000001") is None
        assert cleared == ["000001"]
        assert dialog.note_text() == ""
    finally:
        dialog.close()


def test_fundamental_load_success_populates(qapp, monkeypatch) -> None:
    db = _mem_db()
    dialog = NoteDetailDialog(
        "000001", db=db, fundamental_loader=lambda sym: dict(_full_fund())
    )
    try:
        # 等待后台线程完成 + 刷新队列信号（QThread 信号为队列投递，需事件循环派发）。
        if dialog._worker is not None:
            dialog._worker.wait()
        qapp.processEvents()
        assert dialog.fundamental_placeholder()["名称"] == "平安银行"
        assert dialog.fundamental_placeholder()["总市值"] == "20.00 亿"
        assert "已加载" in dialog.status_text()
        assert "ROE：4.67%" in dialog._summary_label.text()
    finally:
        dialog.close()


def test_fundamental_load_failure_degrades(qapp, monkeypatch) -> None:
    db = _mem_db()

    def boom(_sym):
        raise RuntimeError("网络不可达")

    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=boom)
    try:
        if dialog._worker is not None:
            dialog._worker.wait()
        qapp.processEvents()
        assert dialog.fundamental_placeholder()["名称"] == "—"
        assert "加载失败" in dialog.status_text()
    finally:
        dialog.close()


def test_dirty_close_cancel_keeps_dialog(qapp, monkeypatch) -> None:
    db = _mem_db()
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    try:
        dialog._note_edit.setPlainText("未保存的修改")
        dialog.show()
        qapp.processEvents()

        # 点「取消」→ 关闭被忽略，正文未保存。
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
        )
        from PySide6.QtGui import QCloseEvent

        event = QCloseEvent()
        dialog.closeEvent(event)
        assert not event.isAccepted()
        assert db.get_note("000001") is None
    finally:
        dialog.close()


def test_dirty_close_save_writes(qapp, monkeypatch) -> None:
    db = _mem_db()
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    try:
        dialog._note_edit.setPlainText("关闭前保存")
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save),
        )
        from PySide6.QtGui import QCloseEvent

        event = QCloseEvent()
        dialog.closeEvent(event)
        assert event.isAccepted()
        note = db.get_note("000001")
        assert note is not None and note.text == "关闭前保存"
    finally:
        dialog.close()


def test_dirty_close_discard_keeps_nothing(qapp, monkeypatch) -> None:
    db = _mem_db()
    dialog = NoteDetailDialog("000001", db=db, fundamental_loader=None)
    try:
        dialog._note_edit.setPlainText("要丢弃的")
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
        )
        from PySide6.QtGui import QCloseEvent

        event = QCloseEvent()
        dialog.closeEvent(event)
        assert event.isAccepted()
        assert db.get_note("000001") is None
    finally:
        dialog.close()


# --------------------------------------------------------------------------- #
# 4. WatchlistPanel：显式置 ★ 与详情信号
# --------------------------------------------------------------------------- #
def test_panel_set_note_mark_true_and_false(qapp) -> None:
    db = _mem_db()
    group_id = _seed_watchlist(db, symbol="000001")
    panel = WatchlistPanel(db=db)
    try:
        assert panel.set_note_mark(group_id, "000001", True) is True
        item = _find_symbol_item(panel, "000001")
        assert item is not None and "★" in item.text(0)
        assert panel.set_note_mark(group_id, "000001", False) is True
        item = _find_symbol_item(panel, "000001")
        assert item is not None and "★" not in item.text(0)
    finally:
        panel.close()


def test_panel_set_note_mark_missing_item_false(qapp) -> None:
    db = _mem_db()
    group_id = _seed_watchlist(db, symbol="000001")
    panel = WatchlistPanel(db=db)
    try:
        # 条目不存在 → db.set_note_mark 不抛、方法幂等返回 True（与既有 toggle 语义一致）。
        assert panel.set_note_mark(group_id, "999999", True) is True
    finally:
        panel.close()


def test_panel_detail_menu_action_emits_signal(qapp) -> None:
    """右键「备注/详情…」菜单项存在，且选中后动作连线到 ``detailRequested``。

    半端到端：``QMenu.exec`` 模态离屏不可真实点击，按 P2-S6 既有限制验证
    「菜单项 → 动作连线」（新增的菜单项排在第一位，动作即首项）。
    """
    db = _mem_db()
    group_id = _seed_watchlist(db, symbol="000001")
    panel = WatchlistPanel(db=db)
    try:
        item = _find_symbol_item(panel, "000001")
        assert item is not None

        emitted: list = []
        panel.detailRequested.connect(lambda gid, sym: emitted.append((gid, sym)))

        menu = panel._build_stock_menu(item)
        assert menu is not None
        texts = [act.text() for act in menu.actions()]
        assert texts[0] == "备注/详情…"
        assert texts[1] == "设置预警…"  # P6-S2：股票右键补齐预警入口
        assert texts[2] == "添加到自选…"
        assert texts[3] == "从自选移除"
        # 触发首项动作（等价于用户在菜单中选中该项）。
        menu.actions()[0].trigger()
        assert emitted == [(group_id, "000001")]
    finally:
        panel.close()


def _find_symbol_item(panel: WatchlistPanel, symbol: str):
    """在面板树中定位指定代码的股票条目（供测试）。"""
    for item in panel._symbol_items():
        if str(item.data(0, 256)) == symbol:  # UserRole==256（§4.11）
            return item
    return None


# --------------------------------------------------------------------------- #
# 5. MainWindow：详情弹窗接线
# --------------------------------------------------------------------------- #
class _FakeSignal:
    """模拟 Qt 信号：记录 connect 的处理器，供测试触发。"""

    def __init__(self) -> None:
        self.handlers: list = []

    def connect(self, handler) -> None:
        self.handlers.append(handler)

    def emit(self, *args) -> None:
        for handler in list(self.handlers):
            handler(*args)


def test_open_note_detail_wires_dialog_and_star(qapp, tmp_path, monkeypatch) -> None:
    db = _mem_db()
    group_id = _seed_watchlist(db, symbol="000001")
    win = _make_window(qapp, db)
    try:
        captured: dict = {}

        class _FakeDialog:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs
                captured["dialog"] = self
                self.note_saved = _FakeSignal()
                self.note_cleared = _FakeSignal()

            def exec(self):
                captured["exec_called"] = True
                return 0

            def deleteLater(self):
                captured["delete_later_called"] = True

        monkeypatch.setattr("app.ui.main_window.NoteDetailDialog", _FakeDialog)
        win._open_note_detail(int(group_id), "000001")

        assert captured["kwargs"]["symbol"] == "000001"
        assert captured["kwargs"]["db"] is db
        assert captured["kwargs"]["parent"] is win
        assert captured["kwargs"]["fundamental_loader"] is not None  # 默认走主数据源
        assert captured["exec_called"] is True
        assert captured["delete_later_called"] is True

        dialog = captured["dialog"]
        # ★ 联动：触发 note_saved → set_note_mark(True)；note_cleared → set_note_mark(False)。
        dialog.note_saved.emit("000001")
        item = _find_symbol_item(win.watchlist_panel, "000001")
        assert item is not None and "★" in item.text(0)

        dialog.note_cleared.emit("000001")
        item = _find_symbol_item(win.watchlist_panel, "000001")
        assert item is not None and "★" not in item.text(0)
    finally:
        win.close()


def test_detail_requested_wired_in_main_window(qapp) -> None:
    db = _mem_db()
    _seed_watchlist(db, symbol="000001")
    win = _make_window(qapp, db)
    try:
        panel = win.watchlist_panel
        # 信号已连接：向面板注入一个假处理器，发信号不抛即视为接线正常。
        called: list = []
        win._open_note_detail = lambda gid, sym: called.append((gid, sym))
        panel.detailRequested.emit(1, "000001")
        assert called == [(1, "000001")]
    finally:
        win.close()
