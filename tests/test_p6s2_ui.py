"""P6-S2 快捷键全集 + 右键菜单补齐 —— UI 层测试。

覆盖：
    1. AlertManagerDialog：空表/新增（ma_cross 参数经 add_rule 落库）/删除/启停切换/
       非法参数（expr 空）→ ValueError 拦截不落库；
    2. watchlist_panel 右键：股票菜单含「设置预警…」，触发 → alertRequested；
    3. KLineView 右键：导出菜单含「导出K线图…」，触发 → exportRequested；
    4. ShortcutHelpDialog：构造展示 10 条清单；
    5. main_window：帮助菜单/预警/快捷键说明接线（FakeDialog 注入验证）。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

from app.engine.alert import KIND_EXPR, KIND_MA_CROSS  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.widgets.alert_manager import AlertManagerDialog  # noqa: E402
from app.ui.widgets.kline_view import KLineView  # noqa: E402
from app.ui.widgets.shortcut_help import ShortcutHelpDialog  # noqa: E402
from app.ui.widgets.watchlist_panel import WatchlistPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _seed_watchlist(db: Database, symbol: str = "000001") -> int:
    group_id = db.create_group("我的自选")
    db.add_watch_item(int(group_id), symbol)
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


# --------------------------------------------------------------------------- #
# 1. AlertManagerDialog
# --------------------------------------------------------------------------- #
def test_alert_dialog_empty_then_add(qapp) -> None:
    db = _mem_db()
    from app.engine.alert import AlertService

    service = AlertService(db=db)

    dialog = AlertManagerDialog("000001", db=db, add_rule=service.add_rule)
    try:
        assert dialog.rule_count() == 0
        # 显式选 ma_cross 并填 fast/slow。
        idx = dialog._kind_combo.findData(KIND_MA_CROSS)
        dialog._kind_combo.setCurrentIndex(idx)
        dialog._param_widgets["fast"].setValue(10)
        dialog._param_widgets["slow"].setValue(30)
        assert dialog.save_rule() is True
        assert dialog.rule_count() == 1
        rules = db.list_alert_rules(symbol="000001")
        assert len(rules) == 1
        assert rules[0].kind == KIND_MA_CROSS
        assert "fast" in rules[0].params_json
    finally:
        dialog.close()


def test_alert_dialog_invalid_expr_blocked(qapp, monkeypatch) -> None:
    db = _mem_db()
    from app.engine.alert import AlertService

    dialog = AlertManagerDialog("000001", db=db, add_rule=AlertService(db=db).add_rule)
    # 切到 expr 类型并清空表达式 → 应拦截（不落库）。
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    try:
        idx = dialog._kind_combo.findData(KIND_EXPR)
        dialog._kind_combo.setCurrentIndex(idx)
        dialog._param_widgets["expr"].setText("   ")
        assert dialog.save_rule() is False
        assert dialog.rule_count() == 0
    finally:
        dialog.close()


def test_alert_dialog_delete_and_toggle(qapp) -> None:
    db = _mem_db()
    from app.engine.alert import AlertService

    service = AlertService(db=db)
    service.add_rule("000001", KIND_MA_CROSS, {"fast": 5, "slow": 20})
    dialog = AlertManagerDialog("000001", db=db, add_rule=service.add_rule)
    try:
        assert dialog.rule_count() == 1
        dialog._table.selectRow(0)
        assert dialog.delete_selected() is True
        assert dialog.rule_count() == 0
        # 重新建一条后切换启用。
        service.add_rule("000001", KIND_MA_CROSS, {"fast": 5, "slow": 20})
        dialog._load_rules()
        assert dialog.rule_count() == 1
        dialog._table.selectRow(0)
        assert dialog.toggle_selected() is True
        rules = db.list_alert_rules(symbol="000001")
        assert rules[0].enabled is False
    finally:
        dialog.close()


# --------------------------------------------------------------------------- #
# 2. watchlist 右键「设置预警…」
# --------------------------------------------------------------------------- #
def test_watchlist_menu_has_alert_and_emits(qapp) -> None:
    db = _mem_db()
    group_id = _seed_watchlist(db, "000001")
    panel = WatchlistPanel(db=db)
    try:
        item = None
        for it in panel._symbol_items():
            if str(it.data(0, 256)) == "000001":
                item = it
                break
        assert item is not None
        menu = panel._build_stock_menu(item)
        texts = [a.text() for a in menu.actions()]
        assert "设置预警…" in texts
        assert texts.index("备注/详情…") < texts.index("设置预警…")

        emitted: list = []
        panel.alertRequested.connect(lambda gid, sym: emitted.append((gid, sym)))
        act = next(a for a in menu.actions() if a.text() == "设置预警…")
        act.trigger()
        assert emitted == [(group_id, "000001")]
    finally:
        panel.close()


# --------------------------------------------------------------------------- #
# 3. KLineView 右键导出
# --------------------------------------------------------------------------- #
def test_kline_export_menu_emits(qapp) -> None:
    view = KLineView(db=_mem_db())
    try:
        menu = QMenu()
        emitted: list = []
        view.exportRequested.connect(lambda: emitted.append(True))
        view._build_export_menu(menu)
        assert any("导出K线图" in a.text() for a in menu.actions())
        next(a for a in menu.actions() if "导出K线图" in a.text()).trigger()
        assert emitted == [True]
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 4. ShortcutHelpDialog
# --------------------------------------------------------------------------- #
def test_shortcut_help_shows_all(qapp) -> None:
    dialog = ShortcutHelpDialog(
        [("1", "周期:1分"), ("8", "周期:月"), ("Up", "上一只自选"), ("Down", "下一只自选")]
    )
    try:
        from PySide6.QtWidgets import QTableWidget

        table = dialog.findChild(QTableWidget)
        assert table is not None
        assert table.rowCount() == 4
        assert table.item(0, 0).text() == "1"
        assert table.item(0, 1).text() == "周期:1分"
    finally:
        dialog.close()


# --------------------------------------------------------------------------- #
# 5. MainWindow 接线
# --------------------------------------------------------------------------- #
def test_main_window_alert_wiring(qapp, monkeypatch) -> None:
    db = _mem_db()
    _seed_watchlist(db, "000001")
    win = _make_window(qapp, db)
    try:
        captured: dict = {}

        class _FakeDialog:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def exec(self):
                captured["exec"] = True

            def deleteLater(self):
                captured["delete"] = True

        monkeypatch.setattr("app.ui.main_window.AlertManagerDialog", _FakeDialog)
        win._open_alert_manager(1, "000001")
        assert captured["symbol"] == "000001"
        assert captured["db"] is db
        assert captured["exec"] is True and captured["delete"] is True
    finally:
        win.close()


def test_main_window_help_menu(qapp, monkeypatch) -> None:
    db = _mem_db()
    win = _make_window(qapp, db)
    try:
        help_menu = next(
            (a.menu() for a in win.menuBar().actions() if "帮助" in a.text()), None
        )
        assert help_menu is not None
        assert any("快捷键说明" in a.text() for a in help_menu.actions())

        captured: dict = {}

        class _FakeHelp:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def exec(self):
                captured["exec"] = True

            def deleteLater(self):
                captured["delete"] = True

        monkeypatch.setattr("app.ui.main_window.ShortcutHelpDialog", _FakeHelp)
        win._open_shortcut_help()
        assert captured["exec"] is True
        # shortcuts 参数 = ShortcutManager.shortcuts()（10 条说明）。
        assert len(captured["shortcuts"]) == 10
    finally:
        win.close()