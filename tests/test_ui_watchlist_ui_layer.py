"""P2-S6 UI 触达层回归测试（**真实鼠标点击按钮**，离屏 ``offscreen``）。

本文件补的是上一轮复核漏掉的**信号签名**缺陷（P0）：
    ``QPushButton.clicked`` 自带一个 ``bool checked`` 参数，若直接连到
    ``_on_add_symbol_action(group_id=None)``，点击时 ``group_id`` 会被填成 ``False``
    （``bool`` 是 ``int`` 子类，会静默当成分组 0），导致「＋ 添加自选」按钮**加不进股票**。

覆盖：
    1. 真实点击「＋ 添加自选」→ 代码真的进库，且 ``add_symbol`` 第二参**不是 bool**；
    2. 非法代码经 UI → 不落库，弹窗文案含「格式」，且**不出现**「指定分组不存在」；
    3. 真实点击「＋ 新建分组」→ 分组真的落库；
    4. ``add_symbol`` 防御：显式拒绝 ``bool`` / 非 ``int`` 型 ``group_id``（防同类回归）；
    5. 代码校验只接受 **ASCII** 数字（全角/阿拉伯-印度数字一律拒绝）；
    6. 真实点击「＋ 添加自选」落到**当前选中**分组（无选中则回退默认组）。
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.widgets import watchlist_panel as wp  # noqa: E402
from app.ui.widgets.watchlist_panel import WatchlistPanel, is_valid_symbol  # noqa: E402

_LOG_NAME = "app.ui.widgets.watchlist_panel"


def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _patch_input(monkeypatch, value: str, ok: bool = True) -> None:
    """把模块内 ``QInputDialog.getText`` 替换为固定返回值（离屏不弹模态）。"""
    monkeypatch.setattr(
        wp,
        "QInputDialog",
        types.SimpleNamespace(getText=lambda *a, **k: (value, ok)),
    )


def _patch_messagebox(monkeypatch) -> list[tuple[str, str]]:
    """替换 ``QMessageBox``，记录 ``warning`` 的 ``(title, text)``，``question`` 默认否。"""
    records: list[tuple[str, str]] = []

    def _warn(_parent, title, text, *a, **k):  # noqa: ANN001
        records.append((str(title), str(text)))
        return None

    monkeypatch.setattr(
        wp,
        "QMessageBox",
        types.SimpleNamespace(
            warning=_warn,
            question=lambda *a, **k: 0,
            StandardButton=types.SimpleNamespace(Yes=1, No=0),
        ),
    )
    return records


def _spy_add_symbol(monkeypatch, panel: WatchlistPanel) -> list[tuple]:
    """包一层 spy 记录 ``add_symbol`` 实参 ``(symbol, group_id, type)``。"""
    calls: list[tuple] = []
    original = panel.add_symbol

    def spy(symbol, group_id=None):  # noqa: ANN001
        calls.append((symbol, group_id, type(group_id).__name__))
        return original(symbol, group_id)

    monkeypatch.setattr(panel, "add_symbol", spy)
    return calls


def _group_header(panel: WatchlistPanel, group_id: int):
    """返回指定分组 id 的顶层分组头条目（找不到返回 ``None``）。"""
    root = panel.list_widget.invisibleRootItem()
    for i in range(root.childCount()):
        item = root.child(i)
        if item.data(0, wp._GROUP_ROLE) == group_id:
            return item
    return None


def _symbol_item(panel: WatchlistPanel, symbol: str):
    """返回指定代码的股票条目（找不到返回 ``None``）。"""
    for item in panel._symbol_items():
        if str(item.data(0, wp._SYMBOL_ROLE)) == symbol:
            return item
    return None


# --------------------------------------------------------------------------- #
# 1. 真实点击「＋ 添加自选」→ 真的进库 + 第二参不是 bool
# --------------------------------------------------------------------------- #
def test_btn_add_mouseclick_adds_symbol(tmp_path, qapp, monkeypatch, caplog) -> None:
    db = _open(tmp_path / "ui_add.db")
    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()

    _patch_input(monkeypatch, "600000", ok=True)
    warnings = _patch_messagebox(monkeypatch)
    calls = _spy_add_symbol(monkeypatch, panel)

    with caplog.at_level(logging.WARNING, logger=_LOG_NAME):
        QTest.mouseClick(panel._btn_add, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    # 关键：真的进库
    assert "600000" in panel.symbols()
    assert "600000" in {it.symbol for it in db.list_watch_items()}
    # 关键：第二参不是 bool（修复前是 ('600000', False, 'bool')）
    assert calls, "add_symbol 未被调用"
    assert calls[-1] == ("600000", None, "NoneType")
    assert not isinstance(calls[-1][1], bool)
    # 关键：不再出现误导性的「指定分组不存在」
    assert "指定分组不存在" not in caplog.text
    assert warnings == []  # 无失败弹窗
    panel.close()
    db.close()


# --------------------------------------------------------------------------- #
# 2. 非法代码经 UI → 不落库 + 文案含「格式」
# --------------------------------------------------------------------------- #
def test_btn_add_invalid_code_shows_format_warning(tmp_path, qapp, monkeypatch, caplog) -> None:
    db = _open(tmp_path / "ui_bad.db")
    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()

    _patch_input(monkeypatch, "abcdef", ok=True)
    warnings = _patch_messagebox(monkeypatch)
    calls = _spy_add_symbol(monkeypatch, panel)

    with caplog.at_level(logging.WARNING, logger=_LOG_NAME):
        QTest.mouseClick(panel._btn_add, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert panel.symbols() == []
    assert db.list_watch_items() == []
    assert warnings, "应弹出失败提示"
    assert "格式" in warnings[-1][1]
    assert "半角" in warnings[-1][1]
    # 非法格式在 UI 层被拦下，不进入 add_symbol，也不报「指定分组不存在」
    assert calls == []
    assert "指定分组不存在" not in caplog.text
    panel.close()
    db.close()


# --------------------------------------------------------------------------- #
# 2b. 全角数字经 UI 真实点击 → 被拦下，文案点明「半角」（QA 复验固化场景）
# --------------------------------------------------------------------------- #
def test_btn_add_fullwidth_code_blocked_with_halfwidth_hint(tmp_path, qapp, monkeypatch, caplog) -> None:
    db = _open(tmp_path / "ui_fullwidth.db")
    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()

    _patch_input(monkeypatch, "０００００１", ok=True)  # 全角数字
    warnings = _patch_messagebox(monkeypatch)
    calls = _spy_add_symbol(monkeypatch, panel)

    with caplog.at_level(logging.WARNING, logger=_LOG_NAME):
        QTest.mouseClick(panel._btn_add, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert panel.symbols() == []
    assert db.list_watch_items() == []
    assert calls == []  # 全角在 UI 层即被拦下，不进入 add_symbol
    assert warnings, "应弹出失败提示"
    assert "格式" in warnings[-1][1]
    assert "半角" in warnings[-1][1]
    assert "指定分组不存在" not in caplog.text
    panel.close()
    db.close()


# --------------------------------------------------------------------------- #
# 3. 真实点击「＋ 新建分组」→ 分组真的落库
# --------------------------------------------------------------------------- #
def test_btn_new_group_mouseclick(tmp_path, qapp, monkeypatch) -> None:
    db = _open(tmp_path / "ui_group.db")
    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()

    _patch_input(monkeypatch, "观察池", ok=True)
    _patch_messagebox(monkeypatch)

    QTest.mouseClick(panel._btn_new_group, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert "观察池" in [g.name for g in db.list_groups()]
    panel.close()
    db.close()


# --------------------------------------------------------------------------- #
# 4. add_symbol 防御：显式拒绝 bool 型 group_id
# --------------------------------------------------------------------------- #
def test_add_symbol_rejects_bool_group_id(tmp_path, qapp) -> None:
    db = _open(tmp_path / "ui_bool.db")
    panel = WatchlistPanel(db=db)

    assert panel.add_symbol("600000", False) is False
    assert panel.add_symbol("600000", True) is False
    assert panel.symbols() == []
    assert db.list_watch_items() == []
    db.close()


# --------------------------------------------------------------------------- #
# 5. 校验函数与内部规则一致（供 UI 预校验复用）—— 只接受 ASCII 数字
# --------------------------------------------------------------------------- #
def test_is_valid_symbol_matches_rule() -> None:
    assert is_valid_symbol("600000") is True
    assert is_valid_symbol(" 000001 ") is True
    # 全角 digit / 阿拉伯-印度数字：Python re 默认 re.UNICODE，\d 会误判为 True；须用 [0-9] 拒绝。
    assert is_valid_symbol("０００００１") is False
    assert is_valid_symbol("٠٠٠٠٠١") is False
    assert is_valid_symbol("６０００００") is False
    for bad in ("abcdef", "6000000", "60000", "", " ", "０００００１", "٠٠٠٠٠١", "６０００００"):
        assert is_valid_symbol(bad) is False


# --------------------------------------------------------------------------- #
# 6. 全角/非 ASCII 数字不得落库，且日志给出「非法股票代码」
# --------------------------------------------------------------------------- #
def test_add_symbol_rejects_unicode_digits(tmp_path, qapp, caplog) -> None:
    db = _open(tmp_path / "ui_unicode.db")
    panel = WatchlistPanel(db=db)

    with caplog.at_level(logging.WARNING, logger=_LOG_NAME):
        assert panel.add_symbol("０００００１") is False
    assert panel.symbols() == []
    assert db.list_watch_items() == []
    assert "非法股票代码" in caplog.text
    db.close()


# --------------------------------------------------------------------------- #
# 7. add_symbol 拒绝 str 型 group_id（只允许 int / None）
# --------------------------------------------------------------------------- #
def test_add_symbol_rejects_str_group_id(tmp_path, qapp) -> None:
    db = _open(tmp_path / "ui_str.db")
    g1 = db.create_group("g1", 0)
    panel = WatchlistPanel(db=db)

    assert panel.add_symbol("600000", "1") is False
    assert panel.symbols() == []
    assert db.list_watch_items(g1) == []
    db.close()


# --------------------------------------------------------------------------- #
# 8. 真实点击「＋ 添加自选」→ 落到**当前选中**分组
# --------------------------------------------------------------------------- #
def test_btn_add_lands_in_current_selected_group(tmp_path, qapp, monkeypatch) -> None:
    db = _open(tmp_path / "ui_land.db")
    g1 = db.create_group("g1", 0)
    g2 = db.create_group("g2", 1)
    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()

    hdr2 = _group_header(panel, g2)
    assert hdr2 is not None
    panel.list_widget.setCurrentItem(hdr2)
    qapp.processEvents()
    assert panel.current_group_id() == g2  # 当前选中 g2

    _patch_input(monkeypatch, "600000", ok=True)
    _patch_messagebox(monkeypatch)
    QTest.mouseClick(panel._btn_add, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert "600000" in [it.symbol for it in db.list_watch_items(g2)]
    assert "600000" not in [it.symbol for it in db.list_watch_items(g1)]
    panel.close()
    db.close()


# --------------------------------------------------------------------------- #
# 9. 真实点击「＋ 添加自选」→ 无选中时回退默认组「我的自选」
# --------------------------------------------------------------------------- #
def test_btn_add_falls_back_to_default_group_when_no_selection(tmp_path, qapp, monkeypatch) -> None:
    db = _open(tmp_path / "ui_default.db")  # 空库：无分组、无选中
    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()
    assert panel.list_widget.currentItem() is None
    assert panel.current_group_id() is None

    _patch_input(monkeypatch, "600000", ok=True)
    _patch_messagebox(monkeypatch)
    QTest.mouseClick(panel._btn_add, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    groups = db.list_groups()
    assert len(groups) == 1 and groups[0].name == "我的自选"
    assert "600000" in [it.symbol for it in db.list_watch_items(groups[0].id)]
    panel.close()
    db.close()


# --------------------------------------------------------------------------- #
# 10. 真实点击「＋ 添加自选」→ 选中**股票条目**时加到其所在分组
# --------------------------------------------------------------------------- #
def test_btn_add_with_selected_symbol_stays_in_its_group(tmp_path, qapp, monkeypatch) -> None:
    db = _open(tmp_path / "ui_sym.db")
    g1 = db.create_group("g1", 0)
    db.add_watch_item(g1, "000001")
    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()

    item = _symbol_item(panel, "000001")
    assert item is not None
    panel.list_widget.setCurrentItem(item)
    qapp.processEvents()
    assert panel.current_group_id() == g1

    _patch_input(monkeypatch, "600000", ok=True)
    _patch_messagebox(monkeypatch)
    QTest.mouseClick(panel._btn_add, Qt.MouseButton.LeftButton)
    qapp.processEvents()

    symbols_g1 = [it.symbol for it in db.list_watch_items(g1)]
    assert "000001" in symbols_g1 and "600000" in symbols_g1
    panel.close()
    db.close()

