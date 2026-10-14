"""P2-S5 周期快捷键 + 自选股导航测试（``QT_QPA_PLATFORM=offscreen``，离线）。

**全部使用真实按键事件**（``QTest.keyClick``）驱动，覆盖三种焦点场景：
    * 自选股树聚焦：``↓/↑`` 一次一格、跨分组连续、从不落在分组头、端点不回绕，
      每次移动恰好 1 次 ``symbolSelected``（端点 0 次为预期）；
    * 周期下拉框聚焦：``↓`` 归下拉框自己处理，**不得**切换自选股；
    * K 线图聚焦：图表不消费方向键 → 落到窗口级兜底（如实断言）；
    * 窗口级：``↓`` 由 ``MainWindow.keyPressEvent`` 处理。

注：P2-S6 起主窗口启动会自动选中首只自选股，故本文件初始态为**第一只**（而非「无选中」）。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.shortcuts import (  # noqa: E402
    KEY_NEXT_SYMBOL,
    KEY_PREV_SYMBOL,
    PERIOD_KEY_MAP,
)

# 期望的周期顺序（与 toolbar.PERIOD_ITEMS 一致）。
_EXPECTED_PERIODS = ["1m", "5m", "15m", "30m", "60m", "d", "w", "m"]
# 数字键 Qt 枚举：Qt.Key_1 .. Qt.Key_8
_PERIOD_KEYS = [getattr(Qt.Key, f"Key_{i}") for i in range(1, 9)]


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


def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _seed_three(db: Database) -> None:
    """两个分组、共 3 只股票（含分组头）：默认=[000001,000002]，自选=[600000]。"""
    g1 = db.create_group("默认")
    db.add_watch_item(g1, "000001")
    db.add_watch_item(g1, "000002")
    g2 = db.create_group("自选")
    db.add_watch_item(g2, "600000")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    db = _mem_db()
    _seed_three(db)
    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)
    win.show()
    win.activateWindow()
    qapp.processEvents()
    yield win
    win.close()


def _press_on_tree(window: MainWindow, key: Qt.Key) -> None:
    """在自选股树上按下某键（树聚焦场景）。"""
    tree = window.watchlist_panel.list_widget
    tree.setFocus()
    QApplication.processEvents()
    QTest.keyClick(tree, key)
    QApplication.processEvents()


def _press_on_window(window: MainWindow, key: Qt.Key) -> None:
    """向窗口发送按键（窗口级 / 周期快捷键场景）。"""
    QTest.keyClick(window, key)
    QApplication.processEvents()


def _current(window: MainWindow):
    return window.watchlist_panel.current_symbol()


# --------------------------------------------------------------------------- #
# 1. PERIOD_KEY_MAP
# --------------------------------------------------------------------------- #
def test_period_key_map_covers_1_to_8() -> None:
    assert set(PERIOD_KEY_MAP.keys()) == {str(i) for i in range(1, 9)}
    assert [PERIOD_KEY_MAP[str(i)] for i in range(1, 9)] == _EXPECTED_PERIODS


# --------------------------------------------------------------------------- #
# 2. 周期快捷键（真实按键）→ 切换 + 重绘
# --------------------------------------------------------------------------- #
def test_period_shortcuts_switch_and_repaint(window: MainWindow) -> None:
    # P2-S6：启动已自动选中首只，kline_view 已有标的
    assert window.kline_view.current_symbol() == "000001"

    for key, code in zip(_PERIOD_KEYS, _EXPECTED_PERIODS):
        _press_on_window(window, key)
        assert window.toolbar.current_period() == code
        assert window.kline_view.current_period() == code  # 确实重绘


# --------------------------------------------------------------------------- #
# 3. 树聚焦：一次一格 + 端点不回绕 + 信号计数
# --------------------------------------------------------------------------- #
def test_tree_focused_down_up_one_step_no_wrap(window: MainWindow) -> None:
    seq: list[str] = []
    window.watchlist_panel.symbolSelected.connect(seq.append)

    # P2-S6：启动自动选中首只（不再是「无选中」初始态）
    assert _current(window) == "000001"

    _press_on_tree(window, Qt.Key.Key_Down)          # 恰好 1 格
    assert _current(window) == "000002" and len(seq) == 1

    _press_on_tree(window, Qt.Key.Key_Down)          # 跨分组到第 3 只
    assert _current(window) == "600000" and len(seq) == 2

    _press_on_tree(window, Qt.Key.Key_Down)          # 末尾停住（不回绕，0 信号）
    assert _current(window) == "600000" and len(seq) == 2

    _press_on_tree(window, Qt.Key.Key_Up)
    assert _current(window) == "000002" and len(seq) == 3

    _press_on_tree(window, Qt.Key.Key_Up)
    assert _current(window) == "000001" and len(seq) == 4

    _press_on_tree(window, Qt.Key.Key_Up)            # 首端停住（不回绕，0 信号）
    assert _current(window) == "000001" and len(seq) == 4

    assert seq == ["000002", "600000", "000002", "000001"]


def test_tree_focused_down_single_step_single_signal(window: MainWindow) -> None:
    """直接证据：树聚焦时按 ↓ **恰好跳一格**且 `symbolSelected` 恰好 1 次（防双跳）。"""
    count = {"n": 0}
    window.watchlist_panel.symbolSelected.connect(lambda s: count.__setitem__("n", count["n"] + 1))

    # P2-S6：启动已选中首只
    assert _current(window) == "000001"
    count["n"] = 0  # 仅统计按键后的

    _press_on_tree(window, Qt.Key.Key_Down)
    assert _current(window) == "000002"
    assert count["n"] == 1


# --------------------------------------------------------------------------- #
# 4. 分组头不被选中为股票
# --------------------------------------------------------------------------- #
def test_group_headers_never_selected_as_symbol(window: MainWindow) -> None:
    tree = window.watchlist_panel.list_widget
    _press_on_tree(window, Qt.Key.Key_Down)
    _press_on_tree(window, Qt.Key.Key_Down)
    item = tree.currentItem()
    assert item is not None
    assert item.data(0, int(Qt.ItemDataRole.UserRole)) is not None
    assert window.watchlist_panel.symbols() == ["000001", "000002", "600000"]


# --------------------------------------------------------------------------- #
# 5. shortcuts() 可枚举 / qshortcuts() 只有 8 个周期键
# --------------------------------------------------------------------------- #
def test_shortcuts_enumerable_and_qshortcuts_size(window: MainWindow) -> None:
    specs = window.shortcut_manager.shortcuts()
    keys = [key for key, _ in specs]
    assert len(specs) == 10  # 8 周期 + Up + Down（说明书）
    for i in range(1, 9):
        assert str(i) in keys
    assert KEY_NEXT_SYMBOL in keys
    assert KEY_PREV_SYMBOL in keys
    assert any(desc.startswith("周期:") for _, desc in specs)

    # Up/Down 不注册 QShortcut（避免抢占 QTreeWidget 原生方向键）
    assert len(window.shortcut_manager.qshortcuts()) == 8
    assert len(window.shortcuts) == 8


# --------------------------------------------------------------------------- #
# 6. 1m 键自动切分时页（P2-S3 行为）+ 真实按键
# --------------------------------------------------------------------------- #
def test_key_1_switches_to_tick_tab(window: MainWindow) -> None:
    window.watchlist_panel.select_next()
    QApplication.processEvents()
    assert window.right_tabs.currentIndex() == 0
    _press_on_window(window, Qt.Key.Key_1)  # -> 1m
    assert window.toolbar.current_period() == "1m"
    assert window.right_tabs.currentIndex() == 1
    assert window.right_tabs.currentWidget() is window.tick_view


# --------------------------------------------------------------------------- #
# 7. 周期下拉框聚焦：↓ 不被劫持
# --------------------------------------------------------------------------- #
def test_combo_focused_down_not_hijacked(window: MainWindow) -> None:
    # P2-S6：启动已选中首只
    assert _current(window) == "000001"

    count = {"n": 0}
    window.watchlist_panel.symbolSelected.connect(lambda s: count.__setitem__("n", count["n"] + 1))

    combo = window.toolbar.period_combo
    index_before = combo.currentIndex()
    combo.setFocus()
    QApplication.processEvents()
    QTest.keyClick(combo, Qt.Key.Key_Down)
    QApplication.processEvents()

    # 下拉框自行处理 ↓（索引变化），自选股**不变**、不触发 symbolSelected
    assert _current(window) == "000001"
    assert count["n"] == 0
    assert combo.currentIndex() != index_before  # 由下拉框自己消费的副作用


# --------------------------------------------------------------------------- #
# 8. 窗口级兜底：焦点不在树上（窗口）按 ↓ 仍切自选
# --------------------------------------------------------------------------- #
def test_window_level_down_moves_selection(window: MainWindow) -> None:
    # P2-S6：启动已选中首只
    assert _current(window) == "000001"
    _press_on_window(window, Qt.Key.Key_Down)
    assert _current(window) == "000002"


def test_kline_focused_down_falls_through_to_window(window: MainWindow) -> None:
    """K 线图**不消费**方向键 → 事件冒泡至窗口兜底，切换自选股（如实记录当前行为）。"""
    # P2-S6：启动已选中首只
    assert _current(window) == "000001"

    kline = window.kline_view
    kline.setFocus()
    QApplication.processEvents()
    QTest.keyClick(kline, Qt.Key.Key_Down)
    QApplication.processEvents()
    assert _current(window) == "000002"
