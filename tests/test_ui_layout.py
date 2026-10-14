"""P2-S1 三栏布局 / 工具栏 / 主题 离屏测试（``QT_QPA_PLATFORM=offscreen``）。"""
from __future__ import annotations

import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QSignalSpy  # noqa: E402
from PySide6.QtWidgets import QApplication, QSplitter, QToolBar  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.themes import apply_theme, colors  # noqa: E402
from app.ui.widgets import MainToolBar, WatchlistPanel  # noqa: E402


class _FakeSource:
    """最小假数据源（本测试不触发更新）。"""

    name = "fake"

    def get_kline(self, symbol, period, start, end, adjust=""):
        return pd.DataFrame()

    def get_minute(self, symbol, date_):
        return pd.DataFrame()

    def list_stocks(self):
        return pd.DataFrame()

    def get_spot(self):
        return pd.DataFrame()

    def get_adj_factor(self, symbol, start, end):
        return pd.DataFrame()

    def get_fundamental(self, symbol):
        return {}


def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    db = _mem_db()
    service = UpdateService(db=db, sources={"akshare": _FakeSource(), "baostock": _FakeSource()})
    win = MainWindow(db=db, service=service)
    yield win
    win.close()


# --------------------------------------------------------------------------- #
# (a) 三栏布局
# --------------------------------------------------------------------------- #
def test_central_splitter_two_panes(window) -> None:
    """centralWidget 是横向 QSplitter，含 2 个面板，index(0) 为 WatchlistPanel。"""
    central = window.centralWidget()
    assert isinstance(central, QSplitter)
    assert central.orientation() == Qt.Orientation.Horizontal
    assert central.count() == 2
    assert isinstance(central.widget(0), WatchlistPanel)
    assert central.widget(0) is window.watchlist_panel


# --------------------------------------------------------------------------- #
# (b) 工具栏
# --------------------------------------------------------------------------- #
def test_toolbar_has_update_action(window) -> None:
    """存在 QToolBar 且含「更新数据」动作。"""
    toolbars = window.findChildren(QToolBar)
    assert toolbars, "未找到工具栏"
    assert isinstance(window.toolbar, MainToolBar)
    texts = [a.text() for a in window.toolbar.actions()]
    assert any("更新数据" in t for t in texts)


# --------------------------------------------------------------------------- #
# (c) 主题应用
# --------------------------------------------------------------------------- #
def test_apply_theme_stylesheet_no_placeholder(qapp) -> None:
    """apply_theme 后 styleSheet 非空且不含未替换的占位符。"""
    apply_theme(qapp, "dark_red_green")
    qss = qapp.styleSheet()
    assert qss.strip(), "styleSheet 为空"
    leftover = re.findall(r"\{[A-Za-z_]+\}", qss)
    assert leftover == [], f"存在未替换的占位符：{leftover}"
    # 颜色已注入
    assert colors.BG_DARK in qss
    assert colors.ACCENT_COLOR in qss


# --------------------------------------------------------------------------- #
# (d) 颜色常量（红涨绿跌）
# --------------------------------------------------------------------------- #
def test_color_constants_red_up_green_down() -> None:
    """涨=红、跌=绿。"""
    assert colors.UP_COLOR == "#e74c3c"
    assert colors.DOWN_COLOR == "#2ecc71"
    assert colors.VOL_UP == "#e74c3c"
    assert colors.VOL_DOWN == "#2ecc71"
    assert colors.color_map()["UP_COLOR"] == "#e74c3c"


# --------------------------------------------------------------------------- #
# (e) 空库降级
# --------------------------------------------------------------------------- #
def test_watchlist_empty_db_no_raise(qapp) -> None:
    """空库 refresh() 不抛异常，且出现占位提示。"""
    panel = WatchlistPanel(db=_mem_db())
    panel.refresh()  # 不应抛异常
    tree = panel.list_widget
    assert tree.topLevelItemCount() >= 1


# --------------------------------------------------------------------------- #
# (f) 回归：P1-S6 功能仍在
# --------------------------------------------------------------------------- #
def test_regression_menu_and_status_label(window) -> None:
    """既有「更新数据」菜单动作与状态栏「最后更新时间」标签仍在。"""
    assert window._action_update is not None
    assert "更新数据" in window._action_update.text()
    assert window._last_update_label is not None
    assert window._last_update_label.text().startswith("最后更新时间：")


# --------------------------------------------------------------------------- #
# 附加：信号
# --------------------------------------------------------------------------- #
def test_watchlist_selection_emits_symbol(qapp) -> None:
    """选中股票条目应发出 symbolSelected(symbol)。"""
    db = _mem_db()
    gid = db.create_group("默认")
    db.add_watch_item(gid, "000001")
    panel = WatchlistPanel(db=db)

    captured: list[str] = []
    panel.symbolSelected.connect(captured.append)

    tree = panel.list_widget
    child = None
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        for j in range(top.childCount()):
            c = top.child(j)
            if c.data(0, int(Qt.ItemDataRole.UserRole)):
                child = c
                break
        if child is not None:
            break
    assert child is not None, "未找到股票条目"
    tree.setCurrentItem(child)
    assert captured == ["000001"]


def test_toolbar_signals(qapp) -> None:
    """工具栏：更新动作触发 updateRequested；周期切换触发 periodChanged。"""
    bar = MainToolBar()
    spy_update = QSignalSpy(bar.updateRequested)
    spy_period = QSignalSpy(bar.periodChanged)

    bar.action_update.trigger()
    assert spy_update.count() == 1

    bar.set_period("5m")
    assert spy_period.count() >= 1
    assert bar.current_period() == "5m"
