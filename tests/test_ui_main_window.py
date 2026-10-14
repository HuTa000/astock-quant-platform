"""``MainWindow`` 离屏 UI 测试。

在 ``QT_QPA_PLATFORM=offscreen`` 下构造主窗口，验证：
    * 菜单栏含「数据 → 更新数据」动作；
    * 状态栏存在「最后更新时间」标签，且文案随 meta / 事件更新。
"""
from __future__ import annotations

import os

# 必须在导入任何 Qt 模块之前设置离屏平台。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.core.event_bus import get_bus  # noqa: E402
from app.data.store.db import Database  # noqa: E402
from app.data.update_service import (  # noqa: E402
    EVT_COMPLETED,
    EVT_FAILED,
    META_LAST_UPDATE,
    UpdateService,
)

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402


class _FakeSource:
    """最小假数据源（UI 测试不实际触发更新，仅需满足构造）。"""

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


@pytest.fixture(scope="module")
def qapp():
    """提供（或创建）全局 QApplication。"""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    """构造离屏主窗口（注入内存库与假数据源服务）。"""
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    service = UpdateService(db=db, sources={"akshare": _FakeSource(), "baostock": _FakeSource()})
    win = MainWindow(db=db, service=service)
    yield win
    win.close()


def test_menu_has_update_action(window) -> None:
    """菜单栏菜单标题含「数据」，且含「更新数据」动作。"""
    menu_titles = [a.text() for a in window.menuBar().actions()]
    assert any("数据" in t for t in menu_titles)

    action_texts = [a.text() for a in window._data_menu.actions()]
    assert any("更新数据" in t for t in action_texts)


def test_statusbar_has_last_update_label(window) -> None:
    """状态栏常驻标签存在，初始无 meta 时显示占位符。"""
    assert window._last_update_label is not None
    text = window._last_update_label.text()
    assert text.startswith("最后更新时间：")
    assert text.endswith("—")  # 初始无记录


def test_last_update_label_follows_meta(qapp) -> None:
    """标签文案随 meta 更新（手动刷新 + 完成事件两条路径）。"""
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)
    try:
        db.set_meta(META_LAST_UPDATE, "2026-09-21T10:00:00")
        win._refresh_last_update_time()
        assert "2026-09-21T10:00:00" in win._last_update_label.text()

        # 完成事件路径：改 meta 后发布事件，标签应随之刷新
        db.set_meta(META_LAST_UPDATE, "2026-09-22T11:30:00")
        get_bus().publish(EVT_COMPLETED, {"symbols_count": 0, "bars_written": 0, "failed_count": 0})
        qapp.processEvents()
        assert "2026-09-22T11:30:00" in win._last_update_label.text()
    finally:
        win.close()


def test_progress_updates_status_message(window) -> None:
    """进度事件应更新状态栏文案。"""
    window._on_progress({"symbol": "000001", "current": 1, "total": 3})
    assert "000001" in window.statusBar().currentMessage()


def test_failed_event_shows_symbol_and_friendly_reason(window) -> None:
    """失败事件经事件总线到达 UI 时，状态栏应给出「标的 + 人话原因」。"""
    get_bus().publish(EVT_FAILED, {"symbol": "000001", "error": "网络代理不可用"})
    QApplication.instance().processEvents()  # 跨线程信号排队后需处理一次
    message = window.statusBar().currentMessage()
    assert message == "更新失败：000001（网络代理不可用）"


def test_failed_event_without_reason_still_works(window) -> None:
    """error 缺失时不得抛异常，也应保留「更新失败：」前缀。"""
    window._on_failed({"symbol": "600000"})
    assert window.statusBar().currentMessage() == "更新失败：600000"

    window._on_failed({"symbol": "600000", "error": ""})
    assert window.statusBar().currentMessage() == "更新失败：600000"


def test_watchlist_add_auto_triggers_update(window, monkeypatch) -> None:
    """回归（P6-S4 缺陷修复）：新增自选股且本地无其 K 线 → 自动触发一次更新。"""
    from app.ui.widgets.watchlist_panel import WATCHLIST_CHANGED

    calls: list = []
    monkeypatch.setattr(window, "_on_update_daily", lambda symbols=None: calls.append(symbols))

    # add 且无数据 → 触发。
    get_bus().publish(WATCHLIST_CHANGED, {"action": "add", "symbol": "601869", "group_id": 1})
    assert calls == [["601869"]]


def test_watchlist_add_with_data_no_trigger(window, monkeypatch) -> None:
    """新增标的但本地已有 K 线 → 不重复触发（避免冗余网络请求）。"""
    from app.ui.widgets.watchlist_panel import WATCHLIST_CHANGED

    # 预置一条日线数据。
    frame = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2026-01-05")],
            "open": [10.0], "high": [10.5], "low": [9.9], "close": [10.2],
            "volume": [1000.0], "amount": [1_000_000.0],
        }
    )
    frame["symbol"] = "600000"
    frame["period"] = "d"
    frame["adjust"] = ""
    window._db.upsert_bars(frame)

    calls: list = []
    monkeypatch.setattr(window, "_on_update_daily", lambda symbols=None: calls.append(symbols))
    get_bus().publish(WATCHLIST_CHANGED, {"action": "add", "symbol": "600000", "group_id": 1})
    assert calls == []


def test_watchlist_remove_no_trigger(window, monkeypatch) -> None:
    """非 add 动作（remove / 分组）→ 不触发更新。"""
    from app.ui.widgets.watchlist_panel import WATCHLIST_CHANGED

    calls: list = []
    monkeypatch.setattr(window, "_on_update_daily", lambda symbols=None: calls.append(symbols))
    get_bus().publish(WATCHLIST_CHANGED, {"action": "remove", "symbol": "600000", "group_id": 1})
    get_bus().publish(
        WATCHLIST_CHANGED, {"action": "add", "symbol": "", "group_id": 1, "kind": "group"}
    )
    assert calls == []

