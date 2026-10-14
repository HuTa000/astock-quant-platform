"""P2-S6 自选股管理 + 启动加载测试（离屏 ``offscreen``，**文件库**而非 ``:memory:``）。

覆盖：
    1. 持久化（增/删/备注/分组 CRUD 跨「重启」仍一致，删组级联）；
    2. 非法代码输入不落库；
    3. 空库不自动建组（仍显示占位）；首次添加自动建「我的自选」；
    4. ``WATCHLIST_CHANGED`` 事件 payload（add/remove/分组）；
    5. 回归：``select_next/select_previous`` 真实按键逐格跨组不回绕（P2-S5）；
    6. 启动加载：有本地 K 线→自动选中首只并绘制；无 K 线→降级不抛。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.core.event_bus import get_bus  # noqa: E402
from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.widgets.watchlist_panel import WATCHLIST_CHANGED, WatchlistPanel  # noqa: E402


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


def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_daily(db: Database, symbol: str, n: int = 20) -> None:
    """写入 n 根日线（adjust='qfq'，与 KLineView.load 默认一致）。"""
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    base = np.linspace(10.0, 12.0, n)
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.05,
            "volume": np.full(n, 100.0),
            "amount": np.full(n, 1e5),
        }
    )
    frame["symbol"] = symbol
    frame["period"] = "d"
    frame["adjust"] = "qfq"
    db.upsert_bars(frame)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# 1. 持久化（跨「重启」）
# --------------------------------------------------------------------------- #
def test_persistence_across_restart(tmp_path, qapp) -> None:
    path = tmp_path / "wl.db"

    db1 = _open(path)
    panel1 = WatchlistPanel(db=db1)
    # 空库：占位，且不自动建组
    assert panel1.symbols() == []
    assert db1.list_groups() == []
    assert panel1.list_widget.topLevelItemCount() == 1
    assert "暂无自选股" in panel1.list_widget.topLevelItem(0).text(0)
    # 添加 → 自动建默认组
    assert panel1.add_symbol("000001") is True
    gid = db1.list_groups()[0].id
    db1.close()

    # 重启 1：条目仍在
    db2 = _open(path)
    panel2 = WatchlistPanel(db=db2)
    assert panel2.symbols() == ["000001"]
    # 移除
    assert panel2.remove_symbol(gid, "000001") is True
    db2.close()

    # 重启 2：已消失
    db3 = _open(path)
    panel3 = WatchlistPanel(db=db3)
    assert panel3.symbols() == []
    # 备注标记
    assert panel3.add_symbol("000001") is True
    assert panel3.toggle_note_mark(gid, "000001") is True
    db3.close()

    # 重启 3：备注仍在且条目文本含 ★
    db4 = _open(path)
    panel4 = WatchlistPanel(db=db4)
    assert panel4.symbols() == ["000001"]
    item = panel4._symbol_items()[0]
    assert "★" in item.text(0)
    assert db4.list_watch_items(gid)[0].note_mark is True
    # 分组 CRUD
    gid2 = panel4.create_group("测试组")
    assert gid2 is not None
    assert panel4.rename_group(gid2, "改名组") is True
    assert panel4.add_symbol("600000", gid2) is True
    db4.close()

    # 重启 4：分组名与条目一致
    db5 = _open(path)
    panel5 = WatchlistPanel(db=db5)
    assert "改名组" in [g.name for g in db5.list_groups()]
    assert "600000" in panel5.symbols()
    # 删除分组 → 级联删组内条目，其他组不受影响
    assert panel5.delete_group(gid2) is True
    db5.close()

    # 重启 5：组与条目一并消失
    db6 = _open(path)
    panel6 = WatchlistPanel(db=db6)
    assert "改名组" not in [g.name for g in db6.list_groups()]
    assert "600000" not in panel6.symbols()
    assert "000001" in panel6.symbols()
    db6.close()


# --------------------------------------------------------------------------- #
# 2. 非法输入
# --------------------------------------------------------------------------- #
def test_invalid_symbol_not_persisted(tmp_path, qapp) -> None:
    db = _open(tmp_path / "bad.db")
    panel = WatchlistPanel(db=db)
    before = len(db.list_watch_items())
    for bad in ("6000000", "abcdef", "", "60000", "60000 "):
        assert panel.add_symbol(bad) is False
    assert len(db.list_watch_items()) == before == 0
    assert panel.symbols() == []
    db.close()


# --------------------------------------------------------------------------- #
# 3. 无分组时自动建组 / 空库仍占位
# --------------------------------------------------------------------------- #
def test_auto_create_default_group(tmp_path, qapp) -> None:
    db = _open(tmp_path / "auto.db")
    panel = WatchlistPanel(db=db)
    # 空库刚打开：占位、无组
    assert db.list_groups() == []
    assert panel.list_widget.topLevelItemCount() == 1
    assert "暂无自选股" in panel.list_widget.topLevelItem(0).text(0)
    # 首次添加 → 自动出现「我的自选」且条目在内
    assert panel.add_symbol("600519") is True
    groups = db.list_groups()
    assert len(groups) == 1 and groups[0].name == "我的自选"
    assert panel.symbols() == ["600519"]
    db.close()


# --------------------------------------------------------------------------- #
# 4. WATCHLIST_CHANGED 事件 payload
# --------------------------------------------------------------------------- #
def test_watchlist_changed_events(tmp_path, qapp) -> None:
    db = _open(tmp_path / "ev.db")
    panel = WatchlistPanel(db=db)
    events: list[dict] = []
    get_bus().subscribe(WATCHLIST_CHANGED, lambda p: events.append(dict(p)))

    gid = panel.create_group("G1")
    assert events[-1] == {"action": "add", "symbol": "", "group_id": gid, "kind": "group"}

    assert panel.add_symbol("000001", gid) is True
    assert events[-1] == {"action": "add", "symbol": "000001", "group_id": gid}

    assert panel.toggle_note_mark(gid, "000001") is True
    assert events[-1]["action"] == "add"
    assert events[-1]["symbol"] == "000001"
    assert events[-1]["note_mark"] is True

    assert panel.rename_group(gid, "G2") is True
    assert events[-1]["kind"] == "group" and events[-1]["renamed"] is True

    assert panel.remove_symbol(gid, "000001") is True
    assert events[-1] == {"action": "remove", "symbol": "000001", "group_id": gid}

    assert panel.delete_group(gid) is True
    assert events[-1] == {"action": "remove", "symbol": "", "group_id": gid, "kind": "group"}
    db.close()


# --------------------------------------------------------------------------- #
# 5. 回归：真实按键导航逐格跨组不回绕（P2-S5）
# --------------------------------------------------------------------------- #
def test_navigation_regression_real_keys(tmp_path, qapp) -> None:
    db = _open(tmp_path / "nav.db")
    g1 = db.create_group("G1", 0)
    db.add_watch_item(g1, "000001")
    db.add_watch_item(g1, "000002")
    g2 = db.create_group("G2", 1)
    db.add_watch_item(g2, "600000")

    panel = WatchlistPanel(db=db)
    panel.show()
    qapp.processEvents()
    seq: list[str] = []
    panel.symbolSelected.connect(seq.append)

    def cur():
        return panel.current_symbol()

    assert cur() is None
    QTest.keyClick(panel.list_widget, Qt.Key.Key_Down)
    assert cur() == "000001" and len(seq) == 1
    QTest.keyClick(panel.list_widget, Qt.Key.Key_Down)
    assert cur() == "000002" and len(seq) == 2
    QTest.keyClick(panel.list_widget, Qt.Key.Key_Down)
    assert cur() == "600000" and len(seq) == 3
    QTest.keyClick(panel.list_widget, Qt.Key.Key_Down)  # 端点不回绕
    assert cur() == "600000" and len(seq) == 3
    QTest.keyClick(panel.list_widget, Qt.Key.Key_Up)
    assert cur() == "000002" and len(seq) == 4
    # 当前项必为股票条目（从不落在分组头）
    item = panel.list_widget.currentItem()
    assert item is not None
    assert item.data(0, int(Qt.ItemDataRole.UserRole)) is not None
    panel.close()
    db.close()


def test_remove_current_and_toggle_current(tmp_path, qapp) -> None:
    db = _open(tmp_path / "cur.db")
    panel = WatchlistPanel(db=db)
    panel.add_symbol("000001")
    panel.add_symbol("000002")
    gid = db.list_groups()[0].id

    assert panel.current_symbol() == "000002"  # 新增后自动选中
    assert panel.toggle_note_mark() is True    # 作用于当前条目
    assert panel.remove_current() is True
    assert panel.symbols() == ["000001"]
    # 分组头/无选中 → remove_current False
    panel._tree.setCurrentItem(panel.list_widget.topLevelItem(0))  # 分组头
    assert panel.current_symbol() is None
    assert panel.remove_current() is False
    assert panel.toggle_note_mark() is False
    assert gid is not None
    db.close()


# --------------------------------------------------------------------------- #
# 6. 启动加载
# --------------------------------------------------------------------------- #
def test_startup_loads_first_symbol_with_bars(tmp_path, qapp) -> None:
    db = _open(tmp_path / "start.db")
    gid = db.create_group("默认")
    db.add_watch_item(gid, "000001")
    _seed_daily(db, "000001", n=20)

    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)
    try:
        assert win.watchlist_panel.current_symbol() == "000001"
        assert win.kline_view.current_symbol() == "000001"
        assert win.kline_view.bar_count() > 0
        assert not win.kline_view.is_placeholder_visible()
    finally:
        win.close()
    db.close()


def test_startup_loads_first_symbol_without_bars(tmp_path, qapp) -> None:
    db = _open(tmp_path / "start2.db")
    gid = db.create_group("默认")
    db.add_watch_item(gid, "600000")

    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    win = MainWindow(db=db, service=service)  # 不得抛异常
    try:
        assert win.watchlist_panel.current_symbol() == "600000"
        assert win.kline_view.current_symbol() == "600000"
        assert win.kline_view.bar_count() == 0
        assert win.kline_view.is_placeholder_visible()
        assert "本地无数据" in win.kline_view.placeholder_text()
    finally:
        win.close()
    db.close()
