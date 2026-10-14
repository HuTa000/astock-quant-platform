"""全 A 股代码与名称名录单测（P6 优化第 5 项）。

覆盖：
    * 存储层：``upsert_stock_list``（DataFrame / StockInfo 两种入参）、
      ``list_stocks``（全量 + 代码 / 名称关键词过滤）、``count_stocks``、
      同 symbol 重复 upsert 覆盖 name；
    * 编排层：``UpdateService.update_symbols``（成功、主源失败切兜底、全失败不抛）；
    * UI 层：``StockListDialog``（离屏）初始刷新、搜索过滤、双击 / 加载选中发信号、
      更新请求、总线完成事件驱动自动刷新。

说明：对话框测试在 ``QT_QPA_PLATFORM=offscreen`` 下运行，且**不调用 exec()、不弹 QMessageBox**。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.core.event_bus import get_bus  # noqa: E402
from app.data.models import StockInfo  # noqa: E402
from app.data.store.db import Database  # noqa: E402
from app.data.update_service import (  # noqa: E402
    EVT_COMPLETED,
    META_STOCK_LIST_UPDATE,
    UpdateService,
)

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.widgets.stock_list_dialog import StockListDialog  # noqa: E402


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _mem_db() -> Database:
    """新建并初始化一个内存库。"""
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _stock_frame() -> pd.DataFrame:
    """返回 3 行名录 DataFrame（列 code/name/market）。"""
    return pd.DataFrame(
        [
            {"code": "000001", "name": "平安银行", "market": "sz"},
            {"code": "600000", "name": "浦发银行", "market": "sh"},
            {"code": "600519", "name": "贵州茅台", "market": "sh"},
        ]
    )


class _ListSource:
    """仅实现 ``list_stocks`` 的假数据源（名录更新用）。"""

    def __init__(self, name: str, frame: pd.DataFrame | None = None, fail: bool = False) -> None:
        self.name = name
        self._frame = frame
        self.fail = fail
        self.calls = 0

    def list_stocks(self) -> pd.DataFrame:
        """返回预设名录；``fail=True`` 时抛异常。"""
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"{self.name} 模拟名录失败")
        assert self._frame is not None
        return self._frame


@pytest.fixture()
def bus():
    """隔离事件总线：用例前清空订阅。"""
    b = get_bus()
    b.clear()
    yield b
    b.clear()


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# 存储层
# --------------------------------------------------------------------------- #
def test_upsert_from_dataframe_and_list() -> None:
    """DataFrame 入参：写入行数、全量读取、market 回落。"""
    db = _mem_db()
    assert db.upsert_stock_list(_stock_frame()) == 3
    assert db.count_stocks() == 3
    items = db.list_stocks()
    assert [i.symbol for i in items] == ["000001", "600000", "600519"]  # symbol 升序
    by_symbol = {i.symbol: i for i in items}
    assert by_symbol["000001"].name == "平安银行"
    assert by_symbol["600519"].market == "sh"


def test_upsert_from_stock_info_list_and_empty_symbol_skipped() -> None:
    """StockInfo 序列入参：写入正确，空 symbol 行跳过。"""
    db = _mem_db()
    rows = [
        StockInfo(symbol="000001", name="平安银行", market="sz"),
        StockInfo(symbol="", name="空代码", market=""),
    ]
    assert db.upsert_stock_list(rows) == 1
    assert db.count_stocks() == 1


def test_list_stocks_keyword_filter() -> None:
    """按代码 / 名称关键词过滤。"""
    db = _mem_db()
    db.upsert_stock_list(_stock_frame())
    assert [i.symbol for i in db.list_stocks("600")] == ["600000", "600519"]
    assert [i.symbol for i in db.list_stocks("银行")] == ["000001", "600000"]
    assert db.list_stocks("不存在") == []


def test_upsert_duplicate_symbol_overwrites_name() -> None:
    """同 symbol 重复 upsert 覆盖 name，行数不增长。"""
    db = _mem_db()
    db.upsert_stock_list(pd.DataFrame([{"code": "000001", "name": "平安银行"}]))
    db.upsert_stock_list(pd.DataFrame([{"code": "000001", "name": "平安"}]))
    assert db.count_stocks() == 1
    assert db.list_stocks()[0].name == "平安"


def test_upsert_unsupported_type_raises() -> None:
    """不支持的入参类型抛 TypeError。"""
    db = _mem_db()
    with pytest.raises(TypeError):
        db.upsert_stock_list(123)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 编排层
# --------------------------------------------------------------------------- #
def test_update_symbols_success(bus) -> None:
    """主源成功：库内 2 行、meta 已写、COMPLETED 带 scope=symbols。"""
    db = _mem_db()
    frame = pd.DataFrame(
        [{"code": "000001", "name": "平安银行"}, {"code": "600000", "name": "浦发银行"}]
    )
    src = _ListSource("primary", frame)
    svc = UpdateService(db=db, primary_name="primary", fallback_name="fallback",
                        sources={"primary": src})

    completed: list[dict] = []
    bus.subscribe(EVT_COMPLETED, lambda p: completed.append(p))
    report = svc.update_symbols()

    assert report.ok is True
    assert report.symbols_count == 2
    assert report.bars_written == 2
    assert db.count_stocks() == 2
    assert db.get_meta(META_STOCK_LIST_UPDATE) is not None
    assert len(completed) == 1
    assert completed[0]["scope"] == "symbols"
    assert completed[0]["symbols_count"] == 2


def test_update_symbols_primary_fail_fallback_ok(bus) -> None:
    """主源抛异常：切兜底源成功。"""
    db = _mem_db()
    bad = _ListSource("primary", fail=True)
    good = _ListSource("fallback", pd.DataFrame([{"code": "600519", "name": "贵州茅台"}]))
    svc = UpdateService(db=db, primary_name="primary", fallback_name="fallback",
                        sources={"primary": bad, "fallback": good})

    report = svc.update_symbols()
    assert report.ok is True
    assert report.symbols_count == 1
    assert db.count_stocks() == 1
    assert bad.calls == 1 and good.calls == 1


def test_update_symbols_all_fail_no_raise(bus) -> None:
    """全部数据源失败：report.ok 为 False，且不抛异常。"""
    db = _mem_db()
    bad = _ListSource("primary", fail=True)
    svc = UpdateService(db=db, primary_name="primary", fallback_name="fallback",
                        sources={"primary": bad})

    completed: list[dict] = []
    bus.subscribe(EVT_COMPLETED, lambda p: completed.append(p))
    report = svc.update_symbols()  # 不抛异常

    assert report.ok is False
    assert report.failed and report.failed[0][0] == ""
    assert db.count_stocks() == 0
    assert completed and completed[0]["scope"] == "symbols"
    assert db.get_meta(META_STOCK_LIST_UPDATE) is None


# --------------------------------------------------------------------------- #
# UI 层（离屏）
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dialog(qapp, bus):
    """建一个已填充名录的对话框；用例结束关闭以退订总线。"""
    db = _mem_db()
    db.upsert_stock_list(_stock_frame())
    dlg = StockListDialog(db=db)
    yield dlg
    dlg.close()


def test_dialog_initial_rows(dialog) -> None:
    """初始刷新：表格行数与计数标签。"""
    assert dialog._table.rowCount() == 3
    assert "共 3 只" in dialog._count_label.text()


def test_dialog_search_filter(dialog) -> None:
    """搜索过滤生效（textChanged 自动刷新）。"""
    dialog._search_edit.setText("银行")
    assert dialog._table.rowCount() == 2
    assert "匹配 2 只" in dialog._count_label.text()
    dialog._search_edit.setText("600519")
    assert dialog._table.rowCount() == 1
    assert dialog._table.item(0, 0).text() == "600519"


def test_dialog_double_click_emits_symbol(dialog) -> None:
    """双击行触发 symbolChosen。"""
    chosen: list[str] = []
    dialog.symbolChosen.connect(chosen.append)
    dialog._table.setCurrentCell(0, 0)
    dialog._table.cellDoubleClicked.emit(0, 0)
    assert chosen == ["000001"]


def test_dialog_load_button_emits_symbol(dialog) -> None:
    """「加载选中」触发 symbolChosen。"""
    chosen: list[str] = []
    dialog.symbolChosen.connect(chosen.append)
    dialog._table.setCurrentCell(1, 0)
    dialog._btn_load.click()
    assert chosen == ["600000"]


def test_dialog_load_without_selection_hint(dialog) -> None:
    """无选中时不发信号，只更新提示标签。"""
    chosen: list[str] = []
    dialog.symbolChosen.connect(chosen.append)
    dialog._table.clearSelection()
    dialog._table.setCurrentCell(-1, -1)
    dialog._btn_load.click()
    assert chosen == []
    assert "请先选择一行" in dialog._count_label.text()


def test_dialog_update_button_emits_request(dialog) -> None:
    """「更新名录」触发 updateRequested。"""
    fired: list[int] = []
    dialog.updateRequested.connect(lambda: fired.append(1))
    dialog._btn_update.click()
    assert fired == [1]


def test_dialog_auto_refresh_on_bus_completed(dialog, bus) -> None:
    """发布 EVT_COMPLETED(scope=symbols) 后表格自动刷新。"""
    assert dialog._table.rowCount() == 3
    dialog._db.upsert_stock_list(pd.DataFrame([{"code": "300750", "name": "宁德时代"}]))
    bus.publish(EVT_COMPLETED, {"scope": "symbols", "symbols_count": 4, "bars_written": 4})
    assert dialog._table.rowCount() == 4