"""P3-S5 选股对话框测试（离屏 ``offscreen``）。

覆盖：
    1. 对话框构造 + 默认逻辑 AND；
    2. ``current_conditions()`` 正确收集（添加 2 个条件 → field/op/value 正确）；
    3. 运行选股（注入 fake screener）→ 结果表格行数 + 状态标签正确；
    4. 保存模板（注入 fake db）→ ``save_screener_template`` 收到正确 name + conditions_json；
    5. 加自选（注入 fake add_symbol 回调）→ 选中结果行批量加自选被调用；
    6. 条件校验：空字段 / 空阈值 / 非数值阈值跳过不崩溃；
    7. 空结果：0 命中时表格空、状态标签正确。
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QItemSelectionModel  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.engine.screener import ScreenResult  # noqa: E402
from app.export.excel_exporter import ExcelExportError  # noqa: E402
from app.ui.widgets.screener_dialog import ScreenerDialog  # noqa: E402


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
class FakeScreener:
    """注入用 fake 选股引擎（返回固定 ScreenResult，并记录调用）。"""

    def __init__(self, hits=None, failed=None, total=None) -> None:
        self._hits = list(hits or [])
        self._failed = list(failed or [])
        self._total = len(self._hits) if total is None else int(total)
        self.calls: list[tuple] = []

    def run(self, conditions, logic="AND") -> ScreenResult:
        self.calls.append((list(conditions), logic))
        return ScreenResult(hits=list(self._hits), failed=list(self._failed), total=self._total)


class FakeDb:
    """注入用 fake 存储（记录 save_screener_template 调用）。"""

    def __init__(self) -> None:
        self.saved: list[tuple[str, str]] = []

    def save_screener_template(self, name, conditions_json, template_id=None) -> int:
        self.saved.append((name, conditions_json))
        return len(self.saved)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_dialog(screener=None, db=None, add_symbol=None) -> ScreenerDialog:
    return ScreenerDialog(
        db=db if db is not None else FakeDb(),
        screener=screener if screener is not None else FakeScreener(),
        add_symbol=add_symbol,
    )


def _set_row(dialog: ScreenerDialog, index: int, field: str, op: str, value: str) -> None:
    """设置第 index 行条件的字段 / 操作符 / 阈值。"""
    row = dialog.condition_rows[index]
    row.field_combo.setCurrentText(field)
    row.op_combo.setCurrentText(op)
    row.value_edit.setText(value)


# --------------------------------------------------------------------------- #
# 1. 构造 + 默认逻辑
# --------------------------------------------------------------------------- #
def test_construct_and_default_logic(qapp) -> None:
    dialog = _make_dialog()
    assert dialog.current_logic() == "AND"
    assert dialog.last_result() is None
    assert len(dialog.condition_rows) == 1  # 初始一行
    assert dialog.result_table.rowCount() == 0


# --------------------------------------------------------------------------- #
# 2. current_conditions 收集
# --------------------------------------------------------------------------- #
def test_current_conditions_collects(qapp) -> None:
    dialog = _make_dialog()
    _set_row(dialog, 0, "close", ">", "10")
    dialog.add_condition_row()
    row1 = dialog.condition_rows[1]
    row1.field_combo.setCurrentIndex(0)  # 预置项「市盈率 PE」→ 字段 pe
    row1.op_combo.setCurrentText("<")
    row1.value_edit.setText("30")

    conds = dialog.current_conditions()

    assert len(conds) == 2
    assert [c.field for c in conds] == ["close", "pe"]
    assert [c.op for c in conds] == [">", "<"]
    assert [c.value for c in conds] == [10.0, 30.0]


# --------------------------------------------------------------------------- #
# 3. 运行选股 → 表格 + 状态标签
# --------------------------------------------------------------------------- #
def test_run_screen_fills_table_and_status(qapp) -> None:
    screener = FakeScreener(
        hits=["000001", "600519"],
        failed=[{"symbol": "000002", "reason": "无K线"}],
        total=3,
    )
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "10")

    result = dialog.run_screen()

    assert result is not None
    assert result.hits == ["000001", "600519"]
    assert dialog.last_result() is result
    assert dialog.result_table.rowCount() == 2
    assert dialog.result_table.item(0, 0).text() == "000001"
    assert dialog.result_table.item(0, 1).text() == "000001"  # 无 name_resolver → 回退代码
    assert dialog.result_table.item(1, 0).text() == "600519"
    text = dialog.status_label.text()
    assert "命中 2 只" in text
    assert "候选 3 只" in text
    assert "失败 1 只" in text


def test_run_screen_name_resolver(qapp) -> None:
    screener = FakeScreener(hits=["000001"], total=1)
    dialog = ScreenerDialog(
        db=FakeDb(),
        screener=screener,
        name_resolver=lambda s: "平安银行" if s == "000001" else None,
    )
    _set_row(dialog, 0, "close", ">", "1")
    dialog.run_screen()
    assert dialog.result_table.item(0, 0).text() == "000001"
    assert dialog.result_table.item(0, 1).text() == "平安银行"


# --------------------------------------------------------------------------- #
# 4. 保存模板
# --------------------------------------------------------------------------- #
def test_save_template(qapp) -> None:
    db = FakeDb()
    dialog = _make_dialog(db=db)
    _set_row(dialog, 0, "close", ">", "10")
    dialog.template_name_edit.setText("低价股")

    template_id = dialog.save_template()

    assert template_id == 1
    assert len(db.saved) == 1
    name, conditions_json = db.saved[0]
    assert name == "低价股"
    assert json.loads(conditions_json) == [{"field": "close", "op": ">", "value": 10.0}]


def test_save_template_requires_name_and_conditions(qapp) -> None:
    db = FakeDb()
    dialog = _make_dialog(db=db)
    # 名称空
    assert dialog.save_template() is None
    # 名称有但条件全空
    dialog.template_name_edit.setText("x")
    assert dialog.save_template() is None
    assert db.saved == []


# --------------------------------------------------------------------------- #
# 5. 加自选（批量）
# --------------------------------------------------------------------------- #
def test_add_selected_to_watchlist(qapp) -> None:
    calls: list[str] = []
    screener = FakeScreener(hits=["000001", "600519"], total=2)
    dialog = _make_dialog(screener=screener, add_symbol=lambda s: calls.append(s) or True)
    _set_row(dialog, 0, "close", ">", "1")
    dialog.run_screen()

    sm = dialog.result_table.selectionModel()
    for row in (0, 1):
        index = dialog.result_table.model().index(row, 0)
        sm.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    added = dialog.add_selected_to_watchlist()

    assert added == ["000001", "600519"]
    assert calls == ["000001", "600519"]


# --------------------------------------------------------------------------- #
# 6. 条件校验：空字段 / 空阈值 / 非数值阈值跳过不崩溃
# --------------------------------------------------------------------------- #
def test_condition_validation_skips_invalid(qapp) -> None:
    dialog = _make_dialog()
    # 第一行：空字段 + 空阈值 → 跳过
    dialog.condition_rows[0].field_combo.setCurrentText("")
    dialog.condition_rows[0].value_edit.setText("")
    # 第二行：字段有、阈值非数值 → 跳过
    dialog.add_condition_row()
    _set_row(dialog, 1, "close", ">", "abc")
    # 第三行：合法
    dialog.add_condition_row()
    _set_row(dialog, 2, "close", ">", "10")

    conds = dialog.current_conditions()

    assert len(conds) == 1
    assert conds[0].field == "close"
    assert conds[0].value == 10.0


def test_run_screen_with_no_valid_conditions(qapp) -> None:
    dialog = _make_dialog()
    dialog.condition_rows[0].value_edit.setText("")  # 阈值空

    result = dialog.run_screen()

    assert result is None
    assert dialog.last_result() is None
    assert "没有有效条件" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# 7. 空结果
# --------------------------------------------------------------------------- #
def test_empty_result(qapp) -> None:
    screener = FakeScreener(hits=[], failed=[], total=0)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "10")

    result = dialog.run_screen()

    assert result is not None
    assert dialog.result_table.rowCount() == 0
    assert "命中 0 只" in dialog.status_label.text()
    assert "候选 0 只" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# 8. 导出 Excel 按钮（P4-S3）
# --------------------------------------------------------------------------- #
def test_export_no_result_prompts_and_skips_exporter(qapp, monkeypatch) -> None:
    dialog = _make_dialog()
    calls: list[tuple] = []

    monkeypatch.setattr(
        "app.ui.widgets.screener_dialog.export_screen_result",
        lambda *a, **k: calls.append((a, k)) or "x.xlsx",
    )

    dialog._on_export()

    assert calls == []
    assert "请先运行选股" in dialog.status_label.text()


def test_export_empty_hits_prompts_and_skips_exporter(qapp, monkeypatch) -> None:
    screener = FakeScreener(hits=[], failed=[], total=0)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "10")
    dialog.run_screen()

    calls: list[tuple] = []
    monkeypatch.setattr(
        "app.ui.widgets.screener_dialog.export_screen_result",
        lambda *a, **k: calls.append((a, k)) or "x.xlsx",
    )

    dialog._on_export()

    assert calls == []
    assert "请先运行选股" in dialog.status_label.text()


def test_export_with_result_calls_exporter(qapp, monkeypatch, tmp_path) -> None:
    screener = FakeScreener(hits=["000001", "600519"], total=2)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "10")
    dialog.run_screen()

    path = str(tmp_path / "导出结果.xlsx")
    monkeypatch.setattr(
        "app.ui.widgets.screener_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (path, "Excel 文件 (*.xlsx)"),
    )
    captured: dict = {}

    def fake_export(result, p, name_resolver=None, conditions=None):
        captured["result"] = result
        captured["path"] = p
        captured["name_resolver"] = name_resolver
        captured["conditions"] = conditions
        return p

    monkeypatch.setattr("app.ui.widgets.screener_dialog.export_screen_result", fake_export)

    dialog._on_export()

    assert captured["result"] is dialog.last_result()
    assert captured["path"] == path
    assert captured["name_resolver"] is not None  # 传入 _resolve_name 绑定方法
    assert [c.field for c in captured["conditions"]] == ["close"]
    assert f"已导出到 {path}" in dialog.status_label.text()


def test_export_failure_shows_reason(qapp, monkeypatch, tmp_path) -> None:
    screener = FakeScreener(hits=["000001"], total=1)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "1")
    dialog.run_screen()

    path = str(tmp_path / "x.xlsx")
    monkeypatch.setattr(
        "app.ui.widgets.screener_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (path, ""),
    )

    def boom(*args, **kwargs):
        raise ExcelExportError("磁盘只读")

    monkeypatch.setattr("app.ui.widgets.screener_dialog.export_screen_result", boom)

    dialog._on_export()

    assert "导出失败" in dialog.status_label.text()
    assert "磁盘只读" in dialog.status_label.text()
