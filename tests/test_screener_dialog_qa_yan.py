"""P3-S5 选股对话框 —— QA 严过关 补充对抗性测试（第 1 轮独立验证）。

本文件只做**新增覆盖**，不改任何源码（``app/`` 下零改动）。针对工程师
``tests/test_screener_dialog.py`` 的 10 用例未覆盖的边界做独立取证：

    A. 字段中文映射：预置显示名（市盈率 PE 等）→ 引擎字段名（pe 等）；
    B. 自由输入技术面表达式原样保留（``close.rolling(20).mean()``）；
    C. 阈值边界：浮点 / 负数 / 科学计数法 / 非数值 / 空串；
    D. 重复运行：表格清空旧结果再填新结果（不叠加），``last_result`` 更新；
    E. 加自选语义：未选中安全提示、多选批量、部分失败（add_symbol 返回 False）；
    F. 保存模板：重名不崩溃、``conditions_json`` 可 ``json.loads`` 还原为合法
       :class:`Condition` 列表（round-trip）；
    G. 空状态：无任何有效条件时按钮路径不创建 worker、不崩溃；
    H. 后台线程：``_ScreenWorker`` 完成后信号回 UI、``_worker`` 引用被清理；
    I. 线程生命周期：运行中关闭对话框不崩溃、worker 可正常结束。
"""
from __future__ import annotations

import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QItemSelectionModel  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.engine.screener import Condition, ScreenResult  # noqa: E402
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


class SlowScreener:
    """阻塞式 fake 引擎：``run`` 会阻塞直到 ``gate`` 被释放（用于线程生命周期取证）。"""

    def __init__(self, gate: threading.Event, hits=None, total=None) -> None:
        self.gate = gate
        self.entered = threading.Event()
        self._hits = list(hits or [])
        self._total = len(self._hits) if total is None else int(total)

    def run(self, conditions, logic="AND") -> ScreenResult:
        self.entered.set()
        self.gate.wait(timeout=10)
        return ScreenResult(hits=list(self._hits), failed=[], total=self._total)


class FakeDb:
    """注入用 fake 存储（记录 save_screener_template 调用，模拟「重名新建」）。"""

    def __init__(self) -> None:
        self.saved: list[tuple[str, str]] = []

    def save_screener_template(self, name, conditions_json, template_id=None) -> int:
        self.saved.append((name, conditions_json))
        return len(self.saved)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_dialog(screener=None, db=None, add_symbol=None, parent=None) -> ScreenerDialog:
    return ScreenerDialog(
        db=db if db is not None else FakeDb(),
        screener=screener if screener is not None else FakeScreener(),
        add_symbol=add_symbol,
        parent=parent,
    )


def _set_row(dialog: ScreenerDialog, index: int, field: str, op: str, value: str) -> None:
    row = dialog.condition_rows[index]
    row.field_combo.setCurrentText(field)
    row.op_combo.setCurrentText(op)
    row.value_edit.setText(value)


def _spin_until(predicate, timeout: float = 5.0) -> bool:
    """在 UI 线程循环处理事件，直到 ``predicate`` 为真或超时。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QCoreApplication.processEvents()
    return predicate()


# --------------------------------------------------------------------------- #
# A. 字段中文映射：预置显示名 → 引擎字段名
# --------------------------------------------------------------------------- #
def test_preset_field_labels_map_to_engine_codes(qapp) -> None:
    dialog = _make_dialog()
    expected = {
        "市盈率 PE": "pe",
        "市净率 PB": "pb",
        "净资产收益率 ROE": "roe",
        "营收同比增长": "revenue_yoy",
        "总市值": "total_market_cap",
        "流通市值": "float_market_cap",
    }
    row = dialog.condition_rows[0]
    for label, code in expected.items():
        row.field_combo.setCurrentText(label)
        row.value_edit.setText("1")
        conds = dialog.current_conditions()
        assert conds[0].field == code, f"{label!r} 应映射为 {code!r}"


# --------------------------------------------------------------------------- #
# B. 自由输入技术面表达式原样保留
# --------------------------------------------------------------------------- #
def test_free_input_expression_preserved(qapp) -> None:
    dialog = _make_dialog()
    expr = "close.rolling(20).mean()"
    _set_row(dialog, 0, expr, ">", "10")

    conds = dialog.current_conditions()

    assert len(conds) == 1
    assert conds[0].field == expr


def test_free_input_field_code_preserved(qapp) -> None:
    """直接输入引擎字段名 ``pe``（非中文标签）也应原样保留为 ``pe``。"""
    dialog = _make_dialog()
    _set_row(dialog, 0, "pe", "<", "30")

    conds = dialog.current_conditions()

    assert conds[0].field == "pe"


# --------------------------------------------------------------------------- #
# C. 阈值边界
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.5", 10.5),    # 浮点
        ("-3", -3.0),      # 负数
        ("1e3", 1000.0),   # 科学计数法
        ("0", 0.0),        # 零
    ],
)
def test_threshold_valid_boundaries(qapp, raw, expected) -> None:
    dialog = _make_dialog()
    _set_row(dialog, 0, "close", ">", raw)

    conds = dialog.current_conditions()

    assert len(conds) == 1
    assert conds[0].value == expected


@pytest.mark.parametrize("raw", ["abc", "", "  ", "1,000", "1 000"])
def test_threshold_invalid_skipped(qapp, raw) -> None:
    dialog = _make_dialog()
    _set_row(dialog, 0, "close", ">", raw)

    conds = dialog.current_conditions()

    assert conds == []


# --------------------------------------------------------------------------- #
# D. 重复运行：清空旧结果 + 更新 last_result
# --------------------------------------------------------------------------- #
def test_repeated_run_clears_table_and_updates_last_result(qapp) -> None:
    first = FakeScreener(hits=["000001", "600519"], total=2)
    second = FakeScreener(hits=["000002"], total=1)
    dialog = _make_dialog(screener=first)
    _set_row(dialog, 0, "close", ">", "1")

    dialog.run_screen()
    assert dialog.result_table.rowCount() == 2
    assert dialog.last_result() is not None

    # 换引擎后再跑一次
    dialog._screener = second
    result = dialog.run_screen()

    assert result is not None
    assert result.hits == ["000002"]
    assert dialog.last_result() is result
    assert dialog.result_table.rowCount() == 1
    assert dialog.result_table.item(0, 0).text() == "000002"


# --------------------------------------------------------------------------- #
# E. 加自选语义
# --------------------------------------------------------------------------- #
def test_add_watchlist_no_selection_safe(qapp) -> None:
    calls: list[str] = []
    screener = FakeScreener(hits=["000001"], total=1)
    dialog = _make_dialog(screener=screener, add_symbol=lambda s: calls.append(s) or True)
    _set_row(dialog, 0, "close", ">", "1")
    dialog.run_screen()

    added = dialog.add_selected_to_watchlist()
    dialog._on_add_watchlist()

    assert added == []
    assert calls == []
    assert "未选中结果或加入失败" in dialog.status_label.text()


def test_add_watchlist_multi_select_all_called(qapp) -> None:
    calls: list[str] = []
    screener = FakeScreener(hits=["000001", "600519", "000002"], total=3)
    dialog = _make_dialog(screener=screener, add_symbol=lambda s: calls.append(s) or True)
    _set_row(dialog, 0, "close", ">", "1")
    dialog.run_screen()

    sm = dialog.result_table.selectionModel()
    for row in (0, 2):
        index = dialog.result_table.model().index(row, 0)
        sm.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    added = dialog.add_selected_to_watchlist()

    assert added == ["000001", "000002"]
    assert calls == ["000001", "000002"]


def test_add_watchlist_partial_failure(qapp) -> None:
    """add_symbol 返回 False（如重复添加）时：成功项入 added，失败项不崩溃、不混入。"""
    calls: list[str] = []
    screener = FakeScreener(hits=["000001", "600519"], total=2)

    def cb(symbol: str) -> bool:
        calls.append(symbol)
        return symbol == "000001"  # 只有第一只成功

    dialog = _make_dialog(screener=screener, add_symbol=cb)
    _set_row(dialog, 0, "close", ">", "1")
    dialog.run_screen()

    sm = dialog.result_table.selectionModel()
    for row in (0, 1):
        index = dialog.result_table.model().index(row, 0)
        sm.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    dialog._on_add_watchlist()

    assert calls == ["000001", "600519"]  # 两只都被尝试
    # D3：部分失败时明确提示成功/失败数量并列出失败代码
    text = dialog.status_label.text()
    assert "成功 1 项" in text
    assert "失败 1 项" in text
    assert "600519" in text  # 失败项被点名


# --------------------------------------------------------------------------- #
# F. 保存模板：重名 + round-trip
# --------------------------------------------------------------------------- #
def test_save_template_duplicate_name_no_crash(qapp) -> None:
    db = FakeDb()
    dialog = _make_dialog(db=db)
    _set_row(dialog, 0, "close", ">", "10")
    dialog.template_name_edit.setText("低价股")

    id1 = dialog.save_template()
    id2 = dialog.save_template()

    assert id1 == 1
    assert id2 == 2
    assert len(db.saved) == 2  # 重名 → 新增两行，不崩溃
    assert db.saved[0][0] == db.saved[1][0] == "低价股"


def test_save_template_conditions_json_roundtrip(qapp) -> None:
    db = FakeDb()
    dialog = _make_dialog(db=db)
    _set_row(dialog, 0, "close", ">", "10")
    dialog.add_condition_row()
    dialog.condition_rows[1].field_combo.setCurrentIndex(0)  # 市盈率 PE → pe
    dialog.condition_rows[1].op_combo.setCurrentText("<")
    dialog.condition_rows[1].value_edit.setText("30")
    dialog.template_name_edit.setText("低估值")

    dialog.save_template()

    _, conditions_json = db.saved[0]
    raw = json.loads(conditions_json)
    assert isinstance(raw, list)
    assert len(raw) == 2

    restored = [Condition.from_dict(item) for item in raw]
    originals = dialog.current_conditions()
    assert len(restored) == len(originals)
    for got, want in zip(restored, originals):
        assert got.field == want.field
        assert got.op == want.op
        assert got.value == want.value


# --------------------------------------------------------------------------- #
# G. 空状态：按钮路径不创建 worker
# --------------------------------------------------------------------------- #
def test_run_button_no_conditions_no_worker(qapp) -> None:
    dialog = _make_dialog()
    dialog.condition_rows[0].value_edit.setText("")  # 阈值空 → 无有效条件

    dialog._on_run()

    assert dialog._worker is None
    assert dialog.last_result() is None
    assert "没有有效条件" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# H. 后台线程：完成回 UI + worker 引用清理
# --------------------------------------------------------------------------- #
def test_background_worker_finishes_and_cleans_up(qapp) -> None:
    gate = threading.Event()
    screener = SlowScreener(gate, hits=["000001", "600519"], total=2)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "1")

    dialog._on_run()
    assert screener.entered.wait(2), "worker 未进入 run()"
    assert dialog._worker is not None
    assert dialog._worker.isRunning()

    gate.set()
    ok = _spin_until(lambda: dialog._worker is None, timeout=5.0)
    assert ok, "worker 完成后引用未被清理"

    assert dialog.last_result() is not None
    assert dialog.last_result().hits == ["000001", "600519"]
    assert dialog.result_table.rowCount() == 2
    assert dialog.run_button.isEnabled()  # 按钮恢复


# --------------------------------------------------------------------------- #
# I. 线程生命周期：运行中关闭对话框不崩溃、worker 可结束
# --------------------------------------------------------------------------- #
def test_close_while_worker_running_no_crash(qapp) -> None:
    gate = threading.Event()
    screener = SlowScreener(gate, hits=["000001"], total=1)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "1")

    dialog._on_run()
    assert screener.entered.wait(2), "worker 未进入 run()"

    # 运行中关闭对话框（真实模态流：close 即 hide，exec 返回）
    dialog.close()

    gate.set()
    ok = _spin_until(lambda: dialog._worker is None, timeout=5.0)
    assert ok, "关闭后 worker 未正常结束"

    # 结果信号仍回 UI（对话框未析构），不崩溃
    assert dialog.last_result() is not None
    assert dialog.last_result().hits == ["000001"]
    assert dialog.result_table.rowCount() == 1


def test_worker_parent_is_none(qapp) -> None:
    """取证（D1 修复后）：worker 的 parent 置为 None，对话框析构不再连带销毁运行中的 QThread。"""
    gate = threading.Event()
    screener = SlowScreener(gate, hits=["000001"], total=1)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "1")

    dialog._on_run()
    assert screener.entered.wait(2)

    worker = dialog._worker
    assert worker.parent() is None  # D1 修复：生命周期与对话框解耦
    gate.set()
    _spin_until(lambda: dialog._worker is None, timeout=5.0)


def test_close_and_delete_while_running_no_crash(qapp) -> None:
    """D1+D2：运行中 close 再 deleteLater，不崩、worker 自然结束（无 shiboken RuntimeError）。"""
    gate = threading.Event()
    screener = SlowScreener(gate, hits=["000001"], total=1)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "1")

    dialog._on_run()
    assert screener.entered.wait(2), "worker 未进入 run()"

    # 稍后在后台线程释放 gate；主线程立刻 close（closeEvent 内 wait() 等其结束）
    threading.Timer(0.05, gate.set).start()
    dialog.close()
    dialog.deleteLater()
    for _ in range(20):
        QCoreApplication.processEvents()
        time.sleep(0.01)

    # 走到这里即未抛 shiboken RuntimeError / 未崩溃
    assert screener.entered.is_set()


def test_run_button_no_conditions_clears_result_and_table(qapp) -> None:
    """D4：先跑出结果→清空条件→点运行，应清空 _last_result 与表格，不残留旧结果。"""
    screener = FakeScreener(hits=["000001"], total=1)
    dialog = _make_dialog(screener=screener)
    _set_row(dialog, 0, "close", ">", "1")
    dialog.run_screen()
    assert dialog.last_result() is not None
    assert dialog.result_table.rowCount() == 1

    # 清空条件，走按钮路径
    dialog.condition_rows[0].value_edit.setText("")
    dialog._on_run()

    assert dialog.last_result() is None
    assert dialog.result_table.rowCount() == 0
    assert "没有有效条件" in dialog.status_label.text()
