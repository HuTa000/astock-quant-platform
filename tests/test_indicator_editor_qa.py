"""P3-S3 自定义指标编辑器 —— QA 独立补测（离屏 ``offscreen``，文件库）。

本文件由 QA 严过关补充，针对工程师 6 用例未覆盖的**真实缺陷场景**：

    1. 边界输入：名称/表达式全空格、空分类、超长名称；
    2. 预览错误路径：``close / 0``（Runtime inf）、语法错误、``import os`` / dunder 逃逸
       （Security/Syntax），以及**错误后恢复合法表达式**；
    3. 模板列表交互：空列表删除安全、删除后 template_id 清理、保存后列表即时刷新；
    4. w/m 派生：仅日线数据预览周线，长度 == 日线周分组数；
    5. 重复保存语义：两次「新建」应产生两个模板（**预期暴露源码缺陷**）；
       选中模板后改名保存应更新同 id（既有语义确认）。

约定：只读源码（``app/``），仅新增/修复测试，不修改实现。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.widgets.indicator_editor import IndicatorEditorDialog  # noqa: E402


def _open(path) -> Database:
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_daily(db: Database, symbol: str, n: int = 60) -> None:
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.arange(1.0, n + 1.0)
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 100.0),
            "amount": close * 1000.0,
        }
    )
    frame["symbol"] = symbol
    frame["period"] = "d"
    frame["adjust"] = ""
    db.upsert_bars(frame)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# --------------------------------------------------------------------------- #
# 1. 边界输入
# --------------------------------------------------------------------------- #
def test_save_whitespace_name_rejected(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_ws.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("   ")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        assert dialog.save_template() is None
        assert db.list_indicator_templates() == []
        assert "名称不能为空" in dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


def test_save_whitespace_expr_rejected(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_wsexpr.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("空表达式")
        dialog.expr_edit.setPlainText("   \n\t ")
        assert dialog.save_template() is None
        assert db.list_indicator_templates() == []
        assert "表达式不能为空" in dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


def test_save_empty_category_ok(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_cat.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("无分类指标")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        dialog.category_combo.setCurrentText("")
        assert dialog.current_category() == ""
        tid = dialog.save_template()
        assert tid is not None
        items = db.list_indicator_templates()
        assert items[0].category == ""
        assert "未分类" in dialog.template_list.item(0).text()
    finally:
        dialog.close()
        db.close()


def test_save_long_name_no_crash(tmp_path, qapp) -> None:
    """超长名称：当前无长度上限，仅验证不崩溃、不破坏数据（P3 提示：无长度约束）。"""
    db = _open(tmp_path / "qa_long.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("长" * 500)
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        tid = dialog.save_template()
        assert tid is not None
        assert len(db.list_indicator_templates()) == 1
    finally:
        dialog.close()
        db.close()


# --------------------------------------------------------------------------- #
# 2. 预览错误路径 + 恢复
# --------------------------------------------------------------------------- #
def test_preview_divide_by_zero_runtime(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_div0.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.expr_edit.setPlainText("close / 0")
        result = dialog.preview_result()
        assert result.ok is False
        assert result.error_type == "Runtime"
        dialog._on_preview()
        assert "Runtime" in dialog.status_label.text()
        # 曲线被清空（pyqtgraph 空 x 返回 None，故用「None 或空」断言，不残留上一次曲线）
        xs, ys = dialog.curve.getData()
        assert xs is None or len(xs) == 0
        assert ys is None or len(ys) == 0
    finally:
        dialog.close()
        db.close()


def test_preview_syntax_error(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_syn.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.expr_edit.setPlainText("close.rolling(")
        result = dialog.preview_result()
        assert result.ok is False
        assert result.error_type == "Syntax"
        dialog._on_preview()
        assert "Syntax" in dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


def test_preview_import_os_blocked(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_imp.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.expr_edit.setPlainText("import os")
        result = dialog.preview_result()
        assert result.ok is False
        # import 是语句，mode=eval 下被判 Syntax；绝不落入执行
        assert result.error_type in ("Syntax", "Security")
        dialog._on_preview()
        assert dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


def test_preview_dunder_access_security(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_dunder.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.expr_edit.setPlainText("close.__class__")
        result = dialog.preview_result()
        assert result.ok is False
        assert result.error_type == "Security"
        dialog._on_preview()
        assert "Security" in dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


def test_preview_error_then_valid_recovers(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_rec.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        # 先出错
        dialog.expr_edit.setPlainText("close / 0")
        assert dialog.preview_result().ok is False
        dialog._on_preview()
        assert "Runtime" in dialog.status_label.text()

        # 再恢复合法表达式
        dialog.expr_edit.setPlainText("close.rolling(20).mean()")
        result = dialog.preview_result()
        assert result.ok is True
        dialog._on_preview()
        assert "已生成 60 点曲线" in dialog.status_label.text()
        assert len(dialog.curve.getData()[0]) == 60
    finally:
        dialog.close()
        db.close()


# --------------------------------------------------------------------------- #
# 3. 模板列表交互
# --------------------------------------------------------------------------- #
def test_delete_with_no_selection_safe(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_nodel.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        assert dialog.selected_template_id() is None
        assert dialog.delete_template() is False
        assert "未选中模板" in dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


def test_delete_clears_template_id_then_save_new(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_del.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("旧模板")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        tid = dialog.save_template()
        assert tid is not None
        assert dialog.template_list.count() == 1

        dialog.template_list.setCurrentRow(0)
        assert dialog.selected_template_id() == tid
        assert dialog.delete_template() is True
        assert dialog._template_id is None
        assert dialog.template_list.count() == 0

        # 删除后再保存 → 全新 id，不误更新已删除 id
        dialog.name_edit.setText("新模板")
        dialog.expr_edit.setPlainText("close.rolling(8).mean()")
        new_id = dialog.save_template()
        assert new_id is not None
        assert new_id != tid
        items = db.list_indicator_templates()
        assert len(items) == 1
        assert items[0].name == "新模板"
    finally:
        dialog.close()
        db.close()


def test_save_refreshes_list_widget(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_lw.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        assert dialog.template_list.count() == 0
        dialog.name_edit.setText("T1")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        dialog.save_template()
        assert dialog.template_list.count() == 1
        assert "T1" in dialog.template_list.item(0).text()
    finally:
        dialog.close()
        db.close()


# --------------------------------------------------------------------------- #
# 4. w/m 派生
# --------------------------------------------------------------------------- #
def test_preview_weekly_derives_from_daily(tmp_path, qapp) -> None:
    db = _open(tmp_path / "qa_w.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.period_combo.setCurrentIndex(1)  # 周线
        assert dialog.current_period() == "w"
        result = dialog.preview_result()
        assert result.ok is True, result.error
        assert result.series is not None
        dt = pd.date_range("2024-01-01", periods=60, freq="D")
        expected = dt.to_period("W").nunique()
        assert len(result.series) == expected
        assert len(result.series) < 60
    finally:
        dialog.close()
        db.close()


# --------------------------------------------------------------------------- #
# 5. 重复保存语义
# --------------------------------------------------------------------------- #
def test_save_two_new_templates_creates_two(tmp_path, qapp) -> None:
    """两次「新建」（未选中任何模板）应产生两个模板，而非覆盖前一个。"""
    db = _open(tmp_path / "qa_two.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("均线A")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        dialog.category_combo.setCurrentText("均线")
        id_a = dialog.save_template()
        assert id_a is not None

        dialog.name_edit.setText("均线B")
        dialog.expr_edit.setPlainText("close.rolling(10).mean()")
        dialog.category_combo.setCurrentText("均线")
        id_b = dialog.save_template()
        assert id_b is not None

        items = db.list_indicator_templates()
        names = [t.name for t in items]
        assert len(items) == 2, f"期望 2 个模板，实际 {len(items)}：{names}"
        assert "均线A" in names and "均线B" in names
        assert id_a != id_b
    finally:
        dialog.close()
        db.close()


def test_selected_save_updates_same_id(tmp_path, qapp) -> None:
    """选中模板后改名保存 → 更新同 id（既有语义确认，非新建）。"""
    db = _open(tmp_path / "qa_upd.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("原模板")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        tid = dialog.save_template()
        assert tid is not None

        dialog.template_list.setCurrentRow(0)
        assert dialog.selected_template_id() == tid
        dialog.name_edit.setText("改名模板")
        tid2 = dialog.save_template()
        assert tid2 == tid

        items = db.list_indicator_templates()
        assert len(items) == 1
        assert items[0].name == "改名模板"
    finally:
        dialog.close()
        db.close()


def test_new_save_leaves_no_list_selection(tmp_path, qapp) -> None:
    """新建保存后：列表无高亮（selected_template_id()=None）、_template_id 保持 None。"""
    db = _open(tmp_path / "qa_nosel.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("A")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        tid = dialog.save_template()
        assert tid is not None
        assert dialog._template_id is None
        assert dialog.selected_template_id() is None
        assert dialog.template_list.currentItem() is None

        # 再新建第二个：仍无高亮、全新 id
        dialog.name_edit.setText("B")
        dialog.expr_edit.setPlainText("close.rolling(10).mean()")
        tid2 = dialog.save_template()
        assert tid2 is not None
        assert tid2 != tid
        assert dialog._template_id is None
        assert dialog.selected_template_id() is None
    finally:
        dialog.close()
        db.close()
