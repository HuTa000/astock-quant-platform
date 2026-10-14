"""P3-S3 自定义指标编辑器测试（离屏 ``offscreen``，文件库）。

覆盖：
    1. 数据层 CRUD 回归：save（新增返回 id）→ list（含分类过滤）→ 更新（同 id 覆盖）→ delete；
    2. UI 离屏：对话框可构造、示例表达式预览 ok=True 且曲线非空、非法表达式预览 ok=False 且
       状态标签显示错误、保存后 list 多一条、删除后少一条；
    3. 空库预览：无本地 K 线时预览优雅降级、不抛异常。
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
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_daily(db: Database, symbol: str, n: int = 60) -> None:
    """写入 n 根日线（``adjust=""``，与编辑器 ``db.get_bars(symbol, period)`` 默认一致）。"""
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
# 1. 数据层 CRUD 回归
# --------------------------------------------------------------------------- #
def test_indicator_template_crud(tmp_path) -> None:
    db = _open(tmp_path / "crud.db")

    # 新增返回 id
    tid = db.save_indicator_template("均线20", "均线", "close.rolling(20).mean()")
    assert isinstance(tid, int) and tid > 0

    # list
    items = db.list_indicator_templates()
    assert len(items) == 1
    assert items[0].name == "均线20"
    assert items[0].category == "均线"
    assert items[0].expr == "close.rolling(20).mean()"

    # 分类过滤
    assert len(db.list_indicator_templates("均线")) == 1
    assert len(db.list_indicator_templates("趋势")) == 0

    # 更新（同 id 覆盖）
    tid2 = db.save_indicator_template(
        "均线30", "均线", "close.rolling(30).mean()", template_id=tid
    )
    assert tid2 == tid
    items = db.list_indicator_templates()
    assert len(items) == 1
    assert items[0].name == "均线30"
    assert items[0].expr == "close.rolling(30).mean()"

    # 删除
    db.delete_indicator_template(tid)
    assert db.list_indicator_templates() == []
    db.close()


# --------------------------------------------------------------------------- #
# 2. UI 离屏：预览
# --------------------------------------------------------------------------- #
def test_dialog_preview_example_ok(tmp_path, qapp) -> None:
    db = _open(tmp_path / "prev.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        assert dialog.current_expr() == "close.rolling(20).mean()"
        result = dialog.preview_result()
        assert result.ok is True, result.error
        assert result.series is not None
        assert len(result.series) == 60
    finally:
        dialog.close()
        db.close()


def test_dialog_preview_illegal_expr(tmp_path, qapp) -> None:
    db = _open(tmp_path / "bad.db")
    _seed_daily(db, "000001", n=60)
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.expr_edit.setPlainText("foo(bar)")
        result = dialog.preview_result()
        assert result.ok is False
        assert result.error_type is not None
        # 触发状态标签更新，断言可见错误
        dialog._on_preview()
        assert dialog.status_label.text()
        assert result.error_type in dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


def test_dialog_preview_empty_db_graceful(tmp_path, qapp) -> None:
    db = _open(tmp_path / "empty.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        result = dialog.preview_result()  # 不抛异常
        assert result.ok is False
        assert "无本地 K 线" in result.error
        dialog._on_preview()
        assert "无本地 K 线" in dialog.status_label.text()
    finally:
        dialog.close()
        db.close()


# --------------------------------------------------------------------------- #
# 3. UI 离屏：保存 / 载入 / 删除
# --------------------------------------------------------------------------- #
def test_dialog_save_and_delete(tmp_path, qapp) -> None:
    db = _open(tmp_path / "save.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("我的均线")
        dialog.expr_edit.setPlainText("close.rolling(10).mean()")
        dialog.category_combo.setCurrentText("均线")

        before = len(db.list_indicator_templates())
        tid = dialog.save_template()
        assert tid is not None
        assert len(db.list_indicator_templates()) == before + 1

        # 选中模板（载入编辑区）并删除
        dialog.template_list.setCurrentRow(0)
        assert dialog.selected_template_id() == tid
        assert dialog.current_name() == "我的均线"
        assert dialog.delete_template() is True
        assert len(db.list_indicator_templates()) == before
    finally:
        dialog.close()
        db.close()


def test_dialog_save_validation(tmp_path, qapp) -> None:
    """名称 / 表达式为空时不落库。"""
    db = _open(tmp_path / "validate.db")
    dialog = IndicatorEditorDialog(symbols=["000001"], db=db)
    try:
        dialog.name_edit.setText("")
        dialog.expr_edit.setPlainText("close.rolling(5).mean()")
        assert dialog.save_template() is None
        assert db.list_indicator_templates() == []

        dialog.name_edit.setText("空表达式")
        dialog.expr_edit.setPlainText("   ")
        assert dialog.save_template() is None
        assert db.list_indicator_templates() == []
    finally:
        dialog.close()
        db.close()
