"""预警管理对话框 —— P6-S2。

职责（P4-S1 引擎已有，本步补管理 UI）：
    * 按标的展示全部预警规则（表格：类型 / 参数 / 启用状态）；
    * 新增规则（四种 kind：均线金叉死叉 / MACD 金叉 / 放量 / 自定义表达式）；
    * 删除规则、切换启用 / 禁用。

复用底层：
    * :class:`AlertService`（``add_rule`` 新增，参数校验走引擎）；
    * ``db.list_alert_rules(symbol)`` / ``db.delete_alert_rule(id)`` /
      ``db.set_alert_rule_enabled(id, enabled)``（读写直接走存储层）。

⚠️ UI 层约定（§4.11）：带参槽一律 ``lambda`` 吞掉 Qt 隐式参数（如 ``cellClicked(int,int)``）；
新增规则需在 :meth:`save_rule` 内 try/except 捕获 :class:`ValueError`（引擎参数校验错误）
并以模态提示透出。
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.store.db import Database, get_db
from app.engine.alert import (
    KIND_EXPR,
    KIND_MACD_CROSS,
    KIND_MA_CROSS,
    KIND_VOLUME_SURGE,
    SUPPORTED_KINDS,
)

logger = logging.getLogger(__name__)

# kind → 中文名。
KIND_LABELS: dict[str, str] = {
    KIND_MA_CROSS: "均线交叉",
    KIND_MACD_CROSS: "MACD 金叉",
    KIND_VOLUME_SURGE: "放量",
    KIND_EXPR: "自定义表达式",
}

# 表头（顺序稳定，供测试）。
_COLUMNS: tuple[str, str, str] = ("类型", "参数", "启用")


class AlertManagerDialog(QDialog):
    """单标的预警规则管理对话框。"""

    def __init__(
        self,
        symbol: str,
        db: Optional[Database] = None,
        add_rule: Optional[Callable[[str, str, dict], int]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            symbol: 标的代码。
            db: 存储实例；默认 :func:`get_db`（读 / 删除 / 启停规则）。
            add_rule: 新增规则回调（``(symbol, kind, params) -> rule_id``）；
                ``None`` 时默认用全局 ``AlertService.add_rule``。
            parent: 父控件。
        """
        super().__init__(parent)
        self._symbol = str(symbol)
        self._db: Database = db if db is not None else get_db()
        if add_rule is None:
            from app.engine.alert import get_alert_service

            add_rule = get_alert_service().add_rule
        self._add_rule_cb: Callable[[str, str, dict], int] = add_rule
        self._rules_loaded: list = []

        self.setWindowTitle(f"预警管理 - {self._symbol}")
        self.resize(560, 420)
        self._setup_ui()
        self._load_rules()

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """布局：新增区（顶部）+ 规则表格（中央）+ 操作按钮（底部）。"""
        layout = QVBoxLayout(self)

        # -- 新增规则区 -- #
        add_box = QVBoxLayout()
        add_box.addWidget(QLabel("新增预警规则：", self))

        kind_row = QHBoxLayout()
        self._kind_combo = QComboBox(self)
        for kind in SUPPORTED_KINDS:
            self._kind_combo.addItem(KIND_LABELS.get(kind, kind), kind)
        self._kind_combo.currentIndexChanged.connect(lambda _i: self._on_kind_changed())
        kind_row.addWidget(QLabel("类型：", self))
        kind_row.addWidget(self._kind_combo, 1)
        add_box.addLayout(kind_row)

        # 参数表单（字段随 kind 切换）。
        self._form = QFormLayout()
        self._param_widgets: dict[str, QWidget] = {}
        self._build_param_form()
        add_box.addLayout(self._form)
        layout.addLayout(add_box)

        self._btn_add = QPushButton("添加规则", self)
        self._btn_add.clicked.connect(lambda: self.save_rule())
        layout.addWidget(self._btn_add)

        # -- 规则表格 -- #
        self._table = QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self._table, 1)

        # -- 底部操作 -- #
        buttons = QHBoxLayout()
        self._btn_delete = QPushButton("删除选中", self)
        self._btn_delete.clicked.connect(lambda: self.delete_selected())
        self._btn_toggle = QPushButton("启用/禁用", self)
        self._btn_toggle.clicked.connect(lambda: self.toggle_selected())
        self._btn_close = QPushButton("关闭", self)
        self._btn_close.clicked.connect(lambda: self.close())
        buttons.addWidget(self._btn_delete)
        buttons.addWidget(self._btn_toggle)
        buttons.addStretch(1)
        buttons.addWidget(self._btn_close)
        layout.addLayout(buttons)

    def _build_param_form(self) -> None:
        """按当前 kind 重建参数输入行（每次切类型时调用）。"""
        # 清旧行。
        while self._form.rowCount() > 0:
            self._form.removeRow(0)
        self._param_widgets.clear()

        kind = str(self._kind_combo.currentData())
        if kind == KIND_MA_CROSS:
            self._param_widgets["fast"] = self._make_int_spin(5, "fast")
            self._param_widgets["slow"] = self._make_int_spin(20, "slow")
            self._param_widgets["direction"] = QComboBox(self)
            self._param_widgets["direction"].addItem("金叉", "cross_up")
            self._param_widgets["direction"].addItem("死叉", "cross_down")
            self._form.addRow("快线：", self._param_widgets["fast"])
            self._form.addRow("慢线：", self._param_widgets["slow"])
            self._form.addRow("方向：", self._param_widgets["direction"])
        elif kind == KIND_MACD_CROSS:
            self._param_widgets["fast"] = self._make_int_spin(12, "fast")
            self._param_widgets["slow"] = self._make_int_spin(26, "slow")
            self._param_widgets["signal"] = self._make_int_spin(9, "signal")
            self._form.addRow("快线：", self._param_widgets["fast"])
            self._form.addRow("慢线：", self._param_widgets["slow"])
            self._form.addRow("信号：", self._param_widgets["signal"])
        elif kind == KIND_VOLUME_SURGE:
            double = QDoubleSpinBox(self)
            double.setRange(0.1, 1000.0)
            double.setDecimals(1)
            double.setValue(2.0)
            self._param_widgets["multiple"] = double
            self._param_widgets["window"] = self._make_int_spin(5, "window")
            self._form.addRow("倍数：", self._param_widgets["multiple"])
            self._form.addRow("窗口：", self._param_widgets["window"])
        else:  # expr
            edit = QLineEdit(self)
            edit.setPlaceholderText('如 close > MA(close, 20)')
            self._param_widgets["expr"] = edit
            self._form.addRow("表达式：", self._param_widgets["expr"])

    @staticmethod
    def _make_int_spin(default: int, suffix: str = "") -> QSpinBox:
        """构造整数参数输入框。"""
        spin = QSpinBox()
        spin.setRange(1, 1000)
        spin.setValue(default)
        return spin

    def _on_kind_changed(self) -> None:
        """切换规则类型 → 重建参数表单。"""
        self._build_param_form()

    # ------------------------------------------------------------------ #
    # 规则 CRUD
    # ------------------------------------------------------------------ #
    def _read_params(self, kind: str) -> dict:
        """从当前参数表单收集 params dict（未按引擎语义校验）。"""
        widgets = self._param_widgets
        if kind == KIND_MA_CROSS:
            return {
                "fast": int(widgets["fast"].value()),
                "slow": int(widgets["slow"].value()),
                "direction": str(widgets["direction"].currentData()),
            }
        if kind == KIND_MACD_CROSS:
            return {
                "fast": int(widgets["fast"].value()),
                "slow": int(widgets["slow"].value()),
                "signal": int(widgets["signal"].value()),
            }
        if kind == KIND_VOLUME_SURGE:
            return {
                "multiple": float(widgets["multiple"].value()),
                "window": int(widgets["window"].value()),
            }
        return {"expr": str(widgets["expr"].text()).strip()}

    def save_rule(self) -> bool:
        """从表单新增一条规则（引擎校验失败 → 模态提示并返回 False）。"""
        kind = str(self._kind_combo.currentData())
        params = self._read_params(kind)
        try:
            self._add_rule_cb(self._symbol, kind, params)
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - 写库异常给可见反馈
            logger.exception("新增预警规则失败：%s", exc)
            QMessageBox.warning(self, "添加失败", f"新增预警失败：{exc}")
            return False
        self._load_rules()
        return True

    def _load_rules(self) -> None:
        """从库中读取该 symbol 的全部规则并刷新表格。"""
        try:
            rules = list(self._db.list_alert_rules(symbol=self._symbol))
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取预警规则失败：%s", exc)
            rules = []
        self._rules_loaded = rules
        self._table.setRowCount(0)
        for rule in rules:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(
                row, 0, QTableWidgetItem(KIND_LABELS.get(str(rule.kind), str(rule.kind)))
            )
            try:
                params = json.loads(rule.params_json or "{}")
            except (TypeError, ValueError):
                params = {}
            self._table.setItem(row, 1, QTableWidgetItem(json.dumps(params, ensure_ascii=False)))
            self._table.setItem(row, 2, QTableWidgetItem("启用" if rule.enabled else "禁用"))

    def rule_count(self) -> int:
        """当前规则数量（供测试）。"""
        return int(len(self._rules_loaded))

    def _selected_rule_id(self) -> Optional[int]:
        """当前选中行的规则 id；无选中返回 ``None``。"""
        row = self._table.currentRow()
        if row < 0 or row >= len(self._rules_loaded):
            return None
        return int(getattr(self._rules_loaded[row], "id"))

    def delete_selected(self) -> bool:
        """删除选中规则。"""
        rule_id = self._selected_rule_id()
        if rule_id is None:
            return False
        try:
            self._db.delete_alert_rule(rule_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("删除预警规则失败：%s", exc)
            QMessageBox.warning(self, "删除失败", f"删除失败：{exc}")
            return False
        self._load_rules()
        return True

    def toggle_selected(self) -> bool:
        """切换选中规则的启用 / 禁用状态。"""
        row = self._table.currentRow()
        rule_id = self._selected_rule_id()
        if rule_id is None:
            return False
        current = bool(getattr(self._rules_loaded[row], "enabled"))
        try:
            self._db.set_alert_rule_enabled(rule_id, not current)
        except Exception as exc:  # noqa: BLE001
            logger.exception("切换预警启用失败：%s", exc)
            QMessageBox.warning(self, "操作失败", f"切换失败：{exc}")
            return False
        self._load_rules()
        return True


__all__ = ["AlertManagerDialog", "KIND_LABELS"]