"""选股对话框（P3-S5）。

定位：**模态对话框**，让用户：
    * 编辑「技术面 + 基本面」结构化条件（字段 + 操作符 + 阈值）；
    * 用 AND / OR 逻辑叠加后运行选股（:class:`~app.engine.screener.ScreenerEngine`）；
    * 查看命中结果列表，并「＋ 加入自选」；
    * 把当前条件保存为选股模板（``save_screener_template``）。

架构约定（沿用 ``indicator_editor`` / ``watchlist_panel`` 的可测性思路）：
    * **业务动作 = 无弹窗的纯方法**（:meth:`run_screen` / :meth:`add_selected_to_watchlist` /
      :meth:`save_template`），测试可直接调用并断言；
    * **UI 触达层**（运行 / 加自选 / 保存按钮）负责把结果显示到状态标签 / 表格；
    * **绝不向 UI 抛异常**：引擎 / DB / 加自选异常一律捕获，状态标签显示可读原因；
    * **后台线程**：真实运行时经 :class:`_ScreenWorker`（``QThread``）在后台执行选股，
      结果经 Qt 信号回 UI 线程刷新，**工作线程不碰任何 UI**。

关键契约：
    * 构造函数接收 ``db`` / ``screener`` / ``watchlist_panel`` / ``add_symbol`` /
      ``name_resolver`` 依赖注入，默认取全局单例；
    * 只读访问器 :meth:`current_conditions` / :meth:`current_logic` / :meth:`last_result`
      供测试断言；
    * 条件收集时：字段空 / 操作符非法 / 阈值空或非数值 → 跳过该行（不崩溃），运行前提示。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.store.db import Database, get_db
from app.engine.screener import (
    Condition,
    ScreenResult,
    ScreenerEngine,
    get_screener,
)
from app.export.excel_exporter import ExcelExportError, export_screen_result
from app.ui.themes import colors
from app.ui.widgets.watchlist_panel import WatchlistPanel

logger = logging.getLogger(__name__)

# 对话框标题。
WINDOW_TITLE: str = "选股"

# 预置基本面字段（中文显示名 → 引擎字段名）。
_FIELD_PRESETS: tuple[tuple[str, str], ...] = (
    ("市盈率 PE", "pe"),
    ("市净率 PB", "pb"),
    ("净资产收益率 ROE", "roe"),
    ("营收同比增长", "revenue_yoy"),
    ("总市值", "total_market_cap"),
    ("流通市值", "float_market_cap"),
)

# 中文显示名 → 字段名（用于把用户选中的预置项还原为引擎字段名）。
_LABEL_TO_FIELD: dict[str, str] = {label: code for label, code in _FIELD_PRESETS}

# 固定操作符。
_OPS: tuple[str, ...] = (">", ">=", "<", "<=", "==", "!=")

# 逻辑叠加方式。
_LOGIC_AND: str = "AND"
_LOGIC_OR: str = "OR"


@dataclass
class _ConditionRow:
    """一条条件行的控件集合。"""

    field_combo: QComboBox
    op_combo: QComboBox
    value_edit: QLineEdit
    layout: QHBoxLayout
    remove_button: QPushButton


class _ScreenWorker(QThread):
    """在后台线程执行一次 :meth:`ScreenerEngine.run`（结果经信号回传，不碰 UI）。"""

    done = Signal(object)  # ScreenResult 或 Exception

    def __init__(
        self,
        screener: ScreenerEngine,
        conditions: list[Condition],
        logic: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造后台选股线程。

        Args:
            screener: 选股引擎。
            conditions: 已收集的合法条件列表。
            logic: 逻辑叠加方式（``AND`` / ``OR``）。
            parent: 父对象。
        """
        super().__init__(parent)
        self._screener = screener
        self._conditions = conditions
        self._logic = logic

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：执行选股，成功 / 失败均经 ``done`` 信号回传。"""
        try:
            result = self._screener.run(self._conditions, logic=self._logic)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台选股线程异常")
            self.done.emit(exc)
            return
        self.done.emit(result)


class ScreenerDialog(QDialog):
    """选股对话框。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        screener: Optional[ScreenerEngine] = None,
        watchlist_panel: Optional[WatchlistPanel] = None,
        add_symbol: Optional[Callable[[str], bool]] = None,
        name_resolver: Optional[Callable[[str], Optional[str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`（用于保存选股模板）。
            screener: 选股引擎；默认 :func:`get_screener`。
            watchlist_panel: 自选股面板（用于「＋ 加入自选」）；可改用 ``add_symbol`` 回调注入。
            add_symbol: 「代码 → 是否成功」回调；优先于 ``watchlist_panel``。
            name_resolver: 「代码 → 名称」回调；取不到名时结果表格显示代码。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._screener: ScreenerEngine = screener if screener is not None else get_screener()
        self._watchlist_panel: Optional[WatchlistPanel] = watchlist_panel
        self._add_symbol_cb: Optional[Callable[[str], bool]] = add_symbol
        self._name_resolver: Optional[Callable[[str], Optional[str]]] = name_resolver

        # 动态条件行（每行 = 字段 + 操作符 + 阈值 + 删除）。
        self.condition_rows: list[_ConditionRow] = []
        self._last_result: Optional[ScreenResult] = None
        self._last_conditions: list[Condition] = []
        self._worker: Optional[_ScreenWorker] = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(720, 660)
        self._setup_ui()
        self.add_condition_row()
        self._set_status("点击『运行选股』开始")

    # ------------------------------------------------------------------ #
    # 线程生命周期（D1 修复）
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭前等待后台选股线程自然结束，避免对话框析构连带销毁运行中的 QThread。

        背景（D1）：``_ScreenWorker`` 已把 ``parent`` 置为 ``None``（见 :meth:`_on_run`），
        故本对话框析构不会连带销毁线程对象；但若对话框在 worker 仍运行时被 ``deleteLater``
        销毁，仍会因「销毁运行中的 QThread」触发 Qt 文档明确的未定义行为（崩溃）。
        这里在关闭时 ``wait()`` 让线程体 ``run()`` 先结束，再允许关闭，彻底消除时序风险。

        选股为有限次数的实时抓取（数据源各自带超时），UI 线程短暂阻塞可接受。
        不在此处断开 ``done`` / ``finished`` 信号：Qt 会安全丢弃已删除对象的排队事件，
        而保留连接能让**已排队的在途结果**在事件循环恢复后正常回填 UI（与 QA 验收用例
        ``test_close_while_worker_running_no_crash`` 的「结果仍回 UI」断言一致）。
        """
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait()
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建逻辑区 / 条件区 / 运行区 / 结果表格 / 加自选 / 保存模板区。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # (a) 逻辑切换
        logic_bar = QHBoxLayout()
        logic_bar.setSpacing(6)
        logic_bar.addWidget(QLabel("逻辑：", self))
        self.logic_combo = QComboBox(self)
        self.logic_combo.addItem("全部满足 (AND)", _LOGIC_AND)
        self.logic_combo.addItem("任一满足 (OR)", _LOGIC_OR)
        logic_bar.addWidget(self.logic_combo)
        logic_bar.addStretch(1)
        root.addLayout(logic_bar)

        # (b) 条件编辑区
        root.addWidget(QLabel("选股条件：", self))
        self._conditions_container = QWidget(self)
        self.conditions_layout = QVBoxLayout(self._conditions_container)
        self.conditions_layout.setContentsMargins(0, 0, 0, 0)
        self.conditions_layout.setSpacing(4)
        root.addWidget(self._conditions_container)

        self.add_condition_button = QPushButton("＋ 添加条件", self)
        self.add_condition_button.clicked.connect(self._on_add_condition)
        root.addWidget(self.add_condition_button)

        # (c) 运行区
        self.run_button = QPushButton("运行选股", self)
        self.run_button.clicked.connect(self._on_run)
        root.addWidget(self.run_button)

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        # (d) 结果表格
        self.result_table = QTableWidget(0, 2, self)
        self.result_table.setHorizontalHeaderLabels(["代码", "名称"])
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.result_table, 1)

        # (e) 加自选 + 导出 Excel + 保存模板
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.add_watchlist_button = QPushButton("＋ 加入自选", self)
        self.add_watchlist_button.clicked.connect(self._on_add_watchlist)
        bottom.addWidget(self.add_watchlist_button)

        self.export_button = QPushButton("导出 Excel", self)
        self.export_button.clicked.connect(self._on_export)
        bottom.addWidget(self.export_button)

        bottom.addWidget(QLabel("模板名：", self))
        self.template_name_edit = QLineEdit(self)
        self.template_name_edit.setPlaceholderText("如：低估值高成长")
        bottom.addWidget(self.template_name_edit, 1)
        self.save_button = QPushButton("保存模板", self)
        self.save_button.clicked.connect(self._on_save_template)
        bottom.addWidget(self.save_button)
        root.addLayout(bottom)

    # ------------------------------------------------------------------ #
    # 条件行管理
    # ------------------------------------------------------------------ #
    def add_condition_row(self) -> _ConditionRow:
        """在条件区新增一行（字段 / 操作符 / 阈值 / 删除），返回该行。"""
        field_combo = QComboBox(self)
        field_combo.setEditable(True)
        for label, code in _FIELD_PRESETS:
            field_combo.addItem(label, code)

        op_combo = QComboBox(self)
        for op in _OPS:
            op_combo.addItem(op)

        value_edit = QLineEdit(self)
        value_edit.setPlaceholderText("阈值（数值）")

        remove_button = QPushButton("删除", self)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(4)
        row_layout.addWidget(field_combo, 3)
        row_layout.addWidget(op_combo, 1)
        row_layout.addWidget(value_edit, 1)
        row_layout.addWidget(remove_button)

        row = _ConditionRow(
            field_combo=field_combo,
            op_combo=op_combo,
            value_edit=value_edit,
            layout=row_layout,
            remove_button=remove_button,
        )
        self.condition_rows.append(row)
        self.conditions_layout.addLayout(row_layout)
        remove_button.clicked.connect(lambda _=False, r=row: self._remove_condition_row(r))
        return row

    def _remove_condition_row(self, row: _ConditionRow) -> None:
        """删除一条条件行（**至少保留一行**）。"""
        if len(self.condition_rows) <= 1:
            self._set_status("至少保留一个条件行", ok=False)
            return
        for widget in (row.field_combo, row.op_combo, row.value_edit, row.remove_button):
            row.layout.removeWidget(widget)
            widget.deleteLater()
        self.conditions_layout.removeItem(row.layout)
        self.condition_rows.remove(row)

    def _on_add_condition(self) -> None:
        """「＋ 添加条件」按钮。"""
        self.add_condition_row()

    # ------------------------------------------------------------------ #
    # 只读访问器（供测试 / 上层）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _field_text(field_combo: QComboBox) -> str:
        """取字段名：预置项（如「市盈率 PE」）还原为引擎字段名，自由输入原样返回。

        说明：editable 的 ``QComboBox`` 在用户自由输入（如 ``close``）时 ``currentText()``
        即输入文本、不命中任何预置项；选中预置项时 ``currentText()`` 为中文显示名，
        经 :data:`_LABEL_TO_FIELD` 映射回 ``pe`` 等字段名。
        """
        text = field_combo.currentText().strip()
        return _LABEL_TO_FIELD.get(text, text)

    def current_conditions(self) -> list[Condition]:
        """收集当前全部条件（**跳过**空字段 / 空阈值 / 非数值阈值 / 非法操作符的行）。"""
        conditions: list[Condition] = []
        for row in self.condition_rows:
            field = self._field_text(row.field_combo)
            op = row.op_combo.currentText().strip()
            value_text = row.value_edit.text().strip()
            if not field or not value_text:
                continue
            if op not in _OPS:
                continue
            try:
                value = float(value_text)
            except (TypeError, ValueError):
                continue
            conditions.append(Condition(field=field, op=op, value=value))
        return conditions

    def current_logic(self) -> str:
        """返回当前逻辑叠加方式（``AND`` / ``OR``）。"""
        data = self.logic_combo.currentData()
        return str(data) if data else _LOGIC_AND

    def last_result(self) -> Optional[ScreenResult]:
        """返回最近一次选股结果；尚未运行返回 ``None``。"""
        return self._last_result

    def selected_symbols(self) -> list[str]:
        """返回结果表格当前选中行的代码列表（按行序）。"""
        selection = self.result_table.selectionModel()
        rows = {index.row() for index in selection.selectedRows()}
        symbols: list[str] = []
        for row in sorted(rows):
            item = self.result_table.item(row, 0)
            if item is not None and item.text().strip():
                symbols.append(item.text().strip())
        return symbols

    # ------------------------------------------------------------------ #
    # 运行选股（同步纯方法 + 后台线程触达层）
    # ------------------------------------------------------------------ #
    def run_screen(self) -> Optional[ScreenResult]:
        """同步执行选股并把结果刷新到表格 / 状态标签（**不抛异常**）。

        供测试直接调用（注入 fake screener 时无网络、无线程）。空条件时提示并返回 ``None``。
        """
        conditions = self.current_conditions()
        if not conditions:
            self._last_result = None
            self._set_status("没有有效条件，请先添加", ok=False)
            return None
        logic = self.current_logic()
        try:
            result = self._screener.run(conditions, logic=logic)
        except Exception as exc:  # noqa: BLE001 - 引擎异常不外抛
            logger.exception("选股引擎执行失败")
            self._last_result = None
            self._last_conditions = []
            self._set_status(f"选股失败：{exc}", ok=False)
            return None
        self._last_result = result
        self._last_conditions = list(conditions)
        self._apply_result(result)
        return result

    def _apply_result(self, result: ScreenResult) -> None:
        """把选股结果填充到表格并更新状态标签（UI 线程调用）。"""
        self.result_table.setRowCount(0)
        for symbol in result.hits:
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            code = str(symbol)
            name = self._resolve_name(code) or code
            self.result_table.setItem(row, 0, QTableWidgetItem(code))
            self.result_table.setItem(row, 1, QTableWidgetItem(name))
        self._set_status(
            f"命中 {len(result.hits)} 只 / 候选 {result.total} 只 / 失败 {len(result.failed)} 只",
            ok=True,
        )

    def _resolve_name(self, symbol: str) -> Optional[str]:
        """尽力解析代码 → 名称（取不到返回 ``None``）。"""
        if self._name_resolver is None:
            return None
        try:
            return self._name_resolver(symbol)
        except Exception as exc:  # noqa: BLE001 - 名称解析失败不影响展示
            logger.debug("name_resolver(%s) 失败：%s", symbol, exc)
            return None

    def _on_run(self) -> None:
        """「运行选股」按钮：在后台线程执行，避免基本面实时抓取阻塞 UI。"""
        conditions = self.current_conditions()
        if not conditions:
            self._last_result = None
            self.result_table.setRowCount(0)
            self._set_status("没有有效条件，请先添加", ok=False)
            return
        if self._worker is not None and self._worker.isRunning():
            logger.info("选股已在进行中，忽略重复触发")
            return
        self.run_button.setEnabled(False)
        self._set_status("正在选股…", ok=True)
        # 记录本次运行使用的条件（导出 Excel 时按「实际运行的条件」生成列头）。
        self._last_conditions = list(conditions)
        # worker 的 parent 置为 None（D1）：对话框析构不连带销毁运行中的 QThread，
        # 生命周期统一交由 closeEvent 的 wait() + 本类的 _on_worker_finished 管理。
        self._worker = _ScreenWorker(self._screener, conditions, self.current_logic())
        self._worker.done.connect(self._on_screen_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_screen_finished(self, payload: object) -> None:
        """后台选股结束：恢复按钮并把结果刷新到 UI（本槽运行在 UI 线程）。"""
        self.run_button.setEnabled(True)
        if isinstance(payload, ScreenResult):
            self._last_result = payload
            self._apply_result(payload)
        else:
            self._last_result = None
            self._last_conditions = []
            self._set_status(f"选股失败：{payload}", ok=False)

    def _on_worker_finished(self) -> None:
        """线程结束后释放引用。"""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------ #
    # 加自选（无弹窗纯方法 + UI 触达层）
    # ------------------------------------------------------------------ #
    def _add_symbol(self, symbol: str) -> bool:
        """把单个代码加入自选（优先 ``add_symbol`` 回调，其次 ``watchlist_panel``）。"""
        if self._add_symbol_cb is not None:
            try:
                return bool(self._add_symbol_cb(symbol))
            except Exception as exc:  # noqa: BLE001
                logger.warning("加自选回调失败：%s -> %s", symbol, exc)
                return False
        if self._watchlist_panel is not None:
            try:
                return bool(self._watchlist_panel.add_symbol(symbol))
            except Exception as exc:  # noqa: BLE001
                logger.warning("加自选失败：%s -> %s", symbol, exc)
                return False
        return False

    def add_selected_to_watchlist(self) -> list[str]:
        """把结果表格**选中的行**加入自选，返回成功添加的代码列表。"""
        added: list[str] = []
        for symbol in self.selected_symbols():
            if self._add_symbol(symbol):
                added.append(symbol)
        return added

    def _on_add_watchlist(self) -> None:
        """「＋ 加入自选」按钮：区分成功 / 失败项，失败项显式列出（D3）。"""
        selected = self.selected_symbols()
        added = self.add_selected_to_watchlist()
        failed = [symbol for symbol in selected if symbol not in added]
        if not selected:
            self._set_status("未选中结果或加入失败", ok=False)
        elif failed:
            self._set_status(
                f"成功 {len(added)} 项、失败 {len(failed)} 项：{'、'.join(failed)}",
                ok=False,
            )
        else:
            self._set_status(f"已加入自选：{', '.join(added)}", ok=True)

    # ------------------------------------------------------------------ #
    # 保存模板（无弹窗纯方法 + UI 触达层）
    # ------------------------------------------------------------------ #
    def save_template(self) -> Optional[int]:
        """把当前条件保存为选股模板（名称 / 条件非空校验），返回模板 id；失败返回 ``None``。"""
        name = self.template_name_edit.text().strip()
        conditions = self.current_conditions()
        if not name:
            self._set_status("模板名称不能为空", ok=False)
            return None
        if not conditions:
            self._set_status("没有有效条件可保存", ok=False)
            return None
        try:
            conditions_json = json.dumps(
                [cond.to_dict() for cond in conditions], ensure_ascii=False
            )
            template_id = int(self._db.save_screener_template(name, conditions_json))
        except Exception as exc:  # noqa: BLE001 - 落库失败不外抛
            logger.exception("保存选股模板失败")
            self._set_status(f"保存失败：{exc}", ok=False)
            return None
        self._set_status(f"已保存模板：{name}", ok=True)
        return template_id

    def _on_save_template(self) -> None:
        """「保存模板」按钮。"""
        self.save_template()

    # ------------------------------------------------------------------ #
    # 导出 Excel（薄 UI 触达层，核心走可注入的纯函数）
    # ------------------------------------------------------------------ #
    def _on_export(self) -> None:
        """「导出 Excel」按钮：选路径后调用 :func:`export_screen_result` 写盘。

        设计要点：
            * 「选路径」与「导出写盘」分离——文件对话框仅在本触达层出现，导出核心
              :func:`export_screen_result` 为纯函数，测试可 monkeypatch 文件对话框；
            * 无结果 / 无命中 → 状态标签友好提示，不弹文件对话框；
            * 导出成功 / 失败均回状态标签（含人话原因），不向 UI 抛异常。
        """
        result = self._last_result
        if result is None or not result.hits:
            self._set_status("请先运行选股", ok=False)
            return

        default_name = f"选股结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "导出选股结果", default_name, "Excel 文件 (*.xlsx)"
        )
        if not path:
            return  # 用户取消

        try:
            saved = export_screen_result(
                result,
                path,
                name_resolver=self._resolve_name,
                conditions=self._last_conditions,
            )
        except ExcelExportError as exc:
            self._set_status(f"导出失败：{exc}", ok=False)
            return
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不外抛
            logger.exception("导出选股结果失败")
            self._set_status(f"导出失败：{exc}", ok=False)
            return
        self._set_status(f"已导出到 {saved}", ok=True)

    # ------------------------------------------------------------------ #
    # 状态标签
    # ------------------------------------------------------------------ #
    def _set_status(self, text: str, ok: bool = True) -> None:
        """更新状态标签（成功绿色 / 失败红色，沿用主题红涨绿跌但语义为状态）。"""
        color = colors.DOWN_COLOR if ok else colors.UP_COLOR
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)


__all__ = ["ScreenerDialog", "WINDOW_TITLE"]
