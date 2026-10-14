"""股票代码与名称名录对话框（P6 优化：股票查找与加载）。

定位：**非模态 / 模态均可**的查找对话框，让用户：
    * 按代码或名称关键词搜索本地名录（``stock_list`` 表）；
    * 双击行 / 点「加载选中」把代码交回上层（经 :attr:`StockListDialog.symbolChosen`）；
    * 点「更新名录」请求上层在后台刷新名录（经 :attr:`StockListDialog.updateRequested`）。

架构约定：
    * 对话框**不自己开线程、不联网**：更新交由上层（主窗口）编排，
      完成后再经事件总线广播 ``DATA_UPDATE_COMPLETED``（``scope == "symbols"``）驱动本框刷新；
    * 事件可能来自后台线程：内建 :class:`_Bridge`（``QObject``）把总线回调转成 Qt 信号，
      队列投递到 UI 线程后再触碰控件；
    * 关闭时取消总线上订阅，避免悬挂回调。
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
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

from app.core.event_bus import get_bus
from app.data.store.db import Database, get_db
from app.data.update_service import EVT_COMPLETED

logger = logging.getLogger(__name__)

# 对话框标题。
WINDOW_TITLE: str = "股票代码与名称"


class _Bridge(QObject):
    """把事件总线回调（可能来自后台线程）转成 Qt 信号，交由 UI 线程处理。"""

    completed = Signal(dict)


class StockListDialog(QDialog):
    """全 A 股代码与名称名录对话框。"""

    # 用户选定某只股票代码（双击 / 「加载选中」）。
    symbolChosen = Signal(str)
    # 用户请求更新名录（本框不自己开线程、不联网）。
    updateRequested = Signal()

    def __init__(
        self,
        db: Optional[Database] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(520, 600)
        self._setup_ui()

        # 总线订阅：后台线程发布 → bridge 信号 → UI 线程槽。
        self._bridge = _Bridge(self)
        self._bridge.completed.connect(self._on_bus_completed)
        self._bus_cb = self._on_bus_event
        get_bus().subscribe(EVT_COMPLETED, self._bus_cb)

        self.refresh()

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建顶部搜索行 / 中部表格 / 底部按钮行。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # (a) 顶部：搜索 + 更新 + 计数
        top = QHBoxLayout()
        top.setSpacing(6)
        self._search_edit = QLineEdit(self)
        self._search_edit.setPlaceholderText("输入代码或名称搜索")
        top.addWidget(self._search_edit, 1)

        self._btn_update = QPushButton("更新名录", self)
        self._btn_update.clicked.connect(self._on_update_clicked)
        top.addWidget(self._btn_update)

        self._count_label = QLabel("", self)
        top.addWidget(self._count_label)
        root.addLayout(top)

        # (b) 中部：名录表格
        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["代码", "名称"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, 1)

        # (c) 底部：加载 / 关闭
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        bottom.addStretch(1)
        self._btn_load = QPushButton("加载选中", self)
        self._btn_load.clicked.connect(self._on_load_clicked)
        bottom.addWidget(self._btn_load)

        self._btn_close = QPushButton("关闭", self)
        self._btn_close.clicked.connect(self.close)
        bottom.addWidget(self._btn_close)
        root.addLayout(bottom)

        # 交互接线
        self._search_edit.textChanged.connect(lambda _text: self.refresh())
        self._table.cellDoubleClicked.connect(self._on_double_clicked)

    # ------------------------------------------------------------------ #
    # 刷新
    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """按当前搜索词读取名录并刷新表格 / 计数标签（不抛异常）。"""
        keyword = self._search_edit.text().strip() or None
        try:
            items = self._db.list_stocks(keyword)
        except Exception as exc:  # noqa: BLE001 - 读库失败不外抛
            logger.warning("读取股票名录失败：%s", exc)
            items = []
        self._table.setRowCount(0)
        for info in items:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(info.symbol))
            self._table.setItem(row, 1, QTableWidgetItem(info.name))
        if keyword:
            self._count_label.setText(f"匹配 {len(items)} 只")
        else:
            self._count_label.setText(f"共 {len(items)} 只")

    # ------------------------------------------------------------------ #
    # 选择 / 加载
    # ------------------------------------------------------------------ #
    def _selected_symbol(self) -> str:
        """返回当前选中行的代码；无选中返回空串。"""
        row = self._table.currentRow()
        if row < 0:
            return ""
        item = self._table.item(row, 0)
        return item.text().strip() if item is not None else ""

    def _commit_selection(self) -> None:
        """把选中代码经 :attr:`symbolChosen` 交回上层并关闭；无选中则提示。"""
        symbol = self._selected_symbol()
        if not symbol:
            self._count_label.setText("请先选择一行")
            return
        self.symbolChosen.emit(symbol)
        self.accept()

    def _on_double_clicked(self, _row: int, _col: int) -> None:
        """双击行：等价于「加载选中」。"""
        self._commit_selection()

    def _on_load_clicked(self) -> None:
        """「加载选中」按钮。"""
        self._commit_selection()

    # ------------------------------------------------------------------ #
    # 更新请求 / 总线刷新
    # ------------------------------------------------------------------ #
    def _on_update_clicked(self) -> None:
        """「更新名录」按钮：只发请求，实际更新由上层编排。"""
        self.updateRequested.emit()

    def _on_bus_event(self, payload: dict) -> None:
        """总线回调（可能来自后台线程）：仅转成 Qt 信号，不触碰控件。"""
        self._bridge.completed.emit(payload)

    def _on_bus_completed(self, payload: dict) -> None:
        """UI 线程槽：名录更新完成（``scope == "symbols"``）时自动刷新。"""
        if payload.get("scope") == "symbols":
            self.refresh()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭前取消总线订阅，避免悬挂回调。"""
        try:
            get_bus().unsubscribe(EVT_COMPLETED, self._bus_cb)
        except Exception as exc:  # noqa: BLE001 - 退订失败不影响关闭
            logger.debug("退订名录刷新回调失败：%s", exc)
        super().closeEvent(event)


__all__ = ["StockListDialog", "WINDOW_TITLE"]