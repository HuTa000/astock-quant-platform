"""顶部主工具栏。

P2-S1：提供「更新数据」动作与「周期」下拉框，暴露信号供主窗口接驳；
本步不下接 K 线图（周期切换与图表重绘属 P2-S2/S5）。
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QComboBox, QLabel, QToolBar, QWidget

logger = logging.getLogger(__name__)

# 周期下拉项：(显示文案, 周期编码)。编码与全局 Bar/DataSource 约定一致。
PERIOD_ITEMS: tuple[tuple[str, str], ...] = (
    ("1分", "1m"),
    ("5分", "5m"),
    ("15分", "15m"),
    ("30分", "30m"),
    ("60分", "60m"),
    ("日", "d"),
    ("周", "w"),
    ("月", "m"),
)

# 默认周期：日线。
DEFAULT_PERIOD: str = "d"

# 周期快捷键提示（由 PERIOD_ITEMS 顺序派生，保证与下拉一致）。
PERIOD_SHORTCUT_HINT: str = ", ".join(
    f"{index + 1}={text}" for index, (text, _) in enumerate(PERIOD_ITEMS)
)


class MainToolBar(QToolBar):
    """主工具栏（更新数据 + 周期切换）。"""

    #: 点击「更新数据」时发出。
    updateRequested = Signal()
    #: 周期切换时发出，携带周期编码（如 ``"d"`` / ``"5m"``）。
    periodChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """构造工具栏。

        Args:
            parent: 父控件（通常为主窗口）。
        """
        super().__init__("主工具栏", parent)
        self.setObjectName("mainToolBar")
        self.setMovable(False)

        self._action_update = QAction("更新数据", self)
        self._action_update.setToolTip("从数据源拉取自选股日线并落库（后台执行）")
        # 用 lambda 适配 triggered(bool) -> updateRequested()。
        self._action_update.triggered.connect(lambda: self.updateRequested.emit())
        self.addAction(self._action_update)

        self.addSeparator()
        label = QLabel("周期", self)
        label.setToolTip(f"周期切换（快捷键 1-8）：{PERIOD_SHORTCUT_HINT}")
        self.addWidget(label)

        self._period_combo = QComboBox(self)
        for text, code in PERIOD_ITEMS:
            self._period_combo.addItem(text, code)
        self._period_combo.setCurrentIndex(self._period_index(DEFAULT_PERIOD))
        self._period_combo.setToolTip(f"周期（快捷键 1-8）：{PERIOD_SHORTCUT_HINT}")
        self._period_combo.currentIndexChanged.connect(self._on_period_index_changed)
        self.addWidget(self._period_combo)

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _period_index(code: str) -> int:
        """返回周期编码对应的下拉索引；未找到返回 0。"""
        for idx, (_, item_code) in enumerate(PERIOD_ITEMS):
            if item_code == code:
                return idx
        return 0

    def _on_period_index_changed(self, index: int) -> None:
        """下拉切换时发出 ``periodChanged``。"""
        code = self._period_combo.itemData(index)
        if code:
            logger.info("周期切换：%s", code)
            self.periodChanged.emit(str(code))

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    @property
    def action_update(self) -> QAction:
        """「更新数据」动作（供测试/主窗口访问）。"""
        return self._action_update

    @property
    def period_combo(self) -> QComboBox:
        """周期下拉框。"""
        return self._period_combo

    def current_period(self) -> str:
        """当前选中的周期编码。"""
        data = self._period_combo.currentData()
        return str(data) if data else DEFAULT_PERIOD

    def set_period(self, code: str) -> None:
        """以代码设置当前周期（会触发 ``periodChanged``）。"""
        self._period_combo.setCurrentIndex(self._period_index(code))
