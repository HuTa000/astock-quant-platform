"""快捷键说明对话框 —— P6-S2。

展示 :class:`~app.ui.shortcuts.ShortcutManager.shortcuts` 返回的
``[(键, 说明), ...]`` 清单（共 10 条：8 周期 + ↑/↓）。

纯展示组件；内容由构造参数注入（便于离屏测试，不依赖真实主窗口）。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

# 表头。
_KEY_HEADER: str = "按键"
_DESC_HEADER: str = "功能"


class ShortcutHelpDialog(QDialog):
    """快捷键说明对话框。"""

    def __init__(
        self,
        shortcuts: List[Tuple[str, str]],
        parent: Optional[object] = None,  # noqa: ANN401 - 兼容任意 QWidget 父对象
    ) -> None:
        """构造对话框。

        Args:
            shortcuts: ``[(键, 说明), ...]`` 清单（来自 ShortcutManager.shortcuts()）。
            parent: 父控件。
        """
        super().__init__(parent)  # type: ignore[arg-type]
        self.setWindowTitle("快捷键说明")
        self.resize(360, 320)
        layout = QVBoxLayout(self)
        table = QTableWidget(len(shortcuts), 2, self)
        table.setHorizontalHeaderLabels([_KEY_HEADER, _DESC_HEADER])
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        for row, (key, desc) in enumerate(shortcuts):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(str(desc)))
        layout.addWidget(table)


__all__ = ["ShortcutHelpDialog"]