"""系统托盘（P4-S2）。

主窗口点 X 关闭时隐藏到托盘；托盘右键菜单「显示主窗口」「退出」，双击图标显示主窗口。
「退出」经 :attr:`quit_requested` 信号交由主窗口真正退出（以区分「窗口 X 关闭」）。
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon, QWidget

logger = logging.getLogger(__name__)


class SystemTray(QSystemTrayIcon):
    """系统托盘图标：右键菜单（显示主窗口 / 退出）+ 双击显示主窗口。"""

    quit_requested = Signal()

    def __init__(self, window: QWidget, parent: Optional[QObject] = None) -> None:
        """构造托盘并挂到主窗口。

        Args:
            window: 主窗口（用于「显示主窗口」）。
            parent: QObject 父对象（建议传主窗口，随主窗口销毁）。
        """
        super().__init__(self._build_icon(), parent)
        self._window = window
        self.setToolTip("A股行情分析")

        self._menu = QMenu()
        self._show_action = self._menu.addAction("显示主窗口")
        self._quit_action = self._menu.addAction("退出")
        self._show_action.triggered.connect(self._show_main_window)
        self._quit_action.triggered.connect(self._request_quit)
        self.setContextMenu(self._menu)

        self.activated.connect(self._on_activated)

        if QSystemTrayIcon.isSystemTrayAvailable():
            self.show()
        else:
            logger.info("当前平台无系统托盘，跳过托盘图标显示")

    @staticmethod
    def _build_icon() -> QIcon:
        """用 QStyle 标准图标兜底；无 QApplication / 图标为空时用纯色 QPixmap。"""
        app = QApplication.instance()
        if app is not None:
            try:
                icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
                if not icon.isNull():
                    return icon
            except Exception:  # noqa: BLE001 - 降级到纯色图标
                pass
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor("#e74c3c"))
        return QIcon(pixmap)

    def _show_main_window(self) -> None:
        """显示并激活主窗口。"""
        self._window.showNormal()
        self._window.raise_()
        self._window.activateWindow()

    def _request_quit(self) -> None:
        """托盘「退出」：发信号交由主窗口真正退出（区分 X 关闭）。"""
        self.quit_requested.emit()

    def _on_activated(self, reason) -> None:  # noqa: ANN001 - Qt 约定
        """双击（或平台单击）托盘图标时显示主窗口。"""
        if reason in (
            QSystemTrayIcon.ActivationReason.DoubleClick,
            QSystemTrayIcon.ActivationReason.Trigger,
        ):
            self._show_main_window()


__all__ = ["SystemTray"]
