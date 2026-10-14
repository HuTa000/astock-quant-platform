"""全局异常兜底（P6-S3）。

职责（《细化设计》§4「P6-S3」验收：异常有友好提示、不崩）：
    * :func:`install_excepthook` 覆写 ``sys.excepthook`` —— 未捕获异常：
        1. 记录完整 traceback 到 ``data/app.log``（诊断不丢失）；
        2. 若已有实际 ``QApplication`` 实例，经 ``QTimer.singleShot`` 弹一次性
           非模态 ``QMessageBox.critical``（友好提示 + 提示日志位置）；
        3. 不 ``os._exit``：UI 线程未捕获异常后继续运行可能会导致状态不一致，
           故默认在提示后保持进程存活（异常已记录，用户可继续或手动退出）。
    * :func:`install_qt_message_guard` 覆写 Qt 原生 ``messageHandler`` —— 对
      ``QtCriticalMsg``/``QtFatalMsg`` 级别以上也转入 Python logging，避免
      Qt 侧错误静默（仅补日志，不劫持崩溃语义）。

⚠️ 边界：本模块**不 import 任何 Qt 严格必需之外的东西**；``QApplication.instance()``
为 ``None``（headless/测试）时退化为纯日志，不弹窗（规避离屏模态阻塞挂死——参见
P5-S4 调试教训）。
"""
from __future__ import annotations

import logging
import sys
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

# 已安装标记（幂等）。
_hook_installed: bool = False
_qt_guard_installed: bool = False


def _show_critical(message: str) -> None:
    """有 QApplication 时非模态弹出错误框（无则仅日志）。"""
    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        if app is None:
            return

        def _dialog() -> None:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("程序异常")
            box.setText("发生未捕获异常，请查看日志：")
            box.setInformativeText(message)
            box.setDetailedText(message)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            box.show()
            box.raise_()

        if QThread_is_main(app):
            _dialog()
        else:
            QTimer.singleShot(0, _dialog)
    except Exception as exc:  # noqa: BLE001 - 兜底自身不能崩
        logger.exception("异常提示弹窗失败：%s", exc)


def QThread_is_main(app) -> bool:  # noqa: ANN001 - 传入 QApplication 实例
    """当前是否运行在 ``app`` 的主线程。"""
    from PySide6.QtCore import QThread

    return QThread.currentThread() is app.thread()


def _default_hook(type_, value, tb) -> None:  # noqa: ANN001 - 与 sys.excepthook 一致
    """兜底处理器：完整日志 + 友好弹窗（不退出进程）。"""
    detail = "".join(traceback.format_exception(type_, value, tb))
    logger.critical("未捕获异常：\n%s", detail)
    _show_critical(f"{type_.__name__}: {value}\n\n详见 data/app.log")


def install_excepthook() -> None:
    """安装全局 ``sys.excepthook``（幂等）。

    注意：只覆盖 Python 层未捕获异常；Qt 槽内异常由 PySide6 转给
    ``sys.excepthook`` 或 Qt ``messageHandler``，两者都兜住才完整。
    """
    global _hook_installed
    if _hook_installed:
        return
    sys.excepthook = _default_hook
    _hook_installed = True
    logger.debug("全局 sys.excepthook 已安装")


def install_qt_message_guard() -> None:
    """覆写 Qt ``qInstallMessageHandler``：Qt 错误级别也进 Python logging。"""
    global _qt_guard_installed
    if _qt_guard_installed:
        return

    def _qt_handler(mode, context, message):  # noqa: ANN001 - Qt 回调签名
        if mode == 2:  # QtWarningMsg
            logger.warning("Qt: %s", message)
        elif mode >= 3:  # QtCriticalMsg / QtFatalMsg
            logger.error("Qt[%s]: %s", mode, message)
        else:
            logger.debug("Qt: %s", message)

    try:
        from PySide6.QtCore import qInstallMessageHandler

        qInstallMessageHandler(_qt_handler)
        _qt_guard_installed = True
        logger.debug("Qt messageHandler 已安装")
    except Exception as exc:  # noqa: BLE001 - 环境差异
        logger.debug("安装 Qt messageHandler 失败：%s", exc)


__all__ = ["install_excepthook", "install_qt_message_guard"]