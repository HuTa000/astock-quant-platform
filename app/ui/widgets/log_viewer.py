"""日志查看对话框 —— P6-S3。

职责：从 ``data/app.log`` 读取**末尾 N 行**并展示（只读 QPlainTextEdit），
供用户在界面内排查问题（验收：日志可在界面查看）。

纯展示组件：
    * 内容由 :func:`read_log_tail`（纯函数，可无 GUI 单测）读取；
    * 支持「刷新」重新读尾部 + 「打开日志目录」定位文件。
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)

# 默认读取行数。
_DEFAULT_LINES: int = 500


def read_log_tail(log_file: Path, lines: int = _DEFAULT_LINES) -> List[str]:
    """读取日志文件**末尾** ``lines`` 行（UTF-8 容错）。

    Args:
        log_file: 日志文件路径。
        lines: 读取行数（>=1）。

    Returns:
        末行列表（按时间先后）；文件不存在 / 读取失败返回空列表（不抛）。
    """
    if lines < 1:
        lines = 1
    try:
        if not log_file.is_file():
            return []
        # 读全部再取尾（文件 5MB 上限，可接受）。
        with log_file.open("r", encoding="utf-8", errors="replace") as fh:
            content = fh.read().splitlines()
        return content[-lines:]
    except Exception as exc:  # noqa: BLE001 - 读日志失败不抛
        logger.warning("读取日志失败：%s", exc)
        return []


class LogViewerDialog(QDialog):
    """日志查看对话框（末尾 N 行只读）。"""

    def __init__(self, log_file: Optional[Path] = None, parent: Optional[object] = None) -> None:  # noqa: ANN401
        """构造对话框。

        Args:
            log_file: 日志文件路径；``None`` 用 :func:`app.core.logging.get_log_file`。
            parent: 父控件。
        """
        super().__init__(parent)  # type: ignore[arg-type]
        from app.core.logging import get_log_file

        self._log_file: Path = Path(log_file) if log_file is not None else get_log_file()
        self.setWindowTitle("应用日志")
        self.resize(720, 480)
        layout = QVBoxLayout(self)

        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._text)

        buttons = QHBoxLayout()
        self._btn_refresh = QPushButton("刷新", self)
        self._btn_refresh.clicked.connect(lambda: self.refresh())
        self._btn_open_dir = QPushButton("打开日志目录", self)
        self._btn_open_dir.clicked.connect(lambda: self._open_dir())
        self._btn_close = QPushButton("关闭", self)
        self._btn_close.clicked.connect(lambda: self.close())
        buttons.addWidget(self._btn_refresh)
        buttons.addWidget(self._btn_open_dir)
        buttons.addStretch(1)
        buttons.addWidget(self._btn_close)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self) -> None:
        """重新读取日志尾部并刷新显示。"""
        lines = read_log_tail(self._log_file)
        self._text.setPlainText("\n".join(lines) if lines else "（日志为空或不存在）")

    def current_text(self) -> str:
        """当前显示的日志文本（供测试）。"""
        return self._text.toPlainText()

    def _open_dir(self) -> None:
        """打开日志所在目录（Windows explorer / macOS open / Linux xdg-open）。"""
        directory = self._log_file.parent
        if not directory.is_dir():
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(directory)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except Exception as exc:  # noqa: BLE001 - 打开失败不影响
            logger.warning("打开日志目录失败：%s", exc)


__all__ = ["LogViewerDialog", "read_log_tail"]