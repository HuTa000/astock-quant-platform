"""个股详情（备注）弹窗 —— P4-S5。

职责：
    * 展示个股**基本信息 + 核心财务 + 财报摘要**（数据来自 ``fundamental_loader``，
      默认走主数据源实时抓取，**后台线程加载**、失败降级为「加载失败」占位，不阻塞 UI）；
    * 提供**备注正文**编辑（与现有 ★ 标记联动：保存非空正文 → 自动置 ★，清除正文 → 取消 ★），
      正文持久化复用存储层既有 ``stock_note`` 表（:meth:`Database.get_note/set_note/delete_note`）；
    * **显式保存**：只有点「保存备注」才写库；关闭时若正文已修改未保存 → 弹窗询问
      （保存 / 放弃 / 取消），防误丢。

设计要点（遵循《开发手册》§4.11）：
    * 带参数槽一律用 ``lambda`` 适配，避免 Qt 信号把 ``bool`` 等隐式参数传进槽；
    * 后台线程遵循 P3-S5 教训：worker ``parent=None`` + 关闭时 ``wait()`` 守卫，
      防止「销毁运行中 QThread」；
    * 纯格式化函数（:func:`fmt_cap` / :func:`fmt_pct` / :func:`fmt_num` / :func:`fmt_text`）
      与 Qt 解耦，可无 GUI 单测。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Union

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.data.models import Fundamental
from app.data.store.db import Database, get_db

logger = logging.getLogger(__name__)

# 占位文案。
_PLACEHOLDER: str = "—"
_LOADING_TEXT: str = "正在加载基本面…"
_LOAD_FAILED_TEXT: str = "基本面加载失败"

# 财务字段中文名（键顺序稳定，供测试）。
_FUND_KEYS: tuple[tuple[str, str], ...] = (
    ("name", "名称"),
    ("industry", "行业"),
    ("list_date", "上市日期"),
    ("total_market_cap", "总市值"),
    ("float_market_cap", "流通市值"),
    ("pe", "PE"),
    ("pb", "PB"),
    ("roe", "ROE"),
    ("revenue_yoy", "同比"),
    ("report_period", "报告期"),
)


# --------------------------------------------------------------------------- #
# 纯格式化函数（无 Qt 依赖，可单测）
# --------------------------------------------------------------------------- #
def fmt_num(value: Optional[float]) -> str:
    """数值 → ``1,234.56``；``None``/非法 → ``—``。"""
    if value is None:
        return _PLACEHOLDER
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    return f"{num:,.2f}"


def fmt_cap(value: Optional[float]) -> str:
    """市值（元）→ 可读字符串：≥1 亿显示 ``亿``，否则原样；``None`` → ``—``。"""
    if value is None:
        return _PLACEHOLDER
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if abs(num) >= 1e8:
        return f"{num / 1e8:,.2f} 亿"
    return f"{num:,.2f} 元"


def fmt_pct(value: Optional[float]) -> str:
    """百分比数值（如 ``4.69`` 表示 4.69%）→ ``4.69%``；``None`` → ``—``。"""
    if value is None:
        return _PLACEHOLDER
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    return f"{num:,.2f}%"


def fmt_text(value: Optional[str]) -> str:
    """文本 → ``str``（去首尾空白）；``None``/空 → ``—``。"""
    if value is None:
        return _PLACEHOLDER
    text = str(value).strip()
    return text if text else _PLACEHOLDER


def format_fundamental(data: Optional[dict]) -> dict:
    """把基本面 dict 归一为「字段名 → 显示文本」映射（供 UI 填充，纯函数）。

    Args:
        data: ``get_fundamental`` 返回的 dict（键见 :class:`app.data.models.Fundamental`）。

    Returns:
        形如 ``{"名称": "平安银行", "PE": "12.34", ...}`` 的映射；键缺失时填充占位。
    """
    if not isinstance(data, dict):
        data = {}
    out: dict[str, str] = {}
    for key, label in _FUND_KEYS:
        raw = data.get(key)
        if key in ("total_market_cap", "float_market_cap"):
            out[label] = fmt_cap(raw)
        elif key in ("pe", "pb"):
            out[label] = fmt_num(raw)
        elif key in ("roe", "revenue_yoy"):
            out[label] = fmt_pct(raw)
        else:
            out[label] = fmt_text(raw)
    return out


# --------------------------------------------------------------------------- #
# 后台加载线程
# --------------------------------------------------------------------------- #
class _FundWorker(QThread):
    """后台抓取基本面（``parent=None``，由对话框负责关闭时 ``wait()`` 守卫）。"""

    result_ready = Signal(dict)
    fetch_failed = Signal(str)

    def __init__(self, loader: Callable[[str], dict], symbol: str) -> None:
        """构造后台抓取线程。

        Args:
            loader: 「代码 → 基本面 dict」回调（网络侧）。
            symbol: 标的代码。
        """
        super().__init__(parent=None)
        self._loader = loader
        self._symbol = str(symbol)

    def run(self) -> None:  # noqa: D401 - QThread 约定
        """执行抓取并发出结果 / 失败信号（异常不外抛）。"""
        try:
            data = self._loader(self._symbol)
        except Exception as exc:  # noqa: BLE001 - 失败降级为占位
            logger.warning("加载 %s 基本面失败：%s", self._symbol, exc)
            self.fetch_failed.emit(str(exc))
            return
        self.result_ready.emit(data if isinstance(data, dict) else {})


# --------------------------------------------------------------------------- #
# 详情弹窗
# --------------------------------------------------------------------------- #
class NoteDetailDialog(QDialog):
    """个股详情（备注）弹窗。

    Signals:
        note_saved: 备注正文已保存时发出（携带代码）。
        note_cleared: 备注正文已清除时发出（携带代码）。
    """

    note_saved = Signal(str)
    note_cleared = Signal(str)

    def __init__(
        self,
        symbol: str,
        db: Optional[Database] = None,
        fundamental_loader: Optional[Callable[[str], dict]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造弹窗。

        Args:
            symbol: 标的代码（必填）。
            db: 存储实例；默认 :func:`get_db`（读 / 写备注正文）。
            fundamental_loader: 「代码 → 基本面 dict」回调；为 ``None`` 时仅显示占位、
                不发起抓取（供无网络 / 测试场景）。
            parent: 父控件。
        """
        super().__init__(parent)
        self._symbol = str(symbol)
        self._db: Database = db if db is not None else get_db()
        self._loader: Optional[Callable[[str], dict]] = fundamental_loader
        self._worker: Optional[_FundWorker] = None
        self._dirty: bool = False

        self.setWindowTitle(f"个股详情 - {self._symbol}")
        self.setMinimumWidth(460)
        self._setup_ui()
        self._load_note()
        self._start_fundamental_load()

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建：基本信息 + 核心财务（QFormLayout）+ 财报摘要 + 备注 + 按钮。"""
        layout = QVBoxLayout(self)

        self._status_label = QLabel(self)
        self._status_label.setStyleSheet("color: #9aa0a6;")

        # 基本信息 / 核心财务标签映射。
        self._value_labels: dict[str, QLabel] = {}
        for key, label in _FUND_KEYS:
            value = QLabel(_PLACEHOLDER, self)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._value_labels[key] = value

        form = QFormLayout()
        for key, label in _FUND_KEYS:
            form.addRow(label + "：", self._value_labels[key])
        layout.addLayout(form)

        # 财报摘要。
        self._summary_label = QLabel(_PLACEHOLDER, self)
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(QLabel("财报摘要：", self))
        layout.addWidget(self._summary_label)

        # 备注正文。
        layout.addWidget(QLabel("备注：", self))
        self._note_edit = QTextEdit(self)
        self._note_edit.setPlaceholderText("输入该股票的备注（与列表 ★ 标记联动）…")
        self._note_edit.setMinimumHeight(110)
        self._note_edit.textChanged.connect(self._on_note_changed)
        layout.addWidget(self._note_edit)

        # 按钮。
        buttons = QHBoxLayout()
        self._btn_save = QPushButton("保存备注", self)
        self._btn_save.clicked.connect(lambda: self.save_note())
        self._btn_clear = QPushButton("清除备注", self)
        self._btn_clear.clicked.connect(lambda: self.clear_note())
        self._btn_close = QPushButton("关闭", self)
        self._btn_close.clicked.connect(lambda: self.close())
        buttons.addWidget(self._btn_save)
        buttons.addWidget(self._btn_clear)
        buttons.addStretch(1)
        buttons.addWidget(self._btn_close)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ #
    # 数据加载
    # ------------------------------------------------------------------ #
    def _load_note(self) -> None:
        """从库中读取既有备注正文回填编辑框（失败降级为空）。"""
        try:
            note = self._db.get_note(self._symbol)
        except Exception as exc:  # noqa: BLE001 - 读库失败不阻塞
            logger.warning("读取备注失败：%s", exc)
            note = None
        if note is not None and note.text:
            self._note_edit.setPlainText(str(note.text))
            self._dirty = False

    def _start_fundamental_load(self) -> None:
        """后台加载基本面：无 loader 直接降级占位；有则起 ``_FundWorker``。"""
        if self._loader is None:
            self._status_label.setText("基本面未加载（无数据源）")
            return
        self._status_label.setText(_LOADING_TEXT)
        self._worker = _FundWorker(self._loader, self._symbol)
        self._worker.result_ready.connect(self._on_fund_ready)
        self._worker.fetch_failed.connect(self._on_fund_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_fund_ready(self, data: dict) -> None:
        """基本面加载完成：填充各字段与摘要。"""
        labels = format_fundamental(data)
        for key, label in _FUND_KEYS:
            self._value_labels[key].setText(labels.get(label, _PLACEHOLDER))
        summary = str(data.get("summary") or "").strip()
        self._summary_label.setText(summary if summary else _PLACEHOLDER)
        self._status_label.setText("基本面已加载")

    def _on_fund_failed(self, _error: str) -> None:
        """基本面加载失败：各字段保持占位，状态栏提示。"""
        self._status_label.setText(_LOAD_FAILED_TEXT)

    def _on_worker_finished(self) -> None:
        """后台线程结束：释放引用（线程对象自身由 Qt 回收）。"""
        self._worker = None

    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt 约定
        """关闭守卫：正文已修改未保存 → 询问（保存 / 放弃 / 取消）；后台线程 ``wait()``。"""
        if self._dirty:
            answer = QMessageBox.question(
                self,
                "未保存的备注",
                "备注已修改但尚未保存，关闭前是否保存？",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Save:
                self.save_note()
            elif answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # 备注操作（显式保存）
    # ------------------------------------------------------------------ #
    def _on_note_changed(self) -> None:
        """正文编辑即视为「已修改」（脏标记，供关闭守卫使用）。"""
        self._dirty = True

    def note_text(self) -> str:
        """当前正文（去首尾空白）。"""
        return self._note_edit.toPlainText().strip()

    def save_note(self) -> bool:
        """保存备注正文：非空写库并发 ``note_saved``；空正文等价于清除。

        Returns:
            成功 ``True``（空正文走清除语义时也返回 ``True``）。
        """
        text = self.note_text()
        if not text:
            return self.clear_note()
        try:
            self._db.set_note(self._symbol, text)
        except Exception as exc:  # noqa: BLE001 - 写库失败给可见反馈
            logger.warning("保存备注失败：%s", exc)
            QMessageBox.warning(self, "保存失败", f"备注保存失败：{exc}")
            return False
        self._dirty = False
        self.note_saved.emit(self._symbol)
        self._status_label.setText("备注已保存")
        return True

    def clear_note(self) -> bool:
        """清除备注正文：删库并发 ``note_cleared``。

        Returns:
            成功 ``True``。
        """
        try:
            self._db.delete_note(self._symbol)
        except Exception as exc:  # noqa: BLE001 - 删库失败给可见反馈
            logger.warning("清除备注失败：%s", exc)
            QMessageBox.warning(self, "清除失败", f"备注清除失败：{exc}")
            return False
        self._note_edit.clear()
        self._dirty = False
        self.note_cleared.emit(self._symbol)
        self._status_label.setText("备注已清除")
        return True

    def is_dirty(self) -> bool:
        """是否处于「已修改未保存」状态（供测试）。"""
        return self._dirty

    def status_text(self) -> str:
        """当前状态栏文案（供测试）。"""
        return self._status_label.text()

    def fundamental_placeholder(self) -> dict:
        """当前已填充的基本面文本映射（供测试复核，键为中文标签）。"""
        return {label: self._value_labels[key].text() for key, label in _FUND_KEYS}


__all__ = [
    "NoteDetailDialog",
    "format_fundamental",
    "fmt_cap",
    "fmt_num",
    "fmt_pct",
    "fmt_text",
]
