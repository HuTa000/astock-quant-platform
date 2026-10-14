"""预警通知弹窗（P4-S2）。

职责：
    * 非模态、可叠加、自动消失的右下角通知（``AlertNotification``）；
    * 文案组装（kind 中文名、触发 message、时间）做成**纯函数**，UI 层与测试复用；
    * 不抢焦点（``WA_ShowWithoutActivating``），多条预警纵向堆叠不互相覆盖。

可离屏测试：UI 层在 ``QT_QPA_PLATFORM=offscreen`` 下可实例化；文案与 kind 映射
均为纯函数，无需 QApplication 即可测试。
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.themes import colors

logger = logging.getLogger(__name__)

# 弹窗标题。
TITLE: str = "预警触发"

# kind -> 中文名（对应《规格》P4-S1 四类规则）。
KIND_LABELS: dict[str, str] = {
    "ma_cross": "均线交叉",
    "macd_cross": "MACD 金叉",
    "volume_surge": "放量",
    "expr": "自定义表达式",
}

# 弹窗尺寸 / 间距（像素）。
WIDTH: int = 340
HEIGHT: int = 150
MARGIN: int = 12
GAP: int = 8

# 自动消失时长（毫秒）。
AUTO_CLOSE_MS: int = 8000


def kind_label(kind: str) -> str:
    """把规则 kind 映射为中文名。

    Args:
        kind: 规则类型（``ma_cross`` / ``macd_cross`` / ``volume_surge`` / ``expr``）。

    Returns:
        中文名；未知 kind 原样返回；空串返回 ``"未知"``。
    """
    key = str(kind or "").strip()
    return KIND_LABELS.get(key, key or "未知")


def build_message(payload: dict) -> str:
    """根据 payload 组装触发文案（与 ``AlertService._message`` 语义对齐的纯函数）。

    Args:
        payload: 六键事件负载 ``{symbol, kind, rule_id, params, value, time}``。

    Returns:
        人类可读的触发文案。
    """
    kind = str(payload.get("kind", "") or "")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    value = payload.get("value")

    if kind == "ma_cross":
        direction = str(params.get("direction", "cross_up")).strip()
        fast = params.get("fast", "")
        slow = params.get("slow", "")
        if direction == "cross_down":
            return f"均线死叉：MA{fast} 下穿 MA{slow}"
        return f"均线金叉：MA{fast} 上穿 MA{slow}"
    if kind == "macd_cross":
        return "MACD 金叉（DIF 上穿 DEA）"
    if kind == "volume_surge":
        if value is None:
            return "成交量放大"
        try:
            return f"成交量放大：当前量 {float(value):g}"
        except (TypeError, ValueError):
            return "成交量放大"
    if kind == "expr":
        return f"表达式触发：{params.get('expr', '')}"
    return str(kind)


def build_alert_lines(payload: dict) -> dict:
    """把 payload 组装为展示字段（纯函数，供弹窗与托盘气泡复用）。

    Returns:
        ``{"title", "symbol", "kind_label", "message", "time"}``。
    """
    symbol = str(payload.get("symbol", "") or "")
    time_text = str(payload.get("time", "") or "")
    return {
        "title": TITLE,
        "symbol": symbol,
        "kind_label": kind_label(str(payload.get("kind", "") or "")),
        "message": build_message(payload),
        "time": time_text,
    }


class AlertNotification(QWidget):
    """非模态、可叠加、自动消失的预警通知弹窗。

    设计要点（可叠加不互相覆盖）：
        类级注册表 :attr:`_active` 维护当前可见实例；每次增删实例后调用
        :meth:`_relayout`，把实例按注册顺序**自屏幕右下角向上纵向堆叠**
        （后弹的在上方），因此多条预警不会完全重叠。

    生命周期：
        构造时注册 + 排布 + 启动单次 ``QTimer``（默认 8 秒）；用户点「关闭」或
        定时器到期都会触发 :meth:`closeEvent`，在其中移除自身并重排剩余实例，
        不泄漏引用。
    """

    _active: list["AlertNotification"] = []

    def __init__(
        self,
        payload: dict,
        anchor: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
        auto_close_ms: int = AUTO_CLOSE_MS,
    ) -> None:
        """构造一条通知。

        Args:
            payload: 六键事件负载 ``{symbol, kind, rule_id, params, value, time}``。
            anchor: 用于定位屏幕的参考窗口（默认主窗口）；离屏 / 无屏时兜底。
            parent: 父对象（默认 ``None``，作为顶层窗口弹出）。
            auto_close_ms: 自动消失毫秒数（测试可注入更短值）。
        """
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("AlertNotification")
        # 弹出时不抢焦点（非模态关键）。
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(WIDTH, HEIGHT)

        self._payload = payload
        self._anchor = anchor

        self._build_ui()
        self._apply_style()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(auto_close_ms)
        self._timer.timeout.connect(self.close)

        AlertNotification._active.append(self)
        self._relayout()
        self._timer.start()

    # ------------------------------------------------------------------ #
    # UI 构建
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        """搭建通知内容：标题 + 关闭、symbol、信号类型、message、时间。"""
        lines = build_alert_lines(self._payload)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(6)

        header = QHBoxLayout()
        title_label = QLabel(lines["title"])
        title_label.setObjectName("AlertTitle")
        header.addWidget(title_label)
        header.addStretch(1)
        self._close_btn = QPushButton("关闭")
        self._close_btn.setObjectName("AlertCloseButton")
        self._close_btn.clicked.connect(self.close)
        header.addWidget(self._close_btn)
        root.addLayout(header)

        self._symbol_label = QLabel(lines["symbol"] or "—")
        self._symbol_label.setObjectName("AlertSymbol")
        root.addWidget(self._symbol_label)

        self._kind_label = QLabel(lines["kind_label"])
        self._kind_label.setObjectName("AlertKind")
        root.addWidget(self._kind_label)

        self._message_label = QLabel(lines["message"])
        self._message_label.setObjectName("AlertMessage")
        self._message_label.setWordWrap(True)
        root.addWidget(self._message_label)

        self._time_label = QLabel(lines["time"] or "")
        self._time_label.setObjectName("AlertTime")
        root.addWidget(self._time_label)

    def _apply_style(self) -> None:
        """应用深色主题内联 QSS（复用主题颜色常量）。"""
        self.setStyleSheet(
            f"""
            #AlertNotification {{
                background-color: {colors.PANEL_BG};
                border: 1px solid {colors.GRID_COLOR};
            }}
            QLabel#AlertTitle {{
                color: {colors.ACCENT_COLOR};
                font-weight: bold;
                font-size: 14px;
            }}
            QLabel#AlertSymbol {{
                color: {colors.UP_COLOR};
                font-weight: bold;
                font-size: 16px;
            }}
            QLabel#AlertKind {{
                color: {colors.LIMIT_COLOR};
            }}
            QLabel#AlertMessage {{
                color: {colors.TEXT_COLOR};
            }}
            QLabel#AlertTime {{
                color: {colors.SUBTEXT_COLOR};
                font-size: 11px;
            }}
            QPushButton#AlertCloseButton {{
                background-color: {colors.TOOLBAR_BG};
                color: {colors.TEXT_COLOR};
                border: 1px solid {colors.GRID_COLOR};
                padding: 2px 8px;
            }}
            QPushButton#AlertCloseButton:hover {{
                background-color: {colors.ACCENT_COLOR};
                color: #ffffff;
            }}
            """
        )

    # ------------------------------------------------------------------ #
    # 布局（可叠加）
    # ------------------------------------------------------------------ #
    @classmethod
    def _screen_geometry(cls, anchor: Optional[QWidget]) -> QRect:
        """选择弹窗要贴靠的屏幕可用区；离屏 / 无屏时给兜底矩形。"""
        app = QApplication.instance()
        screen = None
        if anchor is not None:
            try:
                handle = anchor.windowHandle()
                if handle is not None:
                    screen = handle.screen()
            except Exception:  # noqa: BLE001 - 定位失败走主屏
                screen = None
        if screen is None and app is not None:
            screen = app.primaryScreen()
        if screen is not None:
            return screen.availableGeometry()
        return QRect(0, 0, 1024, 768)

    @classmethod
    def _relayout(cls) -> None:
        """把当前可见实例自右下角向上纵向堆叠，避免互相覆盖。"""
        active = cls._active
        if not active:
            return
        geometry = cls._screen_geometry(active[0]._anchor)
        right = geometry.right() - MARGIN
        bottom = geometry.bottom() - MARGIN
        for index, widget in enumerate(active):
            x = right - WIDTH
            y = bottom - HEIGHT - index * (HEIGHT + GAP)
            try:
                widget.move(x, y)
            except Exception:  # noqa: BLE001 - 实例可能已被销毁
                logger.exception("排布预警弹窗失败")

    # ------------------------------------------------------------------ #
    # 生命周期 / 测试辅助
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt 约定
        """关闭时移除自身并重排剩余实例。"""
        if self in AlertNotification._active:
            AlertNotification._active.remove(self)
        self._relayout()
        event.accept()

    @classmethod
    def active_notifications(cls) -> list["AlertNotification"]:
        """返回当前可见实例的副本（测试 / 自检用）。"""
        return list(cls._active)

    @classmethod
    def close_all(cls) -> None:
        """关闭全部通知（测试清理用）。"""
        for widget in list(cls._active):
            try:
                widget.close()
            except Exception:  # noqa: BLE001
                logger.exception("关闭预警弹窗失败")


__all__ = [
    "AlertNotification",
    "build_alert_lines",
    "build_message",
    "kind_label",
    "KIND_LABELS",
    "TITLE",
    "AUTO_CLOSE_MS",
]
