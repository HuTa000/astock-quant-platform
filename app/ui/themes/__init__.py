"""主题子包：颜色常量 + QSS 应用。

对外入口：:func:`apply_theme`（读取 ``app.qss`` → 注入颜色 → 应用到 ``QApplication``）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.ui.themes import colors
from app.ui.themes.colors import color_map

logger = logging.getLogger(__name__)

# QSS 模板路径。
QSS_PATH: Path = Path(__file__).resolve().parent / "app.qss"

# 支持的命名主题（P2-S1 仅深色红涨绿跌）。
DEFAULT_THEME: str = "dark_red_green"


def load_qss() -> str:
    """读取 ``app.qss`` 模板文本。

    Returns:
        QSS 模板字符串（含 ``{占位符}``）。
    """
    return QSS_PATH.read_text(encoding="utf-8")


def build_stylesheet(theme: str = DEFAULT_THEME) -> str:
    """生成最终 QSS（占位符已替换为颜色值）。

    Args:
        theme: 主题名（当前仅 ``"dark_red_green"``）。

    Returns:
        可直接 ``setStyleSheet`` 的 QSS 文本。
    """
    template = load_qss()
    return template.format(**color_map())


def apply_theme(app: Optional[Any], theme: str = DEFAULT_THEME) -> None:
    """把主题应用到 ``QApplication``。

    Args:
        app: ``QApplication`` 实例；为 ``None`` 时跳过并记录日志。
        theme: 主题名；默认 ``"dark_red_green"``（P2-S1 仅支持该主题）。

    Notes:
        任何异常均被捕获并记日志，**不向上抛出**，避免因主题问题导致启动失败。
    """
    if app is None:
        logger.warning("apply_theme：QApplication 为 None，跳过主题应用（theme=%s）", theme)
        return
    try:
        if theme != DEFAULT_THEME:
            logger.info("未知主题 %r，回退到 %s", theme, DEFAULT_THEME)
            theme = DEFAULT_THEME
        stylesheet = build_stylesheet(theme)
        app.setStyleSheet(stylesheet)
        logger.info("主题已应用：%s", theme)
    except Exception as exc:  # noqa: BLE001 - 主题失败不影响启动
        logger.warning("应用主题失败（theme=%s）：%s", theme, exc)


__all__ = [
    "apply_theme",
    "build_stylesheet",
    "load_qss",
    "colors",
    "color_map",
    "DEFAULT_THEME",
    "QSS_PATH",
]
