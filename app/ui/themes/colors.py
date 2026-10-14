"""主题颜色常量（深色 + A 股红涨绿跌）。

取值**严格照抄**《细化设计》§1.4 与《开发手册与交接文档》§4.6，
禁止在本文件之外硬编码颜色。
"""
from __future__ import annotations

# ---- 背景 / 前景 ---- #
BG_DARK: str = "#1e1e1e"       # 主背景
PANEL_BG: str = "#252526"      # 面板背景
TOOLBAR_BG: str = "#2d2d30"    # 工具栏背景
TEXT_COLOR: str = "#d4d4d4"    # 主文字
SUBTEXT_COLOR: str = "#9aa0a6"  # 次文字
GRID_COLOR: str = "#3a3a3a"    # 网格线

# ---- 涨跌（红涨绿跌） ---- #
UP_COLOR: str = "#e74c3c"      # 涨（红）
DOWN_COLOR: str = "#2ecc71"    # 跌（绿）
VOL_UP: str = "#e74c3c"        # 量柱涨
VOL_DOWN: str = "#2ecc71"      # 量柱跌

# ---- 强调 / 标注 ---- #
ACCENT_COLOR: str = "#3498db"    # 强调（选中 / 按钮）
LIMIT_COLOR: str = "#f1c40f"     # 涨跌停 / 缺口标注
LIMIT_UP_COLOR: str = "#ffd700"  # 涨停蜡烛（金色，替代原黄色文字标注）
LIMIT_DOWN_COLOR: str = "#006400"  # 跌停蜡烛（深绿）
TRADE_PRICE_COLOR: str = "#f1c40f"  # 模拟训练交易价参考线（黄色虚线）
DIVIDEND_COLOR: str = "#9b59b6"  # 除权除息标注
CROSS_CURSOR: str = "#c0c0c0"    # 十字光标


def color_map() -> dict[str, str]:
    """返回全部主题常量的 ``{名称: 颜色值}`` 字典（供 QSS ``format`` 注入）。

    Returns:
        形如 ``{"BG_DARK": "#1e1e1e", "UP_COLOR": "#e74c3c", ...}`` 的字典。
    """
    return {
        "BG_DARK": BG_DARK,
        "PANEL_BG": PANEL_BG,
        "TOOLBAR_BG": TOOLBAR_BG,
        "TEXT_COLOR": TEXT_COLOR,
        "SUBTEXT_COLOR": SUBTEXT_COLOR,
        "GRID_COLOR": GRID_COLOR,
        "UP_COLOR": UP_COLOR,
        "DOWN_COLOR": DOWN_COLOR,
        "VOL_UP": VOL_UP,
        "VOL_DOWN": VOL_DOWN,
        "ACCENT_COLOR": ACCENT_COLOR,
        "LIMIT_COLOR": LIMIT_COLOR,
        "LIMIT_UP_COLOR": LIMIT_UP_COLOR,
        "LIMIT_DOWN_COLOR": LIMIT_DOWN_COLOR,
        "TRADE_PRICE_COLOR": TRADE_PRICE_COLOR,
        "DIVIDEND_COLOR": DIVIDEND_COLOR,
        "CROSS_CURSOR": CROSS_CURSOR,
    }


__all__ = [
    "BG_DARK",
    "PANEL_BG",
    "TOOLBAR_BG",
    "TEXT_COLOR",
    "SUBTEXT_COLOR",
    "GRID_COLOR",
    "UP_COLOR",
    "DOWN_COLOR",
    "VOL_UP",
    "VOL_DOWN",
    "ACCENT_COLOR",
    "LIMIT_COLOR",
    "LIMIT_UP_COLOR",
    "LIMIT_DOWN_COLOR",
    "TRADE_PRICE_COLOR",
    "DIVIDEND_COLOR",
    "CROSS_CURSOR",
    "color_map",
]
