"""快捷键集中管理（P2-S5）。

职责
----
把主窗口所需的 :class:`QShortcut` 注册到 :class:`~app.ui.main_window.MainWindow` 上，
并以**可枚举**的形式暴露（``ShortcutManager.shortcuts() -> [(键, 说明), ...]``），
便于测试与后续 **P6-S2「快捷键全集」** 在此扩展。

本步覆盖的快捷键
----------------
* ``1`` .. ``8`` → 周期切换（映射见 :data:`PERIOD_KEY_MAP`），走 :class:`QShortcut`；
* ``Up`` / ``Down`` → 上一只 / 下一只自选股（委托
  :meth:`WatchlistPanel.select_previous` / :meth:`select_next`），**不走 QShortcut**（见下）。

⚠️ 为什么 ``Up`` / ``Down`` 不用 ``QShortcut``（P2-S5 回炉修复）
    最初把 ``Up`` / ``Down`` 也注册成窗口级 ``QShortcut``，结果 **方向键被 QShortcut 吃掉**，
    :class:`~PySide6.QtWidgets.QTreeWidget` 根本收不到 ``keyPress`` → 自选股列表的原生方向键
    导航彻底失效（焦点在树上按 ``↓`` 毫无反应），破坏了 P2-S1 既有的「键盘浏览列表」能力。
    因此本步改为：
        * 树自身用 **事件过滤器**（见 :class:`~app.ui.widgets.watchlist_panel.WatchlistPanel`）
          处理 ``↑/↓``；
        * 主窗口 :meth:`~app.ui.main_window.MainWindow.keyPressEvent` 兜底处理「焦点不在树上、
          且控件不消费方向键」的场景。
    ``Up`` / ``Down`` 仍**登记**在 :meth:`ShortcutManager.shortcuts`（说明书）中，供将来
    P6-S2「快捷键全集」展示，但**不产生** :class:`QShortcut`（故 :meth:`ShortcutManager.qshortcuts`
    只返回 8 个周期快捷键）。

设计要点
--------
* 周期快捷键使用 **``Qt.ShortcutContext.WindowShortcut``**（窗口级，不抢全局）；
* 注册过程统一 ``try/except`` + 日志，**注册失败绝不抛出**；
* 周期快捷键复用既有的 ``MainToolBar.set_period`` → ``periodChanged`` → K 线重载链路，
  因此 ``1m``（键 ``1``）仍会自动切到分时页（P2-S3 既有行为，保留）。

⚠️ 关于 ``1..8``（有意偏离设计文档）
    《细化设计》P2-S5 原文写「``1..7`` 映射周期」，但项目实为 **8 个周期**
    （``1m/5m/15m/30m/60m/d/w/m``，见 :data:`app.ui.widgets.toolbar.PERIOD_ITEMS`），
    故本实现按 **``1..8``** 处理：``1=1分 2=5分 3=15分 4=30分 5=60分 6=日 7=周 8=月``。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut

from app.ui.widgets.toolbar import PERIOD_ITEMS

logger = logging.getLogger(__name__)

# 键 → 周期编码（``"1".."8"``）。**由 PERIOD_ITEMS 顺序派生**，保证与工具栏下拉一致。
PERIOD_KEY_MAP: dict[str, str] = {
    str(index + 1): code for index, (_, code) in enumerate(PERIOD_ITEMS)
}

# 周期编码 → 显示文案（用于快捷键说明）。
_PERIOD_LABEL_MAP: dict[str, str] = {code: text for text, code in PERIOD_ITEMS}

# 自选股上下切换键（Qt 键名）。
KEY_PREV_SYMBOL: str = "Up"
KEY_NEXT_SYMBOL: str = "Down"


def _period_label(code: str) -> str:
    """返回周期编码的中文显示文案（找不到时回退编码本身）。"""
    return _PERIOD_LABEL_MAP.get(code, code)


class ShortcutManager:
    """把主窗口快捷键注册到 ``window`` 并暴露可枚举清单。"""

    def __init__(self, window) -> None:  # noqa: ANN001 - 依赖 MainWindow 接口
        """构造（不注册，调用 :meth:`install` 才注册）。

        Args:
            window: 主窗口实例（需提供 ``toolbar`` / ``watchlist_panel`` 及
                ``_on_period_shortcut`` / ``_on_next_symbol`` / ``_on_prev_symbol``）。
        """
        self._window = window
        self._specs: List[Tuple[str, str]] = []
        self._shortcuts: List[QShortcut] = []

    # ------------------------------------------------------------------ #
    # 注册
    # ------------------------------------------------------------------ #
    def install(self) -> List[QShortcut]:
        """注册全部快捷键（异常仅记日志，不抛出）。

        Returns:
            已注册的 :class:`QShortcut` 列表（**仅 8 个周期键**；``Up``/``Down`` 只登记不注册）。
        """
        try:
            self._install_period_shortcuts()
        except Exception:  # noqa: BLE001 - 注册失败不得影响启动
            logger.exception("注册周期快捷键失败")
        try:
            self._register_navigation_specs()
        except Exception:  # noqa: BLE001
            logger.exception("登记自选导航快捷键说明失败")
        return list(self._shortcuts)

    def _install_period_shortcuts(self) -> None:
        """把 ``1..8`` 注册为周期切换快捷键（真正的 QShortcut）。"""
        for key, code in PERIOD_KEY_MAP.items():
            self._register(key, f"周期:{_period_label(code)}", self._period_handler(code))

    def _register_navigation_specs(self) -> None:
        """登记 ``Up`` / ``Down`` 的说明条目（**不注册 QShortcut**）。

        理由见模块 docstring：``QShortcut`` 会抢占 :class:`QTreeWidget` 的原生方向键，
        导致自选股列表键盘导航失效；实际按键由 ``WatchlistPanel`` 的事件过滤器与
        ``MainWindow.keyPressEvent`` 处理。
        """
        self._specs.append((KEY_NEXT_SYMBOL, "下一只自选"))
        self._specs.append((KEY_PREV_SYMBOL, "上一只自选"))
        logger.debug("登记导航快捷键说明（不注册 QShortcut）：%s / %s",
                     KEY_NEXT_SYMBOL, KEY_PREV_SYMBOL)

    def _period_handler(self, code: str) -> Callable[[], None]:
        """返回一个「切换到 ``code`` 周期」的无参处理函数。"""

        def _handler() -> None:
            try:
                self._window._on_period_shortcut(code)
            except Exception:  # noqa: BLE001 - 交互不外抛
                logger.exception("周期快捷键处理失败：%s", code)

        return _handler

    def _register(self, key: str, description: str, handler: Callable[[], None]) -> None:
        """创建一个窗口级 ``QShortcut`` 并连接处理函数。"""
        shortcut = QShortcut(QKeySequence(key), self._window)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.setObjectName(f"shortcut_{key}")
        shortcut.activated.connect(handler)
        self._shortcuts.append(shortcut)
        self._specs.append((key, description))
        logger.debug("注册快捷键 %s -> %s", key, description)

    # ------------------------------------------------------------------ #
    # 枚举
    # ------------------------------------------------------------------ #
    def shortcuts(self) -> List[Tuple[str, str]]:
        """返回可枚举的 ``[(键, 说明), ...]`` 清单（供测试 / P6 快捷键说明）。"""
        return list(self._specs)

    def qshortcuts(self) -> List[QShortcut]:
        """返回已注册的 :class:`QShortcut` 列表。"""
        return list(self._shortcuts)


def install_shortcuts(window) -> List[QShortcut]:  # noqa: ANN001
    """便捷函数：在主窗口上安装全部快捷键并返回其 :class:`QShortcut` 列表。

    Args:
        window: 主窗口实例。

    Returns:
        已注册的 :class:`QShortcut` 列表。
    """
    return ShortcutManager(window).install()


__all__ = [
    "ShortcutManager",
    "install_shortcuts",
    "PERIOD_KEY_MAP",
    "KEY_PREV_SYMBOL",
    "KEY_NEXT_SYMBOL",
]
