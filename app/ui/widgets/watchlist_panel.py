"""自选股栏（左侧列表）。

演进：
    * P2-S1：展示自选股分组/条目，选中条目时发出 ``symbolSelected``（**不做增删改**）；
    * P2-S5：键盘 ↑/↓ 导航（事件过滤器）、``select_next/select_previous``；
    * P2-S6：**增删 / 分组 / 备注标记**持久化 + 顶部工具条 + 右键菜单。

架构约定（P2-S6 关键，决定可测性）：
    * **业务动作 = 无弹窗的纯方法**（``add_symbol`` / ``remove_symbol`` / ``remove_current`` /
      ``toggle_note_mark`` / ``create_group`` / ``rename_group`` / ``delete_group``），
      可被测试直接调用，返回明确结果；
    * **UI 触达层**（顶部按钮 / 右键菜单）负责弹 :class:`QInputDialog` / :class:`QMessageBox`
      确认后调用业务方法。
    * 业务动作成功后经 :func:`app.core.event_bus.publish` 发 ``WATCHLIST_CHANGED``。

不变式（勿破坏 P2-S5 / P2-S1 已验收行为）：
    * ``symbol`` 始终存于 ``item.data(0, int(Qt.ItemDataRole.UserRole))``（``UserRole == 256``）；
    * ``select_next/select_previous/symbols()/current_symbol()/symbolSelected/_symbol_items()``
      语义保持不变；
    * 空库仍显示占位 ``（暂无自选股）``，``refresh()`` **不会**自动建组（仅在"添加自选且无分组"时建默认组）。
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.event_bus import get_bus
from app.data.store.db import Database, get_db

logger = logging.getLogger(__name__)

# 事件名（设计 §1.3 已定义；本步复用，**不新增事件名**）。
WATCHLIST_CHANGED: str = "WATCHLIST_CHANGED"

# 树条目数据角色。
_SYMBOL_ROLE: int = int(Qt.ItemDataRole.UserRole)        # 代码（str）；分组/占位为 None
_GROUP_ROLE: int = int(Qt.ItemDataRole.UserRole) + 1     # 分组 id（int）；分组头/条目均带
_NOTE_ROLE: int = int(Qt.ItemDataRole.UserRole) + 2      # 备注标记（bool）

# 空库提示文案（P2-S1 既有行为）。
_EMPTY_HINT: str = "（暂无自选股）"

# 无分组时自动创建的默认分组名。
_DEFAULT_GROUP_NAME: str = "我的自选"

# 合法代码：6 位 **ASCII** 数字（用 [0-9] 而非 \d，避免匹配 Unicode 全角/阿拉伯-印度数字）。
_SYMBOL_RE = re.compile(r"^[0-9]{6}$")

# 备注标记后缀。
_NOTE_SUFFIX: str = " ★"


def is_valid_symbol(symbol: str) -> bool:
    """6 位数字代码校验（与 :meth:`WatchlistPanel.add_symbol` 内部同一规则）。

    供 UI 触达层预校验复用，以便把"格式错误"与"分组不存在"两类失败区分开。

    Args:
        symbol: 待校验代码（任意类型，内部会 ``str().strip()``）。

    Returns:
        合法（``^[0-9]{6}$``）返回 ``True``，否则 ``False``。
    """
    return bool(_SYMBOL_RE.match(str(symbol).strip()))



class WatchlistPanel(QWidget):
    """左侧自选股面板（展示 + 导航 + 增删/分组/备注管理）。"""

    #: 选中某只自选股时发出，携带 6 位代码。
    symbolSelected = Signal(str)

    #: 请求打开个股详情（备注）弹窗时发出，携带 ``(group_id, symbol)``（P4-S5）。
    detailRequested = Signal(int, str)

    #: 请求打开个股预警管理时发出，携带 ``(group_id, symbol)``（P6-S2）。
    alertRequested = Signal(int, str)

    def __init__(
        self,
        db: Optional[Database] = None,
        parent: Optional[QWidget] = None,
        name_resolver: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        """构造自选股面板。

        Args:
            db: 存储实例；默认 :func:`get_db`。
            parent: 父控件。
            name_resolver: 可选的「代码 → 名称」解析器（预留后续接入名称缓存）。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._name_resolver = name_resolver
        self._updating: bool = False
        self._setup_ui()
        self.refresh()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建 UI：顶部紧凑工具条 + 单列树状列表（分组为顶层项，股票为其子项）。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 顶部工具条。
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(2, 2, 2, 2)
        toolbar.setSpacing(4)
        self._btn_add = QPushButton("＋ 添加自选", self)
        self._btn_add.setToolTip("输入 6 位股票代码加入自选")
        # 注意：QPushButton.clicked 自带一个 bool checked 位置参数，故用 lambda 吞掉它。
        # 第二个参数传「当前选中分组」：有选中分组/条目时加进该组，无选中时传 None 走旧逻辑
        # （第一个分组 / 无分组则自动建「我的自选」）。
        self._btn_add.clicked.connect(lambda: self._on_add_symbol_action(self.current_group_id()))
        self._btn_new_group = QPushButton("＋ 新建分组", self)
        self._btn_new_group.setToolTip("新建一个自选股分组")
        self._btn_new_group.clicked.connect(lambda: self._on_new_group_action())
        toolbar.addWidget(self._btn_add)
        toolbar.addWidget(self._btn_new_group)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setUniformRowHeights(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemClicked.connect(self._on_item_clicked)
        # ↑/↓ 由本面板处理：焦点在内部 QTreeWidget 上，keyPress 不会冒泡到本容器父控件，
        # 故必须在**树上**安装事件过滤器（见 eventFilter）。
        self._tree.installEventFilter(self)
        layout.addWidget(self._tree)

    @property
    def list_widget(self) -> QTreeWidget:
        """返回底层树控件（供测试/后续 Step 访问）。"""
        return self._tree

    # ------------------------------------------------------------------ #
    # 数据填充
    # ------------------------------------------------------------------ #
    def _list_groups(self) -> list:
        """列出全部自选股分组（真实 API：``Database.list_groups``）。"""
        return list(self._db.list_groups())

    def _list_items(self) -> list:
        """列出全部自选股条目（真实 API：``Database.list_watch_items``）。"""
        return list(self._db.list_watch_items())

    def _display_text(self, symbol: str) -> str:
        """条目显示文案：``代码`` 或 ``代码 名称``（有名称解析器时）。"""
        name: Optional[str] = None
        if self._name_resolver is not None:
            try:
                name = self._name_resolver(symbol)
            except Exception as exc:  # noqa: BLE001 - 名称解析失败不影响展示
                logger.debug("name_resolver(%s) 失败：%s", symbol, exc)
        return f"{symbol} {name}" if name else str(symbol)

    def refresh(self) -> None:
        """从数据库重新填充列表；任何异常均被捕获并降级为空列表。

        会**保留当前选中条目**（按 ``(group_id, symbol)`` 重建后重新选中），
        且在重建期间抑制 ``symbolSelected`` 发射，避免刷新触发多余加载。
        """
        previous = self._capture_selection()
        self._updating = True
        try:
            self._tree.clear()
            groups = self._list_groups()
            items = self._list_items()

            if not groups and not items:
                placeholder = QTreeWidgetItem([_EMPTY_HINT])
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                self._tree.addTopLevelItem(placeholder)
                return

            by_group: dict = {}
            orphans: list = []
            known_ids = {g.id for g in groups}
            for item in items:
                if item.group_id in known_ids:
                    by_group.setdefault(item.group_id, []).append(item)
                else:
                    orphans.append(item)

            for group in groups:
                node = QTreeWidgetItem([str(group.name)])
                node.setFlags(Qt.ItemFlag.ItemIsEnabled)
                node.setData(0, _SYMBOL_ROLE, None)
                node.setData(0, _GROUP_ROLE, int(group.id))
                for item in by_group.get(group.id, []):
                    node.addChild(
                        self._make_symbol_item(
                            item.symbol, group_id=int(group.id), note_mark=bool(item.note_mark)
                        )
                    )
                self._tree.addTopLevelItem(node)
                node.setExpanded(True)

            for item in orphans:
                self._tree.addTopLevelItem(
                    self._make_symbol_item(
                        item.symbol, group_id=item.group_id, note_mark=bool(item.note_mark)
                    )
                )

            self._restore_selection(previous)
        except Exception as exc:  # noqa: BLE001 - 优雅降级
            logger.warning("刷新自选股列表失败：%s", exc)
            try:
                self._tree.clear()
                placeholder = QTreeWidgetItem([_EMPTY_HINT])
                placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                self._tree.addTopLevelItem(placeholder)
            except Exception:  # noqa: BLE001  # pragma: no cover - 极端防御
                pass
        finally:
            self._updating = False

    def _capture_selection(self) -> Optional[tuple]:
        """捕获当前选中股票条目的 ``(symbol, group_id)``；无选中/非股票返回 ``None``。"""
        item = self._tree.currentItem()
        if item is None:
            return None
        symbol = item.data(0, _SYMBOL_ROLE)
        if not symbol:
            return None
        return (str(symbol), item.data(0, _GROUP_ROLE))

    def _restore_selection(self, previous: Optional[tuple]) -> None:
        """按 ``(symbol, group_id)`` 重建后重新选中（调用时 ``_updating`` 为真，不发信号）。"""
        if not previous:
            return
        symbol, group_id = previous
        for item in self._symbol_items():
            if str(item.data(0, _SYMBOL_ROLE)) == symbol and item.data(0, _GROUP_ROLE) == group_id:
                self._tree.setCurrentItem(item)
                self._tree.scrollToItem(item)
                return


    def _make_symbol_item(
        self, symbol: str, group_id: Optional[int] = None, note_mark: bool = False
    ) -> QTreeWidgetItem:
        """构造一个股票条目（携带 symbol / group_id / note_mark 数据）。"""
        text = self._display_text(symbol)
        if note_mark:
            text = f"{text}{_NOTE_SUFFIX}"
        node = QTreeWidgetItem([text])
        node.setData(0, _SYMBOL_ROLE, str(symbol))
        node.setData(0, _GROUP_ROLE, group_id)
        node.setData(0, _NOTE_ROLE, bool(note_mark))
        return node

    # ------------------------------------------------------------------ #
    # 导航（P2-S5：语义保持不变）
    # ------------------------------------------------------------------ #
    def _symbol_items(self) -> list:
        """按显示顺序返回全部**股票条目**（跳过顶层分组头与占位项）。"""
        items: list = []
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            top = root.child(i)
            if top.data(0, _SYMBOL_ROLE):
                items.append(top)  # 无分组的顶层股票条目
                continue
            for j in range(top.childCount()):
                child = top.child(j)
                if child.data(0, _SYMBOL_ROLE):
                    items.append(child)
        return items

    def symbols(self) -> list:
        """返回全部自选股代码（按显示顺序，供测试）。"""
        return [str(item.data(0, _SYMBOL_ROLE)) for item in self._symbol_items()]

    def current_symbol(self) -> Optional[str]:
        """返回当前选中股票条目的代码；分组头/占位/未选中时返回 ``None``。"""
        item = self._tree.currentItem()
        if item is None:
            return None
        symbol = item.data(0, _SYMBOL_ROLE)
        return str(symbol) if symbol else None

    def current_group_id(self) -> Optional[int]:
        """返回当前选中项所属分组 id；无选中/占位项返回 ``None``。

        用于「＋ 添加自选」落到**当前选中**分组：选中某分组头或组内股票条目时取所在组；
        无选中 / 空库占位项（``NoItemFlags``）/ 分组数据缺失（如无分组的孤儿条目）返回 ``None``，
        由 :meth:`add_symbol` 走旧逻辑（第一个分组 / 无分组则自动建默认组）。

        Returns:
            分组 id（``int``），或 ``None``。
        """
        item = self._tree.currentItem()
        if item is None:
            return None
        gid = item.data(0, _GROUP_ROLE)
        return int(gid) if gid is not None else None

    def _activate(self, item) -> Optional[str]:
        """把 ``item`` 设为当前项（触发既有 ``symbolSelected`` 链路）并返回其代码。

        若 ``item`` 已是当前项则不重设（避免端点处重复触发信号）。
        """
        if item is None:
            return None
        if item is not self._tree.currentItem():
            self._tree.setCurrentItem(item)
        self._tree.scrollToItem(item)
        symbol = item.data(0, _SYMBOL_ROLE)
        return str(symbol) if symbol else None

    def select_next(self) -> Optional[str]:
        """在**股票条目**之间后移一格并选中（触发 ``symbolSelected``）。

        语义：**到端点不回绕**——已在最后一只时停在原地；
        当前无选中或选中非股票项时，落到第一只；无股票时不动、不抛。
        """
        items = self._symbol_items()
        if not items:
            return None
        current = self._tree.currentItem()
        index = items.index(current) if current in items else -1
        target = items[0] if index < 0 else items[min(index + 1, len(items) - 1)]
        return self._activate(target)

    def select_previous(self) -> Optional[str]:
        """在**股票条目**之间前移一格并选中（触发 ``symbolSelected``）。

        语义：**到端点不回绕**——已在第一只时停在原地；
        当前无选中或选中非股票项时，落到第一只；无股票时不动、不抛。
        """
        items = self._symbol_items()
        if not items:
            return None
        current = self._tree.currentItem()
        index = items.index(current) if current in items else -1
        target = items[0] if index < 0 else items[max(index - 1, 0)]
        return self._activate(target)

    # ------------------------------------------------------------------ #
    # 业务 API（P2-S6：无弹窗纯方法，可被测试直接调用）
    # ------------------------------------------------------------------ #
    def add_symbol(self, symbol: str, group_id: Optional[int] = None) -> bool:
        """添加一只自选股并持久化。

        Args:
            symbol: 6 位股票代码；**必须匹配 ``^[0-9]{6}$``**，否则返回 ``False`` 且不落库。
            group_id: 目标分组（必须是 ``int`` 或 ``None``）；``None`` 时用第一个分组
                （**无任何分组才自动建默认组**「我的自选」）。

        Returns:
            成功 ``True``；代码非法 / group_id 型别非法 / 指定分组不存在 / 落库异常均 ``False``。
        """
        # 防御：bool 是 int 子类且 int(True)==1，若被信号误传会静默落到分组 id=1；
        # 非 int 的 group_id（如 str"1"、bool）一律拒绝，让同类问题立刻暴露而非法分组不报错。
        # 必须先判 bool：isinstance(True, int) 为 True，顺序颠倒会放过 True。
        if group_id is not None and (isinstance(group_id, bool) or not isinstance(group_id, int)):
            logger.warning("add_symbol 收到非法型别的 group_id：%r（疑似信号误传）", group_id)
            return False
        code = str(symbol).strip()
        if not _SYMBOL_RE.match(code):
            logger.warning("非法股票代码：%r", symbol)
            return False
        target = group_id
        if target is None:
            groups = self._list_groups()
            if groups:
                target = int(groups[0].id)
            else:
                try:
                    target = int(self._db.create_group(_DEFAULT_GROUP_NAME, 0))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("创建默认分组失败：%s", exc)
                    return False
                self._publish_changed("add", "", target, kind="group")
        else:
            known = {int(g.id) for g in self._list_groups()}
            if int(target) not in known:
                logger.warning("指定分组不存在：%s", target)
                return False
        try:
            self._db.add_watch_item(int(target), code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("添加自选股失败：%s -> %s", code, exc)
            return False
        self.refresh()
        self._select_symbol(code, int(target))
        self._publish_changed("add", code, int(target))
        return True

    def remove_symbol(self, group_id: int, symbol: str) -> bool:
        """从指定分组移除一只自选股。

        Args:
            group_id: 分组 id。
            symbol: 股票代码。

        Returns:
            成功移除 ``True``；条目不存在 / 异常 ``False``。
        """
        try:
            existing = {it.symbol for it in self._db.list_watch_items(int(group_id))}
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取自选股失败：%s", exc)
            return False
        if str(symbol) not in existing:
            return False
        try:
            self._db.remove_watch_item(int(group_id), str(symbol))
        except Exception as exc:  # noqa: BLE001
            logger.warning("移除自选股失败：%s -> %s", symbol, exc)
            return False
        self.refresh()
        self._publish_changed("remove", str(symbol), int(group_id))
        return True

    def remove_current(self) -> bool:
        """移除**当前选中**条目；当前项是分组头/占位/无选中 → 返回 ``False``。"""
        item = self._tree.currentItem()
        if item is None:
            return False
        symbol = item.data(0, _SYMBOL_ROLE)
        group_id = item.data(0, _GROUP_ROLE)
        if not symbol or group_id is None:
            return False
        return self.remove_symbol(int(group_id), str(symbol))

    def toggle_note_mark(
        self, group_id: Optional[int] = None, symbol: Optional[str] = None
    ) -> bool:
        """切换备注标记（``note_mark``）并持久化。

        参数缺省时作用于**当前选中**条目。无选中 / 条目不存在 → ``False``。

        Args:
            group_id: 分组 id；``None`` 时取当前选中条目。
            symbol: 代码；``None`` 时取当前选中条目。

        Returns:
            成功 ``True``。
        """
        if symbol is None or group_id is None:
            item = self._tree.currentItem()
            if item is None:
                return False
            sym = item.data(0, _SYMBOL_ROLE)
            gid = item.data(0, _GROUP_ROLE)
            if not sym or gid is None:
                return False
            symbol = str(sym)
            group_id = int(gid)
        try:
            current: Optional[bool] = None
            for it in self._db.list_watch_items(int(group_id)):
                if it.symbol == str(symbol):
                    current = bool(it.note_mark)
                    break
            if current is None:
                return False
            new_mark = not current
            self._db.set_note_mark(int(group_id), str(symbol), new_mark)
        except Exception as exc:  # noqa: BLE001
            logger.warning("切换备注标记失败：%s -> %s", symbol, exc)
            return False
        self.refresh()
        self._publish_changed("add", str(symbol), int(group_id), extra={"note_mark": new_mark})
        return True

    def set_note_mark(
        self, group_id: int, symbol: str, mark: bool
    ) -> bool:
        """**显式设置**备注标记（与 :meth:`toggle_note_mark` 相对，供详情弹窗联动 ★）。

        保存正文 → 传 ``True``；清除正文 → 传 ``False``。条目不存在 / 写库失败 → ``False``。

        Args:
            group_id: 分组 id。
            symbol: 代码。
            mark: 目标标记值。

        Returns:
            成功 ``True``。
        """
        try:
            self._db.set_note_mark(int(group_id), str(symbol), bool(mark))
        except Exception as exc:  # noqa: BLE001
            logger.warning("设置备注标记失败：%s -> %s", symbol, exc)
            return False
        self.refresh()
        self._publish_changed("add", str(symbol), int(group_id), extra={"note_mark": bool(mark)})
        return True

    def create_group(self, name: str) -> Optional[int]:
        """新建分组（``sort_order`` = 现有最大值 + 1）。

        Args:
            name: 分组名；空 / 纯空白返回 ``None`` 且不落库。

        Returns:
            新分组 id；失败 ``None``。
        """
        clean = str(name).strip()
        if not clean:
            return None
        try:
            orders = [int(g.sort_order) for g in self._list_groups()]
            sort_order = (max(orders) + 1) if orders else 0
            gid = int(self._db.create_group(clean, sort_order))
        except Exception as exc:  # noqa: BLE001
            logger.warning("新建分组失败：%s -> %s", name, exc)
            return None
        self.refresh()
        self._publish_changed("add", "", gid, kind="group")
        return gid

    def rename_group(self, group_id: int, name: str) -> bool:
        """重命名分组。

        Args:
            group_id: 分组 id。
            name: 新名称；空 / 纯空白 / 分组不存在返回 ``False``。

        Returns:
            成功 ``True``。
        """
        clean = str(name).strip()
        if not clean:
            return False
        if int(group_id) not in {int(g.id) for g in self._list_groups()}:
            return False
        try:
            self._db.rename_group(int(group_id), clean)
        except Exception as exc:  # noqa: BLE001
            logger.warning("重命名分组失败：%s -> %s", group_id, exc)
            return False
        self.refresh()
        self._publish_changed("add", "", int(group_id), kind="group", extra={"renamed": True})
        return True

    def delete_group(self, group_id: int) -> bool:
        """删除分组（其下条目经外键 ``ON DELETE CASCADE`` 一并删除）。

        Args:
            group_id: 分组 id。

        Returns:
            成功 ``True``；分组不存在 / 异常 ``False``。
        """
        if int(group_id) not in {int(g.id) for g in self._list_groups()}:
            return False
        try:
            self._db.delete_group(int(group_id))  # 级联由 schema 外键负责，勿手写级联
        except Exception as exc:  # noqa: BLE001
            logger.warning("删除分组失败：%s -> %s", group_id, exc)
            return False
        self.refresh()
        self._publish_changed("remove", "", int(group_id), kind="group")
        return True

    # ------------------------------------------------------------------ #
    # 事件发布
    # ------------------------------------------------------------------ #
    def _publish_changed(
        self,
        action: str,
        symbol: str,
        group_id: Optional[int],
        kind: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """发布 ``WATCHLIST_CHANGED``（**绝不抛出**）。

        payload 基础形如 ``{"action": "add"|"remove", "symbol": str, "group_id": int}``；
        分组操作时 ``symbol=""`` 且附 ``"kind": "group"``。
        """
        payload: dict = {"action": action, "symbol": symbol, "group_id": group_id}
        if kind is not None:
            payload["kind"] = kind
        if extra:
            payload.update(extra)
        try:
            get_bus().publish(WATCHLIST_CHANGED, payload)
        except Exception:  # noqa: BLE001 - 发布不得影响业务
            logger.exception("发布 WATCHLIST_CHANGED 失败：%s", payload)

    def _select_symbol(self, symbol: str, group_id: Optional[int] = None) -> Optional[str]:
        """在树中定位并选中 ``(group_id, symbol)`` 条目（触发 ``symbolSelected``）。"""
        for item in self._symbol_items():
            if str(item.data(0, _SYMBOL_ROLE)) != str(symbol):
                continue
            if group_id is None or item.data(0, _GROUP_ROLE) == int(group_id):
                return self._activate(item)
        return None

    # ------------------------------------------------------------------ #
    # 交互：选中信号
    # ------------------------------------------------------------------ #
    def _current_symbol(self) -> Optional[str]:
        """返回当前选中条目的 symbol（分组/占位项返回 ``None``）。"""
        item = self._tree.currentItem()
        if item is None:
            return None
        symbol = item.data(0, _SYMBOL_ROLE)
        return str(symbol) if symbol else None

    def _emit_selected(self) -> None:
        """若当前项为股票条目，发出 ``symbolSelected``。"""
        if self._updating:
            return
        symbol = self._current_symbol()
        if symbol:
            logger.info("选中自选股：%s", symbol)
            self.symbolSelected.emit(symbol)

    def _on_selection_changed(self) -> None:
        """选中变化时发出信号。"""
        self._emit_selected()

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """点击条目时发出信号（保证单击即触发）。"""
        symbol = item.data(0, _SYMBOL_ROLE)
        if symbol and not self._updating:
            self.symbolSelected.emit(str(symbol))

    # ------------------------------------------------------------------ #
    # UI 触达层：顶部按钮 + 右键菜单（弹窗后调用业务 API）
    # ------------------------------------------------------------------ #
    def _on_add_symbol_action(self, group_id: Optional[int] = None) -> None:
        """弹输入框输入 6 位代码并添加；失败按原因给可见反馈。"""
        text, ok = QInputDialog.getText(self, "添加自选", "输入 6 位股票代码：")
        if not ok:
            return
        code = str(text).strip()
        if not is_valid_symbol(code):
            QMessageBox.warning(
                self, "添加失败", f"代码格式不正确：{text!r}（需 6 位半角数字，请勿使用全角数字）"
            )
            return
        if not self.add_symbol(code, group_id):
            QMessageBox.warning(self, "添加失败", f"添加 {code} 失败（目标分组不存在或写入异常）")

    def _on_new_group_action(self) -> None:
        """弹输入框输入组名并新建分组。"""
        text, ok = QInputDialog.getText(self, "新建分组", "输入分组名称：")
        if not ok:
            return
        if self.create_group(text) is None:
            QMessageBox.warning(self, "新建失败", "分组名不能为空。")

    def _on_rename_group_action(self, group_id: int) -> None:
        """弹输入框重命名分组。"""
        current = ""
        for group in self._list_groups():
            if int(group.id) == int(group_id):
                current = str(group.name)
                break
        text, ok = QInputDialog.getText(self, "重命名分组", "新名称：", text=current)
        if not ok:
            return
        if not self.rename_group(group_id, text):
            QMessageBox.warning(self, "重命名失败", "名称不能为空。")

    def _on_delete_group_action(self, group_id: int) -> None:
        """确认后删除分组（提示会删掉组内 N 只）。"""
        count = sum(1 for it in self._list_items() if int(it.group_id) == int(group_id))
        answer = QMessageBox.question(
            self,
            "删除分组",
            f"确定删除该分组吗？将同时删除组内 {count} 只自选股。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self.delete_group(group_id):
            QMessageBox.warning(self, "删除失败", "分组不存在或删除失败。")

    def _on_remove_symbol_action(self, group_id: int, symbol: str) -> None:
        """确认后移除某只自选股。"""
        answer = QMessageBox.question(self, "移除自选", f"确定从自选移除 {symbol} 吗？")
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self.remove_symbol(group_id, symbol):
            QMessageBox.warning(self, "移除失败", "该条目不存在。")

    def _on_toggle_note_action(self, group_id: int, symbol: str) -> None:
        """切换备注标记。"""
        if not self.toggle_note_mark(group_id, symbol):
            QMessageBox.warning(self, "操作失败", "无法切换备注标记。")

    def _on_context_menu(self, pos: QPoint) -> None:
        """右键菜单：按点击对象（股票 / 分组头 / 空白）构造不同菜单。"""
        item = self._tree.itemAt(pos)
        menu = QMenu(self)
        symbol = item.data(0, _SYMBOL_ROLE) if item is not None else None
        group_id = item.data(0, _GROUP_ROLE) if item is not None else None

        if item is not None and symbol:
            # 股票条目：动作均已接线（P4-S5 抽出为可测方法）。
            self._build_stock_menu(item).exec(self._tree.viewport().mapToGlobal(pos))
            return

        if item is not None and group_id is not None:
            # 分组头
            act_add = menu.addAction("添加到自选…")
            act_rename = menu.addAction("重命名分组…")
            act_delete = menu.addAction("删除分组")
            chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
            if chosen is act_add:
                self._on_add_symbol_action(int(group_id))
            elif chosen is act_rename:
                self._on_rename_group_action(int(group_id))
            elif chosen is act_delete:
                self._on_delete_group_action(int(group_id))
            return

        # 空白 / 占位项
        act_new = menu.addAction("新建分组…")
        chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if chosen is act_new:
            self._on_new_group_action()

    def _build_stock_menu(self, item: QTreeWidgetItem) -> QMenu:
        """构造股票条目右键菜单并**接线动作**（P4-S5 抽出，供测试真实触发动作）。

        动作顺序固定：``备注/详情… → 添加到自选… → 从自选移除 → (取消)标记备注``。
        ⚠️ ``QAction.triggered`` 自带 ``bool checked`` 参数，槽一律用 ``lambda`` 吞掉
        （§4.11 信号签名铁律）。

        Args:
            item: 股票条目（携带 ``_SYMBOL_ROLE`` / ``_GROUP_ROLE`` / ``_NOTE_ROLE``）。

        Returns:
            已接线动作的 :class:`QMenu`。
        """
        symbol = str(item.data(0, _SYMBOL_ROLE))
        group_id = int(item.data(0, _GROUP_ROLE))
        menu = QMenu(self)

        act_detail = menu.addAction("备注/详情…")
        act_detail.triggered.connect(lambda: self.detailRequested.emit(group_id, symbol))

        act_alert = menu.addAction("设置预警…")
        act_alert.triggered.connect(lambda: self.alertRequested.emit(group_id, symbol))

        act_add = menu.addAction("添加到自选…")
        act_add.triggered.connect(lambda: self._on_add_symbol_action(group_id))

        act_remove = menu.addAction("从自选移除")
        act_remove.triggered.connect(lambda: self._on_remove_symbol_action(group_id, symbol))

        note_mark = bool(item.data(0, _NOTE_ROLE))
        act_note = menu.addAction("取消备注" if note_mark else "标记备注")
        act_note.triggered.connect(lambda: self._on_toggle_note_action(group_id, symbol))
        return menu

    # ------------------------------------------------------------------ #
    # 方向键导航（事件过滤器，P2-S5 回炉修复）
    # ------------------------------------------------------------------ #
    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001 - Qt 约定
        """拦截自选股树上的 ``↑`` / ``↓``，切换上一只 / 下一只股票。

        为什么用事件过滤器而非 :meth:`keyPressEvent`：焦点位于内部
        :class:`QTreeWidget` 上，其 ``keyPress`` 事件**不会冒泡**到本容器父控件，
        必须在树上安装过滤器才能截获；否则方向键无法处理。

        处理时返回 ``True``（吃掉事件），避免 ``QTreeWidget`` 再做一次原生移动
        （那会导致一次按键跳两格）。

        Args:
            obj: 事件发送对象。
            event: 事件对象。

        Returns:
            已处理返回 ``True``，否则交回 ``super()``。
        """
        if obj is self._tree and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Down:
                self.select_next()
                return True
            if key == Qt.Key.Key_Up:
                self.select_previous()
                return True
        return super().eventFilter(obj, event)


__all__ = ["WatchlistPanel", "WATCHLIST_CHANGED", "is_valid_symbol"]
