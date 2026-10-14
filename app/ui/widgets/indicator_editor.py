"""自定义指标编辑器（P3-S3）。

定位：**模态对话框**，让用户：
    * 编写指标表达式（如 ``close.rolling(20).mean()``）；
    * 选样本股 + 周期，用 :class:`~app.engine.indicator.IndicatorEngine` 预览曲线；
    * 保存为「名称 + 分类」模板，并管理（列表 / 载入 / 删除）。

架构约定（沿用 ``watchlist_panel`` 的可测性思路）：
    * **计算 / 落库动作 = 无弹窗的纯方法**（:meth:`preview_result` / :meth:`save_template` /
      :meth:`delete_template`），测试可直接调用并断言；
    * **UI 触达层**（预览 / 保存 / 删除按钮）负责把结果显示到状态标签 / 图表；
    * **绝不向 UI 抛异常**：``calc`` 与 DB 异常一律捕获，状态标签显示 ``error_type`` + 原因。

关键契约：
    * 构造函数接收 ``symbols``（样本股候选）、``db``、``engine`` 依赖注入，默认取全局单例；
    * 只读访问器 ``current_expr()/current_name()/current_category()/
      selected_template_id()/current_symbol()/current_period()`` 供测试；
    * ``preview_result() -> IndicatorResult`` 供测试直接断言预览结果。
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.data.resample import can_derive, derive_bars
from app.data.store.db import Database, get_db
from app.engine.indicator import IndicatorEngine, IndicatorResult, get_engine
from app.ui.themes import colors

logger = logging.getLogger(__name__)

# 默认示例表达式（首次打开即给出可预览的合法表达式）。
_DEFAULT_EXPR: str = "close.rolling(20).mean()"

# 预置分类项（可自由输入新分类）。
_PRESET_CATEGORIES: tuple[str, ...] = ("均线", "趋势", "摆动", "成交量", "自定义")

# 可预览周期（周/月线无本地数据时由日线派生，见 :mod:`app.data.resample`）。
_PERIODS: tuple[tuple[str, str], ...] = (("d", "日线"), ("w", "周线"), ("m", "月线"))

# 模板列表项里存放模板 id 的数据角色。
_TEMPLATE_ROLE: int = int(Qt.ItemDataRole.UserRole)

# 空库 / 空数据降级文案。
_NO_BARS_TEXT: str = "无本地 K 线，请先更新数据"

# 对话框标题。
WINDOW_TITLE: str = "自定义指标编辑器"


class IndicatorEditorDialog(QDialog):
    """自定义指标编辑器对话框。"""

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        db: Optional[Database] = None,
        engine: Optional[IndicatorEngine] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            symbols: 样本股候选代码列表；``None`` / 空列表时兜底 ``["000001"]``。
            db: 存储实例；默认 :func:`get_db`。
            engine: 指标引擎；默认 :func:`get_engine`。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._engine: IndicatorEngine = engine if engine is not None else get_engine()

        symbols = [str(s).strip() for s in (symbols or []) if str(s).strip()]
        self._symbols: list[str] = symbols if symbols else ["000001"]

        # 当前正在编辑的模板 id（``None`` 表示新建；载入 / 保存后变为对应 id）。
        self._template_id: Optional[int] = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(760, 680)
        self._setup_ui()
        self._refresh_list()
        self._set_status("输入表达式后点击『预览』")

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建表达式区 / 元信息区 / 样本预览区 / 保存管理区。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # (a) 表达式区
        root.addWidget(QLabel("指标表达式：", self))
        self.expr_edit = QPlainTextEdit(self)
        self.expr_edit.setPlainText(_DEFAULT_EXPR)
        self.expr_edit.setMinimumHeight(64)
        root.addWidget(self.expr_edit)

        # (b) 元信息区
        meta = QHBoxLayout()
        meta.setSpacing(6)
        meta.addWidget(QLabel("名称：", self))
        self.name_edit = QLineEdit(self)
        self.name_edit.setPlaceholderText("如：20 日均线")
        meta.addWidget(self.name_edit, 1)
        meta.addWidget(QLabel("分类：", self))
        self.category_combo = QComboBox(self)
        self.category_combo.setEditable(True)
        for cat in _PRESET_CATEGORIES:
            self.category_combo.addItem(cat)
        meta.addWidget(self.category_combo, 1)
        root.addLayout(meta)

        # (c) 样本股预览区
        preview_bar = QHBoxLayout()
        preview_bar.setSpacing(6)
        preview_bar.addWidget(QLabel("样本股：", self))
        self.symbol_combo = QComboBox(self)
        for symbol in self._symbols:
            self.symbol_combo.addItem(symbol)
        preview_bar.addWidget(self.symbol_combo, 1)
        preview_bar.addWidget(QLabel("周期：", self))
        self.period_combo = QComboBox(self)
        for code, label in _PERIODS:
            self.period_combo.addItem(label, code)
        preview_bar.addWidget(self.period_combo, 1)
        self.preview_button = QPushButton("预览", self)
        self.preview_button.clicked.connect(self._on_preview)
        preview_bar.addWidget(self.preview_button)
        root.addLayout(preview_bar)

        self.status_label = QLabel("", self)
        root.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget(parent=self)
        self.plot_widget.setBackground(colors.BG_DARK)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self.plot_widget.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)
        self.curve = self.plot_widget.plot(
            [], [], pen=pg.mkPen(colors.ACCENT_COLOR, width=1.4)
        )
        root.addWidget(self.plot_widget, 1)

        # (d) 保存 / 管理区
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.save_button = QPushButton("保存", self)
        self.save_button.clicked.connect(self._on_save)
        self.delete_button = QPushButton("删除", self)
        self.delete_button.clicked.connect(self._on_delete)
        actions.addWidget(self.save_button)
        actions.addWidget(self.delete_button)
        actions.addStretch(1)
        root.addLayout(actions)

        root.addWidget(QLabel("已保存模板：", self))
        self.template_list = QListWidget(self)
        self.template_list.currentItemChanged.connect(self._on_template_selected)
        root.addWidget(self.template_list, 1)

    # ------------------------------------------------------------------ #
    # 只读访问器（供测试 / 上层）
    # ------------------------------------------------------------------ #
    def current_expr(self) -> str:
        """返回当前表达式文本（已 ``strip``）。"""
        return self.expr_edit.toPlainText().strip()

    def current_name(self) -> str:
        """返回当前指标名称（已 ``strip``）。"""
        return self.name_edit.text().strip()

    def current_category(self) -> str:
        """返回当前分类文本（已 ``strip``）。"""
        return self.category_combo.currentText().strip()

    def current_symbol(self) -> str:
        """返回当前样本股代码。"""
        return self.symbol_combo.currentText().strip()

    def current_period(self) -> str:
        """返回当前周期编码（``d`` / ``w`` / ``m``）。"""
        return str(self.period_combo.currentData() or "d")

    def selected_template_id(self) -> Optional[int]:
        """返回模板列表当前选中项的模板 id；未选中返回 ``None``。"""
        item = self.template_list.currentItem()
        if item is None:
            return None
        value = item.data(_TEMPLATE_ROLE)
        return int(value) if value is not None else None

    # ------------------------------------------------------------------ #
    # 预览（无弹窗纯方法 + UI 触达层）
    # ------------------------------------------------------------------ #
    def _load_bars(self, symbol: str, period: str) -> pd.DataFrame:
        """读取样本股本地 K 线；周/月线本地为空时由日线派生（**不抛异常**）。"""
        try:
            frame = self._db.get_bars(symbol, period)
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取本地 K 线失败：%s -> %s", symbol, exc)
            return pd.DataFrame()

        if (frame is None or frame.empty) and can_derive(period):
            try:
                daily = self._db.get_bars(symbol, "d")
                return derive_bars(daily, period)
            except Exception as exc:  # noqa: BLE001 - 派生失败降级
                logger.warning("派生周月线失败：%s -> %s", symbol, exc)
                return pd.DataFrame()

        return frame if frame is not None else pd.DataFrame()

    def preview_result(self) -> IndicatorResult:
        """按当前表达式 + 样本股 + 周期计算预览结果（**不抛异常**）。

        Returns:
            :class:`IndicatorResult`：成功时 ``ok=True`` 且 ``series`` 为与 bars 等长的
            Series；空表达式 / 空数据 / 计算错误时 ``ok=False`` 且带可读 ``error``。
        """
        expr = self.current_expr()
        if not expr:
            return IndicatorResult.failure("表达式不能为空")
        symbol = self.current_symbol()
        period = self.current_period()
        bars = self._load_bars(symbol, period)
        if bars is None or bars.empty:
            return IndicatorResult.failure(_NO_BARS_TEXT)
        return self._engine.calc(
            expr, bars, ctx={"symbol": symbol, "period": period, "n": 20}
        )

    def _on_preview(self) -> None:
        """预览按钮：计算并把结果画到 ``PlotWidget`` / 状态标签。"""
        result = self.preview_result()
        if result.ok:
            series = result.series
            if series is not None and len(series):
                self._render_curve(series)
                self._set_status(f"已生成 {len(series)} 点曲线", ok=True)
            else:
                self.curve.setData([], [])
                self._set_status(f"结果：{result.data}", ok=True)
            return
        self.curve.setData([], [])
        if result.error_type:
            self._set_status(f"{result.error_type}：{result.error}", ok=False)
        else:
            self._set_status(result.error, ok=False)

    def _render_curve(self, series: pd.Series) -> None:
        """把 Series 画成曲线（X 为序号，NaN 由 pyqtgraph 跳过）。"""
        values = pd.to_numeric(series, errors="coerce")
        xs = list(range(len(values)))
        ys = [float(v) if not pd.isna(v) else float("nan") for v in values]
        self.curve.setData(xs, ys)

    def _set_status(self, text: str, ok: bool = True) -> None:
        """更新状态标签（成功绿色 / 失败红色，沿用主题红涨绿跌但语义为状态）。"""
        color = colors.DOWN_COLOR if ok else colors.UP_COLOR
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)

    # ------------------------------------------------------------------ #
    # 保存 / 删除（无弹窗纯方法 + UI 触达层）
    # ------------------------------------------------------------------ #
    def save_template(self) -> Optional[int]:
        """校验并保存当前编辑内容，返回模板 id；校验失败 / 异常返回 ``None``。

        名称 / 表达式为空时返回 ``None``（不落库）；正在编辑已有模板（``_template_id``
        非空）时走更新（同 id 覆盖）。
        """
        name = self.current_name()
        expr = self.current_expr()
        category = self.current_category()
        if not name:
            self._set_status("指标名称不能为空", ok=False)
            return None
        if not expr:
            self._set_status("表达式不能为空", ok=False)
            return None
        # 保存前是否处于「编辑已有模板」态：只有真正走更新时才保留/刷新 ``_template_id``；
        # 新建成功后保持 ``None``，否则下一次保存会被误判为更新并静默覆盖上一个模板。
        was_update = self._template_id is not None
        try:
            template_id = int(
                self._db.save_indicator_template(name, category, expr, self._template_id)
            )
        except Exception as exc:  # noqa: BLE001 - 落库失败不外抛
            logger.exception("保存指标模板失败")
            self._set_status(f"保存失败：{exc}", ok=False)
            return None
        self._template_id = template_id if was_update else None
        self._refresh_list()
        self._set_status(f"已保存：{name} [{category or '未分类'}]", ok=True)
        return template_id

    def delete_template(self) -> bool:
        """删除当前选中的模板；未选中 / 删除异常返回 ``False``。"""
        template_id = self.selected_template_id()
        if template_id is None:
            self._set_status("未选中模板", ok=False)
            return False
        try:
            self._db.delete_indicator_template(template_id)
        except Exception as exc:  # noqa: BLE001 - 删除失败不外抛
            logger.exception("删除指标模板失败")
            self._set_status(f"删除失败：{exc}", ok=False)
            return False
        self._template_id = None
        self._refresh_list()
        self._set_status("已删除模板", ok=True)
        return True

    def _on_save(self) -> None:
        """保存按钮。"""
        self.save_template()

    def _on_delete(self) -> None:
        """删除按钮。"""
        self.delete_template()

    # ------------------------------------------------------------------ #
    # 模板列表
    # ------------------------------------------------------------------ #
    def _refresh_list(self) -> None:
        """从数据库重新填充模板列表（显示「名称 [分类]」，条目带模板 id）。"""
        self.template_list.clear()
        try:
            templates = self._db.list_indicator_templates()
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取指标模板失败：%s", exc)
            return
        for template in templates:
            item = QListWidgetItem(
                f"{template.name} [{template.category or '未分类'}]"
            )
            item.setData(_TEMPLATE_ROLE, int(template.id))
            self.template_list.addItem(item)

    def _on_template_selected(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        """选中模板 → 载入到编辑区（名称 / 分类 / 表达式），并记录 ``_template_id``。"""
        if current is None:
            return
        template_id = current.data(_TEMPLATE_ROLE)
        if template_id is None:
            return
        template = None
        try:
            for tpl in self._db.list_indicator_templates():
                if int(tpl.id) == int(template_id):
                    template = tpl
                    break
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("载入模板失败：%s", exc)
            return
        if template is None:
            return
        self._template_id = int(template.id)
        self.name_edit.setText(template.name)
        self.expr_edit.setPlainText(template.expr)
        self.category_combo.setCurrentText(template.category or "")


__all__ = ["IndicatorEditorDialog", "WINDOW_TITLE"]
