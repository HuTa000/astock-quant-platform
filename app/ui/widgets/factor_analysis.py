"""因子分析对话框（P7-S3）。

定位：**模态对话框**，把 P7-S3 多因子有效性分析引擎的产出落到界面：
    * 配置分析范围（因子勾选 / 预测期 / 分层层数 / 最少标的数 / 起止日期）；
    * 运行分析（:class:`~app.engine.factor.analysis.FactorAnalyzer`），后台线程执行；
    * 展示：IC 汇总表、IC 衰减表、IC 时序图、分层净值图、分层统计表、
      共线性（相关矩阵 + VIF）、警告与状态。

架构约定（与 ``strategy_analysis`` 同构）：
    * **业务动作 = 无弹窗的纯方法**（:meth:`current_spec` / :meth:`run_analysis`），
      测试可直接调用并断言（注入 fake 分析器时无线程、无网络）；
    * **UI 触达层**（运行按钮）经 :class:`_FactorWorker`（``QThread``）后台执行，
      结果经 Qt 信号回 UI 线程刷新，**工作线程不碰任何 UI**；
    * **绝不向 UI 抛异常**：引擎 / 绘图异常一律捕获，状态标签显示可读原因；
    * 本对话框**只读**本地库（分析引擎不写库），不做任何数据更新。

口径提示（展示给用户）：
    * 因子用 T 日及以前的数据计算，前瞻收益从 **T+1 开盘**起算（规避前视偏差）；
    * 统一前复权；股票池 = 本地库全部有前复权日线的标的（自动排除指数）；
    * 分层组合为每日轮动口径（与回测引擎一致）。
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.store.db import Database, get_db
from app.engine.factor import (
    DEFAULT_FACTOR_NAMES,
    DEFAULT_HORIZONS,
    FactorAnalysisSpec,
    FactorAnalyzer,
    FactorReport,
    ic_decay_table,
)
from app.engine.factor.factors import get_factor
from app.ui.themes import colors
from app.ui.widgets.kline_view import CompactAxisItem, DateAxisItem
from app.ui.widgets.strategy_analysis import (
    format_count,
    format_num,
    format_pct,
)

logger = logging.getLogger(__name__)

# 对话框标题。
WINDOW_TITLE: str = "因子分析"

# IC 汇总表列头。
_IC_HEADERS: tuple[str, ...] = (
    "因子",
    "中文名",
    "类别",
    "预测期",
    "有效天数",
    "IC 均值",
    "IC 标准差",
    "ICIR",
    "IC 胜率",
    "NW t 值",
    "p 值",
    "显著",
)

# 分层统计表列头。
_LAYER_HEADERS: tuple[str, ...] = (
    "层",
    "交易日",
    "日均收益",
    "区间收益",
    "年化收益",
    "年化波动",
    "夏普",
    "最大回撤",
)

# VIF 表列头。
_VIF_HEADERS: tuple[str, ...] = ("因子", "中文名", "VIF", "判定")

# 分层净值曲线配色（Q1 最低 → Q5 最高：冷 → 暖渐变）。
_GROUP_COLORS: tuple[str, ...] = (
    "#5d8aa8",
    "#5dade2",
    "#8e7cc3",
    "#c27ba0",
    "#e06666",
)

# 多空组合（QN − Q1）配色：金色虚线，与主题的强调色系区分。
_LONG_SHORT_COLOR: str = colors.LIMIT_COLOR


# --------------------------------------------------------------------------- #
# 格式化纯函数（模块级，供测试直接对拍）
# --------------------------------------------------------------------------- #
def format_ic(value: Any) -> str:
    """IC / ICIR / t 值 → 4 位小数文本；无效值 → ``"—"``。"""
    return format_num(value, digits=4)


def format_pvalue(value: Any) -> str:
    """p 值 → 4 位小数文本（``<0.0001`` 时显示 ``"<0.0001"``）；无效值 → ``"—"``。"""
    number = _finite(value)
    if number is None:
        return "—"
    if number < 0.0001:
        return "<0.0001"
    return f"{number:.4f}"


def format_flag(value: Any) -> str:
    """布尔标志 → ``"是"`` / ``"否"``。"""
    return "是" if bool(value) else "否"


def _finite(value: Any) -> Optional[float]:
    """尽力转 float；``None`` / 非数值 / NaN / ±inf → ``None``。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_horizons(text: str) -> list[int]:
    """把 ``"5, 10,20"`` 解析为 ``[5, 10, 20]``（去重升序）；非法项忽略。"""
    resolved: list[int] = []
    for token in re.split(r"[,，;；\s]+", str(text)):
        if not token.strip().isdigit():
            continue
        value = int(token.strip())
        if value >= 1 and value not in resolved:
            resolved.append(value)
    resolved.sort()
    return resolved


# --------------------------------------------------------------------------- #
# 后台线程
# --------------------------------------------------------------------------- #
class _FactorWorker(QThread):
    """在后台线程执行一次 :meth:`FactorAnalyzer.run`（结果经信号回传，不碰 UI）。"""

    done = Signal(object)  # FactorReport 或 Exception

    def __init__(
        self,
        analyzer: FactorAnalyzer,
        spec: FactorAnalysisSpec,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造后台分析线程（parent 默认 None，生命周期由 closeEvent 管理）。"""
        super().__init__(parent)
        self._analyzer = analyzer
        self._spec = spec

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：执行分析，成功 / 失败均经 ``done`` 信号回传。"""
        try:
            report = self._analyzer.run(self._spec)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台因子分析线程异常")
            self.done.emit(exc)
            return
        self.done.emit(report)


# --------------------------------------------------------------------------- #
# 对话框
# --------------------------------------------------------------------------- #
class FactorAnalysisDialog(QDialog):
    """因子分析对话框（配置 → 分析 → IC / 分层 / 共线性视图）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        analyzer: Optional[FactorAnalyzer] = None,
        symbols: Optional[list[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`（只读）。
            analyzer: 分析器；默认用 ``db`` 构造 :class:`FactorAnalyzer`。
            symbols: 股票池；``None`` = 本地库全部有前复权日线的标的（排除指数）。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._analyzer: FactorAnalyzer = (
            analyzer if analyzer is not None else FactorAnalyzer(db=self._db)
        )
        self._symbols: Optional[list[str]] = list(symbols) if symbols else None
        self._last_report: Optional[FactorReport] = None
        self._worker: Optional[_FactorWorker] = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1180, 880)
        self._setup_ui()
        self._set_status("勾选因子并设置参数后点击『运行分析』开始", ok=True)

    # ------------------------------------------------------------------ #
    # 线程生命周期（与 StrategyAnalysisDialog 同约定）
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭前等待后台分析线程自然结束，避免析构连带销毁运行中的 QThread。"""
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait()
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建「分析配置 / 曲线 / 明细标签页 / 警告」四段式布局。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        root.addWidget(self._build_config_box())
        root.addLayout(self._build_charts())
        root.addWidget(self._build_detail_tabs(), 1)

        self.warnings_label = QLabel("", self)
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setVisible(False)
        root.addWidget(self.warnings_label)

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    def _build_config_box(self) -> QGroupBox:
        """分析配置区：因子勾选 / 预测期 / 分层 / 闸门 / 起止日期 / 运行。"""
        box = QGroupBox("分析配置", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        hint = QLabel(
            "口径：T 日收盘算因子 → T+1 开盘买入 → T+1+N 开盘卖出（无未来函数）；"
            "前复权；股票池 = 本地库全部有前复权日线的标的（自动排除指数）；"
            "分层组合为每日轮动口径。IC = 横截面秩相关（因子值 vs 未来 N 日收益）。",
            box,
        )
        hint.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        body = QHBoxLayout()
        body.setSpacing(8)

        # 左：因子勾选列表（默认全选）。
        factor_column = QVBoxLayout()
        factor_column.setSpacing(2)
        factor_column.addWidget(QLabel("参与分析的因子（勾选）：", box))
        self.factor_list = QListWidget(box)
        self.factor_list.setMaximumHeight(150)
        for name in DEFAULT_FACTOR_NAMES:
            definition = get_factor(name)
            label = f"{definition.label}（{name}）" if definition else name
            item = QListWidgetItem(label, self.factor_list)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setCheckState(Qt.CheckState.Checked)
        factor_column.addWidget(self.factor_list)
        body.addLayout(factor_column, 3)

        # 右：参数。
        params_column = QVBoxLayout()
        params_column.setSpacing(4)

        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel("预测期（交易日）：", box))
        self.horizons_edit = QLineEdit(box)
        self.horizons_edit.setText(", ".join(str(item) for item in DEFAULT_HORIZONS))
        self.horizons_edit.setPlaceholderText("如：5, 10, 20, 60")
        horizon_row.addWidget(self.horizons_edit, 1)
        params_column.addLayout(horizon_row)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("分层层数：", box))
        self.groups_spin = QSpinBox(box)
        self.groups_spin.setRange(2, 10)
        self.groups_spin.setValue(5)
        shape_row.addWidget(self.groups_spin)
        shape_row.addWidget(QLabel("最少标的数：", box))
        self.min_stocks_spin = QSpinBox(box)
        self.min_stocks_spin.setRange(3, 500)
        self.min_stocks_spin.setValue(10)
        shape_row.addWidget(self.min_stocks_spin)
        shape_row.addStretch(1)
        params_column.addLayout(shape_row)

        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("起始日：", box))
        self.start_date_edit = QDateEdit(QDate(2015, 1, 1), box)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_row.addWidget(self.start_date_edit)
        date_row.addWidget(QLabel("结束日：", box))
        self.end_date_edit = QDateEdit(QDate.currentDate(), box)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        date_row.addWidget(self.end_date_edit)
        params_column.addLayout(date_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("运行分析", box)
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        params_column.addLayout(run_row)
        params_column.addStretch(1)

        body.addLayout(params_column, 4)
        layout.addLayout(body)
        return box

    def _build_charts(self) -> QVBoxLayout:
        """IC 时序图 + 分层净值图（各带因子 / 预测期选择器）。"""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # (a) IC 时序
        ic_bar = QHBoxLayout()
        ic_bar.setSpacing(6)
        ic_label = QLabel("IC 时序 —", self)
        ic_label.setStyleSheet(f"color: {colors.ACCENT_COLOR};")
        ic_bar.addWidget(ic_label)
        ic_bar.addWidget(QLabel("因子：", self))
        self.ic_factor_combo = QComboBox(self)
        self.ic_factor_combo.currentIndexChanged.connect(self._refresh_ic_chart)
        ic_bar.addWidget(self.ic_factor_combo)
        ic_bar.addWidget(QLabel("预测期：", self))
        self.ic_horizon_combo = QComboBox(self)
        self.ic_horizon_combo.currentIndexChanged.connect(self._refresh_ic_chart)
        ic_bar.addWidget(self.ic_horizon_combo)
        ic_bar.addStretch(1)
        layout.addLayout(ic_bar)
        self.ic_axis_label = ic_label

        self.ic_axis = DateAxisItem(orientation="bottom")
        self.ic_plot = pg.PlotWidget(
            axisItems={"left": CompactAxisItem(orientation="left"), "bottom": self.ic_axis},
            parent=self,
        )
        self.ic_plot.setBackground(colors.BG_DARK)
        self.ic_plot.showGrid(x=False, y=True, alpha=0.25)
        self.ic_plot.setMenuEnabled(False)
        self.ic_plot.setMouseEnabled(x=False, y=False)
        self.ic_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self.ic_plot.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)
        self.ic_plot.setMinimumHeight(150)
        self.ic_plot.setMaximumHeight(200)
        self.ic_zero_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(colors.SUBTEXT_COLOR, width=1))
        self.ic_plot.addItem(self.ic_zero_line)
        self.ic_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.ACCENT_COLOR, width=1.5),
            symbol="o", symbolSize=3, symbolBrush=colors.ACCENT_COLOR,
        )
        self.ic_plot.addItem(self.ic_curve)
        layout.addWidget(self.ic_plot)

        # (b) 分层净值
        layer_bar = QHBoxLayout()
        layer_bar.setSpacing(6)
        layer_label = QLabel("分层净值 —", self)
        layer_label.setStyleSheet(f"color: {colors.ACCENT_COLOR};")
        layer_bar.addWidget(layer_label)
        layer_bar.addWidget(QLabel("因子：", self))
        self.layer_factor_combo = QComboBox(self)
        self.layer_factor_combo.currentIndexChanged.connect(self._refresh_layer_chart)
        # 切换因子时曲线与统计表需同步刷新。
        self.layer_factor_combo.currentIndexChanged.connect(self._refresh_layer_tab)
        layer_bar.addWidget(self.layer_factor_combo)
        layer_bar.addStretch(1)
        layout.addLayout(layer_bar)
        self.layer_axis_label = layer_label

        self.layer_axis = DateAxisItem(orientation="bottom")
        self.layer_plot = pg.PlotWidget(
            axisItems={"left": CompactAxisItem(orientation="left"), "bottom": self.layer_axis},
            parent=self,
        )
        self.layer_plot.setBackground(colors.BG_DARK)
        self.layer_plot.showGrid(x=False, y=True, alpha=0.25)
        self.layer_plot.setMenuEnabled(False)
        self.layer_plot.setMouseEnabled(x=False, y=False)
        self.layer_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self.layer_plot.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)
        self.layer_plot.setMinimumHeight(150)
        self.layer_plot.setMaximumHeight(200)
        self.layer_curves: dict[str, pg.PlotDataItem] = {}
        # 多空曲线在构建时加入一次，后续仅切换数据（避免反复增删图元）。
        self.long_short_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(_LONG_SHORT_COLOR, width=1.5, style=Qt.PenStyle.DashLine)
        )
        self.layer_plot.addItem(self.long_short_curve)
        layout.addWidget(self.layer_plot)
        return layout

    def _build_detail_tabs(self) -> QTabWidget:
        """明细标签页：IC 汇总 / IC 衰减 / 分层统计 / 共线性。"""
        tabs = QTabWidget(self)

        self.ic_table = self._make_table(list(_IC_HEADERS), tabs)
        tabs.addTab(self.ic_table, "IC 汇总")

        self.decay_table = QTableWidget(0, 1, tabs)
        self.decay_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.decay_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        tabs.addTab(self.decay_table, "IC 衰减")

        layer_tab = QWidget(tabs)
        layer_layout = QVBoxLayout(layer_tab)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_summary_label = QLabel("分层统计 —", layer_tab)
        self.layer_summary_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        layer_layout.addWidget(self.layer_summary_label)
        self.layer_stats_table = self._make_table(list(_LAYER_HEADERS), layer_tab)
        layer_layout.addWidget(self.layer_stats_table)
        tabs.addTab(layer_tab, "分层统计")

        collinear_tab = QWidget(tabs)
        collinear_layout = QVBoxLayout(collinear_tab)
        collinear_layout.setContentsMargins(0, 0, 0, 0)
        self.corr_table = QTableWidget(0, 1, collinear_tab)
        self.corr_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.corr_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        collinear_layout.addWidget(self.corr_table, 1)
        self.vif_table = self._make_table(list(_VIF_HEADERS), collinear_tab)
        collinear_layout.addWidget(self.vif_table)
        tabs.addTab(collinear_tab, "共线性")

        self.detail_tabs = tabs
        return tabs

    @staticmethod
    def _make_table(headers: list[str], parent: QWidget) -> QTableWidget:
        """构造统一风格的只读表格（首列自适应、其余拉伸）。"""
        table = QTableWidget(0, len(headers), parent)
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = table.horizontalHeader()
        if len(headers) > 1:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            for column in range(1, len(headers)):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        else:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        return table

    # ------------------------------------------------------------------ #
    # 只读访问器（供测试 / 上层）
    # ------------------------------------------------------------------ #
    def selected_factors(self) -> list[str]:
        """当前勾选的因子名（保持注册顺序）。"""
        chosen: list[str] = []
        for index in range(self.factor_list.count()):
            item = self.factor_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return chosen

    def current_horizons(self) -> list[int]:
        """当前预测期列表（解析失败返回空列表）。"""
        return parse_horizons(self.horizons_edit.text())

    def current_dates(self) -> tuple[str, str]:
        """返回起止日期（``YYYY-MM-DD``）。"""
        return (
            self.start_date_edit.date().toString("yyyy-MM-dd"),
            self.end_date_edit.date().toString("yyyy-MM-dd"),
        )

    def last_report(self) -> Optional[FactorReport]:
        """返回最近一次分析结果；尚未运行返回 ``None``。"""
        return self._last_report

    # ------------------------------------------------------------------ #
    # 规格构建（无弹窗纯方法）
    # ------------------------------------------------------------------ #
    def current_spec(self) -> Optional[FactorAnalysisSpec]:
        """用当前界面参数构建 :class:`FactorAnalysisSpec`；无有效因子 / 预测期返回 ``None``。"""
        factors = self.selected_factors()
        if not factors:
            self._set_status("请至少勾选一个因子", ok=False)
            return None
        horizons = self.current_horizons()
        if not horizons:
            self._set_status("预测期非法：请填写 ≥1 的整数（如 5, 10, 20, 60）", ok=False)
            return None
        start, end = self.current_dates()
        return FactorAnalysisSpec(
            symbols=self._symbols,
            factor_names=factors,
            horizons=horizons,
            start=start,
            end=end,
            n_groups=int(self.groups_spin.value()),
            min_stocks=int(self.min_stocks_spin.value()),
        )

    # ------------------------------------------------------------------ #
    # 运行分析（同步纯方法 + 后台线程触达层）
    # ------------------------------------------------------------------ #
    def run_analysis(self) -> Optional[FactorReport]:
        """同步执行分析并把结果刷新到曲线 / 表格 / 状态标签（**不抛异常**）。

        供测试直接调用（注入 fake 分析器时无线程、无网络）。
        """
        spec = self.current_spec()
        if spec is None:
            return None
        try:
            report = self._analyzer.run(spec)
        except Exception as exc:  # noqa: BLE001 - 引擎异常不外抛
            logger.exception("因子分析执行失败")
            self._last_report = None
            self._clear_result_views()
            self._set_status(f"分析失败：{exc}", ok=False)
            return None
        self._last_report = report
        self._apply_result(report)
        return report

    def _on_run(self) -> None:
        """「运行分析」按钮：在后台线程执行，避免全市场分析阻塞 UI。"""
        spec = self.current_spec()
        if spec is None:
            return
        if self._worker is not None and self._worker.isRunning():
            logger.info("因子分析已在进行中，忽略重复触发")
            return
        self.run_button.setEnabled(False)
        self._set_status("正在分析…（全市场因子计算可能需要一些时间）", ok=True)
        # worker 的 parent 置为 None：对话框析构不连带销毁运行中的 QThread，
        # 生命周期统一交由 closeEvent 的 wait() + _on_worker_finished 管理。
        self._worker = _FactorWorker(self._analyzer, spec)
        self._worker.done.connect(self._on_analysis_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_analysis_finished(self, payload: object) -> None:
        """后台分析结束：恢复按钮并把结果刷新到 UI（本槽运行在 UI 线程）。"""
        self.run_button.setEnabled(True)
        if isinstance(payload, FactorReport):
            self._last_report = payload
            self._apply_result(payload)
        else:
            self._last_report = None
            self._clear_result_views()
            self._set_status(f"分析失败：{payload}", ok=False)

    def _on_worker_finished(self) -> None:
        """线程结束后释放引用。"""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------ #
    # 结果刷新
    # ------------------------------------------------------------------ #
    def _apply_result(self, report: FactorReport) -> None:
        """把分析结果填充到曲线 / 表格 / 状态标签（UI 线程调用）。"""
        if not getattr(report, "ok", False):
            self._clear_result_views()
            self._show_warnings(getattr(report, "warnings", None))
            self._set_status(
                f"分析未完成：{getattr(report, 'error', '') or '未知原因'}", ok=False
            )
            return

        self._fill_ic_table(report.ic_table)
        self._fill_decay_table(report.ic_table)
        self._populate_selections(report)
        self._fill_collinearity(report.collinearity)
        self._refresh_layer_tab()
        self._show_warnings(report.warnings)
        labeled = len(report.factors)
        self._set_status(
            f"分析完成：因子 {labeled} 个 / 标的 {len(report.universe)} 只 / "
            f"评估交易日 {report.evaluation_days} 天 / 分层 {report.n_groups} 层",
            ok=True,
        )

    def _populate_selections(self, report: FactorReport) -> None:
        """填充两个图表选择器（保持当前选择，若不可用则回退到第一个可用项）。"""
        factor_pairs = [
            (definition.name, definition.label) for definition in report.factors
        ]
        horizon_values = sorted(
            {horizon for (_, horizon) in report.ic_series.keys()}
        ) or [int(item) for item in report.ic_table["horizon"].tolist()]

        for combo, entries in (
            (self.ic_factor_combo, factor_pairs),
            (self.layer_factor_combo, factor_pairs),
        ):
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for name, label in entries:
                combo.addItem(f"{label}（{name}）", name)
            if previous is not None:
                index = combo.findData(previous)
                combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

        previous_horizon = self.ic_horizon_combo.currentData()
        self.ic_horizon_combo.blockSignals(True)
        self.ic_horizon_combo.clear()
        for horizon in horizon_values:
            self.ic_horizon_combo.addItem(f"{int(horizon)} 日", int(horizon))
        if previous_horizon is not None:
            index = self.ic_horizon_combo.findData(previous_horizon)
            self.ic_horizon_combo.setCurrentIndex(index if index >= 0 else 0)
        self.ic_horizon_combo.blockSignals(False)

        self._refresh_ic_chart()
        self._refresh_layer_chart()

    def _refresh_ic_chart(self, *_args: Any) -> None:
        """按当前「因子 × 预测期」选择刷新 IC 时序图（**不抛异常**）。"""
        try:
            report = self._last_report
            if report is None or not report.ic_series:
                return
            name = self.ic_factor_combo.currentData()
            horizon = self.ic_horizon_combo.currentData()
            series = report.ic_series.get((str(name), int(horizon))) if horizon else None
            if series is None or series.empty:
                self.ic_curve.setData([], [])
                self.ic_axis.set_labels([])
                self.ic_axis_label.setText("IC 时序 —")
                return
            values = pd.to_numeric(series, errors="coerce")
            labels = [
                "" if pd.isna(stamp) else pd.Timestamp(stamp).strftime("%Y-%m-%d")
                for stamp in values.index
            ]
            self.ic_axis.set_labels(labels)
            self.ic_curve.setData(list(range(len(values))), values.to_numpy(dtype=float))
            self.ic_plot.getViewBox().enableAutoRange()
            mean = _finite(values.mean())
            self.ic_axis_label.setText(
                f"IC 时序 {self.ic_factor_combo.currentText()} @ {horizon} 日"
                + (f"（均值 {format_ic(mean)}）" if mean is not None else "")
            )
        except Exception as exc:  # noqa: BLE001 - 绘图失败不影响表格展示
            logger.exception("刷新 IC 曲线失败")
            self._set_status(f"IC 曲线绘制失败：{exc}", ok=False)

    def _refresh_layer_chart(self, *_args: Any) -> None:
        """按当前因子选择刷新分层净值图（Q1~QN + 多空，**不抛异常**）。"""
        try:
            report = self._last_report
            if report is None or not report.layered:
                return
            name = str(self.layer_factor_combo.currentData())
            result = report.layered.get(name)
            if result is None or not result.ok or result.group_nav.empty:
                self._clear_layer_curve()
                self.layer_axis_label.setText("分层净值 —")
                return
            frame = result.group_nav
            labels = [
                "" if pd.isna(stamp) else pd.Timestamp(stamp).strftime("%Y-%m-%d")
                for stamp in frame.index
            ]
            self.layer_axis.set_labels(labels)
            x_values = list(range(len(frame)))
            self._clear_layer_curve()
            for order, group in enumerate(result.group_names):
                color = _GROUP_COLORS[order % len(_GROUP_COLORS)]
                curve = pg.PlotDataItem(
                    x_values,
                    pd.to_numeric(frame[group], errors="coerce").to_numpy(dtype=float),
                    pen=pg.mkPen(color, width=1.4),
                )
                self.layer_plot.addItem(curve)
                self.layer_curves[group] = curve
            if not result.long_short_nav.empty:
                self.long_short_curve.setData(
                    list(range(len(result.long_short_nav))),
                    pd.to_numeric(result.long_short_nav, errors="coerce").to_numpy(dtype=float),
                )
            self.layer_plot.getViewBox().enableAutoRange()
            self.layer_axis_label.setText(
                f"分层净值 {self.layer_factor_combo.currentText()}（Q1 低 → "
                f"{result.group_names[-1]} 高，金色虚线 = 多空）"
            )
        except Exception as exc:  # noqa: BLE001 - 绘图失败不影响表格展示
            logger.exception("刷新分层曲线失败")
            self._set_status(f"分层曲线绘制失败：{exc}", ok=False)

    def _clear_layer_curve(self) -> None:
        """移除分层图的分层曲线并清空多空曲线数据（多空图元常驻）。"""
        for curve in list(self.layer_curves.values()):
            self.layer_plot.removeItem(curve)
        self.layer_curves.clear()
        self.long_short_curve.setData([], [])

    def _refresh_layer_tab(self, *_args: Any) -> None:
        """按当前因子选择刷新分层统计表 + 摘要标签。"""
        report = self._last_report
        if report is None:
            return
        name = str(self.layer_factor_combo.currentData())
        result = report.layered.get(name)
        if result is None or not result.ok:
            self._fill_table(self.layer_stats_table, [])
            self.layer_summary_label.setText(
                f"分层统计 —（{name or '—'} 无有效样本）"
            )
            return
        rows: list[list[str]] = []
        for _, row in result.group_stats.iterrows():
            rows.append(
                [
                    str(row.get("group", "")),
                    format_count(row.get("days")),
                    format_pct(row.get("mean_daily_return")),
                    format_pct(row.get("total_return")),
                    format_pct(row.get("annual_return")),
                    format_pct(row.get("annual_vol")),
                    format_num(row.get("sharpe")),
                    format_pct(row.get("max_drawdown")),
                ]
            )
        self._fill_table(self.layer_stats_table, rows)
        self.layer_summary_label.setText(
            f"分层统计 {self.layer_factor_combo.currentText()}："
            f"多空年化 {format_pct(result.long_short_annual)} / "
            f"多空夏普 {format_num(result.long_short_sharpe)} / "
            f"单调性 {format_num(result.monotonicity)}（+1 完全递增、−1 完全递减）"
        )

    def _fill_ic_table(self, ic_table: pd.DataFrame) -> None:
        """IC 汇总长表（契约列见 ``IC_TABLE_COLUMNS``）。"""
        rows: list[list[str]] = []
        if isinstance(ic_table, pd.DataFrame) and not ic_table.empty:
            for _, row in ic_table.iterrows():
                rows.append(
                    [
                        str(row.get("factor", "")),
                        str(row.get("label", "")),
                        str(row.get("category", "")),
                        f"{int(row.get('horizon'))} 日"
                        if _finite(row.get("horizon")) is not None
                        else "—",
                        format_count(row.get("n_days")),
                        format_ic(row.get("ic_mean")),
                        format_ic(row.get("ic_std")),
                        format_ic(row.get("ic_ir")),
                        format_pct(row.get("ic_win_rate")),
                        format_ic(row.get("t_stat_nw")),
                        format_pvalue(row.get("p_value")),
                        format_flag(row.get("significant")),
                    ]
                )
        self._fill_table(self.ic_table, rows)

    def _fill_decay_table(self, ic_table: pd.DataFrame) -> None:
        """IC 衰减宽表（行 = 因子、列 = 预测期）。"""
        wide = ic_decay_table(ic_table)
        if not isinstance(wide, pd.DataFrame) or wide.empty:
            self.decay_table.setRowCount(0)
            self.decay_table.setColumnCount(1)
            self.decay_table.setHorizontalHeaderLabels(["因子"])
            return
        headers = ["因子"] + [str(column) for column in wide.columns]
        self.decay_table.setColumnCount(len(headers))
        self.decay_table.setHorizontalHeaderLabels(headers)
        self._fill_table(
            self.decay_table,
            [
                [str(index)] + [format_ic(value) for value in row]
                for index, row in zip(wide.index, wide.to_numpy(dtype=object))
            ],
        )

    def _fill_collinearity(self, result: Any) -> None:
        """相关矩阵表 + VIF 表（结果缺失时清空）。"""
        corr = getattr(result, "corr", None)
        vif = getattr(result, "vif", None)
        if not isinstance(corr, pd.DataFrame) or corr.empty:
            self.corr_table.setRowCount(0)
            self.corr_table.setColumnCount(1)
            self.corr_table.setHorizontalHeaderLabels(["相关矩阵"])
            self._fill_table(self.vif_table, [])
            return
        headers = ["因子"] + [str(column) for column in corr.columns]
        self.corr_table.setColumnCount(len(headers))
        self.corr_table.setHorizontalHeaderLabels(headers)
        rows = [
            [str(index)] + [format_ic(value) for value in corr.loc[index].tolist()]
            for index in corr.index
        ]
        self._fill_table(self.corr_table, rows)

        vif_rows: list[list[str]] = []
        if isinstance(vif, pd.Series):
            for name, value in vif.items():
                definition = get_factor(str(name))
                number = _finite(value)
                if number is None:
                    verdict = "严重共线" if str(value) == "inf" else "—"
                elif number >= 10:
                    verdict = "严重共线（建议剔除或合成）"
                elif number >= 5:
                    verdict = "警戒（谨慎同用）"
                else:
                    verdict = "正常"
                vif_rows.append(
                    [
                        str(name),
                        definition.label if definition else "—",
                        format_num(number, digits=4),
                        verdict,
                    ]
                )
        self._fill_table(self.vif_table, vif_rows)

    @staticmethod
    def _fill_table(table: QTableWidget, rows: list[list[str]]) -> None:
        """把字符串行填充到表格（先清空）。"""
        table.setRowCount(0)
        for row in rows:
            index = table.rowCount()
            table.insertRow(index)
            for column, text in enumerate(row):
                table.setItem(index, column, QTableWidgetItem(str(text)))

    def _clear_result_views(self) -> None:
        """清空曲线 / 全部表格 / 选择器 / 警告。"""
        self.ic_curve.setData([], [])
        self.ic_axis.set_labels([])
        self.ic_axis_label.setText("IC 时序 —")
        self._clear_layer_curve()
        self.layer_axis.set_labels([])
        self.layer_axis_label.setText("分层净值 —")
        for combo in (self.ic_factor_combo, self.ic_horizon_combo, self.layer_factor_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        for table in (self.ic_table, self.decay_table, self.layer_stats_table,
                      self.corr_table, self.vif_table):
            table.setRowCount(0)
        self.layer_summary_label.setText("分层统计 —")
        self.warnings_label.setText("")
        self.warnings_label.setVisible(False)

    # ------------------------------------------------------------------ #
    # 警告 / 状态
    # ------------------------------------------------------------------ #
    def _show_warnings(self, warnings: Any) -> None:
        """展示分析警告（无警告则隐藏）。"""
        items = [str(item) for item in (warnings or [])]
        if not items:
            self.warnings_label.setText("")
            self.warnings_label.setVisible(False)
            return
        self.warnings_label.setStyleSheet(f"color: {colors.LIMIT_COLOR};")
        self.warnings_label.setText("分析警告：" + "；".join(items))
        self.warnings_label.setVisible(True)

    def _set_status(self, text: str, ok: bool = True) -> None:
        """更新状态标签（成功绿色 / 失败红色，语义为状态）。"""
        color = colors.DOWN_COLOR if ok else colors.UP_COLOR
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)


__all__ = [
    "FactorAnalysisDialog",
    "WINDOW_TITLE",
    "format_ic",
    "format_pvalue",
    "format_flag",
    "parse_horizons",
]