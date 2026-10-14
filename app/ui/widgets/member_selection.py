"""策略成员筛选与筛选后组合验证对话框（P7-S8）。

定位：**模态对话框**，把 P7-S8「候选策略筛选 + 筛选后组合验证」从引擎落到界面上：
    * 配置样本切分（样本内 2015-01-01 ~ 2020-12-31 为筛选依据；其后为样本外验证段）
      与「合格成员」绝对门槛（跑赢基准 / 回撤 / 夏普 / 分年度稳定性，逐条可关）；
    * 配置候选池（均线网格 × 调仓频率 + 打分因子非空子集，默认 15 个）；
    * 运行 :class:`~app.engine.backtest.member_selection.SelectionRunner`（后台线程）：
      逐候选全期回测 → 分段评估 → 门槛判定（筛选），再构建四个组合变体做样本外验证；
    * 展示：成员筛选明细表（样本内判定 + 样本外对照 + 淘汰原因）、四线组合对比表
      （筛选前 / 筛选后 / 上界参考 / 最优单策略，评估口径 = 样本外段）、四线样本外
      净值曲线（含基准）、警告与状态。

架构约定（与 ``strategy_analysis`` / ``portfolio_backtest`` 同构）：
    * **业务动作 = 无弹窗的纯方法**（:meth:`current_pool` / :meth:`current_split` /
      :meth:`current_criteria` / :meth:`run_selection`），测试可直接调用并断言
      （注入 fake runner 时无线程、无网络）；
    * **UI 触达层**（运行按钮）经 :class:`_SelectionWorker`（``QThread``）后台执行，
      结果经 Qt 信号回 UI 线程刷新，**工作线程不碰任何 UI**；
    * **绝不向 UI 抛异常**：引擎 / 绘图异常一律捕获，状态标签显示可读原因；
    * 本对话框**只读**本地库（筛选与组合回测都不写库），不做任何数据更新。

口径提示（展示给用户）：
    * **样本内筛选 + 样本外验证**：合格判定只看样本内段；样本外段**只展示、不参与
      筛选判定**——这是对付「数据窥探 / 选择偏差」的时间闸门；
    * **上界参考**：另按全样本口径筛一遍作为上界，与「筛选后」的差距即选择偏差量级；
    * 组合对比从样本内起点跑到样本外终点、评估只取样本外段（成员带完整历史预热，
      才是对实盘状态的正确模拟）；T 日收盘信号 → T+1 开盘成交（无未来函数）；
      前复权；100 股整手；
    * 均线族标的池默认 PIT 并集（含窗口内曾进出指数名单的标的，显著减弱幸存者偏差）。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.store.db import Database, get_db
from app.data.universe import DEFAULT_INDEX_CODE, pit_members, universe_union
from app.engine.backtest import (
    DEFAULT_IN_END,
    DEFAULT_IN_START,
    DEFAULT_INITIAL_CASH,
    DEFAULT_MA_REBALANCES,
    DEFAULT_MA_TOP_N,
    DEFAULT_MA_WINDOWS,
    DEFAULT_SCORED_TOP_N,
    REBALANCE_LABELS,
    VARIANT_ALL,
    VARIANT_BEST_SINGLE,
    VARIANT_SELECTED_FULL,
    VARIANT_SELECTED_IN,
    MemberCandidate,
    SampleSplit,
    SelectionComparison,
    SelectionCriteria,
    SelectionReport,
    SelectionRunner,
    build_candidate_pool,
    slice_equity,
)
from app.engine.factor import DEFAULT_SCORING_FACTORS
from app.ui.themes import colors
from app.ui.widgets.kline_view import CompactAxisItem, DateAxisItem
from app.ui.widgets.strategy_analysis import format_date, format_num, format_pct

logger = logging.getLogger(__name__)

# 对话框标题。
WINDOW_TITLE: str = "策略成员筛选与组合验证"

# 变体展示顺序（曲线 / 图例固定顺序：筛选前 → 筛选后 → 上界 → 最优单策略）。
_VARIANT_ORDER: tuple[str, ...] = (
    VARIANT_ALL,
    VARIANT_SELECTED_IN,
    VARIANT_SELECTED_FULL,
    VARIANT_BEST_SINGLE,
)

# 变体曲线配色 / 线宽（键 → (颜色, 线宽)；「筛选后」为主结论、加粗强调）。
_VARIANT_STYLES: dict[str, tuple[str, float]] = {
    VARIANT_ALL: (colors.TEXT_COLOR, 1.2),
    VARIANT_SELECTED_IN: (colors.ACCENT_COLOR, 1.8),
    VARIANT_SELECTED_FULL: (colors.DIVIDEND_COLOR, 1.2),
    VARIANT_BEST_SINGLE: (colors.UP_COLOR, 1.2),
}

# 变体图例中文短名（键 → 短名；图例与清空还原共用）。
_VARIANT_CAPTIONS: dict[str, str] = {
    VARIANT_ALL: "筛选前·全候选等权",
    VARIANT_SELECTED_IN: "筛选后·样本内合格等权",
    VARIANT_SELECTED_FULL: "上界参考·全样本合格等权",
    VARIANT_BEST_SINGLE: "对照·最优单策略",
}

# 均线族标的池口径显示名（data 为对话框取值）。
_POOL_LABELS: tuple[tuple[str, str], ...] = (
    ("PIT 成分股并集（推荐，消除幸存者偏差）", "pit"),
    ("本地全部前复权标的", "local"),
)

# 筛选明细表列头（与 SELECTION_TABLE_COLUMNS 一一对应）。
_SELECTION_HEADERS: tuple[str, ...] = (
    "候选",
    "族",
    "样本内收益",
    "样本内基准",
    "样本内回撤",
    "样本内夏普",
    "样本内正年占比",
    "样本内合格",
    "样本外收益",
    "样本外回撤",
    "样本外夏普",
    "全样本合格",
    "淘汰原因",
)

# 组合对比表列头（与 SELECTION_COMPARISON_COLUMNS 一一对应）。
_COMPARISON_HEADERS: tuple[str, ...] = (
    "变体",
    "名称",
    "成员",
    "成员数",
    "样本外收益",
    "样本外基准",
    "样本外回撤",
    "样本外夏普",
    "说明",
)

# 「是 / 否」布尔列显示文本。
_YES: str = "是"
_NO: str = "否"


# --------------------------------------------------------------------------- #
# 后台线程
# --------------------------------------------------------------------------- #
class _SelectionWorker(QThread):
    """后台线程执行一次「筛选 + 组合对比」（结果经信号回传，不碰 UI）。"""

    done = Signal(object)  # (SelectionReport, Optional[SelectionComparison]) 或 Exception

    def __init__(
        self,
        runner: SelectionRunner,
        pool: list[MemberCandidate],
        split: SampleSplit,
        criteria: SelectionCriteria,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造后台筛选线程（parent 默认 None，生命周期由 closeEvent 管理）。"""
        super().__init__(parent)
        self._runner = runner
        self._pool = pool
        self._split = split
        self._criteria = criteria

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：筛选 + （筛选有效时）组合对比，成功 / 失败均经 ``done`` 回传。"""
        try:
            report = self._runner.run(self._pool, self._split, self._criteria)
            comparison = (
                self._runner.compare(report, self._split) if report.ok else None
            )
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台成员筛选线程异常")
            self.done.emit(exc)
            return
        self.done.emit((report, comparison))


# --------------------------------------------------------------------------- #
# 对话框
# --------------------------------------------------------------------------- #
class MemberSelectionDialog(QDialog):
    """策略成员筛选对话框（切分 + 门槛 + 候选池 → 筛选 → 四线组合验证）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        runner: Optional[SelectionRunner] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`（只读）。
            runner: 筛选编排器；默认用 ``db`` 构造 :class:`SelectionRunner`。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._runner: SelectionRunner = (
            runner if runner is not None else SelectionRunner(db=self._db)
        )
        self._last_report: Optional[SelectionReport] = None
        self._last_comparison: Optional[SelectionComparison] = None
        self._worker: Optional[_SelectionWorker] = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1280, 940)
        self._setup_ui()
        self._refresh_pool_hint()
        self._set_status("设置切分、门槛与候选池后点击『运行成员筛选』开始", ok=True)

    # ------------------------------------------------------------------ #
    # 线程生命周期（与 PortfolioBacktestDialog 同约定）
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭前等待后台筛选线程自然结束，避免析构连带销毁运行中的 QThread。"""
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
        """搭建「配置 / 四线净值曲线 / 明细标签页 / 警告」四段式布局。"""
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
        """配置区：样本切分与资金 / 合格门槛 / 候选池 / 运行。"""
        box = QGroupBox("筛选配置（样本内筛选 → 样本外验证）", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        hint = QLabel(
            "口径：合格判定只看样本内段（样本外段只展示、不参与筛选判定——防数据窥探）；"
            "组合对比从样本内起点跑到样本外终点、评估只取样本外段（成员带完整历史预热）；"
            "「上界参考」用全样本口径再筛一遍，与「筛选后」的差距即选择偏差量级；"
            "T 日收盘信号 → T+1 开盘成交（无未来函数）；前复权；100 股整手。",
            box,
        )
        hint.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addLayout(self._build_scope_row(box))
        layout.addLayout(self._build_criteria_row(box))
        layout.addLayout(self._build_pool_row(box))

        self.pool_hint_label = QLabel("", box)
        self.pool_hint_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        layout.addWidget(self.pool_hint_label)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("运行成员筛选", box)
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)
        return box

    def _build_scope_row(self, parent: QWidget) -> QHBoxLayout:
        """样本切分与组合总资金：样本内起 / 样本内止 / 样本外止 / 总资金。"""
        row = QHBoxLayout()
        row.addWidget(QLabel("样本内起：", parent))
        self.in_start_edit = self._make_date_edit(QDate.fromString(DEFAULT_IN_START, "yyyy-MM-dd"), parent)
        row.addWidget(self.in_start_edit)
        row.addWidget(QLabel("样本内止：", parent))
        self.in_end_edit = self._make_date_edit(QDate.fromString(DEFAULT_IN_END, "yyyy-MM-dd"), parent)
        row.addWidget(self.in_end_edit)
        row.addWidget(QLabel("样本外止：", parent))
        self.out_end_edit = self._make_date_edit(QDate.currentDate(), parent)
        row.addWidget(self.out_end_edit)
        row.addWidget(QLabel("组合总资金：", parent))
        self.total_cash_spin = QDoubleSpinBox(parent)
        self.total_cash_spin.setRange(10_000.0, 100_000_000.0)
        self.total_cash_spin.setDecimals(0)
        self.total_cash_spin.setSingleStep(100_000.0)
        self.total_cash_spin.setSuffix(" 元")
        self.total_cash_spin.setValue(DEFAULT_INITIAL_CASH)
        row.addWidget(self.total_cash_spin)
        row.addStretch(1)
        return row

    @staticmethod
    def _make_date_edit(value: QDate, parent: QWidget) -> QDateEdit:
        """构造统一风格的日期输入框（日历弹出、``yyyy-MM-dd``）。"""
        edit = QDateEdit(value, parent)
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("yyyy-MM-dd")
        edit.setMinimumDate(QDate(2000, 1, 1))
        return edit

    def _build_criteria_row(self, parent: QWidget) -> QHBoxLayout:
        """合格门槛：跑赢基准 / 回撤不深于基准 / 回撤上限 / 夏普 / 分年度稳定性。"""
        row = QHBoxLayout()
        row.addWidget(QLabel("合格门槛（全部满足才算合格；不勾选 = 不启用）：", parent))

        self.beat_check = QCheckBox("跑赢基准", parent)
        self.beat_check.setChecked(True)
        row.addWidget(self.beat_check)

        self.dd_vs_bench_check = QCheckBox("回撤不深于基准", parent)
        self.dd_vs_bench_check.setChecked(True)
        row.addWidget(self.dd_vs_bench_check)

        self.dd_limit_check = QCheckBox("回撤上限：", parent)
        self.dd_limit_check.setChecked(False)
        row.addWidget(self.dd_limit_check)
        self.dd_limit_spin = QDoubleSpinBox(parent)
        self.dd_limit_spin.setRange(5.0, 100.0)
        self.dd_limit_spin.setDecimals(0)
        self.dd_limit_spin.setSingleStep(5.0)
        self.dd_limit_spin.setSuffix(" %")
        self.dd_limit_spin.setValue(50.0)
        row.addWidget(self.dd_limit_spin)

        self.sharpe_check = QCheckBox("夏普 ≥", parent)
        self.sharpe_check.setChecked(True)
        row.addWidget(self.sharpe_check)
        self.sharpe_spin = QDoubleSpinBox(parent)
        self.sharpe_spin.setRange(-3.0, 5.0)
        self.sharpe_spin.setDecimals(2)
        self.sharpe_spin.setSingleStep(0.1)
        self.sharpe_spin.setValue(0.50)
        row.addWidget(self.sharpe_spin)

        self.win_ratio_check = QCheckBox("正收益年份占比 ≥", parent)
        self.win_ratio_check.setChecked(True)
        row.addWidget(self.win_ratio_check)
        self.win_ratio_spin = QDoubleSpinBox(parent)
        self.win_ratio_spin.setRange(0.0, 100.0)
        self.win_ratio_spin.setDecimals(0)
        self.win_ratio_spin.setSingleStep(10.0)
        self.win_ratio_spin.setSuffix(" %")
        self.win_ratio_spin.setValue(50.0)
        row.addWidget(self.win_ratio_spin)

        row.addStretch(1)
        return row

    def _build_pool_row(self, parent: QWidget) -> QHBoxLayout:
        """候选池：均线窗口 × 调仓频率 + 打分因子子集 + 标的池口径。"""
        row = QHBoxLayout()
        row.addWidget(QLabel("均线窗口：", parent))
        self.window_checks: dict[int, QCheckBox] = {}
        for window in DEFAULT_MA_WINDOWS:
            check = QCheckBox(f"{window}日", parent)
            check.setChecked(True)
            check.toggled.connect(self._refresh_pool_hint)
            row.addWidget(check)
            self.window_checks[int(window)] = check

        row.addWidget(QLabel("调仓：", parent))
        self.rebalance_checks: dict[str, QCheckBox] = {}
        for code in DEFAULT_MA_REBALANCES:
            check = QCheckBox(REBALANCE_LABELS.get(code, code), parent)
            check.setChecked(True)
            check.toggled.connect(self._refresh_pool_hint)
            row.addWidget(check)
            self.rebalance_checks[str(code)] = check

        row.addWidget(QLabel("均线最多持有：", parent))
        self.ma_top_n_spin = QSpinBox(parent)
        self.ma_top_n_spin.setRange(1, 50)
        self.ma_top_n_spin.setValue(DEFAULT_MA_TOP_N)
        self.ma_top_n_spin.setSuffix(" 只")
        row.addWidget(self.ma_top_n_spin)

        row.addWidget(QLabel("打分因子：", parent))
        self.factor_checks: dict[str, QCheckBox] = {}
        for name in DEFAULT_SCORING_FACTORS:
            check = QCheckBox(str(name), parent)
            check.setChecked(True)
            check.toggled.connect(self._refresh_pool_hint)
            row.addWidget(check)
            self.factor_checks[str(name)] = check

        row.addWidget(QLabel("打分最多持有：", parent))
        self.scored_top_n_spin = QSpinBox(parent)
        self.scored_top_n_spin.setRange(1, 200)
        self.scored_top_n_spin.setValue(DEFAULT_SCORED_TOP_N)
        self.scored_top_n_spin.setSuffix(" 只")
        row.addWidget(self.scored_top_n_spin)

        row.addWidget(QLabel("均线族标的池：", parent))
        self.pool_combo = QComboBox(parent)
        for label, code in _POOL_LABELS:
            self.pool_combo.addItem(label, code)
        row.addWidget(self.pool_combo)
        row.addStretch(1)
        return row

    def _build_charts(self) -> QVBoxLayout:
        """四线样本外净值曲线（筛选前 / 筛选后 / 上界参考 / 最优单策略 + 基准）。"""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        legend = QHBoxLayout()
        legend.setSpacing(8)
        self.variant_labels: dict[str, QLabel] = {}
        for key in _VARIANT_ORDER:
            label = QLabel(f"{_VARIANT_CAPTIONS[key]} —", self)
            label.setStyleSheet(f"color: {_VARIANT_STYLES[key][0]};")
            legend.addWidget(label)
            self.variant_labels[key] = label
        self.benchmark_label = QLabel("基准 —", self)
        self.benchmark_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        legend.addWidget(self.benchmark_label)
        legend.addStretch(1)
        self.nav_scope_label = QLabel("样本外段 —", self)
        self.nav_scope_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        legend.addWidget(self.nav_scope_label)
        layout.addLayout(legend)

        self.nav_axis = DateAxisItem(orientation="bottom")
        self.nav_plot = pg.PlotWidget(
            axisItems={"left": CompactAxisItem(orientation="left"), "bottom": self.nav_axis},
            parent=self,
        )
        self.nav_plot.setBackground(colors.BG_DARK)
        self.nav_plot.showGrid(x=False, y=True, alpha=0.25)
        self.nav_plot.setMenuEnabled(False)
        self.nav_plot.setMouseEnabled(x=False, y=False)
        self.nav_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self.nav_plot.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)
        self.nav_plot.setMinimumHeight(200)
        self.nav_plot.setMaximumHeight(280)

        self.variant_curves: dict[str, pg.PlotDataItem] = {}
        for key in _VARIANT_ORDER:
            color, width = _VARIANT_STYLES[key]
            curve = pg.PlotDataItem([], [], pen=pg.mkPen(color, width=width))
            self.nav_plot.addItem(curve)
            self.variant_curves[key] = curve
        self.benchmark_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.SUBTEXT_COLOR, width=1.0)
        )
        self.nav_plot.addItem(self.benchmark_curve)
        layout.addWidget(self.nav_plot)
        return layout

    def _build_detail_tabs(self) -> QTabWidget:
        """明细标签页：成员筛选明细 / 组合对比（样本外）。"""
        tabs = QTabWidget(self)

        self.selection_table = self._make_table(list(_SELECTION_HEADERS), tabs)
        tabs.addTab(self.selection_table, "成员筛选明细")

        self.comparison_table = self._make_table(list(_COMPARISON_HEADERS), tabs)
        tabs.addTab(self.comparison_table, "组合对比（样本外）")

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
    def selected_ma_windows(self) -> list[int]:
        """当前勾选的均线窗口（保持界面顺序）。"""
        return [window for window, check in self.window_checks.items() if check.isChecked()]

    def selected_ma_rebalances(self) -> list[str]:
        """当前勾选的均线族调仓频率（保持界面顺序）。"""
        return [code for code, check in self.rebalance_checks.items() if check.isChecked()]

    def selected_factors(self) -> list[str]:
        """当前勾选的打分因子名（保持注册顺序）。"""
        return [name for name, check in self.factor_checks.items() if check.isChecked()]

    def current_pool_choice(self) -> str:
        """返回均线族标的池口径（``pit`` / ``local``）。"""
        data = self.pool_combo.currentData()
        return str(data) if data else "pit"

    def last_report(self) -> Optional[SelectionReport]:
        """返回最近一次筛选报告；尚未运行返回 ``None``。"""
        return self._last_report

    def last_comparison(self) -> Optional[SelectionComparison]:
        """返回最近一次组合对比；尚未运行 / 未构建返回 ``None``。"""
        return self._last_comparison

    def _pool_counts(self) -> tuple[int, int, int]:
        """候选数构成（均线 / 打分 / 合计），供提示标签使用。"""
        ma_count = len(self.selected_ma_windows()) * len(self.selected_ma_rebalances())
        factor_count = len(self.selected_factors())
        scored_count = (2**factor_count - 1) if factor_count else 0
        return ma_count, scored_count, ma_count + scored_count

    def _refresh_pool_hint(self) -> None:
        """刷新候选数提示（每次勾选变化时调用）。"""
        ma_count, scored_count, total = self._pool_counts()
        self.pool_hint_label.setText(
            f"候选数：{total}（均线 {ma_count} 个 = 窗口 × 调仓；打分 {scored_count} 个 = "
            "因子非空子集）；每个候选都会全期回测一次，候选越多耗时越长"
        )

    # ------------------------------------------------------------------ #
    # 规格构建（无弹窗纯方法）
    # ------------------------------------------------------------------ #
    def current_split(self) -> Optional[SampleSplit]:
        """用当前界面日期构建 :class:`SampleSplit`；非法返回 ``None``（**不抛异常**）。"""
        try:
            return SampleSplit(
                in_start=self.in_start_edit.date().toString("yyyy-MM-dd"),
                in_end=self.in_end_edit.date().toString("yyyy-MM-dd"),
                out_end=self.out_end_edit.date().toString("yyyy-MM-dd"),
            )
        except ValueError as exc:
            self._set_status(f"样本切分不合法：{exc}", ok=False)
            return None

    def current_criteria(self) -> SelectionCriteria:
        """用当前界面勾选构建 :class:`SelectionCriteria`（控件范围已约束取值合法性）。"""
        return SelectionCriteria(
            beat_benchmark=self.beat_check.isChecked(),
            drawdown_not_worse_than_benchmark=self.dd_vs_bench_check.isChecked(),
            max_drawdown_limit=(
                float(self.dd_limit_spin.value()) / 100.0
                if self.dd_limit_check.isChecked()
                else None
            ),
            min_sharpe=(
                float(self.sharpe_spin.value()) if self.sharpe_check.isChecked() else None
            ),
            min_win_year_ratio=(
                float(self.win_ratio_spin.value()) / 100.0
                if self.win_ratio_check.isChecked()
                else None
            ),
        )

    def _candidate_symbols(self, split: SampleSplit) -> Optional[list[str]]:
        """均线族标的池：PIT 并集（推荐）或本地全库（``symbols=None``）。

        PIT 并集 = 「样本内起点起全部快照出现过的成分股」∪「样本内起点时点成分」，
        覆盖窗口内曾进出指数名单的标的（补数据口径，显著减弱幸存者偏差）。

        Raises:
            ValueError: 选择 PIT 并集但库内无成分股快照 / 名单读取失败
                （消息面向用户，调用方捕获后显示到状态标签）。
        """
        if self.current_pool_choice() != "pit":
            return None  # 引擎回退「本地全部前复权标的」
        try:
            records = self._db.list_index_constituents(DEFAULT_INDEX_CODE)
        except Exception as exc:  # noqa: BLE001 - 读库失败不外抛
            raise ValueError(f"读取沪深300成分股名单失败：{exc}") from exc
        union = set(
            universe_union(records, DEFAULT_INDEX_CODE, start=split.in_start, end=None)
        )
        union |= set(pit_members(records, split.in_start, DEFAULT_INDEX_CODE))
        if not union:
            raise ValueError(
                "库内无沪深300成分股快照，均线族无法按 PIT 并集取池"
                "（请先「数据 → 更新数据」拉取成分股，或把标的池切为本地全部前复权标的）"
            )
        return sorted(union)

    def current_pool(self) -> Optional[list[MemberCandidate]]:
        """用当前界面参数构建候选池；参数非法返回 ``None``（**不抛异常**）。"""
        split = self.current_split()
        if split is None:
            return None
        windows = self.selected_ma_windows()
        if not windows:
            self._set_status("请至少勾选一个均线窗口（候选池需要均线族）", ok=False)
            return None
        rebalances = self.selected_ma_rebalances()
        if not rebalances:
            self._set_status("请至少勾选一个调仓频率（候选池需要均线族）", ok=False)
            return None
        factors = self.selected_factors()
        if not factors:
            self._set_status("请至少勾选一个打分因子（候选池需要打分族）", ok=False)
            return None
        try:
            symbols = self._candidate_symbols(split)
        except ValueError as exc:
            self._set_status(str(exc), ok=False)
            return None
        try:
            return build_candidate_pool(
                symbols=symbols,
                ma_windows=windows,
                ma_rebalances=rebalances,
                ma_top_n=int(self.ma_top_n_spin.value()),
                factors=factors,
                scored_top_n=int(self.scored_top_n_spin.value()),
            )
        except ValueError as exc:  # noqa: PERF203 - 参数非法不外抛
            logger.warning("候选池参数非法：%s", exc)
            self._set_status(f"候选池参数不合法：{exc}", ok=False)
            return None

    # ------------------------------------------------------------------ #
    # 运行筛选（同步纯方法 + 后台线程触达层）
    # ------------------------------------------------------------------ #
    def run_selection(self) -> Optional[SelectionReport]:
        """同步执行「筛选 + 组合对比」并把结果刷新到表格 / 曲线 / 状态（**不抛异常**）。

        供测试直接调用（注入 fake runner 时无线程、无网络）。参数非法时提示并
        返回 ``None``；引擎异常一律捕获后返回 ``None``。
        """
        pool = self.current_pool()
        if pool is None:
            return None
        split = self.current_split()
        if split is None:
            return None
        criteria = self.current_criteria()
        try:
            report = self._runner.run(pool, split, criteria)
            comparison = self._runner.compare(report, split) if report.ok else None
        except Exception as exc:  # noqa: BLE001 - 引擎异常不外抛
            logger.exception("成员筛选执行失败")
            self._last_report = None
            self._last_comparison = None
            self._clear_result_views()
            self._set_status(f"成员筛选失败：{exc}", ok=False)
            return None
        self._last_report = report
        self._last_comparison = comparison
        self._apply_result(report, comparison)
        return report

    def _on_run(self) -> None:
        """「运行成员筛选」按钮：在后台线程执行，避免长区间多候选回测阻塞 UI。"""
        pool = self.current_pool()
        if pool is None:
            return
        split = self.current_split()
        if split is None:
            return
        criteria = self.current_criteria()
        if self._worker is not None and self._worker.isRunning():
            logger.info("成员筛选已在进行中，忽略重复触发")
            return
        self.run_button.setEnabled(False)
        self._set_status(
            f"正在筛选 {len(pool)} 个候选（逐候选全期回测 + 门槛判定），"
            "随后构建四线组合对比…（候选多 / 区间长时需要一些时间）",
            ok=True,
        )
        # worker 的 parent 置为 None：对话框析构不连带销毁运行中的 QThread，
        # 生命周期统一交由 closeEvent 的 wait() + _on_worker_finished 管理。
        self._worker = _SelectionWorker(self._runner, pool, split, criteria)
        self._worker.done.connect(self._on_run_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_run_finished(self, payload: object) -> None:
        """后台筛选结束：恢复按钮并把结果刷新到 UI（本槽运行在 UI 线程）。"""
        self.run_button.setEnabled(True)
        if isinstance(payload, tuple) and len(payload) == 2:
            report, comparison = payload
            self._last_report = report
            self._last_comparison = comparison
            self._apply_result(report, comparison)
        else:
            self._last_report = None
            self._last_comparison = None
            self._clear_result_views()
            self._set_status(f"成员筛选失败：{payload}", ok=False)

    def _on_worker_finished(self) -> None:
        """线程结束后释放引用。"""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------ #
    # 结果刷新
    # ------------------------------------------------------------------ #
    def _apply_result(
        self,
        report: Optional[SelectionReport],
        comparison: Optional[SelectionComparison],
    ) -> None:
        """把筛选报告与组合对比填充到表格 / 曲线 / 状态标签（UI 线程调用）。"""
        if report is None or not getattr(report, "ok", False):
            self._clear_result_views()
            self._show_warnings(getattr(report, "warnings", None))
            self._set_status(
                f"成员筛选未完成：{getattr(report, 'error', '') or '未知原因'}", ok=False
            )
            return

        self._fill_selection_table(report)
        self._fill_comparison_table(comparison)
        self._plot_variants(comparison, report.split)

        warnings = list(report.warnings)
        if comparison is not None:
            warnings.extend(comparison.warnings)
        self._show_warnings(warnings)

        ok_count = sum(1 for evaluation in report.evaluations if evaluation.ok)
        message = (
            f"成员筛选完成：候选 {len(report.evaluations)} 个（成功回测 {ok_count}）/ "
            f"样本内合格 {len(report.passed_in)} / 全样本合格 {len(report.passed_full)}"
        )
        if comparison is not None and comparison.ok:
            selected = next(
                (
                    variant
                    for variant in comparison.variants
                    if variant.key == VARIANT_SELECTED_IN
                ),
                None,
            )
            if selected is not None and selected.ok and selected.metrics_out is not None:
                metrics = selected.metrics_out
                message += (
                    f"；筛选后组合（样本外自 {format_date(report.split.out_start)}）："
                    f"收益 {format_pct(metrics.total_return)} / "
                    f"回撤 {format_pct(metrics.max_drawdown)} / "
                    f"夏普 {format_num(metrics.sharpe)}"
                )
            else:
                reason = ""
                if selected is not None:
                    reason = str(selected.error or "")
                message += f"；筛选后组合未构建（{reason or '样本内无合格成员'}）"
        else:
            message += "；组合对比未完成"
        self._set_status(message, ok=True)

    def _clear_result_views(self) -> None:
        """清空全部结果视图（失败 / 重跑前调用）。"""
        self.selection_table.setRowCount(0)
        self.comparison_table.setRowCount(0)
        self._clear_charts()

    def _clear_charts(self) -> None:
        """清空曲线与图例（**不抛异常**）。"""
        try:
            for curve in list(self.variant_curves.values()) + [self.benchmark_curve]:
                curve.setData([], [])
            for key, label in self.variant_labels.items():
                label.setText(f"{_variant_caption(key)} —")
            self.benchmark_label.setText("基准 —")
            self.nav_scope_label.setText("样本外段 —")
        except Exception:  # noqa: BLE001 - 绘图清理异常不外抛
            logger.exception("清空四线曲线失败")

    # ------------------------------------------------------------------ #
    # 表格填充
    # ------------------------------------------------------------------ #
    def _fill_selection_table(self, report: SelectionReport) -> None:
        """填充成员筛选明细表：候选 × 样本内判定 + 样本外对照 + 淘汰原因。"""
        try:
            frame = report.to_frame()
            self.selection_table.setRowCount(len(frame))
            for position, (_, record) in enumerate(frame.iterrows()):
                values = [
                    str(record["candidate"]),
                    str(record["family"]),
                    format_pct(record["in_return"]),
                    format_pct(record["in_benchmark_return"]),
                    format_pct(record["in_max_drawdown"]),
                    format_num(record["in_sharpe"]),
                    format_pct(record["in_win_year_ratio"]),
                    _YES if record["in_passed"] else _NO,
                    format_pct(record["out_return"]),
                    format_pct(record["out_max_drawdown"]),
                    format_num(record["out_sharpe"]),
                    _YES if record["full_passed"] else _NO,
                    str(record["reasons"]),
                ]
                for column, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    if 2 <= column <= 6 or 8 <= column <= 10:
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                        )
                    elif column in (7, 11):
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.selection_table.setItem(position, column, item)
        except Exception:  # noqa: BLE001 - 表格填充异常不外抛
            logger.exception("填充成员筛选明细表失败")
            self.selection_table.setRowCount(0)

    def _fill_comparison_table(self, comparison: Optional[SelectionComparison]) -> None:
        """填充组合对比表：四个变体的样本外表现与口径说明。"""
        try:
            if comparison is None:
                self.comparison_table.setRowCount(0)
                return
            frame = comparison.to_frame()
            self.comparison_table.setRowCount(len(frame))
            for position, (_, record) in enumerate(frame.iterrows()):
                values = [
                    str(record["variant"]),
                    str(record["name"]),
                    str(record["members"]),
                    str(record["member_count"]),
                    format_pct(record["out_return"]),
                    format_pct(record["out_benchmark_return"]),
                    format_pct(record["out_max_drawdown"]),
                    format_num(record["out_sharpe"]),
                    str(record["note"]),
                ]
                for column, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    if 3 <= column <= 7:
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                        )
                    self.comparison_table.setItem(position, column, item)
        except Exception:  # noqa: BLE001 - 表格填充异常不外抛
            logger.exception("填充组合对比表失败")
            self.comparison_table.setRowCount(0)

    # ------------------------------------------------------------------ #
    # 曲线刷新
    # ------------------------------------------------------------------ #
    def _plot_variants(
        self,
        comparison: Optional[SelectionComparison],
        split: SampleSplit,
    ) -> None:
        """刷新四线样本外净值曲线（各变体切片归一 + 基准，**不抛异常**）。"""
        try:
            self._clear_charts()
            if comparison is None or not comparison.ok:
                return
            labels: Optional[list[str]] = None
            scope_text = f"样本外段 {format_date(split.out_start)} 起（段首归一 1.0）"
            for variant in comparison.variants:
                curve = self.variant_curves.get(variant.key)
                label = self.variant_labels.get(variant.key)
                if curve is None or label is None:
                    continue
                frame = pd.DataFrame()
                if variant.ok and variant.result is not None:
                    frame = slice_equity(
                        variant.result.equity, split.out_start, split.out_end
                    )
                if frame.empty:
                    curve.setData([], [])
                    label.setText(f"{variant.name} —")
                    continue
                nav = pd.to_numeric(frame["nav"], errors="coerce").to_numpy(dtype=float)
                if labels is None:
                    stamps = pd.to_datetime(frame["date"], errors="coerce")
                    labels = [
                        "" if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")
                        for stamp in stamps
                    ]
                    self.nav_axis.set_labels(labels)
                    if "benchmark_nav" in frame.columns:
                        bench = pd.to_numeric(
                            frame["benchmark_nav"], errors="coerce"
                        ).to_numpy(dtype=float)
                        if not pd.isna(bench).all():
                            self.benchmark_curve.setData(
                                list(range(len(bench))), list(bench)
                            )
                            self.benchmark_label.setText(
                                f"基准（样本外归一）{format_num(bench[-1])}"
                            )
                curve.setData(list(range(len(nav))), list(nav))
                last = nav[-1] if len(nav) else float("nan")
                label.setText(
                    f"{variant.name} {format_num(last)}"
                    if not pd.isna(last)
                    else f"{variant.name} —"
                )
            if labels:
                self.nav_scope_label.setText(scope_text)
        except Exception:  # noqa: BLE001 - 绘图异常不外抛
            logger.exception("刷新四线曲线失败")
            self._clear_charts()

    # ------------------------------------------------------------------ #
    # 警告与状态
    # ------------------------------------------------------------------ #
    def _show_warnings(self, warnings: Any) -> None:
        """把聚合警告显示到警告标签（空 / 无 → 隐藏）。"""
        try:
            items = [str(text) for text in (warnings or []) if str(text).strip()]
        except TypeError:
            items = []
        if not items:
            self.warnings_label.setText("")
            self.warnings_label.setVisible(False)
            return
        # 上限 20 条：极端情况下避免标签撑爆窗口。
        shown = items[:20]
        suffix = f"（共 {len(items)} 条，此处显示前 {len(shown)} 条）" if len(items) > 20 else ""
        self.warnings_label.setText("筛选警告：" + "；".join(shown) + suffix)
        self.warnings_label.setVisible(True)

    def _set_status(self, message: str, ok: bool) -> None:
        """写状态标签并着色（成功 → 主文字色；失败 → 涨色警示）。"""
        self.status_label.setText(message)
        color = colors.SUBTEXT_COLOR if ok else colors.UP_COLOR
        self.status_label.setStyleSheet(f"color: {color};")


def _variant_caption(key: str) -> str:
    """变体键 → 图例中文短名（供清空曲线时还原图例文本）。"""
    return _VARIANT_CAPTIONS.get(key, key)


__all__ = [
    "WINDOW_TITLE",
    "MemberSelectionDialog",
]