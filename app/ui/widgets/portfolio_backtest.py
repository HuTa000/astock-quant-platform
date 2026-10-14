"""多策略组合回测对话框（P7-S7）。

定位：**模态对话框**，把 P7-S7「资金分账 · 等权组合」从引擎落到界面上：
    * 配置两个现有基线成员——「站上 N 日线」（条件式选股）与「多因子打分」
      （ICIR 加权打分选股）各自的参数与资金权重；
    * 配置组合级窗口（起止日期 / 总资金），运行
      :class:`~app.engine.backtest.portfolio.PortfolioRunner`（后台线程执行）；
    * 展示：组合净值曲线（组合 vs 基准 vs 成员）、绩效指标、逐年表现、牛熊拆分、
      成员对照、成员相关性矩阵、贡献归因（含最大回撤期分解）、组合成交明细、
      警告与状态。

架构约定（与 ``strategy_analysis`` / ``score_backtest`` 同构）：
    * **业务动作 = 无弹窗的纯方法**（:meth:`current_spec` / :meth:`run_portfolio_backtest`），
      测试可直接调用并断言（注入 fake runner 时无线程、无网络）；
    * **UI 触达层**（运行按钮）经 :class:`_PortfolioWorker`（``QThread``）后台执行，
      结果经 Qt 信号回 UI 线程刷新，**工作线程不碰任何 UI**；
    * **绝不向 UI 抛异常**：引擎 / 绘图异常一律捕获，状态标签显示可读原因；
    * 本对话框**只读**本地库（组合回测不写库），不做任何数据更新。

口径提示（展示给用户）：
    * **资金分账 · 等权**：组合总资金按权重切成子账户，成员各自独立调仓、独立扣费，
      组合净值 = Σ 子账户资产 / 总资金；权重随各自净值漂移，子账户间**不再平衡**；
    * T 日收盘信号 → T+1 开盘成交（无未来函数）；前复权；100 股整手；
    * 相关性用成员两两**共同交易日**的真实日收益计算；贡献归因 Σ = 组合总收益，
      回撤期分解 Σ = 组合最大回撤；
    * 条件式成员的「PIT 并集池」含窗口内曾进出指数名单的标的（补数据口径，
      显著减弱幸存者偏差；严格逐日掩码由打分成员的 PIT 口径承担）；
    * A/B 实测（2018-01~2026-10、564 只 PIT 池）：坏成员拖累收益与回撤，分散
      不能补偿其负 alpha——界面如实披露，等权仅作研究起点。
"""
from __future__ import annotations

import logging
import math
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
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
from app.data.universe import DEFAULT_INDEX_CODE, universe_union
from app.engine.backtest import (
    DEFAULT_INITIAL_CASH,
    DEFAULT_SCORED_REBALANCE,
    DEFAULT_SCORED_TOP_N,
    DEFAULT_SCORED_UNIVERSE,
    PORTFOLIO_TRADE_COLUMNS,
    REBALANCE_CHOICES,
    PortfolioMember,
    PortfolioResult,
    PortfolioRunner,
    PortfolioSpec,
    ScoredBacktestSpec,
    StrategySpec,
    yearly_summary,
)
from app.engine.factor import (
    DEFAULT_FACTOR_NAMES,
    DEFAULT_IC_HORIZON,
    DEFAULT_MIN_IC_DAYS,
    DEFAULT_SCORING_FACTORS,
    get_factor,
)
from app.engine.screener import Condition
from app.ui.themes import colors
from app.ui.widgets.kline_view import CompactAxisItem, DateAxisItem
from app.ui.widgets.strategy_analysis import (
    format_count,
    format_date,
    format_num,
    format_pct,
)

logger = logging.getLogger(__name__)

# 对话框标题。
WINDOW_TITLE: str = "多策略组合回测"

# 成员类型中文显示名（data 为引擎取值 ``condition`` / ``scored``）。
_KIND_LABELS: dict[str, str] = {"condition": "条件式", "scored": "打分"}

# 调仓频率中文显示名。
_REBALANCE_LABELS: dict[str, str] = {
    "daily": "日频",
    "weekly": "周频",
    "monthly": "月频",
}

# 条件式成员的标的池口径显示名（data 为对话框取值）。
_CONDITION_POOL_LABELS: tuple[tuple[str, str], ...] = (
    ("PIT 成分股并集（推荐，消除幸存者偏差）", "pit"),
    ("本地全部前复权标的", "local"),
)

# 打分成员股票池口径显示名（data 为引擎取值）。
_UNIVERSE_LABELS: tuple[tuple[str, str], ...] = (
    ("PIT 成分股（推荐，消除幸存者偏差）", "pit"),
    ("当前可见标的", "current"),
)

# 成交方向中文显示名。
_SIDE_LABELS: dict[str, str] = {"buy": "买入", "sell": "卖出"}

# 绩效指标卡行（中文标签 / PerformanceMetrics 字段 / 格式类型）。
_METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("区间总收益", "total_return", "pct"),
    ("年化收益", "annual_return", "pct"),
    ("年化波动率", "annual_volatility", "pct"),
    ("夏普比率", "sharpe", "num"),
    ("最大回撤", "max_drawdown", "pct"),
    ("Calmar 比率", "calmar", "num"),
    ("年化 α（选股超额）", "alpha", "pct"),
    ("β（大盘敏感度）", "beta", "num"),
    ("跟踪误差", "tracking_error", "pct"),
    ("信息比率", "information_ratio", "num"),
    ("基准总收益", "benchmark_total_return", "pct"),
    ("超额收益（总收益−基准）", "excess_return", "pct"),
    ("胜率（已平仓）", "win_rate", "pct"),
    ("盈亏比", "profit_loss_ratio", "num"),
    ("已平仓笔数（笔）", "trade_count", "int"),
)

# 逐年表现表列头。
_YEARLY_HEADERS: tuple[str, ...] = (
    "年份",
    "交易日",
    "策略收益",
    "基准收益",
    "超额收益",
    "策略最深回撤",
    "基准最深回撤",
)

# 牛熊拆分表列头。
_REGIME_HEADERS: tuple[str, ...] = (
    "环境",
    "交易日",
    "天数占比",
    "策略收益",
    "基准收益",
    "超额收益",
)

# 成员对照表列头（子策略净值对照：组合层不改写成员自身结果）。
_MEMBER_HEADERS: tuple[str, ...] = (
    "成员",
    "类型",
    "权重",
    "初始资金",
    "期末资产",
    "区间收益",
    "最大回撤",
    "夏普",
    "状态",
)

# 贡献归因表列头（与 CONTRIBUTION_COLUMNS 一一对应）。
_CONTRIBUTION_HEADERS: tuple[str, ...] = (
    "成员",
    "权重",
    "初始资金",
    "期末资产",
    "盈亏",
    "成员收益",
    "贡献（占组合总资金）",
    "回撤期贡献",
)

# 相关性矩阵表首列表头（其余列 = 成员名）。
_CORRELATION_FIRST_HEADER: str = "成员"


def _format_by_kind(value: Any, kind: str) -> str:
    """按格式类型渲染单元格文本。"""
    if kind == "pct":
        return format_pct(value)
    if kind == "int":
        return format_count(value)
    return format_num(value)


# --------------------------------------------------------------------------- #
# 后台线程
# --------------------------------------------------------------------------- #
class _PortfolioWorker(QThread):
    """后台线程执行一次 :meth:`PortfolioRunner.run`（结果经信号回传，不碰 UI）。"""

    done = Signal(object)  # PortfolioResult 或 Exception

    def __init__(
        self,
        runner: PortfolioRunner,
        spec: PortfolioSpec,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造后台组合回测线程（parent 默认 None，生命周期由 closeEvent 管理）。"""
        super().__init__(parent)
        self._runner = runner
        self._spec = spec

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：执行组合回测，成功 / 失败均经 ``done`` 信号回传。"""
        try:
            result = self._runner.run(self._spec)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台组合回测线程异常")
            self.done.emit(exc)
            return
        self.done.emit(result)


# --------------------------------------------------------------------------- #
# 对话框
# --------------------------------------------------------------------------- #
class PortfolioBacktestDialog(QDialog):
    """多策略组合回测对话框（成员配置 → 分账回放 → 曲线 + 明细视图）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        runner: Optional[PortfolioRunner] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`（只读）。
            runner: 组合编排器；默认用 ``db`` 构造 :class:`PortfolioRunner`。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._runner: PortfolioRunner = (
            runner if runner is not None else PortfolioRunner(db=self._db)
        )
        self._last_result: Optional[PortfolioResult] = None
        self._worker: Optional[_PortfolioWorker] = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1160, 900)
        self._setup_ui()
        self._set_status("设置成员与组合参数后点击『运行组合回测』开始", ok=True)

    # ------------------------------------------------------------------ #
    # 线程生命周期（与 StrategyAnalysisDialog 同约定）
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭前等待后台回测线程自然结束，避免析构连带销毁运行中的 QThread。"""
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
        """搭建「成员配置 / 净值曲线 / 明细标签页 / 警告」四段式布局。"""
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
        """成员配置区：条件式成员 / 打分成员 / 组合级窗口与总资金 / 运行。"""
        box = QGroupBox("组合配置（资金分账 · 成员独立调仓）", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        hint = QLabel(
            "口径：组合总资金按权重切成子账户，成员各自独立调仓、独立扣费，"
            "组合净值 = Σ 子账户资产 / 总资金；权重随各自净值漂移、子账户间不再平衡；"
            "T 日收盘信号 → T+1 开盘成交（无未来函数）；前复权；100 股整手；"
            "成员自身风控规格保留生效。",
            box,
        )
        hint.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addLayout(self._build_condition_member(box), 3)
        body.addLayout(self._build_scored_member(box), 4)
        layout.addLayout(body)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("起始日：", box))
        self.start_date_edit = QDateEdit(QDate(2015, 1, 1), box)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        scope_row.addWidget(self.start_date_edit)
        scope_row.addWidget(QLabel("结束日：", box))
        self.end_date_edit = QDateEdit(QDate.currentDate(), box)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        scope_row.addWidget(self.end_date_edit)
        scope_row.addWidget(QLabel("组合总资金：", box))
        self.total_cash_spin = QDoubleSpinBox(box)
        self.total_cash_spin.setRange(10_000.0, 100_000_000.0)
        self.total_cash_spin.setDecimals(0)
        self.total_cash_spin.setSingleStep(100_000.0)
        self.total_cash_spin.setSuffix(" 元")
        self.total_cash_spin.setValue(DEFAULT_INITIAL_CASH)
        scope_row.addWidget(self.total_cash_spin)
        scope_row.addStretch(1)
        layout.addLayout(scope_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("运行组合回测", box)
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)
        return box

    def _build_condition_member(self, parent: QWidget) -> QVBoxLayout:
        """成员一（条件式）参数列：均线窗口 / Top-N / 调仓 / 标的池 / 权重。"""
        column = QVBoxLayout()
        column.setSpacing(4)
        title = QLabel("成员一 · 站上 N 日线（条件式选股）", parent)
        title.setStyleSheet(f"color: {colors.UP_COLOR};")
        column.addWidget(title)

        row = QHBoxLayout()
        row.addWidget(QLabel("均线窗口：", parent))
        self.cond_window_spin = QSpinBox(parent)
        self.cond_window_spin.setRange(5, 250)
        self.cond_window_spin.setValue(20)
        self.cond_window_spin.setSuffix(" 日")
        row.addWidget(self.cond_window_spin)
        row.addWidget(QLabel("Top-N：", parent))
        self.cond_top_n_spin = QSpinBox(parent)
        self.cond_top_n_spin.setRange(1, 50)
        self.cond_top_n_spin.setValue(3)
        row.addWidget(self.cond_top_n_spin)
        row.addWidget(QLabel("调仓：", parent))
        self.cond_rebalance_combo = QComboBox(parent)
        for code in REBALANCE_CHOICES:
            self.cond_rebalance_combo.addItem(_REBALANCE_LABELS.get(code, code), code)
        self.cond_rebalance_combo.setCurrentIndex(
            max(0, list(REBALANCE_CHOICES).index("weekly"))
        )
        row.addWidget(self.cond_rebalance_combo)
        row.addStretch(1)
        column.addLayout(row)

        pool_row = QHBoxLayout()
        pool_row.addWidget(QLabel("标的池：", parent))
        self.cond_pool_combo = QComboBox(parent)
        for label, code in _CONDITION_POOL_LABELS:
            self.cond_pool_combo.addItem(label, code)
        pool_row.addWidget(self.cond_pool_combo, 1)
        column.addLayout(pool_row)

        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("资金权重：", parent))
        self.cond_weight_spin = self._make_weight_spin(parent)
        weight_row.addWidget(self.cond_weight_spin)
        weight_row.addStretch(1)
        column.addLayout(weight_row)

        note = QLabel(
            "标的池：PIT 并集 = 窗口内所有成分股名单的并集（含已退出标的，减弱幸存者偏差）。",
            parent,
        )
        note.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        note.setWordWrap(True)
        column.addWidget(note)
        column.addStretch(1)
        return column

    def _build_scored_member(self, parent: QWidget) -> QVBoxLayout:
        """成员二（打分）参数列：因子勾选 / Top-N / 调仓 / 股票池 / 口径 / 权重。"""
        column = QVBoxLayout()
        column.setSpacing(4)
        title = QLabel("成员二 · 多因子打分（ICIR 加权打分选股）", parent)
        title.setStyleSheet(f"color: {colors.DIVIDEND_COLOR};")
        column.addWidget(title)

        column.addWidget(QLabel("参与打分的因子（勾选）：", parent))
        self.factor_list = QListWidget(parent)
        self.factor_list.setMaximumHeight(150)
        for name in DEFAULT_FACTOR_NAMES:
            definition = get_factor(name)
            label = f"{definition.label}（{name}）" if definition else name
            item = QListWidgetItem(label, self.factor_list)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setCheckState(
                Qt.CheckState.Checked
                if name in DEFAULT_SCORING_FACTORS
                else Qt.CheckState.Unchecked
            )
        column.addWidget(self.factor_list)

        row = QHBoxLayout()
        row.addWidget(QLabel("Top-N：", parent))
        self.scored_top_n_spin = QSpinBox(parent)
        self.scored_top_n_spin.setRange(1, 200)
        self.scored_top_n_spin.setValue(DEFAULT_SCORED_TOP_N)
        row.addWidget(self.scored_top_n_spin)
        row.addWidget(QLabel("调仓：", parent))
        self.scored_rebalance_combo = QComboBox(parent)
        for code in REBALANCE_CHOICES:
            self.scored_rebalance_combo.addItem(_REBALANCE_LABELS.get(code, code), code)
        self.scored_rebalance_combo.setCurrentIndex(
            max(0, list(REBALANCE_CHOICES).index(DEFAULT_SCORED_REBALANCE))
        )
        row.addWidget(self.scored_rebalance_combo)
        row.addWidget(QLabel("股票池：", parent))
        self.scored_universe_combo = QComboBox(parent)
        for label, code in _UNIVERSE_LABELS:
            self.scored_universe_combo.addItem(label, code)
        self.scored_universe_combo.setCurrentIndex(
            max(0, [code for _, code in _UNIVERSE_LABELS].index(DEFAULT_SCORED_UNIVERSE))
        )
        row.addWidget(self.scored_universe_combo, 1)
        column.addLayout(row)

        gate_row = QHBoxLayout()
        gate_row.addWidget(QLabel("资金权重：", parent))
        self.scored_weight_spin = self._make_weight_spin(parent)
        gate_row.addWidget(self.scored_weight_spin)
        gate_row.addWidget(QLabel("权重前瞻期（交易日）：", parent))
        self.ic_horizon_spin = QSpinBox(parent)
        self.ic_horizon_spin.setRange(1, 120)
        self.ic_horizon_spin.setValue(DEFAULT_IC_HORIZON)
        gate_row.addWidget(self.ic_horizon_spin)
        gate_row.addWidget(QLabel("最少 IC 样本：", parent))
        self.min_ic_days_spin = QSpinBox(parent)
        self.min_ic_days_spin.setRange(20, 1000)
        self.min_ic_days_spin.setValue(DEFAULT_MIN_IC_DAYS)
        gate_row.addWidget(self.min_ic_days_spin)
        gate_row.addStretch(1)
        column.addLayout(gate_row)
        column.addStretch(1)
        return column

    @staticmethod
    def _make_weight_spin(parent: QWidget) -> QDoubleSpinBox:
        """构造统一风格的资金权重输入框（0.05 ~ 10.00，两位小数；构造时整体归一化）。"""
        spin = QDoubleSpinBox(parent)
        spin.setRange(0.05, 10.00)
        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setValue(1.00)
        return spin

    def _build_charts(self) -> QVBoxLayout:
        """净值曲线（组合 vs 基准 vs 成员），图例风格与策略分析保持一致。"""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        legend = QHBoxLayout()
        legend.setSpacing(8)
        self.nav_strategy_label = QLabel("组合净值 —", self)
        self.nav_strategy_label.setStyleSheet(f"color: {colors.ACCENT_COLOR};")
        self.nav_benchmark_label = QLabel("基准净值 —", self)
        self.nav_benchmark_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        self.nav_member1_label = QLabel("成员一 —", self)
        self.nav_member1_label.setStyleSheet(f"color: {colors.UP_COLOR};")
        self.nav_member2_label = QLabel("成员二 —", self)
        self.nav_member2_label.setStyleSheet(f"color: {colors.DIVIDEND_COLOR};")
        legend.addWidget(self.nav_strategy_label)
        legend.addWidget(self.nav_benchmark_label)
        legend.addWidget(self.nav_member1_label)
        legend.addWidget(self.nav_member2_label)
        legend.addStretch(1)
        self.nav_benchmark_info = QLabel("基准口径 —", self)
        self.nav_benchmark_info.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        legend.addWidget(self.nav_benchmark_info)
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
        self.nav_plot.setMinimumHeight(190)
        self.nav_plot.setMaximumHeight(260)
        self.nav_strategy_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.ACCENT_COLOR, width=1.6)
        )
        self.nav_benchmark_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.SUBTEXT_COLOR, width=1.2)
        )
        self.nav_member1_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.UP_COLOR, width=1.0)
        )
        self.nav_member2_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.DIVIDEND_COLOR, width=1.0)
        )
        for curve in (
            self.nav_strategy_curve,
            self.nav_benchmark_curve,
            self.nav_member1_curve,
            self.nav_member2_curve,
        ):
            self.nav_plot.addItem(curve)
        layout.addWidget(self.nav_plot)
        return layout

    def _build_detail_tabs(self) -> QTabWidget:
        """明细标签页：绩效 / 逐年 / 牛熊 / 成员对照 / 相关性 / 贡献归因 / 成交明细。"""
        tabs = QTabWidget(self)

        self.metrics_table = self._make_table(["指标", "数值"], tabs)
        tabs.addTab(self.metrics_table, "绩效指标")

        self.yearly_table = self._make_table(list(_YEARLY_HEADERS), tabs)
        tabs.addTab(self.yearly_table, "逐年表现")

        self.regime_table = self._make_table(list(_REGIME_HEADERS), tabs)
        tabs.addTab(self.regime_table, "牛熊拆分")

        self.members_table = self._make_table(list(_MEMBER_HEADERS), tabs)
        tabs.addTab(self.members_table, "成员对照")

        self.correlation_table = self._make_table(
            [_CORRELATION_FIRST_HEADER, "成员一", "成员二"], tabs
        )
        tabs.addTab(self.correlation_table, "成员相关性")

        self.contribution_table = self._make_table(list(_CONTRIBUTION_HEADERS), tabs)
        tabs.addTab(self.contribution_table, "贡献归因")

        self.trades_table = self._make_table(list(PORTFOLIO_TRADE_COLUMNS), tabs)
        tabs.addTab(self.trades_table, "成交明细")

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
        """当前勾选的打分因子名（保持注册顺序）。"""
        chosen: list[str] = []
        for index in range(self.factor_list.count()):
            item = self.factor_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return chosen

    def current_dates(self) -> tuple[str, str]:
        """返回组合起止日期（``YYYY-MM-DD``）。"""
        return (
            self.start_date_edit.date().toString("yyyy-MM-dd"),
            self.end_date_edit.date().toString("yyyy-MM-dd"),
        )

    def current_cond_pool(self) -> str:
        """返回条件式成员标的池口径（``pit`` / ``local``）。"""
        data = self.cond_pool_combo.currentData()
        return str(data) if data else "pit"

    def current_cond_rebalance(self) -> str:
        """返回条件式成员调仓频率（``daily`` / ``weekly`` / ``monthly``）。"""
        data = self.cond_rebalance_combo.currentData()
        return str(data) if data else "weekly"

    def current_scored_rebalance(self) -> str:
        """返回打分成员调仓频率（``daily`` / ``weekly`` / ``monthly``）。"""
        data = self.scored_rebalance_combo.currentData()
        return str(data) if data else str(DEFAULT_SCORED_REBALANCE)

    def current_scored_universe(self) -> str:
        """返回打分成员股票池口径（``pit`` / ``current``）。"""
        data = self.scored_universe_combo.currentData()
        return str(data) if data else str(DEFAULT_SCORED_UNIVERSE)

    def last_result(self) -> Optional[PortfolioResult]:
        """返回最近一次组合回测结果；尚未运行返回 ``None``。"""
        return self._last_result

    # ------------------------------------------------------------------ #
    # 规格构建（无弹窗纯方法）
    # ------------------------------------------------------------------ #
    def _condition_pool(self, start: str, end: str) -> Optional[list[str]]:
        """条件式成员的标的池：PIT 并集（推荐）或本地全库（``symbols=None``）。

        Raises:
            ValueError: 选择 PIT 并集但库内无成分股快照 / 名单读取失败
                （消息面向用户，调用方捕获后显示到状态标签）。
        """
        if self.current_cond_pool() != "pit":
            return None  # 引擎回退「本地全部前复权标的」
        try:
            records = self._db.list_index_constituents(DEFAULT_INDEX_CODE)
        except Exception as exc:  # noqa: BLE001 - 读库失败不外抛
            raise ValueError(f"读取沪深300成分股名单失败：{exc}") from exc
        pool = universe_union(records, DEFAULT_INDEX_CODE, start=start, end=end)
        if not pool:
            raise ValueError(
                "库内无沪深300成分股快照，条件式成员无法按 PIT 并集取池"
                "（请先「数据 → 更新数据」拉取成分股，或把标的池切为本地全部前复权标的）"
            )
        return pool

    def current_spec(self) -> Optional[PortfolioSpec]:
        """用当前界面参数构建 :class:`PortfolioSpec`；参数非法返回 ``None``（**不抛异常**）。"""
        factors = self.selected_factors()
        if not factors:
            self._set_status("请至少勾选一个打分因子", ok=False)
            return None
        start, end = self.current_dates()
        window = int(self.cond_window_spin.value())
        try:
            pool = self._condition_pool(start, end)
        except ValueError as exc:
            self._set_status(str(exc), ok=False)
            return None

        condition_spec = StrategySpec(
            name=f"站上{window}日线",
            conditions=[
                Condition(field=f"close - close.rolling({window}).mean()", op=">", value=0.0)
            ],
            symbols=pool,
            rebalance=self.current_cond_rebalance(),
            top_n=int(self.cond_top_n_spin.value()),
            rank_by="close",
        )
        scored_spec = ScoredBacktestSpec(
            factors=factors,
            top_n=int(self.scored_top_n_spin.value()),
            rebalance=self.current_scored_rebalance(),
            universe=self.current_scored_universe(),
            start=start,
            end=end,
            ic_horizon=int(self.ic_horizon_spin.value()),
            min_ic_days=int(self.min_ic_days_spin.value()),
        )
        try:
            spec = PortfolioSpec(
                name="条件式 + 打分 组合",
                members=[
                    PortfolioMember(
                        name=f"站上{window}日线",
                        spec=condition_spec,
                        weight=float(self.cond_weight_spin.value()),
                    ),
                    PortfolioMember(
                        name="多因子打分",
                        spec=scored_spec,
                        weight=float(self.scored_weight_spin.value()),
                    ),
                ],
                start=start,
                end=end,
                initial_cash=float(self.total_cash_spin.value()),
            )
        except Exception as exc:  # noqa: BLE001 - 参数非法不外抛
            logger.warning("组合参数非法：%s", exc)
            self._set_status(f"组合参数不合法：{exc}", ok=False)
            return None
        return spec

    # ------------------------------------------------------------------ #
    # 运行组合回测（同步纯方法 + 后台线程触达层）
    # ------------------------------------------------------------------ #
    def run_portfolio_backtest(self) -> Optional[PortfolioResult]:
        """同步执行组合回测并把结果刷新到曲线 / 表格 / 状态标签（**不抛异常**）。

        供测试直接调用（注入 fake runner 时无线程、无网络）。参数非法时提示并
        返回 ``None``；引擎异常一律捕获后返回 ``None``。
        """
        spec = self.current_spec()
        if spec is None:
            return None
        try:
            result = self._runner.run(spec)
        except Exception as exc:  # noqa: BLE001 - 引擎异常不外抛
            logger.exception("组合回测执行失败")
            self._last_result = None
            self._clear_result_views()
            self._set_status(f"组合回测失败：{exc}", ok=False)
            return None
        self._last_result = result
        self._apply_result(result)
        return result

    def _on_run(self) -> None:
        """「运行组合回测」按钮：在后台线程执行，避免长区间回测阻塞 UI。"""
        spec = self.current_spec()
        if spec is None:
            return
        if self._worker is not None and self._worker.isRunning():
            logger.info("组合回测已在进行中，忽略重复触发")
            return
        self.run_button.setEnabled(False)
        self._set_status("正在逐成员回测并合并组合…（全池打分可能需要一些时间）", ok=True)
        # worker 的 parent 置为 None：对话框析构不连带销毁运行中的 QThread，
        # 生命周期统一交由 closeEvent 的 wait() + _on_worker_finished 管理。
        self._worker = _PortfolioWorker(self._runner, spec)
        self._worker.done.connect(self._on_run_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_run_finished(self, payload: object) -> None:
        """后台回测结束：恢复按钮并把结果刷新到 UI（本槽运行在 UI 线程）。"""
        self.run_button.setEnabled(True)
        if isinstance(payload, PortfolioResult):
            self._last_result = payload
            self._apply_result(payload)
        else:
            self._last_result = None
            self._clear_result_views()
            self._set_status(f"组合回测失败：{payload}", ok=False)

    def _on_worker_finished(self) -> None:
        """线程结束后释放引用。"""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------ #
    # 结果刷新
    # ------------------------------------------------------------------ #
    def _apply_result(self, result: PortfolioResult) -> None:
        """把组合回测结果填充到曲线 / 表格 / 状态标签（UI 线程调用）。"""
        if not getattr(result, "ok", False):
            self._clear_result_views()
            self._show_warnings(getattr(result, "warnings", None))
            self._set_status(
                f"组合回测未完成：{getattr(result, 'error', '') or '未知原因'}", ok=False
            )
            return

        equity = result.equity if isinstance(result.equity, pd.DataFrame) else pd.DataFrame()
        self._fill_metrics(result.metrics)
        self._fill_yearly(yearly_summary(equity))
        regime = result.regime if isinstance(result.regime, pd.DataFrame) else pd.DataFrame()
        self._fill_regime(regime)
        self._fill_members(result.members, result.contribution)
        self._fill_correlation(result.correlation)
        self._fill_contribution(result.contribution)
        self._fill_trades(result.trades)
        self._plot_equity(equity, result)
        self._show_warnings(result.warnings)

        contrib_sum: Optional[float] = None
        if isinstance(result.contribution, pd.DataFrame) and not result.contribution.empty:
            values = pd.to_numeric(result.contribution["contribution"], errors="coerce")
            contrib_sum = float(values.fillna(0.0).sum())
        dd_text = ""
        if result.dd_window is not None:
            peak, trough = result.dd_window
            dd_text = f"（{format_date(peak)} → {format_date(trough)}）"
        metrics = result.metrics
        self._set_status(
            f"组合回测完成：成员 {len(result.members)} 个 / 净值 {len(equity)} 天 / "
            f"总收益 {format_pct(getattr(metrics, 'total_return', None))} / "
            f"年化 {format_pct(getattr(metrics, 'annual_return', None))} / "
            f"夏普 {format_num(getattr(metrics, 'sharpe', None))} / "
            f"最大回撤 {format_pct(getattr(metrics, 'max_drawdown', None))}{dd_text} / "
            f"Σ成员贡献 {format_pct(contrib_sum)}（对账口径 = 组合总收益）",
            ok=True,
        )

    # ------------------------------------------------------------------ #
    # 图表刷新
    # ------------------------------------------------------------------ #
    def _plot_equity(self, equity: pd.DataFrame, result: PortfolioResult) -> None:
        """刷新净值曲线（组合 vs 基准 vs 成员），标题展示基准代码 / 口径（**不抛异常**）。"""
        try:
            symbol = str(getattr(result, "benchmark_symbol", "") or "—")
            kind = str(getattr(result, "benchmark_kind", "") or "—")
            self.nav_benchmark_info.setText(f"基准 {symbol}（{kind}）")

            if not isinstance(equity, pd.DataFrame) or equity.empty:
                self._clear_charts()
                return
            if "date" in equity.columns:
                stamps = pd.to_datetime(equity["date"], errors="coerce")
            else:
                stamps = pd.to_datetime(equity.index, errors="coerce")
            nav = pd.to_numeric(equity["nav"], errors="coerce").to_numpy(dtype=float)
            labels = [
                "" if pd.isna(stamp) else stamp.strftime("%Y-%m-%d") for stamp in stamps
            ]
            x_values = list(range(len(nav)))

            self.nav_axis.set_labels(labels)
            self.nav_strategy_curve.setData(x_values, nav)

            benchmark: Optional[list[float]] = None
            if "benchmark_nav" in equity.columns:
                values = pd.to_numeric(equity["benchmark_nav"], errors="coerce").to_numpy(
                    dtype=float
                )
                if not pd.isna(values).all():
                    benchmark = list(values)
            if benchmark is not None:
                self.nav_benchmark_curve.setData(x_values, benchmark)
            else:
                self.nav_benchmark_curve.setData([], [])

            # 成员曲线：按组合日历对齐（成员日历不一致时留空档，仅连有效点）。
            stamp_index = pd.DatetimeIndex(stamps)
            members = list(result.members or [])
            member_curves = (self.nav_member1_curve, self.nav_member2_curve)
            member_labels = (self.nav_member1_label, self.nav_member2_label)
            for position, curve in enumerate(member_curves):
                member = members[position] if position < len(members) else None
                series = self._member_nav_series(member, stamp_index)
                if series is None:
                    curve.setData([], [])
                    member_labels[position].setText(
                        f"{member.name} —" if member is not None else "成员 —"
                    )
                    continue
                values = series.to_numpy(dtype=float)
                xs = [index for index, value in enumerate(values) if not math.isnan(value)]
                ys = [float(values[index]) for index in xs]
                curve.setData(xs, ys)
                name = member.name if member is not None else f"成员{position + 1}"
                last = ys[-1] if ys else float("nan")
                member_labels[position].setText(
                    f"{name} {format_num(last)}（末值）"
                    if not math.isnan(last)
                    else f"{name} —"
                )

            self.nav_plot.getViewBox().enableAutoRange()

            nav_last = nav[-1] if len(nav) else float("nan")
            self.nav_strategy_label.setText(
                f"组合净值 {format_num(nav_last)}（末值）"
                if not math.isnan(nav_last)
                else "组合净值 —"
            )
            if benchmark is not None and len(benchmark):
                bench_last = benchmark[-1]
                self.nav_benchmark_label.setText(
                    f"基准净值 {format_num(bench_last)}（末值）"
                    if not math.isnan(bench_last)
                    else "基准净值 —"
                )
            else:
                self.nav_benchmark_label.setText("基准净值 —")
        except Exception as exc:  # noqa: BLE001 - 绘图失败不影响表格展示
            logger.exception("刷新组合净值曲线失败")
            self._set_status(f"曲线绘制失败：{exc}", ok=False)

    @staticmethod
    def _member_nav_series(member: Any, stamps: pd.DatetimeIndex) -> Optional[pd.Series]:
        """成员净值序列按组合日历对齐（缺失日为 NaN）；不可用时返回 ``None``。"""
        backtest = getattr(member, "backtest", None)
        equity = getattr(backtest, "equity", None)
        if not isinstance(equity, pd.DataFrame) or equity.empty or "nav" not in equity.columns:
            return None
        index = pd.DatetimeIndex(pd.to_datetime(equity["date"], errors="coerce"))
        values = pd.to_numeric(equity["nav"], errors="coerce").to_numpy(dtype=float)
        series = pd.Series(values, index=index)
        series = series[~series.index.isna()]
        return series.reindex(stamps)

    def _clear_charts(self) -> None:
        """清空全部净值曲线与轴标签。"""
        for curve in (
            self.nav_strategy_curve,
            self.nav_benchmark_curve,
            self.nav_member1_curve,
            self.nav_member2_curve,
        ):
            curve.setData([], [])
        self.nav_axis.set_labels([])
        self.nav_strategy_label.setText("组合净值 —")
        self.nav_benchmark_label.setText("基准净值 —")
        self.nav_member1_label.setText("成员一 —")
        self.nav_member2_label.setText("成员二 —")
        self.nav_benchmark_info.setText("基准口径 —")

    # ------------------------------------------------------------------ #
    # 表格刷新
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fill_table(table: QTableWidget, rows: list[list[str]]) -> None:
        """把字符串行填充到表格（先清空）。"""
        table.setRowCount(0)
        for row in rows:
            index = table.rowCount()
            table.insertRow(index)
            for column, text in enumerate(row):
                table.setItem(index, column, QTableWidgetItem(str(text)))

    def _fill_metrics(self, metrics: Any) -> None:
        """绩效指标卡（PerformanceMetrics 15 字段）。"""
        rows: list[list[str]] = []
        for label, attribute, kind in _METRIC_ROWS:
            rows.append([label, _format_by_kind(getattr(metrics, attribute, None), kind)])
        self._fill_table(self.metrics_table, rows)

    def _fill_yearly(self, frame: pd.DataFrame) -> None:
        """逐年表现表（契约列见 ``attribution.YEARLY_COLUMNS``）。"""
        rows: list[list[str]] = []
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, row in frame.iterrows():
                rows.append(
                    [
                        format_count(row.get("year")),
                        format_count(row.get("days")),
                        format_pct(row.get("strategy_return")),
                        format_pct(row.get("benchmark_return")),
                        format_pct(row.get("excess_return")),
                        format_pct(row.get("strategy_max_dd")),
                        format_pct(row.get("benchmark_max_dd")),
                    ]
                )
        self._fill_table(self.yearly_table, rows)

    def _fill_regime(self, frame: pd.DataFrame) -> None:
        """牛熊拆分表（契约列见 ``regime.REGIME_COLUMNS``）。"""
        rows: list[list[str]] = []
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, row in frame.iterrows():
                rows.append(
                    [
                        str(row.get("regime", "")),
                        format_count(row.get("days")),
                        format_pct(row.get("share")),
                        format_pct(row.get("strategy_return")),
                        format_pct(row.get("benchmark_return")),
                        format_pct(row.get("excess_return")),
                    ]
                )
        self._fill_table(self.regime_table, rows)

    def _fill_members(self, members: list[Any], contribution: Any) -> None:
        """成员对照表（子策略净值对照：收益 / 回撤 / 夏普取成员自身回测指标）。"""
        final_assets: dict[str, Any] = {}
        if isinstance(contribution, pd.DataFrame) and not contribution.empty:
            for _, row in contribution.iterrows():
                final_assets[str(row.get("member", ""))] = row.get("final_assets")
        rows: list[list[str]] = []
        for member in members or []:
            metrics = getattr(getattr(member, "backtest", None), "metrics", None)
            rows.append(
                [
                    str(getattr(member, "name", "")),
                    _KIND_LABELS.get(str(getattr(member, "kind", "")), str(getattr(member, "kind", ""))),
                    format_pct(getattr(member, "weight", None)),
                    format_num(getattr(member, "initial_cash", None)),
                    format_num(final_assets.get(str(getattr(member, "name", "")))),
                    format_pct(getattr(metrics, "total_return", None)),
                    format_pct(getattr(metrics, "max_drawdown", None)),
                    format_num(getattr(metrics, "sharpe", None)),
                    "成功" if getattr(member, "ok", False) else f"失败：{getattr(member, 'error', '')}",
                ]
            )
        self._fill_table(self.members_table, rows)

    def _fill_correlation(self, frame: Any) -> None:
        """成员相关性矩阵表（行 / 列为成员名，动态重建表头）。"""
        table = self.correlation_table
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            table.setColumnCount(1)
            table.setHorizontalHeaderLabels([_CORRELATION_FIRST_HEADER])
            table.setRowCount(0)
            return
        names = [str(column) for column in frame.columns]
        headers = [_CORRELATION_FIRST_HEADER] + names
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(1, len(headers)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        rows: list[list[str]] = []
        for name in frame.index:
            values = [
                format_num(frame.loc[name, column]) for column in frame.columns
            ]
            rows.append([str(name)] + values)
        self._fill_table(table, rows)

    def _fill_contribution(self, frame: Any) -> None:
        """贡献归因表（契约列 :data:`CONTRIBUTION_COLUMNS`）。"""
        rows: list[list[str]] = []
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, row in frame.iterrows():
                rows.append(
                    [
                        str(row.get("member", "")),
                        format_pct(row.get("weight")),
                        format_num(row.get("initial_cash")),
                        format_num(row.get("final_assets")),
                        format_num(row.get("pnl")),
                        format_pct(row.get("member_return")),
                        format_pct(row.get("contribution")),
                        format_pct(row.get("dd_contribution")),
                    ]
                )
        self._fill_table(self.contribution_table, rows)

    def _fill_trades(self, trades: Any) -> None:
        """组合成交明细表（契约列 :data:`PORTFOLIO_TRADE_COLUMNS`，末列为成员名）。"""
        rows: list[list[str]] = []
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            for _, row in trades.iterrows():
                side = row.get("side")
                rows.append(
                    [
                        format_date(row.get("date")),
                        str(row.get("symbol", "")),
                        _SIDE_LABELS.get(str(side), str(side)),
                        format_num(row.get("price")),
                        format_count(row.get("qty")),
                        format_num(row.get("amount")),
                        format_num(row.get("cost")),
                        format_num(row.get("pnl")),
                        str(row.get("member", "")),
                    ]
                )
        self._fill_table(self.trades_table, rows)

    def _clear_result_views(self) -> None:
        """清空曲线 / 全部表格 / 警告。"""
        self._clear_charts()
        for table in (
            self.metrics_table,
            self.yearly_table,
            self.regime_table,
            self.members_table,
            self.correlation_table,
            self.contribution_table,
            self.trades_table,
        ):
            table.setRowCount(0)
        self.warnings_label.setText("")
        self.warnings_label.setVisible(False)

    # ------------------------------------------------------------------ #
    # 警告 / 状态
    # ------------------------------------------------------------------ #
    def _show_warnings(self, warnings: Any) -> None:
        """展示回测警告（无警告则隐藏）。"""
        items = [str(item) for item in (warnings or [])]
        if not items:
            self.warnings_label.setText("")
            self.warnings_label.setVisible(False)
            return
        self.warnings_label.setStyleSheet(f"color: {colors.LIMIT_COLOR};")
        self.warnings_label.setText("组合回测警告：" + "；".join(items))
        self.warnings_label.setVisible(True)

    def _set_status(self, text: str, ok: bool = True) -> None:
        """更新状态标签（成功绿色 / 失败红色，语义为状态）。"""
        color = colors.DOWN_COLOR if ok else colors.UP_COLOR
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)


__all__ = [
    "PortfolioBacktestDialog",
    "WINDOW_TITLE",
]