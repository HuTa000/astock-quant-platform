"""多因子打分选股对话框（P7-S5）。

定位：**模态对话框**，把 P7-S5「ICIR 动态加权打分 → Top-N 选股 → 逐日回放」
从研究结论落到界面上：
    * 配置打分口径（因子勾选 / Top-N / 调仓频率 / 股票池 / 权重前瞻期 /
      最少 IC 样本 / 起止日期）；
    * 配置风控参数（P7-S6，可选：止损 / 固定止盈 / 移动止盈 / 冷却期 /
      牛熊环境开关，默认全部关闭）；
    * 运行（:class:`~app.engine.backtest.scored.ScoredBacktestRunner`），后台线程执行；
    * 展示：净值曲线（策略 vs 基准）、绩效指标、逐年表现、牛熊拆分、
      最新权重（:data:`WEIGHT_TABLE_COLUMNS`）、最新选股名单、成交明细、
      警告与状态。

架构约定（与 ``strategy_analysis`` / ``factor_analysis`` 同构）：
    * **业务动作 = 无弹窗的纯方法**（:meth:`current_spec` / :meth:`run_scored_backtest`），
      测试可直接调用并断言（注入 fake runner 时无线程、无网络）；
    * **UI 触达层**（运行按钮）经 :class:`_ScoreWorker`（``QThread``）后台执行，
      结果经 Qt 信号回 UI 线程刷新，**工作线程不碰任何 UI**；
    * **绝不向 UI 抛异常**：引擎 / 绘图异常一律捕获，状态标签显示可读原因；
    * 本对话框**只读**本地库（打分与回测都不写库），不做任何数据更新。

口径提示（展示给用户）：
    * T 日收盘打分 → T+1 开盘成交（无未来函数）；前复权；
    * 权重只用「已兑现」的历史 IC（扩展窗口 ICIR + 兑现滞后），预热期不建仓；
    * PIT 成分股掩码消除幸存者偏差（库内无快照时显式失败而非静默降级）；
    * 100 股整手 + 默认现实费率（含双边最低佣金 5 元）+ 全额再平衡；
    * 风控（可选）：收盘判定 → 次日开盘成交（无未来函数）；A/B 实测本策略上
      各规则均降低收益（P7-S6 结论），界面对用户如实披露、默认全部关闭。
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
    QCheckBox,
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
from app.engine.backtest import (
    DEFAULT_COOLDOWN_DAYS,
    DEFAULT_REGIME_EXPOSURE,
    DEFAULT_REGIME_WINDOW,
    DEFAULT_STOP_LOSS_PCT,
    REBALANCE_CHOICES,
    TRADE_COLUMNS,
    RiskSpec,
    ScoredBacktestResult,
    ScoredBacktestRunner,
    ScoredBacktestSpec,
    yearly_summary,
)
from app.engine.factor import (
    DEFAULT_FACTOR_NAMES,
    DEFAULT_IC_HORIZON,
    DEFAULT_MIN_IC_DAYS,
    DEFAULT_SCORING_FACTORS,
    WEIGHT_TABLE_COLUMNS,
    get_factor,
)
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
WINDOW_TITLE: str = "多因子打分选股"

# 调仓频率中文显示名。
_REBALANCE_LABELS: dict[str, str] = {
    "daily": "日频",
    "weekly": "周频",
    "monthly": "月频",
}

# 股票池口径显示名（data 为引擎取值）。
_UNIVERSE_LABELS: tuple[tuple[str, str], ...] = (
    ("PIT 成分股（推荐，消除幸存者偏差）", "pit"),
    ("当前可见标的", "current"),
)

# 成交方向中文显示名。
_SIDE_LABELS: dict[str, str] = {"buy": "买入", "sell": "卖出"}

# 风控参数区起始建议值（止损取引擎默认；固定止盈 20% / 移动止盈 15% 为研究起点）。
_DEFAULT_TAKE_PROFIT_PCT: float = 20.0
_DEFAULT_TRAILING_STOP_PCT: float = 15.0

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
class _ScoreWorker(QThread):
    """在后台线程执行一次 :meth:`ScoredBacktestRunner.run`（结果经信号回传，不碰 UI）。"""

    done = Signal(object)  # ScoredBacktestResult 或 Exception

    def __init__(
        self,
        runner: ScoredBacktestRunner,
        spec: ScoredBacktestSpec,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造后台打分回测线程（parent 默认 None，生命周期由 closeEvent 管理）。"""
        super().__init__(parent)
        self._runner = runner
        self._spec = spec

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：执行打分回测，成功 / 失败均经 ``done`` 信号回传。"""
        try:
            result = self._runner.run(self._spec)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台打分回测线程异常")
            self.done.emit(exc)
            return
        self.done.emit(result)


# --------------------------------------------------------------------------- #
# 对话框
# --------------------------------------------------------------------------- #
class ScoreBacktestDialog(QDialog):
    """多因子打分选股对话框（配置 → 打分回测 → 曲线 + 明细视图）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        runner: Optional[ScoredBacktestRunner] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`（只读）。
            runner: 打分回测编排器；默认用 ``db`` 构造 :class:`ScoredBacktestRunner`。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._runner: ScoredBacktestRunner = (
            runner if runner is not None else ScoredBacktestRunner(db=self._db)
        )
        self._last_result: Optional[ScoredBacktestResult] = None
        self._worker: Optional[_ScoreWorker] = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1120, 860)
        self._setup_ui()
        self._set_status("设置打分参数后点击『运行打分并回测』开始", ok=True)

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
        """搭建「打分配置 / 净值曲线 / 明细标签页 / 警告」四段式布局。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        root.addWidget(self._build_config_box())
        root.addWidget(self._build_risk_box())
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
        """打分配置区：因子勾选 / Top-N / 调仓 / 股票池 / 口径 / 起止日期 / 运行。"""
        box = QGroupBox("打分配置", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        hint = QLabel(
            "口径：T 日收盘打分 → T+1 开盘成交（无未来函数）；前复权；PIT 成分掩码；"
            "权重只用到「已兑现」的历史 IC（预热期不建仓）；100 股整手 + 默认现实费率"
            "（含双边最低佣金 5 元）+ 全额再平衡。",
            box,
        )
        hint.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        body = QHBoxLayout()
        body.setSpacing(8)

        # 左：因子勾选列表（列出全部注册因子，默认勾选核心三因子）。
        factor_column = QVBoxLayout()
        factor_column.setSpacing(2)
        factor_column.addWidget(QLabel("参与打分的因子（勾选）：", box))
        self.factor_list = QListWidget(box)
        self.factor_list.setMaximumHeight(180)
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
        factor_column.addWidget(self.factor_list)
        body.addLayout(factor_column, 3)

        # 右：参数。
        params_column = QVBoxLayout()
        params_column.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Top-N（每日持有）：", box))
        self.top_n_spin = QSpinBox(box)
        self.top_n_spin.setRange(1, 200)
        self.top_n_spin.setValue(30)
        top_row.addWidget(self.top_n_spin)
        top_row.addWidget(QLabel("调仓频率：", box))
        self.rebalance_combo = QComboBox(box)
        for code in REBALANCE_CHOICES:
            self.rebalance_combo.addItem(_REBALANCE_LABELS.get(code, code), code)
        self.rebalance_combo.setCurrentIndex(
            max(0, list(REBALANCE_CHOICES).index("monthly"))
        )
        top_row.addWidget(self.rebalance_combo)
        top_row.addStretch(1)
        params_column.addLayout(top_row)

        universe_row = QHBoxLayout()
        universe_row.addWidget(QLabel("股票池：", box))
        self.universe_combo = QComboBox(box)
        for label, code in _UNIVERSE_LABELS:
            self.universe_combo.addItem(label, code)
        universe_row.addWidget(self.universe_combo, 1)
        params_column.addLayout(universe_row)

        gate_row = QHBoxLayout()
        gate_row.addWidget(QLabel("权重前瞻期（交易日）：", box))
        self.ic_horizon_spin = QSpinBox(box)
        self.ic_horizon_spin.setRange(1, 120)
        self.ic_horizon_spin.setValue(DEFAULT_IC_HORIZON)
        gate_row.addWidget(self.ic_horizon_spin)
        gate_row.addWidget(QLabel("最少 IC 样本：", box))
        self.min_ic_days_spin = QSpinBox(box)
        self.min_ic_days_spin.setRange(20, 1000)
        self.min_ic_days_spin.setValue(DEFAULT_MIN_IC_DAYS)
        gate_row.addWidget(self.min_ic_days_spin)
        gate_row.addStretch(1)
        params_column.addLayout(gate_row)

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
        self.run_button = QPushButton("运行打分并回测", box)
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        params_column.addLayout(run_row)
        params_column.addStretch(1)

        body.addLayout(params_column, 4)
        layout.addLayout(body)
        return box

    def _build_risk_box(self) -> QGroupBox:
        """风控参数区（P7-S6，可选）：止损 / 固定止盈 / 移动止盈 / 冷却 / 牛熊环境开关。

        默认**全部不勾选**（不启用任何风控）：真实库 A/B 实测（本策略
        「短期反转 + 流动性」）显示各规则均降低收益，界面如实披露，
        用户可勾选研究各规则的独立效果。
        """
        box = QGroupBox("风控参数（可选，默认全部关闭）", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        hint = QLabel(
            "实测结论：本策略（短期反转 + 流动性）上止损 / 止盈 / 环境开关均降低收益，"
            "默认关闭；如需研究请勾选启用。口径：收盘判定 → 次日开盘成交（无未来函数），"
            "触发退出后冷却期内该标的不再买回。",
            box,
        )
        hint.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        rules_row = QHBoxLayout()
        rules_row.setSpacing(6)

        self.stop_loss_check = QCheckBox("止损", box)
        self.stop_loss_spin = self._make_pct_spin(DEFAULT_STOP_LOSS_PCT * 100.0, box)
        rules_row.addWidget(self.stop_loss_check)
        rules_row.addWidget(self.stop_loss_spin)

        self.take_profit_check = QCheckBox("固定止盈", box)
        self.take_profit_spin = self._make_pct_spin(_DEFAULT_TAKE_PROFIT_PCT, box)
        rules_row.addWidget(self.take_profit_check)
        rules_row.addWidget(self.take_profit_spin)

        self.trailing_stop_check = QCheckBox("移动止盈", box)
        self.trailing_stop_spin = self._make_pct_spin(_DEFAULT_TRAILING_STOP_PCT, box)
        rules_row.addWidget(self.trailing_stop_check)
        rules_row.addWidget(self.trailing_stop_spin)

        rules_row.addWidget(QLabel("冷却期：", box))
        self.cooldown_spin = QSpinBox(box)
        self.cooldown_spin.setRange(0, 250)
        self.cooldown_spin.setValue(DEFAULT_COOLDOWN_DAYS)
        self.cooldown_spin.setSuffix(" 个交易日")
        rules_row.addWidget(self.cooldown_spin)
        rules_row.addStretch(1)
        layout.addLayout(rules_row)

        regime_row = QHBoxLayout()
        regime_row.setSpacing(6)
        self.regime_filter_check = QCheckBox("牛熊环境开关", box)
        regime_row.addWidget(self.regime_filter_check)
        regime_row.addWidget(QLabel("均线窗口：", box))
        self.regime_window_spin = QSpinBox(box)
        self.regime_window_spin.setRange(20, 500)
        self.regime_window_spin.setValue(DEFAULT_REGIME_WINDOW)
        self.regime_window_spin.setSuffix(" 日")
        regime_row.addWidget(self.regime_window_spin)
        regime_row.addWidget(QLabel("熊市仓位上限：", box))
        self.regime_exposure_spin = QSpinBox(box)
        self.regime_exposure_spin.setRange(5, 100)
        self.regime_exposure_spin.setValue(int(round(DEFAULT_REGIME_EXPOSURE * 100)))
        self.regime_exposure_spin.setSuffix(" %")
        regime_row.addWidget(self.regime_exposure_spin)
        regime_row.addStretch(1)
        layout.addLayout(regime_row)

        for checkbox in (
            self.stop_loss_check,
            self.take_profit_check,
            self.trailing_stop_check,
            self.regime_filter_check,
        ):
            checkbox.toggled.connect(self._sync_risk_enabled)
        self._sync_risk_enabled()
        return box

    @staticmethod
    def _make_pct_spin(value: float, parent: QWidget) -> QDoubleSpinBox:
        """构造统一风格的百分比输入框（1% ~ 99%，一位小数）。"""
        spin = QDoubleSpinBox(parent)
        spin.setRange(1.0, 99.0)
        spin.setDecimals(1)
        spin.setSuffix(" %")
        spin.setValue(value)
        return spin

    def _sync_risk_enabled(self) -> None:
        """按勾选状态联动启用 / 禁用各风控参数控件（未勾选项不可编辑，防误读）。"""
        self.stop_loss_spin.setEnabled(self.stop_loss_check.isChecked())
        self.take_profit_spin.setEnabled(self.take_profit_check.isChecked())
        self.trailing_stop_spin.setEnabled(self.trailing_stop_check.isChecked())
        exits_on = (
            self.stop_loss_check.isChecked()
            or self.take_profit_check.isChecked()
            or self.trailing_stop_check.isChecked()
        )
        self.cooldown_spin.setEnabled(exits_on)
        regime_on = self.regime_filter_check.isChecked()
        self.regime_window_spin.setEnabled(regime_on)
        self.regime_exposure_spin.setEnabled(regime_on)

    def _build_charts(self) -> QVBoxLayout:
        """净值曲线（策略 vs 基准），图例风格与策略分析保持一致。"""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        legend = QHBoxLayout()
        legend.setSpacing(8)
        self.nav_strategy_label = QLabel("策略净值 —", self)
        self.nav_strategy_label.setStyleSheet(f"color: {colors.ACCENT_COLOR};")
        self.nav_benchmark_label = QLabel("基准净值 —", self)
        self.nav_benchmark_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        legend.addWidget(self.nav_strategy_label)
        legend.addWidget(self.nav_benchmark_label)
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
        self.nav_plot.setMinimumHeight(180)
        self.nav_plot.setMaximumHeight(240)
        self.nav_strategy_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.ACCENT_COLOR, width=1.6)
        )
        self.nav_benchmark_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.SUBTEXT_COLOR, width=1.2)
        )
        self.nav_plot.addItem(self.nav_strategy_curve)
        self.nav_plot.addItem(self.nav_benchmark_curve)
        layout.addWidget(self.nav_plot)
        return layout

    def _build_detail_tabs(self) -> QTabWidget:
        """明细标签页：绩效 / 逐年 / 牛熊 / 最新权重 / 最新名单 / 成交明细。"""
        tabs = QTabWidget(self)

        self.metrics_table = self._make_table(["指标", "数值"], tabs)
        tabs.addTab(self.metrics_table, "绩效指标")

        self.yearly_table = self._make_table(list(_YEARLY_HEADERS), tabs)
        tabs.addTab(self.yearly_table, "逐年表现")

        self.regime_table = self._make_table(list(_REGIME_HEADERS), tabs)
        tabs.addTab(self.regime_table, "牛熊拆分")

        self.weights_table = self._make_table(list(WEIGHT_TABLE_COLUMNS), tabs)
        tabs.addTab(self.weights_table, "最新权重")

        self.picks_table = self._make_table(["序号", "代码"], tabs)
        tabs.addTab(self.picks_table, "最新选股名单")

        self.trades_table = self._make_table(list(TRADE_COLUMNS), tabs)
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
        """当前勾选的因子名（保持注册顺序）。"""
        chosen: list[str] = []
        for index in range(self.factor_list.count()):
            item = self.factor_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return chosen

    def current_rebalance(self) -> str:
        """返回当前调仓频率（``daily`` / ``weekly`` / ``monthly``）。"""
        data = self.rebalance_combo.currentData()
        return str(data) if data else str(REBALANCE_CHOICES[0])

    def current_universe(self) -> str:
        """返回当前股票池口径（``pit`` / ``current``）。"""
        data = self.universe_combo.currentData()
        return str(data) if data else "pit"

    def current_dates(self) -> tuple[str, str]:
        """返回起止日期（``YYYY-MM-DD``）。"""
        return (
            self.start_date_edit.date().toString("yyyy-MM-dd"),
            self.end_date_edit.date().toString("yyyy-MM-dd"),
        )

    def last_result(self) -> Optional[ScoredBacktestResult]:
        """返回最近一次打分回测结果；尚未运行返回 ``None``。"""
        return self._last_result

    # ------------------------------------------------------------------ #
    # 规格构建（无弹窗纯方法）
    # ------------------------------------------------------------------ #
    def _risk_any_checked(self) -> bool:
        """风控参数区是否有任一规则被勾选。"""
        return (
            self.stop_loss_check.isChecked()
            or self.take_profit_check.isChecked()
            or self.trailing_stop_check.isChecked()
            or self.regime_filter_check.isChecked()
        )

    def current_risk_spec(self) -> Optional[RiskSpec]:
        """用当前风控参数区构建 :class:`RiskSpec`；全部未勾选返回 ``None``（不启用风控）。

        未勾选的退出规则**显式传 ``None``**（覆盖 :class:`RiskSpec` 的止损默认值，
        防止「只勾环境开关」时意外带入默认止损）；冷却 / 环境参数一并带上
        （对应规则未启用时引擎不会使用）。参数非法时状态标签提示并返回
        ``None``（**不抛异常**）。
        """
        if not self._risk_any_checked():
            return None
        try:
            spec = RiskSpec(
                stop_loss_pct=(
                    float(self.stop_loss_spin.value()) / 100.0
                    if self.stop_loss_check.isChecked()
                    else None
                ),
                take_profit_pct=(
                    float(self.take_profit_spin.value()) / 100.0
                    if self.take_profit_check.isChecked()
                    else None
                ),
                trailing_stop_pct=(
                    float(self.trailing_stop_spin.value()) / 100.0
                    if self.trailing_stop_check.isChecked()
                    else None
                ),
                cooldown_days=int(self.cooldown_spin.value()),
                regime_filter=self.regime_filter_check.isChecked(),
                regime_window=int(self.regime_window_spin.value()),
                regime_exposure=float(self.regime_exposure_spin.value()) / 100.0,
            )
        except Exception as exc:  # noqa: BLE001 - 参数非法不外抛
            logger.warning("风控参数非法：%s", exc)
            self._set_status(f"风控参数不合法：{exc}", ok=False)
            return None
        return spec

    def current_spec(self) -> Optional[ScoredBacktestSpec]:
        """用当前界面参数构建 :class:`ScoredBacktestSpec`；无勾选因子 / 参数非法返回 ``None``。"""
        factors = self.selected_factors()
        if not factors:
            self._set_status("请至少勾选一个因子", ok=False)
            return None
        start, end = self.current_dates()
        risk = self.current_risk_spec()
        if risk is None and self._risk_any_checked():
            return None  # 风控参数非法：current_risk_spec 已提示（防御性分支）
        try:
            spec = ScoredBacktestSpec(
                factors=factors,
                top_n=int(self.top_n_spin.value()),
                rebalance=self.current_rebalance(),
                universe=self.current_universe(),
                start=start,
                end=end,
                ic_horizon=int(self.ic_horizon_spin.value()),
                min_ic_days=int(self.min_ic_days_spin.value()),
                risk=risk,
            )
        except Exception as exc:  # noqa: BLE001 - 参数非法不外抛
            logger.warning("打分参数非法：%s", exc)
            self._set_status(f"打分参数不合法：{exc}", ok=False)
            return None
        return spec

    # ------------------------------------------------------------------ #
    # 运行打分回测（同步纯方法 + 后台线程触达层）
    # ------------------------------------------------------------------ #
    def run_scored_backtest(self) -> Optional[ScoredBacktestResult]:
        """同步执行打分回测并把结果刷新到曲线 / 表格 / 状态标签（**不抛异常**）。

        供测试直接调用（注入 fake runner 时无线程、无网络）。无有效因子 / 参数
        非法时提示并返回 ``None``；引擎异常一律捕获后返回 ``None``。
        """
        spec = self.current_spec()
        if spec is None:
            return None
        try:
            result = self._runner.run(spec)
        except Exception as exc:  # noqa: BLE001 - 引擎异常不外抛
            logger.exception("打分回测执行失败")
            self._last_result = None
            self._clear_result_views()
            self._set_status(f"回测失败：{exc}", ok=False)
            return None
        self._last_result = result
        self._apply_result(result)
        return result

    def _on_run(self) -> None:
        """「运行打分并回测」按钮：在后台线程执行，避免长区间回测阻塞 UI。"""
        spec = self.current_spec()
        if spec is None:
            return
        if self._worker is not None and self._worker.isRunning():
            logger.info("打分回测已在进行中，忽略重复触发")
            return
        self.run_button.setEnabled(False)
        self._set_status("正在打分并回测…（全市场打分可能需要一些时间）", ok=True)
        # worker 的 parent 置为 None：对话框析构不连带销毁运行中的 QThread，
        # 生命周期统一交由 closeEvent 的 wait() + _on_worker_finished 管理。
        self._worker = _ScoreWorker(self._runner, spec)
        self._worker.done.connect(self._on_run_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_run_finished(self, payload: object) -> None:
        """后台回测结束：恢复按钮并把结果刷新到 UI（本槽运行在 UI 线程）。"""
        self.run_button.setEnabled(True)
        if isinstance(payload, ScoredBacktestResult):
            self._last_result = payload
            self._apply_result(payload)
        else:
            self._last_result = None
            self._clear_result_views()
            self._set_status(f"回测失败：{payload}", ok=False)

    def _on_worker_finished(self) -> None:
        """线程结束后释放引用。"""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------ #
    # 结果刷新
    # ------------------------------------------------------------------ #
    def _apply_result(self, result: ScoredBacktestResult) -> None:
        """把打分回测结果填充到曲线 / 表格 / 状态标签（UI 线程调用）。"""
        if not getattr(result, "ok", False):
            self._clear_result_views()
            self._show_warnings(getattr(result, "warnings", None))
            self._set_status(
                f"回测未完成：{getattr(result, 'error', '') or '未知原因'}", ok=False
            )
            return

        backtest = result.backtest
        scoring = result.scoring
        equity = pd.DataFrame()
        if backtest is not None and isinstance(backtest.equity, pd.DataFrame):
            equity = backtest.equity

        if backtest is not None:
            self._fill_metrics(backtest.metrics)
            self._fill_yearly(yearly_summary(equity))
            regime = (
                backtest.regime if isinstance(backtest.regime, pd.DataFrame) else pd.DataFrame()
            )
            self._fill_regime(regime)
            self._fill_trades(backtest.trades)
        else:
            self._fill_table(self.metrics_table, [])
            self._fill_table(self.yearly_table, [])
            self._fill_table(self.regime_table, [])
            self._fill_table(self.trades_table, [])

        self._fill_weights(scoring.latest_weights if scoring is not None else None)
        self._fill_picks(result.picks)
        self._plot_equity(equity, backtest)
        self._show_warnings(result.warnings)

        universe_count = len(scoring.universe) if scoring is not None else 0
        scored_days = int(scoring.scored_days) if scoring is not None else 0
        pick_days = len(result.picks)
        metrics = backtest.metrics if backtest is not None else None
        self._set_status(
            f"打分回测完成：研究池 {universe_count} 只 / 打分 {scored_days} 天 / "
            f"信号名单 {pick_days} 天 / 总收益 {format_pct(getattr(metrics, 'total_return', None))} / "
            f"年化 {format_pct(getattr(metrics, 'annual_return', None))} / "
            f"夏普 {format_num(getattr(metrics, 'sharpe', None))} / "
            f"最大回撤 {format_pct(getattr(metrics, 'max_drawdown', None))}",
            ok=True,
        )

    # ------------------------------------------------------------------ #
    # 图表刷新
    # ------------------------------------------------------------------ #
    def _plot_equity(self, equity: pd.DataFrame, backtest: Any) -> None:
        """刷新净值曲线（策略 vs 基准），标题展示基准代码 / 口径（**不抛异常**）。"""
        try:
            symbol = str(getattr(backtest, "benchmark_symbol", "") or "—")
            kind = str(getattr(backtest, "benchmark_kind", "") or "—")
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
            self.nav_plot.getViewBox().enableAutoRange()

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

            nav_last = nav[-1] if len(nav) else float("nan")
            self.nav_strategy_label.setText(
                f"策略净值 {format_num(nav_last)}（末值）"
                if not math.isnan(nav_last)
                else "策略净值 —"
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
            logger.exception("刷新净值曲线失败")
            self._set_status(f"曲线绘制失败：{exc}", ok=False)

    def _clear_charts(self) -> None:
        """清空两条净值曲线与轴标签。"""
        self.nav_strategy_curve.setData([], [])
        self.nav_benchmark_curve.setData([], [])
        self.nav_axis.set_labels([])
        self.nav_strategy_label.setText("策略净值 —")
        self.nav_benchmark_label.setText("基准净值 —")
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

    def _fill_weights(self, frame: Any) -> None:
        """最新权重诊断表（契约列 :data:`WEIGHT_TABLE_COLUMNS`）。"""
        rows: list[list[str]] = []
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, row in frame.iterrows():
                rows.append(
                    [
                        str(row.get("factor", "")),
                        str(row.get("label", "")),
                        format_count(row.get("n_ic_days")),
                        format_num(row.get("ic_mean")),
                        format_num(row.get("ic_std")),
                        format_num(row.get("ic_ir")),
                        format_num(row.get("weight")),
                    ]
                )
        self._fill_table(self.weights_table, rows)

    def _fill_picks(self, picks: dict[pd.Timestamp, list[str]]) -> None:
        """最新选股名单（结束日打分窗口最后一天的 Top-N，每行 序号 / 代码）。"""
        rows: list[list[str]] = []
        if picks:
            last_day = max(picks)
            for order, symbol in enumerate(picks[last_day], start=1):
                rows.append([str(order), str(symbol)])
        self._fill_table(self.picks_table, rows)

    def _fill_trades(self, trades: Any) -> None:
        """成交明细表（契约列 :data:`TRADE_COLUMNS`）。"""
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
            self.weights_table,
            self.picks_table,
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
        self.warnings_label.setText("回测警告：" + "；".join(items))
        self.warnings_label.setVisible(True)

    def _set_status(self, text: str, ok: bool = True) -> None:
        """更新状态标签（成功绿色 / 失败红色，语义为状态）。"""
        color = colors.DOWN_COLOR if ok else colors.UP_COLOR
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)


__all__ = [
    "ScoreBacktestDialog",
    "WINDOW_TITLE",
]