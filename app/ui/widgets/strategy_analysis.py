"""策略分析对话框（P7-S2）。

定位：**模态对话框**，把 P7-S1 回测引擎 + P7-S2 归因模块的产出落到界面上：
    * 配置策略（技术面表达式条件 + 调仓频率 + 持有上限 + 排序表达式 + 起止日期）；
    * 运行回测（:class:`~app.engine.backtest.engine.BacktestEngine`），后台线程执行；
    * 展示：净值曲线（策略 vs 基准）、回撤曲线、绩效指标卡、逐年拆解、
      牛熊环境拆分、回撤区间明细、α/β 收益来源分解、回测警告。

架构约定（沿用 ``screener_dialog`` 的可测性思路）：
    * **业务动作 = 无弹窗的纯方法**（:meth:`current_conditions` / :meth:`build_spec` /
      :meth:`run_backtest`），测试可直接调用并断言（注入 fake 引擎时无线程、无网络）；
    * **UI 触达层**（运行按钮）经 :class:`_BacktestWorker`（``QThread``）后台执行，
      结果经 Qt 信号回 UI 线程刷新，**工作线程不碰任何 UI**；
    * **绝不向 UI 抛异常**：引擎 / 归因 / 绘图异常一律捕获，状态标签显示可读原因；
    * 本对话框**只读**本地库（回测引擎不写库），不做任何数据更新。

口径提示（展示给用户）：
    * 信号用 T 日及以前的历史数据计算，**T+1 开盘成交**（规避前视偏差）；
    * 前复权价格、100 股整手、全额再平衡（先卖后买）；
    * 成交成本默认含佣金（万 2.5、双边最低 5 元）+ 印花税（卖出万 5）+
      过户费（双边万 0.1）+ 滑点；
    * 股票池默认取当前自选股（空则用本地库全部有日线的标的）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QThread, Signal
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
    REBALANCE_CHOICES,
    BacktestEngine,
    BacktestResult,
    StrategySpec,
    drawdown_episodes,
    drawdown_series,
    jensen_decomposition,
    yearly_summary,
)
from app.engine.screener import Condition
from app.ui.themes import colors
from app.ui.widgets.kline_view import CompactAxisItem, DateAxisItem

logger = logging.getLogger(__name__)

# 对话框标题。
WINDOW_TITLE: str = "策略分析"

# 技术面表达式预置项（中文显示名 → 表达式；表达式可自由编辑）。
_EXPRESSION_PRESETS: tuple[tuple[str, str], ...] = (
    ("站上 20 日均线", "close - close.rolling(20).mean()"),
    ("站上 60 日均线", "close - close.rolling(60).mean()"),
    ("5 日动量", "close / close.shift(5) - 1"),
    ("20 日动量", "close / close.shift(20) - 1"),
    ("量比（相对 20 日均量）", "volume / volume.rolling(20).mean()"),
    ("收盘价", "close"),
)

# 固定操作符。
_OPS: tuple[str, ...] = (">", ">=", "<", "<=", "==", "!=")

# 调仓频率中文显示名。
_REBALANCE_LABELS: dict[str, str] = {
    "daily": "每日",
    "weekly": "每周",
    "monthly": "每月",
}

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

# 归因分解行（中文标签 / DECOMPOSITION_KEYS 键 / 格式类型）。
_DECOMPOSITION_ROWS: tuple[tuple[str, str, str], ...] = (
    ("策略年化收益", "strategy_annual", "pct"),
    ("基准年化收益", "benchmark_annual", "pct"),
    ("年化超额（策略 − 基准）", "excess_annual", "pct"),
    ("年化 α（选股贡献）", "alpha_annual", "pct"),
    ("β（大盘敏感度）", "beta", "num"),
    ("β 偏离贡献 (β−1)×基准年化", "beta_component", "pct"),
    ("α 贡献（= 年化 α）", "alpha_component", "pct"),
    ("残差（近似误差 + 噪声）", "residual", "pct"),
)

# 表格列头。
_YEARLY_HEADERS: tuple[str, ...] = (
    "年份",
    "交易日",
    "策略收益",
    "基准收益",
    "超额收益",
    "策略最深回撤",
    "基准最深回撤",
)
_REGIME_HEADERS: tuple[str, ...] = (
    "环境",
    "交易日",
    "天数占比",
    "策略收益",
    "基准收益",
    "超额收益",
)
_DRAWDOWN_HEADERS: tuple[str, ...] = (
    "峰值日",
    "谷底日",
    "修复日",
    "深度",
    "下跌交易日",
    "修复交易日",
    "已修复",
)


# --------------------------------------------------------------------------- #
# 格式化纯函数（模块级，供测试直接对拍）
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    """尽力把值转为 float；``None`` / 非数值 / NaN / ±inf → ``None``。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def format_pct(value: Any, digits: int = 2) -> str:
    """比率 → 百分比文本（如 ``0.1234 → "12.34%"``）；无效值 → ``"—"``。"""
    number = _to_float(value)
    if number is None:
        return "—"
    return f"{number * 100:.{digits}f}%"


def format_num(value: Any, digits: int = 4) -> str:
    """数值 → 定点文本（如 ``1.5 → "1.5000"``）；无效值 → ``"—"``。"""
    number = _to_float(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def format_count(value: Any) -> str:
    """计数 → 整数文本；无效值 → ``"—"``。"""
    number = _to_float(value)
    if number is None:
        return "—"
    return str(int(number))


def format_date(value: Any) -> str:
    """日期 → ``"YYYY-MM-DD"``；``None`` / 无效 → ``"—"``。"""
    if value is None:
        return "—"
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return str(value)
    return stamp.strftime("%Y-%m-%d")


def _format_by_kind(value: Any, kind: str) -> str:
    """按格式类型渲染单元格文本。"""
    if kind == "pct":
        return format_pct(value)
    if kind == "int":
        return format_count(value)
    return format_num(value)


# --------------------------------------------------------------------------- #
# 条件行
# --------------------------------------------------------------------------- #
@dataclass
class _ConditionRow:
    """一条条件行的控件集合（表达式 + 操作符 + 阈值 + 删除）。"""

    field_combo: QComboBox
    op_combo: QComboBox
    value_edit: QLineEdit
    layout: QHBoxLayout
    remove_button: QPushButton


class _BacktestWorker(QThread):
    """在后台线程执行一次 :meth:`BacktestEngine.run`（结果经信号回传，不碰 UI）。"""

    done = Signal(object)  # BacktestResult 或 Exception

    def __init__(
        self,
        engine: BacktestEngine,
        spec: StrategySpec,
        start: str,
        end: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造后台回测线程。

        Args:
            engine: 回测引擎。
            spec: 已构建的策略规格。
            start: 回测起始日（``YYYY-MM-DD``）。
            end: 回测结束日（``YYYY-MM-DD``）。
            parent: 父对象（默认 None，生命周期由对话框 closeEvent 管理）。
        """
        super().__init__(parent)
        self._engine = engine
        self._spec = spec
        self._start = start
        self._end = end

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：执行回测，成功 / 失败均经 ``done`` 信号回传。"""
        try:
            result = self._engine.run(self._spec, start=self._start, end=self._end)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台回测线程异常")
            self.done.emit(exc)
            return
        self.done.emit(result)


class StrategyAnalysisDialog(QDialog):
    """策略分析对话框（配置 → 回测 → 曲线 + 归因表）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        engine: Optional[BacktestEngine] = None,
        symbols: Optional[list[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`（回测只读）。
            engine: 回测引擎；默认用 ``db`` 构造 :class:`BacktestEngine`。
            symbols: 策略股票池；``None`` / 空列表 = 用本地库全部有日线的标的。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._engine: BacktestEngine = (
            engine if engine is not None else BacktestEngine(db=self._db)
        )
        self._symbols: Optional[list[str]] = list(symbols) if symbols else None

        # 动态条件行（与选股对话框同构）。
        self.condition_rows: list[_ConditionRow] = []
        self._last_result: Optional[BacktestResult] = None
        self._worker: Optional[_BacktestWorker] = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1120, 820)
        self._setup_ui()
        self.add_condition_row()
        self._set_status("设置策略参数后点击『运行回测』开始", ok=True)

    # ------------------------------------------------------------------ #
    # 线程生命周期（与 ScreenerDialog 同约定）
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
        """搭建「策略配置 / 曲线 / 明细标签页 / 警告」四段式布局。"""
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
        """策略配置区：名称 / 条件行 / 调仓 / 上限 / 排序 / 起止日期 / 运行。"""
        box = QGroupBox("策略配置", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        hint = QLabel(
            "口径：T 日收盘出信号 → T+1 开盘成交（无未来函数）；前复权；100 股整手；"
            "全额再平衡；含佣金 / 印花税 / 过户费 / 滑点；股票池 = 当前自选股（空则用本地库全部）。",
            box,
        )
        hint.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # (a) 条件编辑区
        self._conditions_container = QWidget(box)
        self.conditions_layout = QVBoxLayout(self._conditions_container)
        self.conditions_layout.setContentsMargins(0, 0, 0, 0)
        self.conditions_layout.setSpacing(4)
        layout.addWidget(self._conditions_container)

        self.add_condition_button = QPushButton("＋ 添加条件", box)
        self.add_condition_button.clicked.connect(self._on_add_condition)
        layout.addWidget(self.add_condition_button)

        # (b) 参数行
        params = QHBoxLayout()
        params.setSpacing(6)
        params.addWidget(QLabel("策略名：", box))
        self.strategy_name_edit = QLineEdit(box)
        self.strategy_name_edit.setPlaceholderText("如：站上20日线 · 周频 Top3")
        params.addWidget(self.strategy_name_edit, 2)

        params.addWidget(QLabel("调仓：", box))
        self.rebalance_combo = QComboBox(box)
        for code in REBALANCE_CHOICES:
            self.rebalance_combo.addItem(_REBALANCE_LABELS.get(code, code), code)
        self.rebalance_combo.setCurrentIndex(max(0, list(REBALANCE_CHOICES).index("weekly")))
        params.addWidget(self.rebalance_combo)

        params.addWidget(QLabel("最多持有：", box))
        self.top_n_spin = QSpinBox(box)
        self.top_n_spin.setRange(0, 100)
        self.top_n_spin.setValue(3)
        self.top_n_spin.setSpecialValueText("不限制")
        params.addWidget(self.top_n_spin)

        params.addWidget(QLabel("排序表达式：", box))
        self.rank_by_edit = QLineEdit(box)
        self.rank_by_edit.setPlaceholderText("值大者优先，如 close")
        self.rank_by_edit.setText("close")
        params.addWidget(self.rank_by_edit, 1)
        layout.addLayout(params)

        # (c) 日期 + 运行
        run_bar = QHBoxLayout()
        run_bar.setSpacing(6)
        run_bar.addWidget(QLabel("起始日：", box))
        self.start_date_edit = QDateEdit(QDate(2015, 1, 1), box)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        run_bar.addWidget(self.start_date_edit)
        run_bar.addWidget(QLabel("结束日：", box))
        self.end_date_edit = QDateEdit(QDate.currentDate(), box)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        run_bar.addWidget(self.end_date_edit)

        self.run_button = QPushButton("运行回测", box)
        self.run_button.clicked.connect(self._on_run)
        run_bar.addWidget(self.run_button)
        run_bar.addStretch(1)
        layout.addLayout(run_bar)
        return box

    def _build_charts(self) -> QVBoxLayout:
        """净值曲线（策略 vs 基准）+ 回撤曲线。"""
        layout = QVBoxLayout()
        layout.setSpacing(4)

        legend = QHBoxLayout()
        legend.setSpacing(8)
        self.nav_strategy_label = QLabel("策略净值 —", self)
        self.nav_strategy_label.setStyleSheet(f"color: {colors.ACCENT_COLOR};")
        self.nav_benchmark_label = QLabel("基准净值 —", self)
        self.nav_benchmark_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        self.dd_label = QLabel("回撤（峰值以来）—", self)
        self.dd_label.setStyleSheet(f"color: {colors.DOWN_COLOR};")
        legend.addWidget(self.nav_strategy_label)
        legend.addWidget(self.nav_benchmark_label)
        legend.addWidget(self.dd_label)
        legend.addStretch(1)
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
        self.nav_plot.setMinimumHeight(170)
        self.nav_plot.setMaximumHeight(220)
        self.nav_strategy_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.ACCENT_COLOR, width=1.6)
        )
        self.nav_benchmark_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors.SUBTEXT_COLOR, width=1.2)
        )
        self.nav_plot.addItem(self.nav_strategy_curve)
        self.nav_plot.addItem(self.nav_benchmark_curve)
        layout.addWidget(self.nav_plot)

        self.dd_axis = DateAxisItem(orientation="bottom")
        self.dd_plot = pg.PlotWidget(
            axisItems={"left": CompactAxisItem(orientation="left"), "bottom": self.dd_axis},
            parent=self,
        )
        self.dd_plot.setBackground(colors.BG_DARK)
        self.dd_plot.showGrid(x=False, y=True, alpha=0.25)
        self.dd_plot.setMenuEnabled(False)
        self.dd_plot.setMouseEnabled(x=False, y=False)
        self.dd_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self.dd_plot.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)
        self.dd_plot.setMinimumHeight(110)
        self.dd_plot.setMaximumHeight(150)
        self.dd_curve = pg.PlotDataItem([], [], pen=pg.mkPen(colors.DOWN_COLOR, width=1.4))
        self.dd_plot.addItem(self.dd_curve)
        layout.addWidget(self.dd_plot)
        return layout

    def _build_detail_tabs(self) -> QTabWidget:
        """明细标签页：指标 / 逐年 / 牛熊 / 回撤 / 归因。"""
        tabs = QTabWidget(self)

        self.metrics_table = self._make_table(["指标", "数值"], tabs)
        tabs.addTab(self.metrics_table, "绩效指标")

        self.yearly_table = self._make_table(list(_YEARLY_HEADERS), tabs)
        tabs.addTab(self.yearly_table, "逐年拆解")

        self.regime_table = self._make_table(list(_REGIME_HEADERS), tabs)
        tabs.addTab(self.regime_table, "牛熊拆分")

        self.drawdown_table = self._make_table(list(_DRAWDOWN_HEADERS), tabs)
        tabs.addTab(self.drawdown_table, "回撤区间")

        self.decomposition_table = self._make_table(["项目", "数值"], tabs)
        tabs.addTab(self.decomposition_table, "收益来源（α/β）")

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
    # 条件行管理
    # ------------------------------------------------------------------ #
    def add_condition_row(self) -> _ConditionRow:
        """在条件区新增一行（表达式 / 操作符 / 阈值 / 删除），返回该行。"""
        field_combo = QComboBox(self)
        field_combo.setEditable(True)
        for label, expression in _EXPRESSION_PRESETS:
            field_combo.addItem(label, expression)

        op_combo = QComboBox(self)
        for op in _OPS:
            op_combo.addItem(op)

        value_edit = QLineEdit(self)
        value_edit.setPlaceholderText("阈值（数值）")
        value_edit.setText("0")

        remove_button = QPushButton("删除", self)

        row_layout = QHBoxLayout()
        row_layout.setSpacing(4)
        row_layout.addWidget(field_combo, 3)
        row_layout.addWidget(op_combo, 1)
        row_layout.addWidget(value_edit, 1)
        row_layout.addWidget(remove_button)

        row = _ConditionRow(
            field_combo=field_combo,
            op_combo=op_combo,
            value_edit=value_edit,
            layout=row_layout,
            remove_button=remove_button,
        )
        self.condition_rows.append(row)
        self.conditions_layout.addLayout(row_layout)
        remove_button.clicked.connect(lambda _=False, r=row: self._remove_condition_row(r))
        return row

    def _remove_condition_row(self, row: _ConditionRow) -> None:
        """删除一条条件行（**至少保留一行**）。"""
        if len(self.condition_rows) <= 1:
            self._set_status("至少保留一个条件行", ok=False)
            return
        for widget in (row.field_combo, row.op_combo, row.value_edit, row.remove_button):
            row.layout.removeWidget(widget)
            widget.deleteLater()
        self.conditions_layout.removeItem(row.layout)
        self.condition_rows.remove(row)

    def _on_add_condition(self) -> None:
        """「＋ 添加条件」按钮。"""
        self.add_condition_row()

    # ------------------------------------------------------------------ #
    # 只读访问器（供测试 / 上层）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expression_text(field_combo: QComboBox) -> str:
        """取表达式：选中的预置项还原为其 ``itemData``，自由输入原样返回。

        说明：editable 的 ``QComboBox`` 在自由输入（如 ``close``）时当前文本与
        预置项文本不一致，此时应原样使用输入文本；只有当前文本恰好等于某个
        预置项文本（用户从下拉中选中）时才取该项的 ``itemData`` 表达式。
        """
        text = field_combo.currentText().strip()
        index = field_combo.currentIndex()
        if 0 <= index < field_combo.count() and field_combo.itemText(index) == text:
            data = field_combo.itemData(index)
            if data:
                return str(data).strip()
        return text

    def current_conditions(self) -> list[Condition]:
        """收集当前全部条件（**跳过**空表达式 / 空阈值 / 非数值阈值 / 非法操作符的行）。"""
        conditions: list[Condition] = []
        for row in self.condition_rows:
            field = self._expression_text(row.field_combo)
            op = row.op_combo.currentText().strip()
            value_text = row.value_edit.text().strip()
            if not field or not value_text:
                continue
            if op not in _OPS:
                continue
            try:
                value = float(value_text)
            except (TypeError, ValueError):
                continue
            conditions.append(Condition(field=field, op=op, value=value))
        return conditions

    def current_rebalance(self) -> str:
        """返回当前调仓频率（``daily`` / ``weekly`` / ``monthly``）。"""
        data = self.rebalance_combo.currentData()
        return str(data) if data else str(REBALANCE_CHOICES[0])

    def current_top_n(self) -> Optional[int]:
        """返回最多持有只数；``0``（界面「不限制」）→ ``None``。"""
        value = int(self.top_n_spin.value())
        return value if value > 0 else None

    def current_dates(self) -> tuple[str, str]:
        """返回起止日期（``YYYY-MM-DD``）。"""
        return (
            self.start_date_edit.date().toString("yyyy-MM-dd"),
            self.end_date_edit.date().toString("yyyy-MM-dd"),
        )

    def last_result(self) -> Optional[BacktestResult]:
        """返回最近一次回测结果；尚未运行返回 ``None``。"""
        return self._last_result

    # ------------------------------------------------------------------ #
    # 策略规格构建（无弹窗纯方法）
    # ------------------------------------------------------------------ #
    def build_spec(self) -> Optional[StrategySpec]:
        """用当前界面参数构建 :class:`StrategySpec`；无有效条件 / 参数非法返回 ``None``。"""
        conditions = self.current_conditions()
        if not conditions:
            self._set_status("没有有效条件，请先添加", ok=False)
            return None
        rank_by = self.rank_by_edit.text().strip() or None
        try:
            spec = StrategySpec(
                name=self.strategy_name_edit.text().strip(),
                conditions=conditions,
                logic="AND",
                symbols=self._symbols,
                rebalance=self.current_rebalance(),
                top_n=self.current_top_n(),
                rank_by=rank_by,
            )
        except Exception as exc:  # noqa: BLE001 - 参数非法不外抛
            logger.warning("策略参数非法：%s", exc)
            self._set_status(f"策略参数不合法：{exc}", ok=False)
            return None
        return spec

    # ------------------------------------------------------------------ #
    # 运行回测（同步纯方法 + 后台线程触达层）
    # ------------------------------------------------------------------ #
    def run_backtest(self) -> Optional[BacktestResult]:
        """同步执行回测并把结果刷新到曲线 / 表格 / 状态标签（**不抛异常**）。

        供测试直接调用（注入 fake 引擎时无线程、无网络）。策略无有效条件时
        提示并返回 ``None``。
        """
        spec = self.build_spec()
        if spec is None:
            return None
        start, end = self.current_dates()
        try:
            result = self._engine.run(spec, start=start, end=end)
        except Exception as exc:  # noqa: BLE001 - 引擎异常不外抛
            logger.exception("回测执行失败")
            self._last_result = None
            self._clear_result_views()
            self._set_status(f"回测失败：{exc}", ok=False)
            return None
        self._last_result = result
        self._apply_result(result)
        return result

    def _apply_result(self, result: BacktestResult) -> None:
        """把回测结果填充到曲线 / 表格 / 状态标签（UI 线程调用）。"""
        if not getattr(result, "ok", False):
            self._clear_result_views()
            self._show_warnings(getattr(result, "warnings", None))
            self._set_status(f"回测未完成：{getattr(result, 'error', '') or '未知原因'}", ok=False)
            return

        equity = result.equity if isinstance(result.equity, pd.DataFrame) else pd.DataFrame()
        self._fill_metrics(result.metrics)
        self._fill_yearly(yearly_summary(equity))
        regime = result.regime if isinstance(result.regime, pd.DataFrame) else pd.DataFrame()
        self._fill_regime(regime)
        self._fill_drawdown(drawdown_episodes(self._equity_nav(equity)))
        self._fill_decomposition(jensen_decomposition(equity))
        self._plot_equity(equity)
        self._show_warnings(result.warnings)
        self._set_status(
            f"回测完成：股票池 {len(result.universe)} 只 / 成交 {len(result.trades)} 笔 / "
            f"基准口径 {result.benchmark_kind or '—'}",
            ok=True,
        )

    def _on_run(self) -> None:
        """「运行回测」按钮：在后台线程执行，避免长区间回测阻塞 UI。"""
        spec = self.build_spec()
        if spec is None:
            return
        if self._worker is not None and self._worker.isRunning():
            logger.info("回测已在进行中，忽略重复触发")
            return
        start, end = self.current_dates()
        self.run_button.setEnabled(False)
        self._set_status("正在回测…", ok=True)
        # worker 的 parent 置为 None：对话框析构不连带销毁运行中的 QThread，
        # 生命周期统一交由 closeEvent 的 wait() + _on_worker_finished 管理。
        self._worker = _BacktestWorker(self._engine, spec, start, end)
        self._worker.done.connect(self._on_backtest_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_backtest_finished(self, payload: object) -> None:
        """后台回测结束：恢复按钮并把结果刷新到 UI（本槽运行在 UI 线程）。"""
        self.run_button.setEnabled(True)
        if isinstance(payload, BacktestResult):
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
    # 图表刷新
    # ------------------------------------------------------------------ #
    @staticmethod
    def _equity_nav(equity: pd.DataFrame) -> pd.Series:
        """从净值表提取策略净值序列（``DatetimeIndex``）；非法输入返回空 Series。"""
        if not isinstance(equity, pd.DataFrame) or equity.empty or "nav" not in equity.columns:
            return pd.Series(dtype=float)
        if "date" in equity.columns:
            stamps = pd.to_datetime(equity["date"], errors="coerce")
        else:
            stamps = pd.to_datetime(equity.index, errors="coerce")
        series = pd.Series(
            pd.to_numeric(equity["nav"], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(stamps),
        )
        return series[~series.index.isna()]

    def _plot_equity(self, equity: pd.DataFrame) -> None:
        """刷新净值曲线（策略 vs 基准）与回撤曲线（**不抛异常**）。"""
        try:
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
            self.dd_axis.set_labels(labels)
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

            if "drawdown" in equity.columns:
                ddown = pd.to_numeric(equity["drawdown"], errors="coerce").to_numpy(dtype=float)
            else:
                ddown = drawdown_series(pd.Series(nav)).to_numpy(dtype=float)
            self.dd_curve.setData(x_values, ddown)
            self.dd_plot.getViewBox().enableAutoRange()

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
            dd_min = self._opt_min(ddown)
            self.dd_label.setText(
                f"最深回撤 {format_pct(dd_min)}" if dd_min is not None else "回撤（峰值以来）—"
            )
        except Exception as exc:  # noqa: BLE001 - 绘图失败不影响表格展示
            logger.exception("刷新曲线失败")
            self._set_status(f"曲线绘制失败：{exc}", ok=False)

    @staticmethod
    def _opt_min(values: Any) -> Optional[float]:
        """数组最小值（去 NaN）；全 NaN / 空返回 ``None``。"""
        series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.min())

    def _clear_charts(self) -> None:
        """清空两条曲线与轴标签。"""
        for curve in (self.nav_strategy_curve, self.nav_benchmark_curve, self.dd_curve):
            curve.setData([], [])
        self.nav_axis.set_labels([])
        self.dd_axis.set_labels([])
        self.nav_strategy_label.setText("策略净值 —")
        self.nav_benchmark_label.setText("基准净值 —")
        self.dd_label.setText("回撤（峰值以来）—")

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
            value = getattr(metrics, attribute, None)
            rows.append([label, _format_by_kind(value, kind)])
        self._fill_table(self.metrics_table, rows)

    def _fill_yearly(self, frame: pd.DataFrame) -> None:
        """逐年拆解表（契约列见 ``attribution.YEARLY_COLUMNS``）。"""
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

    def _fill_drawdown(self, frame: pd.DataFrame) -> None:
        """回撤区间明细表（契约列见 ``attribution.DRAWDOWN_COLUMNS``）。"""
        rows: list[list[str]] = []
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, row in frame.iterrows():
                rows.append(
                    [
                        format_date(row.get("peak_date")),
                        format_date(row.get("trough_date")),
                        format_date(row.get("recovery_date")),
                        format_pct(row.get("depth")),
                        format_count(row.get("drawdown_days")),
                        format_count(row.get("recovery_days")),
                        "是" if bool(row.get("recovered")) else "否",
                    ]
                )
        self._fill_table(self.drawdown_table, rows)

    def _fill_decomposition(self, result: dict[str, Optional[float]]) -> None:
        """α/β 收益来源分解表（键见 ``attribution.DECOMPOSITION_KEYS``）。"""
        rows: list[list[str]] = []
        for label, key, kind in _DECOMPOSITION_ROWS:
            rows.append([label, _format_by_kind(result.get(key), kind)])
        self._fill_table(self.decomposition_table, rows)

    def _clear_result_views(self) -> None:
        """清空曲线 / 全部表格 / 警告。"""
        self._clear_charts()
        for table in (
            self.metrics_table,
            self.yearly_table,
            self.regime_table,
            self.drawdown_table,
            self.decomposition_table,
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
    "StrategyAnalysisDialog",
    "WINDOW_TITLE",
    "format_pct",
    "format_num",
    "format_count",
    "format_date",
]