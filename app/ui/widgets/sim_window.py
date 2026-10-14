"""模拟训练窗口（P5-S4 初版；P6 收尾优化：周期/起始日/固定显示区/按日回放/盈亏曲线；
P6 再优化：非模态多窗口 / 快捷下单 / 多标的切换）。

职责（《细化设计》§3.5 / §4「P5-S4」；本轮优化见《开发手册》§10 Changelog）：
    * 选股 + **周期**（5/15/30/60 分钟 / 日K；1 分钟链路不可用，选项保留但禁用）
      + **起始日**（默认本地数据首日，可自选）→ 从本地库构造
      :class:`~app.engine.simulator.SimContext` 回放；
    * **非模态独立窗口**（2026-10-02 用户要求）：由 ``MainWindow`` 以多窗口方式管理
      （``show()`` 而非 ``exec()``，主界面可继续操作），互不阻塞；
    * **视图受限 as_of**：主图只接收 ``SimContext.as_of()`` 返回的可见数据——未来数据
      **不可能**出现在图上（look-ahead 防护落到 UI 层）；
    * **固定显示区**：显示窗口槽位数固定（分钟周期 = 一个交易日 240min/周期，
      5m=48 / 15m=16 / 30m=8 / 60m=4；日K = 滚动 60 根），逐根显现时**图表宽度恒定**
      （解决「数据越多图越窄」的观感问题）；
    * **最新K线居中**（2026-10-07）：每次显现新根，视窗以**最新根为中心**水平滑动
      （宽度恒定 = 槽位数），无需鼠标移入即可跟随，避免大幅波动时最新 K 线跳出视野；
    * **自定义交易价**（2026-10-07）：下单前用滑钮在**当日（截至游标）震荡范围**
      ``[最低, 最高]`` 内选择成交价，主图以**黄色虚线**标注；买卖 / 平仓 / 快捷买入
      均按该价成交（未拖动时默认 = 最新收盘价，保持旧版行为）；
    * **按日回放**：图上每次只画**当前交易日**的 K 线（日K 一根 = 一天）；当日放完
      自动暂停并点亮「下一天」，用户主动点击才进入下一交易日；
    * QTimer 驱动 :meth:`SimEngine.tick`，倍速由 :func:`interval_for_speed` 决定
      （1x = 400ms，2026-10-01 用户要求调慢到原 1/2）；
    * **快捷下单辅助**（2026-10-02）：面板显示「可买 / 可卖股数」，并提供
      全仓 / 1/2仓 / 1/4仓 / 按金额 快捷买入按钮（股数按整手向下取整）；
    * **多标的**（2026-10-02）：回放中在标的下拉框输入/选择其它 6 位代码即切换回放
      标的（**账户与持仓不变**），可对多只股票分别买卖、分别持仓（各自 T+1），
      成交表带「标的」列、状态面板显示全部持仓；
    * 手动买入 / 卖出 / 平仓 + **条件单**（价格 / 涨跌幅，一次性；绑定添加时的标的）；
    * 右侧面板：现金 / 持仓 / 已实现盈亏 / 浮动盈亏 / 总资产 / 收益率 + **盈亏曲线**
      （随回放逐根记录累计盈亏与浮动盈亏）。

数据来源与空数据策略（需求方决策，2026-09-30 / 2026-10-01）：
    * **只读本地库**；分钟周期优先级：库内该周期（未复权）→ 15/30/60m 缺失时从本地
      5m **本地派生**（:func:`~app.data.resample.derive_minute_bars`，不落库）；日线
      未复权缺失时回退前复权；
    * 库中无数据 → 占位提示 + 开始按钮禁用（1 分钟链路依赖 pytdx，R12 未真机验证，
      故不做 1m 回放）。

⚠️ UI 层约定（§4.11）：带参槽一律 ``lambda`` 吞掉 Qt 隐式参数（如 ``clicked(bool)``）；
QTimer 在 UI 线程回调，引擎回调（``on_tick/on_fill/on_finished``）同步执行，无跨线程风险。
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import List, Optional, Tuple

import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QTimer, Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.data.resample import derive_minute_bars
from app.data.store.db import BAR_COLUMNS, Database, get_db
from app.engine.simulator import (
    BUY,
    SELL,
    ConditionalOrder,
    SimContext,
    SimEngine,
    max_buy_qty,
    qty_for_amount,
)
from app.ui.themes import colors
from app.ui.widgets.kline_view import PERIOD_LABELS, CompactAxisItem, KLineView

logger = logging.getLogger(__name__)

# 全局 pyqtgraph 选项（与 KLineView / TickView 一致；关闭抗锯齿以保性能）。
pg.setConfigOptions(antialias=False, background=colors.BG_DARK, foreground=colors.TEXT_COLOR)

# ---------------- 周期与固定显示区（纯函数，便于单测） ---------------- #
#: 周期下拉项 ``(编码, 文案)``；1m 保留但禁用（链路不可用，见 :data:`_DISABLED_PERIOD_HINT`）。
_PERIOD_CHOICES: tuple[tuple[str, str], ...] = (
    ("1m", "1分钟"),
    ("5m", "5分钟"),
    ("15m", "15分钟"),
    ("30m", "30分钟"),
    ("60m", "60分钟"),
    ("d", "日K"),
)
#: 禁用周期 → 悬停提示（QComboBox setItemData ToolTipRole）。
_DISABLED_PERIOD_HINT: dict[str, str] = {"1m": "1 分钟链路不可用，请用 5m"}
#: 默认周期（本地 5m 数据最全）。
_DEFAULT_PERIOD: str = "5m"
#: 分钟周期编码集合（区分「分钟」与「日」文案）。
_MINUTE_PERIODS: frozenset[str] = frozenset({"1m", "5m", "15m", "30m", "60m"})
#: 一个交易日的开市分钟数（上午 120 + 下午 120）。
_SESSION_MINUTES: int = 240
#: 分钟周期 → 单日 bar 数（240 / 周期分钟）：5m=48 / 15m=16 / 30m=8 / 60m=4。
_BARS_PER_DAY: dict[str, int] = {"5m": 48, "15m": 16, "30m": 8, "60m": 4}
#: 日K 滚动窗口根数（一根 = 一天；窗口固定，历史 K 线保留作参照）。
_DAILY_WINDOW_BARS: int = 60
#: 可由本地 5m 派生的分钟周期（库内缺失时使用，不落库）。
_DERIVE_FROM_5M: frozenset[str] = frozenset({"15m", "30m", "60m"})

# 空数据提示（分钟 / 日线分开，指向对应菜单项）。
_NO_MINUTE_TEXT: str = "本地暂无分钟数据，请先用『数据→更新分钟数据』拉取 5/15/30/60 分钟线"
_NO_DAILY_TEXT: str = "本地暂无日线数据，请先用『数据→更新数据』拉取日线"
# 占位 / 就绪提示。
_READY_TEXT: str = "就绪：选择标的 / 周期 / 起始日并点击「开始回放」"
_RUNNING_TEXT: str = "回放中…"
_PAUSED_TEXT: str = "已暂停"
_FINISHED_TEXT: str = "回放完成"
_DASH: str = "—"

# 倍速选项。
_SPEED_CHOICES: tuple[int, ...] = (1, 2, 5, 10)

# 条件单触发类型选项。
_TRIGGER_CHOICES: tuple[tuple[str, str], ...] = (("price", "价格"), ("pct", "涨跌幅"))


def window_slots(period: str) -> int:
    """固定显示窗口的 bar 槽位数（纯函数）。

    * 分钟周期 → **一个交易日**的 bar 数（``240 / 周期分钟``）：5m=48、15m=16、30m=8、60m=4；
    * 日K → 滚动窗口 :data:`_DAILY_WINDOW_BARS` 根（一根 = 一天）。

    Args:
        period: 周期编码。

    Returns:
        槽位数（正整数）。
    """
    return _BARS_PER_DAY.get(str(period), _DAILY_WINDOW_BARS)


def is_rolling_window(period: str) -> bool:
    """是否「滚动窗口」周期（日K：保留历史 K 线做参照，窗口整体左移）。

    Args:
        period: 周期编码。

    Returns:
        日K（非分钟周期）返回 ``True``；分钟周期返回 ``False``（仅显示当前交易日）。
    """
    return str(period) not in _BARS_PER_DAY


def _pnl_color(value: float) -> str:
    """盈亏数值配色：正红 / 负绿 / 零次文字（A 股红涨绿跌）。"""
    if value > 0:
        return colors.UP_COLOR
    if value < 0:
        return colors.DOWN_COLOR
    return colors.SUBTEXT_COLOR


class SimWindow(QDialog):
    """模拟训练窗口（**非模态**独立窗口：选股 → 回放 → 下单 → 复盘；可多开）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        symbols: Optional[List[str]] = None,
        default_symbol: Optional[str] = None,
        parent: Optional[QWidget] = None,
        initial_cash: float = 100_000.0,
    ) -> None:
        """构造训练窗口。

        Args:
            db: 存储实例；默认 :func:`get_db`。
            symbols: 可选标的候选（自选股列表）；空则允许手动输入任意 6 位代码。
            default_symbol: 默认标的。
            parent: 父控件。
            initial_cash: 初始资金（默认 100000；可在界面修改）。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._candidates: List[str] = [str(s) for s in (symbols or [])]
        self._engine: Optional[SimEngine] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: self._on_timer())
        # ---- 回放会话状态（按日回放 + 盈亏曲线）---- #
        self._bars: pd.DataFrame = pd.DataFrame(columns=BAR_COLUMNS)  # 本次会话全量（已按起始日切片）
        self._slots: int = 0                      # 本次会话的固定显示槽位数（含实测单日最大根数）
        self._day: Optional[date] = None          # 当前回放交易日
        self._day_complete: bool = False          # 当日已放完（等「下一天」）
        self._shown_this_day: int = 0             # 当前图上根数
        self._total_days: int = 0                 # 本次会话交易日数
        self._initial_cash: float = float(initial_cash)
        self._equity: List[Tuple[datetime, float, float]] = []  # (游标, 总资产, 浮动盈亏)
        self._trade_price: Optional[float] = None  # 自定义成交价（None → 用最新收盘价）

        self.setWindowTitle("模拟训练")
        self.resize(1240, 780)
        self._setup_ui()
        self._populate_symbols(default_symbol)
        self._set_status(_READY_TEXT)

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建布局：参数行 + 操作行 + 左（回放K线）+ 右（状态/盈亏曲线/下单/条件单/成交）。"""
        root = QVBoxLayout(self)

        # ---- 参数行：标的 / 周期 / 起始日 / 初始资金 / 倍速 ----
        params = QHBoxLayout()
        self._symbol_combo = QComboBox(self)
        self._symbol_combo.setEditable(True)
        self._symbol_combo.setPlaceholderText("6 位代码")
        self._symbol_combo.setToolTip("回放中可切换到其它 6 位代码（账户与持仓不变，可同时买卖多只）")
        # 换标的：回放中 → 切换回放标的；未开始 → 用本地数据范围刷新「起始日」。
        self._symbol_combo.currentTextChanged.connect(lambda t: self._on_symbol_text_changed(t))

        self._period_combo = QComboBox(self)
        for code, text in _PERIOD_CHOICES:
            self._period_combo.addItem(text, code)
        self._disable_period_items()
        default_idx = self._period_combo.findData(_DEFAULT_PERIOD)
        self._period_combo.setCurrentIndex(max(0, default_idx))
        # 换周期 → 本地数据范围随周期变化，刷新「起始日」。
        self._period_combo.currentIndexChanged.connect(lambda _i: self._refresh_date_range())

        self._date_edit = QDateEdit(self)
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDisplayFormat("yyyy-MM-dd")
        self._date_edit.setToolTip("回放起始日（含）；默认本地数据首日，可自选")

        self._cash_spin = QSpinBox(self)
        self._cash_spin.setRange(1000, 100_000_000)
        self._cash_spin.setValue(int(self._initial_cash))
        self._cash_spin.setSingleStep(10_000)
        self._cash_spin.setSuffix(" 元")

        self._speed_combo = QComboBox(self)
        for speed in _SPEED_CHOICES:
            self._speed_combo.addItem(f"{speed}x", speed)
        self._speed_combo.setCurrentIndex(0)
        # 倍速变更 → 同步引擎并刷新定时器间隔（§4.11：带参槽用 lambda 吞参数）。
        self._speed_combo.currentIndexChanged.connect(lambda _i: self._on_speed_changed())

        params.addWidget(QLabel("标的：", self))
        params.addWidget(self._symbol_combo, 1)
        params.addWidget(QLabel("周期：", self))
        params.addWidget(self._period_combo)
        params.addWidget(QLabel("起始日：", self))
        params.addWidget(self._date_edit)
        params.addWidget(QLabel("初始资金：", self))
        params.addWidget(self._cash_spin)
        params.addWidget(QLabel("倍速：", self))
        params.addWidget(self._speed_combo)
        root.addLayout(params)

        # ---- 操作行：开始 / 暂停 / 下一天 / 重置 + 状态 ----
        actions = QHBoxLayout()
        self._btn_start = QPushButton("开始回放", self)
        self._btn_start.clicked.connect(lambda: self.start_replay())
        self._btn_pause = QPushButton("暂停", self)
        self._btn_pause.clicked.connect(lambda: self.toggle_pause())
        self._btn_pause.setEnabled(False)
        self._btn_next_day = QPushButton("下一天", self)
        self._btn_next_day.setToolTip("当日 K 线放完后点击，进入下一交易日（每次只显示一天）")
        self._btn_next_day.clicked.connect(lambda: self.next_day())
        self._btn_next_day.setEnabled(False)
        self._btn_stop = QPushButton("重置", self)
        self._btn_stop.clicked.connect(lambda: self.stop_replay())
        self._btn_stop.setEnabled(False)

        actions.addWidget(self._btn_start)
        actions.addWidget(self._btn_pause)
        actions.addWidget(self._btn_next_day)
        actions.addWidget(self._btn_stop)
        actions.addStretch(1)
        self._status_label = QLabel("", self)
        self._status_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        actions.addWidget(self._status_label)
        root.addLayout(actions)

        # ---- 主体：左图右板 ----
        body = QHBoxLayout()
        root.addLayout(body, 1)

        # 左：回放 K 线主图（复用 KLineView：蜡烛 + 量柱 + 固定显示区）。
        self._chart = KLineView(db=self._db, parent=self)
        # 训练窗口不接导出右键菜单（MainWindow 的导出接线不在此处，避免死菜单）。
        self._chart.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        body.addWidget(self._chart, 2)

        # 右：状态 / 盈亏曲线 / 下单 / 条件单 / 成交记录。
        right = QVBoxLayout()
        body.addLayout(right, 1)
        right.addWidget(self._build_state_box(), 0)
        right.addWidget(self._build_equity_box(), 0)
        right.addWidget(self._build_order_box(), 0)
        right.addWidget(self._build_condition_box(), 0)
        right.addWidget(self._build_trades_box(), 1)

    def _disable_period_items(self) -> None:
        """禁用不可用周期项（1m）并挂悬停提示。"""
        model = self._period_combo.model()
        for row in range(self._period_combo.count()):
            code = str(self._period_combo.itemData(row))
            hint = _DISABLED_PERIOD_HINT.get(code)
            if hint is None:
                continue
            item = model.item(row) if isinstance(model, QStandardItemModel) else None
            if item is not None:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self._period_combo.setItemData(row, hint, Qt.ItemDataRole.ToolTipRole)

    # -- 状态面板 -- #
    def _build_state_box(self) -> QGroupBox:
        box = QGroupBox("账户状态", self)
        layout = QVBoxLayout(box)
        self._cash_label = QLabel("现金：—", box)
        self._position_label = QLabel("持仓：—", box)
        self._all_pos_label = QLabel("全部持仓：—", box)
        self._realized_label = QLabel("已实现盈亏：—", box)
        self._unrealized_label = QLabel("浮动盈亏：—", box)
        self._asset_label = QLabel("总资产：—", box)
        self._return_label = QLabel("收益率：—", box)
        self._cursor_label = QLabel("游标：—", box)
        for label in (
            self._cash_label,
            self._position_label,
            self._all_pos_label,
            self._realized_label,
            self._unrealized_label,
            self._asset_label,
            self._return_label,
            self._cursor_label,
        ):
            layout.addWidget(label)
        return box

    # -- 盈亏曲线 -- #
    def _build_equity_box(self) -> QGroupBox:
        box = QGroupBox("盈亏曲线（随回放逐根记录）", self)
        layout = QVBoxLayout(box)
        legend = QHBoxLayout()
        self._eq_cum_label = QLabel("累计盈亏 —", box)
        self._eq_cum_label.setStyleSheet(f"color: {colors.LIMIT_COLOR};")
        self._eq_float_label = QLabel("浮动盈亏 —", box)
        self._eq_float_label.setStyleSheet(f"color: {colors.ACCENT_COLOR};")
        legend.addWidget(self._eq_cum_label)
        legend.addWidget(self._eq_float_label)
        legend.addStretch(1)
        layout.addLayout(legend)

        # 左轴用紧凑单位（累计盈亏可达百万级，避免出现 1e+ 科学计数法）
        self._equity_plot = pg.PlotWidget(
            axisItems={"left": CompactAxisItem(orientation="left")}, parent=box
        )
        self._equity_plot.setBackground(colors.BG_DARK)
        self._equity_plot.showGrid(x=False, y=True, alpha=0.25)
        self._equity_plot.setMenuEnabled(False)
        self._equity_plot.setMouseEnabled(x=False, y=False)
        self._equity_plot.getAxis("bottom").setStyle(showValues=False)
        self._equity_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self._equity_plot.setMinimumHeight(120)
        self._equity_plot.setMaximumHeight(150)
        # 累计盈亏 = 总资产 - 期初资金（含浮盈）；浮动盈亏 = 持仓市值 - 成本。
        self._equity_curve = pg.PlotDataItem([], [], pen=pg.mkPen(colors.LIMIT_COLOR, width=1.4))
        self._float_curve = pg.PlotDataItem([], [], pen=pg.mkPen(colors.ACCENT_COLOR, width=1.2))
        self._zero_line = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            movable=False,
            pen=pg.mkPen(colors.SUBTEXT_COLOR, width=1, style=Qt.PenStyle.DashLine),
        )
        self._equity_plot.addItem(self._equity_curve)
        self._equity_plot.addItem(self._float_curve)
        self._equity_plot.addItem(self._zero_line, ignoreBounds=True)
        layout.addWidget(self._equity_plot)
        return box

    # -- 手动下单 -- #
    def _build_order_box(self) -> QGroupBox:
        box = QGroupBox("手动下单", self)
        layout = QVBoxLayout(box)

        # 第一行：数量 + 买 / 卖 / 平仓（数量可输入，也可用下方快捷按钮回填）。
        row1 = QHBoxLayout()
        self._qty_spin = QSpinBox(self)
        self._qty_spin.setRange(100, 10_000_000)
        self._qty_spin.setSingleStep(100)
        self._qty_spin.setValue(100)
        self._qty_spin.setSuffix(" 股")
        self._btn_buy = QPushButton("买入", self)
        self._btn_buy.clicked.connect(lambda: self._on_place_order(BUY))
        self._btn_sell = QPushButton("卖出", self)
        self._btn_sell.clicked.connect(lambda: self._on_place_order(SELL))
        self._btn_close = QPushButton("平仓", self)
        self._btn_close.clicked.connect(lambda: self._on_close_position())
        row1.addWidget(QLabel("数量：", box))
        row1.addWidget(self._qty_spin)
        row1.addWidget(self._btn_buy)
        row1.addWidget(self._btn_sell)
        row1.addWidget(self._btn_close)
        layout.addLayout(row1)

        # 交易价滑钮：刻度以「分」为单位（0.01 元精度），范围 = 当日（截至游标）
        # 震荡范围 [最低, 最高]；K 线图上以黄色虚线标注该价。
        row_price = QHBoxLayout()
        self._price_slider = QSlider(Qt.Orientation.Horizontal, box)
        self._price_slider.setRange(0, 1000)
        self._price_slider.setEnabled(False)
        self._price_slider.setToolTip(
            "在当日（截至当前）震荡范围内选择成交价，K 线图上以黄色虚线标注"
        )
        self._price_slider.valueChanged.connect(lambda _v: self._on_trade_price_changed())
        self._price_label = QLabel(_DASH, box)
        self._price_label.setMinimumWidth(48)
        row_price.addWidget(QLabel("交易价：", box))
        row_price.addWidget(self._price_slider, 1)
        row_price.addWidget(self._price_label)
        layout.addLayout(row_price)

        # 可用股数提示（可买按现金整手向下取整；可卖遵循 T+1）。
        self._avail_label = QLabel("可买 — 股 · 可卖 — 股", box)
        self._avail_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        layout.addWidget(self._avail_label)

        # 第二行：快捷买入（全仓 / 1/2仓 / 1/4仓 / 按金额），点击直接买入。
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("快捷买入：", box))
        for text, fraction in (("全仓", 1.0), ("1/2仓", 0.5), ("1/4仓", 0.25)):
            btn = QPushButton(text, box)
            btn.setToolTip(f"按可用现金的 {fraction:.0%} 折算整手股数并直接买入")
            btn.clicked.connect(lambda _checked=False, f=fraction: self._quick_buy(f))
            row2.addWidget(btn)
        self._btn_by_amount = QPushButton("按金额…", box)
        self._btn_by_amount.setToolTip("输入买入金额（默认全部现金），折算整手股数并直接买入")
        self._btn_by_amount.clicked.connect(lambda: self._on_buy_by_amount())
        row2.addWidget(self._btn_by_amount)
        layout.addLayout(row2)
        return box

    # -- 条件单 -- #
    def _build_condition_box(self) -> QGroupBox:
        box = QGroupBox("条件单（一次性，触发即成交）", self)
        layout = QHBoxLayout(box)
        self._cond_side = QComboBox(self)
        self._cond_side.addItem("买入", BUY)
        self._cond_side.addItem("卖出", SELL)
        self._cond_type = QComboBox(self)
        for value, text in _TRIGGER_CHOICES:
            self._cond_type.addItem(text, value)
        self._cond_trigger = QDoubleSpinBox(self)
        self._cond_trigger.setRange(-1e6, 1e6)
        self._cond_trigger.setDecimals(2)
        self._cond_trigger.setValue(10.0)
        self._cond_qty = QSpinBox(self)
        self._cond_qty.setRange(100, 10_000_000)
        self._cond_qty.setSingleStep(100)
        self._cond_qty.setValue(100)
        self._btn_add_cond = QPushButton("添加条件单", self)
        self._btn_add_cond.clicked.connect(lambda: self._on_add_condition())
        layout.addWidget(self._cond_side)
        layout.addWidget(self._cond_type)
        layout.addWidget(self._cond_trigger)
        layout.addWidget(self._cond_qty)
        layout.addWidget(self._btn_add_cond)
        return box

    # -- 成交记录 -- #
    def _build_trades_box(self) -> QGroupBox:
        box = QGroupBox("成交记录", self)
        layout = QVBoxLayout(box)
        self._trades_table = QTableWidget(0, 5, box)
        self._trades_table.setHorizontalHeaderLabels(["时间", "标的", "方向", "价格", "数量"])
        self._trades_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._trades_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._trades_table)
        return box

    def _populate_symbols(self, default_symbol: Optional[str]) -> None:
        """填充标的候选下拉（自选股 + 默认选中）；新增空项允许手动输入。"""
        self._symbol_combo.clear()
        self._symbol_combo.addItem("")
        for symbol in self._candidates:
            self._symbol_combo.addItem(symbol, symbol)
        if default_symbol:
            idx = self._symbol_combo.findData(str(default_symbol))
            if idx >= 0:
                self._symbol_combo.setCurrentIndex(idx)
        self._refresh_date_range()

    def _set_status(self, text: str) -> None:
        """更新状态行。"""
        self._status_label.setText(text)

    # ------------------------------------------------------------------ #
    # 周期 / 起始日 / 数据加载
    # ------------------------------------------------------------------ #
    def current_symbol(self) -> Optional[str]:
        """当前选中的标的代码（文本）。"""
        text = str(self._symbol_combo.currentText()).strip()
        return text or None

    def _on_symbol_text_changed(self, text: str) -> None:
        """标的框变更：回放中且为 6 位代码且不同于当前标的 → **切换回放标的**；
        其余情况（未开始 / 输入中）→ 刷新「起始日」可用范围。
        """
        code = str(text).strip()
        if self._engine is not None and len(code) == 6 and code.isdigit():
            ctx = self._engine_ctx()
            if ctx is not None and code != ctx.symbol:
                self.switch_symbol(code)
                return
        self._refresh_date_range()

    def _revert_symbol_combo(self, symbol: str) -> None:
        """把标的框回退到指定代码（不触发切换）。"""
        self._symbol_combo.blockSignals(True)
        index = self._symbol_combo.findData(symbol)
        if index >= 0:
            self._symbol_combo.setCurrentIndex(index)
        else:
            self._symbol_combo.setEditText(symbol)
        self._symbol_combo.blockSignals(False)

    def _set_params_locked(self, locked: bool) -> None:
        """回放会话中锁定「周期 / 起始日 / 初始资金」（标的框保持可用以便切换）。"""
        for widget in (self._period_combo, self._date_edit, self._cash_spin):
            widget.setEnabled(not locked)

    def current_period(self) -> str:
        """当前选中的周期编码（``5m``/``15m``/``30m``/``60m``/``d``）。"""
        return str(self._period_combo.currentData() or _DEFAULT_PERIOD)

    def window_slots(self) -> int:
        """当前固定显示窗口的槽位数（会话中为实测槽位数，否则按周期常量）。"""
        if self._slots:
            return int(self._slots)
        return window_slots(self.current_period())

    def _start_date_text(self) -> str:
        """「起始日」选择器的 ISO 文本（``YYYY-MM-DD``）。"""
        return self._date_edit.date().toString("yyyy-MM-dd")

    def _read_bars(self, symbol: str, period: str) -> pd.DataFrame:
        """读本地该周期数据（未复权优先）。

        降级链：库内该周期（``adjust=""``）→ 15/30/60m 从本地 5m **派生**（不落库）
        → 日线回退前复权（``qfq``）。读失败 / 空返回空 Bar 契约表（不抛）。
        """
        try:
            frame = self._db.get_bars(symbol, period, adjust="")
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取 %s(%s) 失败：%s", symbol, period, exc)
            frame = None
        if frame is not None and not frame.empty:
            return frame

        if period in _DERIVE_FROM_5M:
            try:
                base = self._db.get_bars(symbol, "5m", adjust="")
            except Exception as exc:  # noqa: BLE001 - 读库失败降级
                logger.warning("读取 %s(5m) 失败：%s", symbol, exc)
                base = None
            if base is not None and not base.empty:
                return derive_minute_bars(base, "5m", period)
            return pd.DataFrame(columns=BAR_COLUMNS)

        if period == "d":
            try:
                frame = self._db.get_bars(symbol, "d", adjust="qfq")
            except Exception as exc:  # noqa: BLE001 - 读库失败降级
                logger.warning("读取 %s(d/qfq) 失败：%s", symbol, exc)
                frame = None
            if frame is not None and not frame.empty:
                return frame
        return pd.DataFrame(columns=BAR_COLUMNS)

    @staticmethod
    def _slice_from(frame: pd.DataFrame, start: Optional[str]) -> pd.DataFrame:
        """按起始日（含）切片并重置索引；``start`` 为 ``None`` / 非法时原样返回。"""
        if frame is None or frame.empty or not start:
            return frame
        try:
            begin = pd.Timestamp(str(start)).normalize()
        except Exception:  # noqa: BLE001 - 非法日期不切片
            return frame
        dts = pd.to_datetime(frame["datetime"], errors="coerce")
        return frame[dts >= begin].reset_index(drop=True)

    @staticmethod
    def _effective_slots(period: str, bars: pd.DataFrame) -> int:
        """固定显示槽位数：周期常量与「实测单日最大根数」取大者（防个别多根被裁掉）。"""
        slots = window_slots(period)
        if is_rolling_window(period) or bars is None or bars.empty:
            return slots
        per_day = pd.to_datetime(bars["datetime"]).dt.normalize().value_counts()
        return max(slots, int(per_day.max()))

    def _refresh_date_range(self) -> None:
        """标的/周期变化 → 用本地可用数据范围刷新「起始日」选择器（默认数据首日）。

        回放会话中跳过（周期/起始日在会话内被锁定，避免与回放口径打架）。
        """
        if self._engine is not None:
            return
        symbol = self.current_symbol()
        if not symbol:
            return
        frame = self._read_bars(symbol, self.current_period())
        if frame is None or frame.empty:
            return
        first = pd.to_datetime(frame["datetime"].iloc[0], errors="coerce")
        last = pd.to_datetime(frame["datetime"].iloc[-1], errors="coerce")
        if pd.isna(first) or pd.isna(last):
            return
        begin = QDate(int(first.year), int(first.month), int(first.day))
        finish = QDate(int(last.year), int(last.month), int(last.day))
        self._date_edit.setDateRange(begin, finish)
        self._date_edit.setDate(begin)

    # ------------------------------------------------------------------ #
    # 回放控制
    # ------------------------------------------------------------------ #
    def start_replay(self) -> None:
        """按「标的 + 周期 + 起始日」载入本地数据并按日回放；无数据则提示并禁用开始。"""
        self.stop_replay()
        symbol = self.current_symbol()
        if not symbol:
            self._set_status("请先输入标的代码")
            return
        period = self.current_period()
        frame = self._read_bars(symbol, period)
        frame = self._slice_from(frame, self._start_date_text())
        if frame is None or frame.empty:
            self._set_status(_NO_MINUTE_TEXT if period in _MINUTE_PERIODS else _NO_DAILY_TEXT)
            self._btn_start.setEnabled(False)
            return

        self._bars = frame.reset_index(drop=True)
        first_dt = pd.to_datetime(self._bars["datetime"].iloc[0])
        self._day = first_dt.date()
        self._slots = self._effective_slots(period, self._bars)
        self._total_days = int(
            pd.to_datetime(self._bars["datetime"]).dt.normalize().nunique()
        )
        self._initial_cash = float(self._cash_spin.value())
        self._equity = [(first_dt, self._initial_cash, 0.0)]
        # 交易价滑钮：新会话重置为 None（默认 = 最新收盘价），随首根显现同步范围。
        self._trade_price = None
        self._price_label.setText(_DASH)
        self._price_slider.setEnabled(True)

        # 只喂本次会话数据（已按起始日切片），未来数据由 as_of 后视（look-ahead 防护）。
        # _frames 预登记初始标的帧，回放中切换到其它标的时账户/持仓不变。
        ctx = SimContext(
            symbol=symbol,
            start_date=str(self._day),
            cash=self._initial_cash,
            initial_cash=self._initial_cash,
            _bars=self._bars,
            _frames={symbol: self._bars},
        )
        self._engine = SimEngine(
            ctx,
            speed=int(self._speed_combo.currentData() or 1),
            emit_events=False,
            on_tick=lambda _payload: self._on_engine_tick(),
            on_finished=lambda: self._on_engine_finished(),
            on_fill=lambda _payload: self._on_engine_fill(),
        )
        self._chart.set_header(symbol, PERIOD_LABELS.get(period, period))
        self.setWindowTitle(f"模拟训练 - {symbol}")
        self._set_params_locked(True)
        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_pause.setText("暂停")
        self._btn_next_day.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._update_timer()
        self._timer.start()
        self._refresh_state()
        self._refresh_equity()
        self._set_status(_RUNNING_TEXT)

    def _on_timer(self) -> None:
        """QTimer 驱动：推进一根；当日放完则暂停等待「下一天」。"""
        if self._engine is None or self._day_complete:
            return
        if not self._engine.tick():
            self._finish_replay()
            return
        self._after_advance()

    def _after_advance(self) -> None:
        """推进后的当日判定：下一根已跨日 → 当日结束（停表 + 点亮「下一天」）。"""
        nxt = self._next_bar_date()
        if nxt is None:
            self._finish_replay()
            return
        if nxt != self._day:
            self._day_complete = True
            self._timer.stop()
            self._btn_next_day.setEnabled(True)
            self._set_status(
                f"{self._day} 当日 K 线已放完（{self._shown_this_day} 根），点击「下一天」继续"
            )

    def _next_bar_date(self) -> Optional[date]:
        """下一根待显现 bar 的日期；已无更多数据返回 ``None``（严格基于 as_of 计数）。"""
        ctx = self._engine_ctx()
        if ctx is None or self._bars.empty:
            return None
        shown = len(ctx.as_of())  # 已显现根数
        if shown >= len(self._bars):
            return None
        value = pd.to_datetime(self._bars["datetime"].iloc[shown], errors="coerce")
        return None if pd.isna(value) else value.date()

    def next_day(self) -> None:
        """「下一天」：当日放完后点击 → 进入下一交易日并继续逐根回放。"""
        if self._engine is None:
            self._set_status("请先点击「开始回放」")
            return
        nxt = self._next_bar_date()
        if nxt is None:
            self._finish_replay()
            return
        self._day = nxt
        self._day_complete = False
        self._trade_price = None  # 新交易日重置交易价（默认 = 当日最新收盘）
        self._btn_next_day.setEnabled(False)
        if not self._engine.tick():  # 立即显现新交易日第 1 根
            self._finish_replay()
            return
        self._after_advance()
        if self._day_complete:  # 日K：一根即一天，等下一次点击
            return
        self._update_timer()
        self._timer.start()
        self._btn_pause.setText("暂停")
        self._set_status(_RUNNING_TEXT)

    def toggle_pause(self) -> None:
        """暂停 / 继续（当日已放完时提示用「下一天」）。"""
        if self._engine is None:
            return
        if self._day_complete:
            self._set_status("当日 K 线已放完，请点击「下一天」继续")
            return
        if self._timer.isActive():
            self._timer.stop()
            self._btn_pause.setText("继续")
            self._set_status(_PAUSED_TEXT)
        else:
            self._update_timer()
            self._timer.start()
            self._btn_pause.setText("暂停")
            self._set_status(_RUNNING_TEXT)

    def _update_timer(self) -> None:
        """按当前倍速刷新定时器间隔。"""
        if self._engine is None:
            return
        interval = self._engine.interval_ms()
        if interval > 0:
            self._timer.setInterval(interval)

    def _finish_replay(self) -> None:
        """回放结束（数据全部放完）：停表、禁用「下一天」并写入收尾状态。"""
        self._day_complete = True
        self._timer.stop()
        self._btn_next_day.setEnabled(False)
        ctx = self._engine_ctx()
        symbol = ctx.symbol if ctx is not None else self.current_symbol()
        self._set_status(
            f"{_FINISHED_TEXT}：{symbol} 共 {len(self._bars)} 根 / "
            f"{self._total_days} 个交易日（可继续下单复盘或切换其它标的）"
        )

    def stop_replay(self) -> None:
        """重置回放（停止计时器并清空引擎与会话状态；面板复位）。"""
        self._timer.stop()
        self._engine = None
        self._bars = pd.DataFrame(columns=BAR_COLUMNS)
        self._slots = 0
        self._day = None
        self._day_complete = False
        self._shown_this_day = 0
        self._total_days = 0
        self._equity = []
        self._trade_price = None
        self._price_slider.setEnabled(False)
        self._price_label.setText(_DASH)
        self._chart.set_trade_price(None)
        self._set_params_locked(False)
        self.setWindowTitle("模拟训练")
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_pause.setText("暂停")
        self._btn_next_day.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._chart.set_bars(None)
        self._refresh_state()
        self._refresh_equity()
        self._set_status(_READY_TEXT)

    def switch_symbol(self, symbol: str) -> bool:
        """**回放中切换标的**（账户与持仓不变，可同时买卖多只股票）。

        流程：按「同周期 + 同起始日」读取新标的本地数据（缺失则回退标的框并提示）
        → :meth:`SimContext.set_symbol` 换帧 → :meth:`SimEngine.rebase` 重建回放时钟
        （游标时间不变，从新标的对应位置继续）→ 刷新图 / 面板 / 标题。

        Returns:
            是否切换成功。
        """
        code = str(symbol).strip()
        if self._engine is None:
            self._set_status("请先点击「开始回放」")
            return False
        ctx = self._engine_ctx()
        if ctx is None or code == ctx.symbol:
            return False
        period = self.current_period()
        frame = self._read_bars(code, period)
        frame = self._slice_from(frame, self._start_date_text())
        if frame is None or frame.empty:
            self._revert_symbol_combo(ctx.symbol)
            self._set_status(f"{code} 本地暂无该周期数据（请先『数据→更新数据』）")
            return False

        was_active = self._timer.isActive()
        self._timer.stop()
        frame = frame.reset_index(drop=True)
        ctx.set_symbol(code, frame)
        self._engine.rebase()
        self._bars = frame
        self._slots = self._effective_slots(period, frame)
        first_dt = pd.to_datetime(frame["datetime"].iloc[0])
        self._day = ctx.cursor_dt.date() if ctx.cursor_dt is not None else first_dt.date()
        self._total_days = int(pd.to_datetime(frame["datetime"]).dt.normalize().nunique())
        self._day_complete = False
        self._chart.set_header(code, PERIOD_LABELS.get(period, period))
        self.setWindowTitle(f"模拟训练 - {code}")
        self._trade_price = None  # 换标的重置交易价（默认 = 新标的当日最新收盘）
        self._refresh_view()
        self._refresh_state()
        self._record_equity()
        self._refresh_equity()

        nxt = self._next_bar_date()
        if nxt is None:
            self._finish_replay()
            self._set_status(f"已切换至 {code}：该标的数据已放完（账户与持仓不变）")
            return True
        if nxt != self._day:
            # 新标的当日数据已到末根（游标未动）→ 沿用按日回放语义，等「下一天」。
            self._day_complete = True
            self._btn_next_day.setEnabled(True)
            self._btn_pause.setText("暂停")
            self._set_status(f"已切换至 {code}：当日已放完，点「下一天」继续（账户与持仓不变）")
            return True
        self._btn_next_day.setEnabled(False)
        if was_active:
            self._update_timer()
            self._timer.start()
            self._btn_pause.setText("暂停")
            self._set_status(f"{_RUNNING_TEXT}（已切换至 {code}，账户与持仓不变）")
        else:
            self._btn_pause.setText("继续")
            self._set_status(f"已切换至 {code}（已暂停；点「继续」恢复回放，账户与持仓不变）")
        return True

    def _on_speed_changed(self) -> None:
        """倍速下拉变更：同步引擎速度并刷新定时器间隔（运行中立即生效）。"""
        if self._engine is None:
            return
        speed = int(self._speed_combo.currentData() or 1)
        self._engine.set_speed(speed)
        self._update_timer()

    def speed(self) -> int:
        """当前倍速（引擎存在时）。"""
        return self._engine.speed() if self._engine is not None else 0

    # ------------------------------------------------------------------ #
    # 引擎回调 → 界面刷新
    # ------------------------------------------------------------------ #
    def _on_engine_tick(self) -> None:
        """每 tick 刷新：主图（as_of + 当日切片）+ 账户面板 + 盈亏曲线。"""
        self._refresh_view()
        self._record_equity()
        self._refresh_state()
        self._refresh_equity()

    def _on_engine_fill(self) -> None:
        """成交后刷新：账户 + 成交记录表 + 盈亏曲线。"""
        self._refresh_state()
        self._refresh_trades()
        self._record_equity()
        self._refresh_equity()

    def _on_engine_finished(self) -> None:
        """回放结束（引擎回调）：进入收尾态。"""
        self._finish_replay()

    # ------------------------------------------------------------------ #
    # 下单
    # ------------------------------------------------------------------ #
    def current_trade_price(self) -> Optional[float]:
        """当前滑钮选择的成交价（未开始 / 未同步返回 ``None`` → 成交回退最新价）。"""
        return self._trade_price

    def _on_trade_price_changed(self) -> None:
        """交易价滑钮变更：更新数值标签、主图黄色虚线参考线与可买股数提示。"""
        self._trade_price = self._price_slider.value() / 100.0
        self._price_label.setText(f"{self._trade_price:.2f}")
        self._chart.set_trade_price(self._trade_price)
        self._refresh_state()

    def _sync_trade_price(self, frame: pd.DataFrame) -> None:
        """按当日（截至游标）震荡范围刷新交易价滑钮与主图参考线。

        滑钮刻度以「分」为单位（0.01 元精度）；默认值 = 最新收盘价（未拖动时行为与
        旧版一致），拖动后保持不随 tick 重置；范围随数据扩展，``high == low``
        （一字板）时退化为单值也不崩。
        """
        if frame is None or frame.empty:
            return
        try:
            low = float(pd.to_numeric(frame["low"], errors="coerce").min())
            high = float(pd.to_numeric(frame["high"], errors="coerce").max())
            last = float(frame["close"].iloc[-1])
        except (KeyError, TypeError, ValueError):
            return
        if not (math.isfinite(low) and math.isfinite(high) and math.isfinite(last)):
            return
        if high < low:
            low, high = high, low
        if self._trade_price is None:
            self._trade_price = min(max(last, low), high)
        else:
            self._trade_price = min(max(self._trade_price, low), high)
        lo_value = int(round(low * 100.0))
        hi_value = max(lo_value, int(round(high * 100.0)))
        value = min(max(int(round(self._trade_price * 100.0)), lo_value), hi_value)
        slider = self._price_slider
        slider.blockSignals(True)
        slider.setRange(lo_value, hi_value)
        slider.setValue(value)
        slider.blockSignals(False)
        self._price_label.setText(f"{self._trade_price:.2f}")
        self._chart.set_trade_price(self._trade_price)

    def _on_place_order(self, side: str) -> None:
        """手动下单（按滑钮交易价；未同步时回退当前最新价）。"""
        if self._engine is None:
            self._set_status("请先点击「开始回放」")
            return
        qty = int(self._qty_spin.value())
        ok, reason = self._engine.place_order(side, qty, price=self._trade_price)
        if ok:
            self._on_engine_fill()
            self._set_status(f"{'买入' if side == BUY else '卖出'} {qty} 股成功")
        else:
            QMessageBox.warning(self, "下单失败", reason)
            self._set_status(reason)

    def _on_close_position(self) -> None:
        """平仓：卖出当前标的全部持仓（按滑钮交易价；未同步时回退最新价）。"""
        if self._engine is None:
            self._set_status("请先点击「开始回放」")
            return
        ctx = self._engine_ctx()
        pos = ctx.position(ctx.symbol)
        if pos is None:
            self._set_status("当前无持仓")
            return
        ok, reason = self._engine.place_order(SELL, int(pos.qty), price=self._trade_price)
        if ok:
            self._on_engine_fill()
        else:
            QMessageBox.warning(self, "平仓失败", reason)
            self._set_status(reason)

    # -- 快捷买入（② 可买股数 / 全仓 / 1/2仓 / 1/4仓 / 按金额）-- #
    def _quick_buy(self, fraction: float) -> None:
        """快捷买入：按可用现金比例折算**整手**股数并直接买入（全仓 / 1/2 / 1/4）。"""
        if self._engine is None:
            self._set_status("请先点击「开始回放」")
            return
        ctx = self._engine_ctx()
        price = self._order_price(ctx)
        if price is None:
            self._set_status("当前无最新价（回放未开始或数据为空）")
            return
        self._place_quick_buy(max_buy_qty(float(ctx.cash) * float(fraction), float(price)))

    def _on_buy_by_amount(self) -> None:
        """按金额买入：弹窗输入金额（默认全部现金）→ 折算整手股数并直接买入。"""
        if self._engine is None:
            self._set_status("请先点击「开始回放」")
            return
        ctx = self._engine_ctx()
        if ctx is None:
            return
        amount, ok = QInputDialog.getDouble(
            self, "按金额买入", "买入金额（元）：", float(ctx.cash), 0.0, 1e12, 2
        )
        if not ok:
            return
        price = self._order_price(ctx)
        if price is None:
            self._set_status("当前无最新价（回放未开始或数据为空）")
            return
        self._place_quick_buy(qty_for_amount(float(amount), float(price)))

    def _order_price(self, ctx) -> Optional[float]:  # noqa: ANN001 - SimContext
        """下单参考价：优先滑钮交易价，未同步（``None``）时回退当前最新价。"""
        if self._trade_price is not None:
            return float(self._trade_price)
        return ctx.last_price() if ctx is not None else None

    def _place_quick_buy(self, qty: int) -> None:
        """快捷买入公共落单：``qty<=0``（不足一手 / 价非法）提示；否则回填数量并买入。"""
        if qty <= 0:
            self._set_status("资金不足一手（100 股）或价格无效")
            return
        self._qty_spin.setValue(int(qty))
        self._on_place_order(BUY)

    def _sellable_qty(self) -> int:
        """当前标的**可卖**股数（T+1：当日买入不可卖；无持仓为 0）。"""
        ctx = self._engine_ctx()
        if ctx is None:
            return 0
        pos = ctx.position(ctx.symbol)
        if pos is None:
            return 0
        if ctx.cursor_dt is not None and str(pos.buy_date) >= str(ctx.cursor_dt.date()):
            return 0
        return int(pos.qty)

    def _on_add_condition(self) -> None:
        """添加条件单（一次性；**绑定添加时的标的**，切换标的后依然按原标的评估）。"""
        if self._engine is None:
            self._set_status("请先点击「开始回放」")
            return
        ctx = self._engine_ctx()
        order = ConditionalOrder(
            side=str(self._cond_side.currentData()),
            trigger_type=str(self._cond_type.currentData()),
            trigger=float(self._cond_trigger.value()),
            qty=int(self._cond_qty.value()),
            symbol=ctx.symbol if ctx is not None else None,
        )
        self._engine.add_conditional_order(order)
        self._set_status(
            f"条件单已添加（{order.symbol or '当前标的'}，待触发 "
            f"{self._engine.conditional_order_count()} 个）"
        )

    # ------------------------------------------------------------------ #
    # 视图 / 面板刷新
    # ------------------------------------------------------------------ #
    def _engine_ctx(self):
        """当前引擎的 SimContext（测试用）。"""
        return self._engine._ctx if self._engine is not None else None

    def _refresh_view(self) -> None:
        """主图只接收 ``as_of()`` 可见数据，且分钟周期**只画当前交易日**。

        视窗为**以最新根为中心的滑窗**（宽度恒定 = 槽位数）：每显现一根 K 线，
        无需鼠标移入即自动跟随居中，避免大幅波动时最新 K 线跳出视野；日K 为滚动窗口
        （保留历史 K 线做参照，一根 = 一天）。
        """
        ctx = self._engine_ctx()
        if ctx is None:
            return
        visible = ctx.as_of()
        if visible.empty:
            return
        period = self.current_period()
        if is_rolling_window(period):
            frame = visible.tail(self.window_slots()).reset_index(drop=True)
        else:
            mask = pd.to_datetime(visible["datetime"]).dt.date == self._day
            frame = visible[mask].reset_index(drop=True)
        self._shown_this_day = int(len(frame))
        slots = self.window_slots()
        newest = float(len(frame) - 1)
        half = slots / 2.0
        self._chart.set_bars(frame, x_range=(newest - half, newest + half))
        self._sync_trade_price(frame)

    def _record_equity(self) -> None:
        """逐根 / 逐笔记录（游标时间, 总资产, 浮动盈亏）供盈亏曲线。"""
        ctx = self._engine_ctx()
        if ctx is None or ctx.cursor_dt is None:
            return
        self._equity.append(
            (ctx.cursor_dt, float(ctx.total_asset()), float(ctx.unrealized_pnl()))
        )

    def _refresh_equity(self) -> None:
        """重绘盈亏曲线：累计盈亏（总资产 - 期初资金）与浮动盈亏，含 0 基准线。"""
        base = self._initial_cash
        cum = [total - base for _dt, total, _unrealized in self._equity]
        floating = [unrealized for _dt, _total, unrealized in self._equity]
        xs = list(range(len(self._equity)))
        self._equity_curve.setData(xs, cum)
        self._float_curve.setData(xs, floating)
        # 横轴固定为本次会话长度（不随记录数跳动），纵轴按数据自适应（含 0 基准）。
        span = max(10, len(self._bars))
        self._equity_plot.setXRange(0, span, padding=0.02)
        values = cum + floating + [0.0]
        low, high = min(values), max(values)
        pad = max(1.0, (high - low) * 0.15)
        self._equity_plot.setYRange(low - pad, high + pad, padding=0.0)
        if self._equity:
            latest_cum = float(self._equity[-1][1]) - base
            latest_float = float(self._equity[-1][2])
            self._eq_cum_label.setText(f"累计盈亏 {latest_cum:+,.2f}")
            self._eq_float_label.setText(f"浮动盈亏 {latest_float:+,.2f}")
        else:
            self._eq_cum_label.setText("累计盈亏 —")
            self._eq_float_label.setText("浮动盈亏 —")

    def _refresh_state(self) -> None:
        """刷新账户状态面板（现金 / 持仓 / 盈亏 / 总资产 / 收益率 / 游标）。"""
        ctx = self._engine_ctx()
        if ctx is None:
            self._cash_label.setText("现金：—")
            self._position_label.setText("持仓：—")
            self._all_pos_label.setText("全部持仓：—")
            self._avail_label.setText("可买 — 股 · 可卖 — 股")
            self._realized_label.setText("已实现盈亏：—")
            self._unrealized_label.setText("浮动盈亏：—")
            self._asset_label.setText("总资产：—")
            self._return_label.setText("收益率：—")
            self._cursor_label.setText("游标：—")
            for label in (self._realized_label, self._unrealized_label, self._return_label):
                label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
            return
        cursor = ctx.cursor_dt
        rolling = is_rolling_window(self.current_period())
        if cursor is None:
            cursor_text = _DASH
        elif rolling:
            cursor_text = f"{cursor:%Y-%m-%d}（近 {self._shown_this_day}/{self.window_slots()} 根）"
        else:
            cursor_text = (
                f"{cursor:%m-%d %H:%M}（当日 {self._shown_this_day}/{self.window_slots()} 根）"
            )
        self._cursor_label.setText(f"游标：{cursor_text}")
        self._cash_label.setText(f"现金：{ctx.cash:,.2f}")
        pos = ctx.position(ctx.symbol)
        if pos is None:
            self._position_label.setText("持仓：空仓")
        else:
            self._position_label.setText(
                f"持仓：{pos.qty} 股 @ {pos.cost:.2f}（买日 {pos.buy_date}）"
            )
        if ctx.positions:
            parts = [
                f"{sym} {p.qty}股@{p.cost:.2f}" for sym, p in sorted(ctx.positions.items())
            ]
            self._all_pos_label.setText("全部持仓：" + " · ".join(parts))
        else:
            self._all_pos_label.setText("全部持仓：空仓")
        # 快捷下单辅助：可买 = 现金整手向下取整（按滑钮交易价，未同步则按最新价）；
        # 可卖 = 当前标的 T+1 后可卖股数。
        price = self._order_price(ctx)
        if price is None:
            self._avail_label.setText(f"可买 — 股 · 可卖 {self._sellable_qty()} 股")
        else:
            basis = "交易价" if self._trade_price is not None else "最新价"
            self._avail_label.setText(
                f"可买 {max_buy_qty(float(ctx.cash), float(price))} 股 · "
                f"可卖 {self._sellable_qty()} 股（按{basis} {float(price):.2f}）"
            )
        realized = float(ctx.realized_pnl())
        unrealized = float(ctx.unrealized_pnl())
        asset = float(ctx.total_asset())
        base = float(ctx.initial_cash) if ctx.initial_cash else self._initial_cash
        returns = (asset / base - 1.0) * 100.0 if base else 0.0
        self._realized_label.setText(f"已实现盈亏：{realized:+,.2f}")
        self._unrealized_label.setText(f"浮动盈亏：{unrealized:+,.2f}")
        self._asset_label.setText(f"总资产：{asset:,.2f}")
        self._return_label.setText(f"收益率：{returns:+.2f}%")
        self._realized_label.setStyleSheet(f"color: {_pnl_color(realized)};")
        self._unrealized_label.setStyleSheet(f"color: {_pnl_color(unrealized)};")
        self._return_label.setStyleSheet(f"color: {_pnl_color(returns)};")

    def _refresh_trades(self) -> None:
        """刷新成交记录表。"""
        ctx = self._engine_ctx()
        if ctx is None or not ctx.orders:
            self._trades_table.setRowCount(0)
            return
        fmt = "%Y-%m-%d" if is_rolling_window(self.current_period()) else "%m-%d %H:%M"
        self._trades_table.setRowCount(0)
        for trade in reversed(ctx.orders):
            row = self._trades_table.rowCount()
            self._trades_table.insertRow(row)
            time_text = trade.dt.strftime(fmt) if trade.dt else _DASH
            side_text = "买入" if trade.side == BUY else "卖出"
            values = [
                time_text,
                str(getattr(trade, "symbol", "") or ""),
                side_text,
                f"{trade.price:.2f}",
                str(trade.qty),
            ]
            for col, value in enumerate(values):
                self._trades_table.setItem(row, col, QTableWidgetItem(value))

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: ANN001 - Qt 约定
        """关闭前停表（引擎为纯 Python，无 QThread 风险）。"""
        self._timer.stop()
        super().closeEvent(event)


__all__ = ["SimWindow", "SimContext", "is_rolling_window", "window_slots"]