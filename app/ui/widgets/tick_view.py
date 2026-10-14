"""分时图独立页（PyQtGraph）：价格线 + 均价线 + 成交量副图 + 现价/涨跌幅。

定位：**纯展示组件**（不联网、不写库）。
    * :meth:`TickView.set_minute` 接收 1 分钟 Bar 契约 DataFrame 并绘制；
    * :meth:`TickView.load` 仅从**本地库**读取后展示（读取 1 分钟线 + 日线昨收）。

设计要点（遵循《细化设计》§4「P2-S3」/《开发手册》§4.2、§4.6）：
    * **价格线**：用 1 分钟 ``close`` 绘制折线；
    * **均价线（VWAP）**：``cumsum(amount) / (cumsum(volume) * 100)``——``volume`` 单位「手」，
      ``amount`` 单位「元」，须 **×100** 把「手」换算为「股」才是真实成交均价（差 100 倍）；
    * **成交量副图**：量柱按现价相对昨收（无昨收则相对当日首笔开盘）红涨绿跌着色，
      与主图 **X 轴联动**（``setXLink``），主副图高度约 3:1；
    * **头部信息**：``代码 · 日期`` + 现价 + 涨跌幅 + 均价，颜色随涨跌刷新；
    * **成交量轴 / 角标「量」按量级带单位**（k / m / b / t，复用
      :func:`app.ui.widgets.kline_view.format_compact_number`），不出现 ``1e+`` 科学计数法；
    * 交互：右键框选缩放、滚轮缩放、左键拖拽平移 + 十字光标。

跨源单位约定（《开发手册》§4.2）：
    * ``volume`` 统一为「手」；``amount`` 统一为「元」。
    * 因 1 手 = 100 股，故 ``VWAP = amount / (volume * 100)``。
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.data.store.db import BAR_COLUMNS, Database, get_db
from app.ui.themes import colors
from app.ui.widgets.kline_view import CompactAxisItem, DateAxisItem, format_compact_number

logger = logging.getLogger(__name__)

# 全局 pyqtgraph 选项（与 KLineView 一致；关闭抗锯齿以保性能）。
pg.setConfigOptions(antialias=False, background=colors.BG_DARK, foreground=colors.TEXT_COLOR)

# 量柱宽度（X 序号单位）。
_VOL_WIDTH: float = 0.6

# 1 手 = 100 股（用于 VWAP 换算）。
_SHARES_PER_LOT: float = 100.0

# 昨收 / 均价参考线的横线颜色（沿用主题常量，勿在别处硬编码颜色）。
_AVG_COLOR: str = colors.LIMIT_COLOR        # 均价线（黄）
_PRICE_COLOR: str = colors.TEXT_COLOR       # 价格线（浅灰）
_PREV_COLOR: str = colors.SUBTEXT_COLOR     # 昨收虚线

# 占位文案。
_EMPTY_TEXT: str = "无数据"
# 1 分钟线占位：本版本「更新数据」仅拉日线，点它也拿不到分钟数据，故给出与
# kline_view 一致的专门文案（避免误导）。
# ⚠️ 该句须与 kline_view._NO_LOCAL_MINUTE_TEXT **一字不差**。
_NO_LOCAL_MINUTE_TEXT: str = "本地暂无分钟数据（当前版本『更新数据』仅拉取日线）"
_DASH: str = "—"


# --------------------------------------------------------------------------- #
# 纯函数（便于单测）
# --------------------------------------------------------------------------- #
def avg_price_series(df: Optional[pd.DataFrame]) -> pd.Series:
    """计算分时**均价线（VWAP）**：``cumsum(amount) / (cumsum(volume) * 100)``。

    单位换算至关重要：``volume`` 为「手」、``amount`` 为「元」，1 手 = 100 股，
    因此分母须 **×100** 转为「股」，否则均价会偏大 100 倍。

    Args:
        df: 含 ``volume``（手）、``amount``（元）两列的 1 分钟 Bar 契约 DataFrame；
            ``None`` / 空表 / 缺列时返回全空 ``Series``。

    Returns:
        与 ``df`` 等长、逐根累计的均价序列（``float``）；累计成交量为 0 处为 ``NaN``。
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    if "volume" not in df.columns or "amount" not in df.columns:
        logger.warning("avg_price_series 收到缺少 volume/amount 的数据")
        return pd.Series([float("nan")] * len(df), dtype=float)

    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cum_vol = np.cumsum(volume)
    cum_amt = np.cumsum(amount)
    denom = cum_vol * _SHARES_PER_LOT  # 手 -> 股
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(denom > 0.0, cum_amt / denom, np.nan)
    return pd.Series(vwap, dtype=float)


def change_pct(price: Optional[float], prev_close: Optional[float]) -> Optional[float]:
    """计算涨跌幅百分比 ``(price - prev_close) / prev_close * 100``。

    Args:
        price: 现价。
        prev_close: 昨收价。

    Returns:
        涨跌幅百分比（``float``）；``price`` 或 ``prev_close`` 缺失、``prev_close`` 为 0
        时返回 ``None``（避免除零 / 伪数据）。
    """
    if price is None or prev_close is None:
        return None
    try:
        base = float(prev_close)
        cur = float(price)
    except (TypeError, ValueError):
        return None
    if base == 0.0 or np.isnan(base):
        return None
    return (cur - base) / base * 100.0


def tick_color(
    price: Optional[float],
    prev_close: Optional[float],
    first_open: Optional[float],
) -> str:
    """按现价相对参考价决定颜色（红涨绿跌）。

    参考价优先取 ``prev_close``（昨收）；无昨收时退化为 ``first_open``（当日首笔开盘）。

    Args:
        price: 现价（或某根收盘价）。
        prev_close: 昨收价；可为 ``None``。
        first_open: 当日首笔开盘价；可为 ``None``。

    Returns:
        ``price >= 参考价`` → :data:`UP_COLOR`（红）；``price < 参考价`` → :data:`DOWN_COLOR`（绿）；
        参考价与现价均不可用时返回 :data:`TEXT_COLOR`（中性）。
    """
    reference: Optional[float] = prev_close if prev_close is not None else first_open
    if price is None or reference is None:
        return colors.TEXT_COLOR
    try:
        return colors.UP_COLOR if float(price) >= float(reference) else colors.DOWN_COLOR
    except (TypeError, ValueError):
        return colors.TEXT_COLOR


# --------------------------------------------------------------------------- #
# 视图
# --------------------------------------------------------------------------- #
class TickView(QWidget):
    """分时图（价格线 + 均价线 + 成交量副图）。"""

    def __init__(self, db: Optional[Database] = None, parent: Optional[QWidget] = None) -> None:
        """构造视图。

        Args:
            db: 存储实例；默认 :func:`get_db`。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._symbol: Optional[str] = None
        self._date: Optional[str] = None
        self._bars: pd.DataFrame = pd.DataFrame(columns=BAR_COLUMNS)
        self._avg: pd.Series = pd.Series(dtype=float)
        self._prev_close: Optional[float] = None
        self._first_open: Optional[float] = None
        self._setup_ui()
        self._show_placeholder(_EMPTY_TEXT)

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建头部信息 + 价格/均价主图 + 成交量副图 + 占位页。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QStackedWidget(self)
        layout.addWidget(self._stack)

        # --- 图表页 ---
        self._chart_page = QWidget(self)
        chart_layout = QVBoxLayout(self._chart_page)
        chart_layout.setContentsMargins(2, 2, 2, 2)
        chart_layout.setSpacing(2)

        # 头部信息行：代码·日期 | 现价 | 涨跌幅 | 均价
        header = QHBoxLayout()
        self._title_label = QLabel("", self._chart_page)
        self._title_label.setStyleSheet(f"color: {colors.TEXT_COLOR}; font-weight: bold;")
        self._price_label = QLabel("", self._chart_page)
        self._change_label = QLabel("", self._chart_page)
        self._avg_label = QLabel("", self._chart_page)
        self._avg_label.setStyleSheet(f"color: {_AVG_COLOR};")
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(self._price_label)
        header.addWidget(self._change_label)
        header.addWidget(self._avg_label)
        chart_layout.addLayout(header)

        # 十字光标角标文本行。
        self._cursor_label = QLabel("", self._chart_page)
        self._cursor_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        chart_layout.addWidget(self._cursor_label)

        # 主图（价格线 + 均价线）
        self._main_axis = DateAxisItem(orientation="bottom")
        self._main_plot = pg.PlotWidget(axisItems={"bottom": self._main_axis}, parent=self._chart_page)
        self._main_plot.setBackground(colors.BG_DARK)
        self._main_plot.showGrid(x=True, y=True, alpha=0.25)
        self._main_plot.setMenuEnabled(False)
        self._main_plot.setMouseEnabled(x=True, y=True)
        self._main_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self._main_plot.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)
        self._main_plot.getAxis("bottom").setStyle(showValues=False)  # 时间仅在副图显示

        self._price_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(_PRICE_COLOR, width=1.2)
        )
        self._avg_curve = pg.PlotDataItem(
            [], [], pen=pg.mkPen(_AVG_COLOR, width=1.2)
        )
        self._main_plot.addItem(self._price_curve)
        self._main_plot.addItem(self._avg_curve)

        # 昨收参考虚线。
        self._prev_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(_PREV_COLOR, width=1, style=Qt.PenStyle.DashLine)
        )
        self._main_plot.addItem(self._prev_line, ignoreBounds=True)
        self._prev_line.setVisible(False)

        # 十字光标（主图）
        cross_pen = pg.mkPen(colors.CROSS_CURSOR, width=1, style=Qt.PenStyle.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self._main_plot.addItem(self._vline, ignoreBounds=True)
        self._main_plot.addItem(self._hline, ignoreBounds=True)
        self._vline.setVisible(False)
        self._hline.setVisible(False)

        # 成交量副图（左轴用紧凑单位，避免大数值出现 1e+ 科学计数法）
        self._vol_axis = DateAxisItem(orientation="bottom")
        self._volume_plot = pg.PlotWidget(
            axisItems={
                "bottom": self._vol_axis,
                "left": CompactAxisItem(orientation="left"),
            },
            parent=self._chart_page,
        )
        self._volume_plot.setBackground(colors.BG_DARK)
        self._volume_plot.showGrid(x=True, y=True, alpha=0.25)
        self._volume_plot.setMenuEnabled(False)
        self._volume_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self._volume_plot.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)

        self._volume_item = pg.BarGraphItem(x=[], height=[], width=_VOL_WIDTH)
        self._volume_plot.addItem(self._volume_item)

        vol_cross_pen = pg.mkPen(colors.CROSS_CURSOR, width=1, style=Qt.PenStyle.DashLine)
        self._vol_vline = pg.InfiniteLine(angle=90, movable=False, pen=vol_cross_pen)
        self._volume_plot.addItem(self._vol_vline, ignoreBounds=True)
        self._vol_vline.setVisible(False)

        # X 轴联动
        self._volume_plot.setXLink(self._main_plot)

        # 主副图高度约 3:1
        chart_layout.addWidget(self._main_plot, 3)
        chart_layout.addWidget(self._volume_plot, 1)

        # 鼠标十字光标
        self._main_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # --- 占位页 ---
        self._placeholder_page = QWidget(self)
        ph_layout = QVBoxLayout(self._placeholder_page)
        self._placeholder_label = QLabel(_EMPTY_TEXT, self._placeholder_page)
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        ph_layout.addWidget(self._placeholder_label)

        self._stack.addWidget(self._chart_page)
        self._stack.addWidget(self._placeholder_page)

    # ------------------------------------------------------------------ #
    # 数据入口
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """把入参规范为升序、仅含 Bar 契约列；None/空返回空表。"""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        frame = df.copy()
        missing = [c for c in BAR_COLUMNS if c not in frame.columns]
        if missing:
            logger.warning("TickView 收到非法列：缺 %s", missing)
            return pd.DataFrame(columns=BAR_COLUMNS)
        frame = frame[BAR_COLUMNS].copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime", kind="stable")
        return frame.reset_index(drop=True)

    def set_minute(
        self,
        df_1m: Optional[pd.DataFrame],
        prev_close: Optional[float] = None,
    ) -> None:
        """用 1 分钟 Bar 契约 DataFrame 刷新分时图；空/None → 显示占位，不抛异常。

        Args:
            df_1m: 1 分钟 Bar 契约 DataFrame
                （列 ``[datetime, open, high, low, close, volume, amount]``，``datetime`` 升序，
                ``volume`` 单位「手」、``amount`` 单位「元」）。
            prev_close: 昨收价；``None`` 时涨跌幅显示 ``—`` 且颜色参考当日首笔开盘。
        """
        try:
            self._bars = self._normalize(df_1m)
            if self._bars.empty:
                self._show_placeholder(_EMPTY_TEXT)
                return
            self._prev_close = None if prev_close is None else float(prev_close)
            self._first_open = float(self._bars.iloc[0]["open"])
            self._avg = avg_price_series(self._bars)
            if self._date is None:
                last_dt = pd.to_datetime(self._bars["datetime"]).max()
                self._date = None if pd.isna(last_dt) else last_dt.strftime("%Y-%m-%d")
            self._render()
        except Exception as exc:  # noqa: BLE001 - 展示层不外抛
            logger.exception("渲染分时图失败")
            self._show_placeholder(f"渲染失败：{exc}")

    def load(self, symbol: str, date_: Optional[str] = None) -> None:
        """从**本地库**读取某标的某交易日的 1 分钟线并展示（不联网）。

        读取规则：
            * 1 分钟线取 ``db.get_bars(symbol, '1m')``；``date_`` 为 ``None`` 时
              取库中**最新交易日**，否则取指定交易日；
            * 昨收取 ``db.get_bars(symbol, 'd')`` 中 ``dt < 该日`` 的最后一根 ``close``；
              优先 ``qfq`` 口径，空则回退未复权口径。

        Args:
            symbol: 标的代码。
            date_: 交易日 ``YYYY-MM-DD``；``None`` 表示取本地最新交易日。
        """
        self._symbol = str(symbol)
        frame: Optional[pd.DataFrame] = None
        try:
            frame = self._db.get_bars(self._symbol, "1m")
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取本地 1 分钟线失败：%s", exc)
            frame = None

        normalized = self._normalize(frame)
        if normalized.empty:
            self._date = None if date_ is None else str(date_)
            self._prev_close = None
            self._first_open = None
            self._update_title()
            self._show_placeholder(_NO_LOCAL_MINUTE_TEXT)
            return

        dts = pd.to_datetime(normalized["datetime"], errors="coerce")
        if date_ is not None:
            day = pd.Timestamp(str(date_)).normalize()
        else:
            max_dt = dts.max()
            day = pd.Timestamp(max_dt).normalize()

        mask = dts.dt.normalize() == day
        day_frame = normalized[mask].reset_index(drop=True)
        if day_frame.empty:
            self._date = day.strftime("%Y-%m-%d")
            self._prev_close = None
            self._update_title()
            self._show_placeholder(_NO_LOCAL_MINUTE_TEXT)
            return

        self._date = day.strftime("%Y-%m-%d")
        self._prev_close = self._prev_close_from_daily(self._symbol, day)
        self.set_minute(day_frame, prev_close=self._prev_close)

    def clear(self) -> None:
        """清空图表并显示占位（保留当前 symbol/date 标题）。"""
        self._bars = pd.DataFrame(columns=BAR_COLUMNS)
        self._avg = pd.Series(dtype=float)
        self._show_placeholder(_EMPTY_TEXT)

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    def current_symbol(self) -> Optional[str]:
        """当前标的代码（未加载则为 ``None``）。"""
        return self._symbol

    def current_date(self) -> Optional[str]:
        """当前交易日文本 ``YYYY-MM-DD``（未加载则为 ``None``）。"""
        return self._date

    def bar_count(self) -> int:
        """当前展示的 1 分钟根数（供测试）。"""
        return int(len(self._bars))

    def is_placeholder_visible(self) -> bool:
        """占位页是否可见（供测试）。"""
        return self._stack.currentWidget() is self._placeholder_page

    def placeholder_text(self) -> str:
        """当前占位文案（供测试）。"""
        return self._placeholder_label.text()

    def current_price(self) -> Optional[float]:
        """当前现价（最后一根 ``close``）；无数据返回 ``None``。"""
        if self._bars.empty:
            return None
        return float(self._bars.iloc[-1]["close"])

    def current_change_pct(self) -> Optional[float]:
        """当前涨跌幅百分比；无昨收或现价时返回 ``None``。"""
        return change_pct(self.current_price(), self._prev_close)

    def current_avg_price(self) -> Optional[float]:
        """当前均价（最后一根 VWAP）；无数据返回 ``None``。"""
        if self._avg is None or self._avg.empty:
            return None
        value = self._avg.iloc[-1]
        return None if pd.isna(value) else float(value)

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def _render(self) -> None:
        """根据 ``self._bars`` 重绘价格线、均价线与量柱，并刷新头部。"""
        frame = self._bars
        count = len(frame)
        xs = list(range(count))
        closes = frame["close"].to_numpy(dtype=float)
        volumes = frame["volume"].to_numpy(dtype=float)
        avg = self._avg.to_numpy(dtype=float) if not self._avg.empty else np.full(count, np.nan)

        # 价格线 + 均价线
        self._price_curve.setData(xs, [float(v) for v in closes])
        self._avg_curve.setData(xs, [float(v) for v in avg])

        # 昨收参考线
        if self._prev_close is not None:
            self._prev_line.setValue(float(self._prev_close))
            self._prev_line.setVisible(True)
        else:
            self._prev_line.setVisible(False)

        # 量柱（红涨绿跌）
        brushes = [
            pg.mkBrush(tick_color(float(closes[i]), self._prev_close, self._first_open))
            for i in range(count)
        ]
        self._volume_plot.removeItem(self._volume_item)
        self._volume_item = pg.BarGraphItem(
            x=xs,
            height=[float(v) for v in volumes],
            width=_VOL_WIDTH,
            brushes=brushes,
            pen=None,
        )
        self._volume_plot.addItem(self._volume_item)

        # 时间轴标签
        labels = self._format_labels(frame["datetime"])
        self._main_axis.set_labels(labels)
        self._vol_axis.set_labels(labels)

        # 十字光标启用
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        self._vol_vline.setVisible(True)

        self._update_title()
        self._update_header()

        self._stack.setCurrentWidget(self._chart_page)
        self._main_plot.setXRange(-1, count, padding=0.0)
        self._volume_plot.setXRange(-1, count, padding=0.0)
        self._main_plot.enableAutoRange(axis="y", enable=True)

    def _format_labels(self, series: pd.Series) -> List[str]:
        """把 datetime 序列格式化为 ``HH:MM`` 时间轴标签。"""
        out: List[str] = []
        for value in pd.to_datetime(series, errors="coerce"):
            out.append(value.strftime("%H:%M") if not pd.isna(value) else "")
        return out

    def _update_title(self) -> None:
        """刷新标题为「代码 · 日期」。"""
        symbol = self._symbol or ""
        date_text = self._date or ""
        if symbol and date_text:
            self._title_label.setText(f"{symbol} · {date_text}")
        elif symbol:
            self._title_label.setText(symbol)
        else:
            self._title_label.setText(date_text)

    def _update_header(self) -> None:
        """刷新头部「现价 / 涨跌幅 / 均价」（颜色随涨跌）。"""
        price = self.current_price()
        pct = self.current_change_pct()
        avg = self.current_avg_price()
        color = tick_color(price, self._prev_close, self._first_open)
        style = f"color: {color}; font-weight: bold;"
        self._price_label.setStyleSheet(style)
        self._change_label.setStyleSheet(style)
        self._price_label.setText(f"现价 {price:.2f}" if price is not None else f"现价 {_DASH}")
        self._change_label.setText(f"{pct:+.2f}%" if pct is not None else _DASH)
        self._avg_label.setText(f"均价 {avg:.2f}" if avg is not None else f"均价 {_DASH}")

    def _prev_close_from_daily(self, symbol: str, day: pd.Timestamp) -> Optional[float]:
        """读取 ``dt < day`` 的最后一根日线收盘价作为昨收。

        优先取 ``qfq`` 复权口径（与日线落库默认一致），为空则回退未复权口径。

        Args:
            symbol: 标的代码。
            day: 当日（``00:00``）。

        Returns:
            昨收价；无可用数据返回 ``None``。
        """
        for adjust in ("qfq", ""):
            try:
                daily = self._db.get_bars(symbol, "d", adjust=adjust)
            except Exception as exc:  # noqa: BLE001 - 读库失败降级
                logger.debug("读取日线（adjust=%r）失败：%s", adjust, exc)
                continue
            if daily is None or daily.empty:
                continue
            dts = pd.to_datetime(daily["datetime"], errors="coerce")
            before = daily[dts < day]
            if before.empty:
                continue
            value = pd.to_numeric(before.iloc[-1]["close"], errors="coerce")
            if pd.isna(value):
                continue
            return float(value)
        return None

    def _show_placeholder(self, text: str) -> None:
        """切换到占位页并清空图表项（保留标题）。"""
        self._placeholder_label.setText(text)
        try:
            self._price_curve.setData([], [])
            self._avg_curve.setData([], [])
            self._volume_plot.removeItem(self._volume_item)
            self._volume_item = pg.BarGraphItem(x=[], height=[], width=_VOL_WIDTH)
            self._volume_plot.addItem(self._volume_item)
        except Exception:  # noqa: BLE001  # pragma: no cover - 防御
            pass
        self._prev_line.setVisible(False)
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._vol_vline.setVisible(False)
        self._price_label.setText("")
        self._change_label.setText("")
        self._avg_label.setText("")
        self._cursor_label.setText("")
        self._stack.setCurrentWidget(self._placeholder_page)

    # ------------------------------------------------------------------ #
    # 交互：十字光标
    # ------------------------------------------------------------------ #
    def _on_mouse_moved(self, pos) -> None:  # noqa: ANN001 - pyqtgraph 场景坐标
        """鼠标移动：更新十字光标与角标文本。"""
        if self._bars.empty or self._stack.currentWidget() is not self._chart_page:
            return
        if not self._main_plot.sceneBoundingRect().contains(pos):
            return
        try:
            point = self._main_plot.getViewBox().mapSceneToView(pos)
            index = int(round(point.x()))
            if index < 0 or index >= len(self._bars):
                return
            self._vline.setPos(index)
            self._hline.setPos(point.y())
            self._vol_vline.setPos(index)
            row = self._bars.iloc[index]
            dt = pd.to_datetime(row["datetime"])
            avg_value = self._avg.iloc[index] if index < len(self._avg) else float("nan")
            avg_text = f"{avg_value:.2f}" if not pd.isna(avg_value) else _DASH
            self._cursor_label.setText(
                f"{dt.strftime('%H:%M')}  价{row['close']:.2f} 均价{avg_text} 量{format_compact_number(row['volume'])}"
            )
        except Exception as exc:  # noqa: BLE001 - 交互不外抛
            logger.debug("分时十字光标更新失败：%s", exc)


__all__ = ["TickView", "avg_price_series", "change_pct", "tick_color"]
