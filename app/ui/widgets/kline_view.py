"""K 线主图 + 成交量副图（PyQtGraph）。

定位：**纯展示组件**（不联网、不写库）。
    * :meth:`KLineView.set_bars` 接收 Bar 契约 DataFrame 并绘制；
    * :meth:`KLineView.load` 仅从**本地库**读取后展示。

设计要点（遵循《细化设计》§4「P2-S2」/《开发手册》§4.2、§4.6）：
    * 蜡烛图用自定义 :class:`CandlestickItem`（``QPicture`` 绘制）——X 轴按**序号**定位，
      跳空/停牌日不占位；Y 轴为价格。
    * **红涨绿跌**：``close >= open`` → :data:`UP_COLOR`（红）；否则 :data:`DOWN_COLOR`（绿）。
      涨跌→颜色抽为纯函数 :func:`candle_color` 便于单测。
    * **涨停蜡烛金色 / 跌停蜡烛深绿**（2026-10-07 用户要求）：涨跌停不再画黄色文字
      标注（用户实测文字 y 位置仍有偏差），改为**直接给蜡烛着色**——判定序列（未复权）
      命中涨停的 bar 用 :data:`LIMIT_UP_COLOR`（金）、跌停用 :data:`LIMIT_DOWN_COLOR`
      （深绿）；关闭标注层开关时恢复普通红/绿。
    * X 轴刻度用自定义 :class:`DateAxisItem`：把序号映射为日期/时间文本，稀疏显示。
    * 成交量副图与主图 **X 轴联动**（``setXLink``），量柱按涨跌着色；主副图高度约 3:1。
    * 交互：右键框选缩放、滚轮缩放、左键拖拽平移（PyQtGraph ViewBox 默认），
      十字光标两条 ``InfiniteLine`` + 角标文本；悬停气泡显示于**鼠标左上方**。
    * **数值轴刻度按量级带单位**（k / m / b / t，见 :func:`format_compact_number`）：
      成交量轴与盈亏曲线轴的大数值不再出现 ``1e+09`` 科学计数法；十字光标角标中的
      「量」同样压缩展示（如 ``17079 → 17k``、``2076245988 → 2.07b``）。
    * **Y 轴随数据即时自适应**：:meth:`KLineView._render` 末尾主动触发 ViewBox 的
      ``updateAutoRange``——pyqtgraph 仅在「增删图元」时自动排队重算范围，自定义蜡烛项
      改数据不发信号；若不显式触发，模拟训练逐根回放时新 K 线会**跳出纵向视野**
      （须鼠标移入图区才恢复，见 2026-10-07 用户实测）。
    * **交易价参考线**：:meth:`KLineView.set_trade_price` 在指定价格处画**黄色虚线**
      （``ignoreBounds``，不参与坐标轴范围；模拟训练用，表示自定义成交价）。

P2-S4：叠加**辅助标注**（涨跌停 / 跳空缺口 / 除权除息）。
    * **判定序列必须用「未复权」（``adjust=""``）**：前复权价在除权日不连续，直接用复权价
      计算相邻 bar 的 ±10% / 跳空会**误判**（复权因子跳变处会产生假事件、真实事件被抹平）。
      故 :meth:`load` 除加载显示序列外，**再读一次未复权序列**作为判定序列；两者按
      ``datetime`` 精确对齐到显示索引。无未复权数据时回退用显示序列并告警（精度受限）。
    * 判定逻辑见 :mod:`app.ui.widgets.kline_annotations`（纯函数，便于单测）。
    * **涨跌停改为蜡烛着色**（2026-10-07）：不再渲染「涨停 / 跌停」黄色文字标注（用户
      实测其 y 位置在高分红股上仍有偏差），改为蜡烛直接着色；事件仍缓存于
      :meth:`annotation_events`（``price`` 为换算后的涨跌停价，供测试 / 后续复用）。
      缺口矩形与除权虚线照旧。
    * **标注 y 值按「显示收盘 / 判定收盘」比例换算到显示刻度**：判定序列为未复权、显示
      序列默认为前复权，两者价格刻度不同；若直接把未复权价画到前复权坐标轴上，高分红
      股会出现「标注浮高」（实测 601398 日 K 达 1.7 倍）。故标注价 / 缺口上下界
      均乘该比例（无比例或比例非法时保持原值不变）。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPicture, QPixmap
from PySide6.QtWidgets import (
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.data.resample import can_derive, derive_bars
from app.data.store.db import BAR_COLUMNS, Database, get_db
from app.export.image_exporter import dpi_scale, scale_pixmap
from app.ui.themes import colors
from app.ui.widgets.kline_annotations import (
    detect_ex_dividend,
    detect_gaps,
    detect_limit_events,
)

logger = logging.getLogger(__name__)

# 全局 pyqtgraph 选项（关闭抗锯齿以保性能）。
pg.setConfigOptions(antialias=False, background=colors.BG_DARK, foreground=colors.TEXT_COLOR)

# 周期编码 → 显示文案。
PERIOD_LABELS: dict[str, str] = {
    "1m": "1分",
    "5m": "5分",
    "15m": "15分",
    "30m": "30分",
    "60m": "60分",
    "d": "日",
    "w": "周",
    "m": "月",
}

# 分钟级周期的 datetime 文本格式（含时间），其余仅日期。
_MINUTE_PERIODS: frozenset[str] = frozenset({"1m", "5m", "15m", "30m", "60m"})

# 蜡烛/量柱宽度（X 序号单位）。
_BAR_WIDTH: float = 0.6

# 占位文案。
_EMPTY_TEXT: str = "无数据"
_NO_LOCAL_TEXT: str = "本地无数据，请点击『更新数据』"
# 分钟级周期占位：本版本「更新数据」仅拉日线，点它也拿不到分钟数据，故文案区分（避免误导）。
# ⚠️ 该句须与 tick_view._NO_LOCAL_MINUTE_TEXT **一字不差**。
_NO_LOCAL_MINUTE_TEXT: str = "本地暂无分钟数据（当前版本『更新数据』仅拉取日线）"

# 标注事件类别（键顺序稳定，供测试钩子返回）。
_ANNOTATION_KINDS: tuple[str, ...] = (
    "limit_up",
    "limit_down",
    "touch_up",
    "touch_down",
    "gap_up",
    "gap_down",
    "ex_div",
)

# 缺口矩形填充透明度（0-255）。
_GAP_ALPHA: int = 50


def candle_color(open_price: float, close_price: float) -> str:
    """按收盘/开盘决定蜡烛颜色（红涨绿跌）。

    Args:
        open_price: 开盘价。
        close_price: 收盘价。

    Returns:
        ``close >= open`` → :data:`UP_COLOR`（红）；否则 :data:`DOWN_COLOR`（绿）。
    """
    return colors.UP_COLOR if close_price >= open_price else colors.DOWN_COLOR


def volume_color(open_price: float, close_price: float) -> str:
    """按收盘/开盘决定量柱颜色（红涨绿跌）。

    Args:
        open_price: 开盘价。
        close_price: 收盘价。

    Returns:
        ``close >= open`` → :data:`VOL_UP`；否则 :data:`VOL_DOWN`。
    """
    return colors.VOL_UP if close_price >= open_price else colors.VOL_DOWN


#: 紧凑数值单位表（1000 进制；小写后缀，与用户约定一致）。
_COMPACT_UNITS: tuple[tuple[float, str], ...] = (
    (1e12, "t"),
    (1e9, "b"),
    (1e6, "m"),
    (1e3, "k"),
)


def format_compact_number(value: Any) -> str:
    """把数值按量级压缩为带单位文本（k / m / b / t），多余尾数**截断省略**。

    例：``17079 → "17k"``、``2076245988 → "2.07b"``、``999 → "999"``。用于坐标
    轴刻度 / 角标等大数值展示处，避免出现 ``1e+09`` 之类科学计数法。

    规则：绝对值 ``>= 1000`` 时取最大适用单位（k=1e3、m=1e6、b=1e9、t=1e12），
    保留 **3 位有效数字**（直接截断、不进位）再剔除尾零；``< 1000`` 原样显示
    （最多 2 位小数、剔尾零）；``None`` / ``NaN`` / ``Inf`` 返回 ``"—"``。

    Args:
        value: 任意数值（int / float / numpy 数值均可用）。

    Returns:
        紧凑文本，如 ``"17k"`` / ``"2.07b"`` / ``"-1.5m"`` / ``"999"``。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if number == 0:
        return "0"
    magnitude = abs(number)
    scaled = magnitude
    unit = ""
    for scale, suffix in _COMPACT_UNITS:
        if magnitude >= scale:
            scaled = magnitude / scale
            unit = suffix
            break
    if scaled >= 1:
        decimals = max(0, 2 - int(math.floor(math.log10(scaled))))
    else:
        decimals = 2
    truncated = math.trunc(scaled * 10**decimals) / 10**decimals
    text = f"{truncated:.{decimals}f}"
    if "." in text:  # 仅对小数剔除尾零（整数如 "120" 不可被误裁为 "12"）
        text = text.rstrip("0").rstrip(".")
    return f"{'-' if number < 0 else ''}{text or '0'}{unit}"


def hover_pct_change(prev_close: Optional[float], close: float) -> float:
    """计算相对**上一根收盘价**的涨跌幅百分比（供悬停气泡显示）。

    Args:
        prev_close: 上一根 bar 的收盘价；``None`` / ``0`` / ``NaN`` / 非法值均视为
            基准不可用。
        close: 当前 bar 的收盘价。

    Returns:
        涨跌幅百分比（如 ``1.23`` 表示 ``+1.23%``）；基准不可用时返回 ``0.0``。
    """
    try:
        if prev_close is None:
            return 0.0
        prev = float(prev_close)
        cur = float(close)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(prev) or not np.isfinite(cur) or prev == 0.0:
        return 0.0
    return (cur - prev) / prev * 100.0


def build_hover_html(dt_text: str, open_price: float, close_price: float, pct: float) -> str:
    """构造悬停气泡的 rich text HTML（日期 / 开盘 / 收盘 / 着色涨跌幅）。

    Args:
        dt_text: 日期（分钟周期含时间）文本。
        open_price: 开盘价。
        close_price: 收盘价。
        pct: 涨跌幅百分比（如 ``1.23`` / ``-1.23``）。

    Returns:
        HTML 字符串。涨跌幅 ``>0`` 用 :data:`UP_COLOR`（红）、``<0`` 用
        :data:`DOWN_COLOR`（绿）、``=0`` 用 :data:`SUBTEXT_COLOR`（灰）；
        开盘 / 收盘文本用 :data:`TEXT_COLOR`。
    """
    if pct > 0:
        pct_color = colors.UP_COLOR
    elif pct < 0:
        pct_color = colors.DOWN_COLOR
    else:
        pct_color = colors.SUBTEXT_COLOR
    return (
        f'<div style="color: {colors.TEXT_COLOR};">'
        f"<div>{dt_text}</div>"
        f"<div>开盘 {open_price:.2f}</div>"
        f"<div>收盘 {close_price:.2f}</div>"
        f'<div><span style="color: {pct_color};">{pct:+.2f}%</span></div>'
        f"</div>"
    )


class CandlestickItem(pg.GraphicsObject):
    """蜡烛图项（``QPicture`` 预绘制，X 轴按序号）。"""

    def __init__(self) -> None:
        """初始化为空。"""
        super().__init__()
        self._picture: QPicture = QPicture()
        self._bounds: QRectF = QRectF()
        self._data: List[tuple] = []  # (x, open, high, low, close, color)

    def set_data(self, data: List[tuple]) -> None:
        """设置蜡烛数据 ``[(x, open, high, low, close, color), ...]`` 并重绘。"""
        self._data = list(data)
        self._generate_picture()
        self.prepareGeometryChange()
        self.update()

    def data_count(self) -> int:
        """当前蜡烛根数（供测试）。"""
        return len(self._data)

    def _generate_picture(self) -> None:
        """把全部蜡烛画进 ``QPicture`` 并计算包围盒。"""
        picture = QPicture()
        painter = QPainter(picture)
        half = _BAR_WIDTH / 2.0
        xs: List[float] = []
        ys: List[float] = []
        for x, o, h, l, c, color in self._data:
            painter.setPen(pg.mkPen(color))
            # 影线
            painter.drawLine(QPointF(x, l), QPointF(x, h))
            top = max(o, c)
            bottom = min(o, c)
            body = top - bottom
            if body == 0:
                painter.drawLine(QPointF(x - half, top), QPointF(x + half, top))
            else:
                painter.setBrush(pg.mkBrush(color))
                painter.drawRect(QRectF(x - half, bottom, _BAR_WIDTH, body))
            xs.append(x)
            ys.extend((l, h))
        painter.end()
        self._picture = picture
        if xs:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            self._bounds = QRectF(min_x - 1.0, min_y, (max_x - min_x) + 2.0, (max_y - min_y) or 1.0)
        else:
            self._bounds = QRectF()

    def paint(self, painter: QPainter, *args) -> None:  # noqa: D401 - pyqtgraph 约定
        """绘制预生成的 ``QPicture``。"""
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        """返回数据包围盒。"""
        return self._bounds


class DateAxisItem(pg.AxisItem):
    """把 X 序号映射为日期/时间文本的自定义轴，稀疏显示。"""

    def __init__(self, *args, **kwargs) -> None:
        """初始化（标签表默认为空）。"""
        super().__init__(*args, **kwargs)
        self._labels: List[str] = []

    def set_labels(self, labels: List[str]) -> None:
        """设置 ``序号 -> 标签`` 映射。"""
        self._labels = list(labels)
        self.picture = None  # 强制重绘
        self.update()

    def label_count(self) -> int:
        """标签数量（供测试）。"""
        return len(self._labels)

    def tickValues(self, minVal, maxVal, size):  # noqa: N802 - Qt 约定
        """生成稀疏的整数刻度位置。"""
        if not self._labels:
            return []
        lo = max(0, int(np.floor(minVal)))
        hi = min(len(self._labels) - 1, int(np.ceil(maxVal)))
        if hi < lo:
            return []
        span = hi - lo + 1
        step = max(1, span // 8)  # 目标约 8 个刻度
        values = [float(v) for v in range(lo, hi + 1, step)]
        return [(float(step), values)]

    def tickStrings(self, values, scale, spacing):  # noqa: N802 - Qt 约定
        """把序号刻度转为日期/时间文本；越界返回空串。"""
        out: List[str] = []
        for value in values:
            idx = int(round(value))
            if 0 <= idx < len(self._labels) and abs(value - idx) < 1e-6:
                out.append(self._labels[idx])
            else:
                out.append("")
        return out


class CompactAxisItem(pg.AxisItem):
    """数值轴：刻度按量级带单位（k / m / b / t）显示。

    覆盖 :meth:`tickStrings`，避免 pyqtgraph 默认对 ``|v| >= 10000`` 使用 ``%g``
    产生 ``1e+09`` 科学计数法（详见 :func:`format_compact_number`）。
    """

    def tickStrings(self, values, scale, spacing):  # noqa: N802 - Qt 约定
        """把刻度值（``v * scale``）格式化为紧凑文本。"""
        return [format_compact_number(v * scale) for v in values]


class KLineView(QWidget):
    """K 线主图 + 成交量副图（纯展示）。"""

    #: 触发「导出K线图」打开的请求信号（P6-S2 右键菜单）。
    exportRequested = Signal()

    def __init__(self, db: Optional[Database] = None, parent: Optional[QWidget] = None) -> None:
        """构造视图。

        Args:
            db: 存储实例；默认 :func:`get_db`。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._symbol: Optional[str] = None
        self._period: str = "d"
        self._adjust: str = "qfq"
        self._bars: pd.DataFrame = pd.DataFrame(columns=BAR_COLUMNS)
        # 受限渲染视窗（P6 优化：模拟训练固定显示区）；None = 默认铺满全量数据。
        self._x_range: Optional[tuple[float, float]] = None
        # 标注（P2-S4）。
        self._annotations_enabled: bool = True
        self._annotation_events: Dict[str, List[dict]] = {
            kind: [] for kind in _ANNOTATION_KINDS
        }
        self._annotation_items: List[object] = []
        # 交易价参考线（模拟训练自定义成交价；None = 不显示）。
        self._trade_price_value: Optional[float] = None
        self._setup_ui()
        self._show_placeholder(_EMPTY_TEXT)

    # ------------------------------------------------------------------ #
    # UI 搭建
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建标题行 + 主副图 + 占位页。"""
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

        header = QHBoxLayout()
        self._title_label = QLabel("", self._chart_page)
        self._title_label.setStyleSheet(f"color: {colors.TEXT_COLOR}; font-weight: bold;")
        self._info_label = QLabel("", self._chart_page)
        self._info_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(self._info_label)
        chart_layout.addLayout(header)

        # 主图
        self._main_axis = DateAxisItem(orientation="bottom")
        self._main_plot = pg.PlotWidget(axisItems={"bottom": self._main_axis}, parent=self._chart_page)
        self._main_plot.setBackground(colors.BG_DARK)
        self._main_plot.showGrid(x=True, y=True, alpha=0.25)
        self._main_plot.setMenuEnabled(False)
        self._main_plot.setMouseEnabled(x=True, y=True)
        self._main_plot.getAxis("left").setTextPen(colors.SUBTEXT_COLOR)
        self._main_plot.getAxis("bottom").setTextPen(colors.SUBTEXT_COLOR)
        self._main_plot.getAxis("bottom").setStyle(showValues=False)  # 日期仅在副图显示

        self._candle_item = CandlestickItem()
        self._main_plot.addItem(self._candle_item)

        # 十字光标（主图）
        cross_pen = pg.mkPen(colors.CROSS_CURSOR, width=1, style=Qt.PenStyle.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        self._main_plot.addItem(self._vline, ignoreBounds=True)
        self._main_plot.addItem(self._hline, ignoreBounds=True)
        self._vline.setVisible(False)
        self._hline.setVisible(False)

        # 交易价参考线（黄色虚线；模拟训练自定义成交价，默认隐藏，不影响坐标轴范围）。
        self._trade_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(colors.TRADE_PRICE_COLOR, width=1, style=Qt.PenStyle.DashLine),
        )
        self._main_plot.addItem(self._trade_line, ignoreBounds=True)
        self._trade_line.setVisible(False)

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

        self._volume_item = pg.BarGraphItem(x=[], height=[], width=_BAR_WIDTH)
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

        # 悬停气泡（浮于图表页之上的自由控件，不参与布局）。
        self._hover_label = QLabel(self._chart_page)
        self._hover_label.setTextFormat(Qt.TextFormat.RichText)
        self._hover_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._hover_label.setWordWrap(False)
        self._hover_label.setStyleSheet(
            f"QLabel {{ background-color: #2a2e39; color: {colors.TEXT_COLOR};"
            f" border: 1px solid {colors.SUBTEXT_COLOR}; border-radius: 4px; padding: 6px; }}"
        )
        self._hover_label.hide()

        # P6-S2：右键菜单（导出K线图）。
        from PySide6.QtWidgets import QMenu

        self._context_menu: Optional["QMenu"] = None
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda pos: self._menu_export(QMenu(self), pos)
        )

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
            logger.warning("KLineView 收到非法列：缺 %s", missing)
            return pd.DataFrame(columns=BAR_COLUMNS)
        frame = frame[BAR_COLUMNS].copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime", kind="stable")
        return frame.reset_index(drop=True)

    def set_bars(
        self,
        df: Optional[pd.DataFrame],
        x_range: Optional[tuple[float, float]] = None,
    ) -> None:
        """用 Bar 契约 DataFrame 刷新图表；空/None → 显示「无数据」占位，不抛异常。

        Args:
            df: Bar 契约 DataFrame（列 ``[datetime, open, high, low, close, volume, amount]``，
                ``datetime`` 升序；``volume`` 单位「手」）。
            x_range: **固定 X 轴视窗** ``(lo, hi)``（X 轴单位为 bar 序号）；``None``（默认）
                时按数据量铺满（普通浏览）；传入时视窗**不随数据量变化**——模拟训练逐根
                显现回放时图表宽度恒定（P6 优化：解决「由宽变窄」观感问题）。
        """
        try:
            self._bars = self._normalize(df)
            self._x_range = x_range
            self._clear_annotations()  # 新数据先清空旧标注层
            if self._bars.empty:
                self._show_placeholder(_EMPTY_TEXT)
                return
            self._render()
        except Exception as exc:  # noqa: BLE001 - 展示层不外抛
            logger.exception("渲染 K 线失败")
            self._show_placeholder(f"渲染失败：{exc}")

    def set_header(self, symbol: str, period: str) -> None:
        """仅设置标题上下文（``代码 · 周期``），**不改动数据**。

        供受限渲染场景（模拟训练回放）复用本视图时刷新标题；普通浏览仍走 :meth:`load`。
        """
        self._symbol = str(symbol)
        self._period = str(period)
        self._update_title()

    def load(self, symbol: str, period: str, adjust: str = "qfq") -> None:
        """从**本地库**读取某标的某周期 K 线并展示（不联网），并叠加辅助标注。

        周线 / 月线（``'w'`` / ``'m'``）在本地为空时，**从同日线（同 ``adjust``）本地派生**
        （见 :mod:`app.data.resample`），派生结果**绝不落库**。日线也为空（派生必空）时
        回退占位，此时文案仍是「更新数据」指引（该指引对日线有效）。

        Args:
            symbol: 标的代码。
            period: 周期（``'d'|'w'|'m'|'1m'|...``）。
            adjust: 复权口径，默认 ``'qfq'``（用于**显示**）。
        """
        self._symbol = str(symbol)
        self._period = str(period)
        self._adjust = str(adjust)
        self._update_title()

        normalized = self._load_normalized(self._symbol, self._period, adjust)
        if normalized.empty and can_derive(self._period):
            # 周/月线本地无数据 → 用**同日线（同 adjust）**本地派生；派生结果不写库。
            daily = self._load_normalized(self._symbol, "d", adjust)
            normalized = self._normalize(derive_bars(daily, self._period))

        if normalized.empty:
            self._bars = normalized
            self._clear_annotations()
            self._show_placeholder(self._no_local_text(self._period))
            return

        # 判定序列：**未复权**（adjust=""）。复权价在除权日不连续，直接用其判定 ±10%/跳空会误判。
        # 周/月线本地没有未复权周期数据（更新仅拉日线），故 w/m 从**未复权日线**派生判定序列，
        # 而不是回退用前复权显示序列（否则除权日附近会误标涨停/缺口——用户实测「黄K与涨停偏移」）。
        det_norm = self._normalize(self._read_unadjusted(self._symbol, self._period))
        if det_norm.empty and can_derive(self._period):
            det_daily = self._load_normalized(self._symbol, "d", "")
            det_norm = self._normalize(derive_bars(det_daily, self._period))
        if det_norm.empty:
            logger.warning(
                "标的 %s 无未复权数据，回退用显示序列判定，涨跌停/缺口精度受限", self._symbol
            )
            det_norm = normalized

        self.set_bars(normalized)  # 渲染蜡烛/量柱，并清空标注层
        self._compute_annotations(self._symbol, det_norm, normalized)
        self._render()  # 判定完成后重绘蜡烛：涨停 → 金色、跌停 → 深绿
        self._render_annotations()

    def _read_unadjusted(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """读取**未复权**（``adjust=""``）序列作为标注判定序列。"""
        try:
            return self._db.get_bars(symbol, period, adjust="")
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取未复权 K 线失败：%s", exc)
            return None

    def _load_normalized(self, symbol: str, period: str, adjust: str) -> pd.DataFrame:
        """读本地 K 线并规范为 Bar 契约（读库失败降级为空表，**不抛**）。"""
        try:
            frame = self._db.get_bars(symbol, period, adjust=adjust)
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取本地 K 线失败：%s", exc)
            frame = None
        return self._normalize(frame)

    @staticmethod
    def _no_local_text(period: str) -> str:
        """按周期返回「本地无数据」占位文案（分钟级提示不同，避免误导）。"""
        return _NO_LOCAL_MINUTE_TEXT if period in _MINUTE_PERIODS else _NO_LOCAL_TEXT

    def clear(self) -> None:
        """清空图表并显示占位（保留当前 symbol/period 标题）。"""
        self._bars = pd.DataFrame(columns=BAR_COLUMNS)
        self._clear_annotations()
        self._show_placeholder(_EMPTY_TEXT)

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    def current_symbol(self) -> Optional[str]:
        """当前标的代码（未加载则为 ``None``）。"""
        return self._symbol

    def current_period(self) -> str:
        """当前周期编码。"""
        return self._period

    def export_pixmap(self, dpi: int = 300) -> Optional[QPixmap]:
        """抓取图表页并按 DPI 放大，供高清图导出使用（P4-S4）。

        抓图目标为图表页 ``_chart_page``（含标题 / 主图 / 副图 / 标注层），
        抓图逻辑封装在本方法内，**不向 exporter 泄漏私有属性**。当无数据
        （占位页显示）时返回 ``None``，调用方据此提示「当前无K线数据」。

        Args:
            dpi: 目标 DPI（默认 300）；实际像素尺寸 = 图表页逻辑尺寸 × ``dpi / 96``。

        Returns:
            按 DPI 放大后的 ``QPixmap``；无数据或抓图失败返回 ``None``。
        """
        if self.is_placeholder_visible():
            return None
        try:
            raw = self._chart_page.grab()
        except Exception as exc:  # noqa: BLE001 - 抓图失败降级为 None
            logger.warning("抓取 K 线图失败：%s", exc)
            return None
        if raw is None or raw.isNull():
            return None
        return scale_pixmap(raw, dpi_scale(dpi))

    def bar_count(self) -> int:
        """当前展示的 K 线根数（供测试）。"""
        return int(len(self._bars))

    def _menu_export(self, menu, pos) -> None:  # noqa: ANN001 - Qt 约定
        """右键菜单：导出K线图（弹菜单，选「导出K线图…」→ ``exportRequested``）。"""
        self._build_export_menu(menu).exec(self.mapToGlobal(pos))

    def _build_export_menu(self, menu) -> object:  # noqa: ANN001 - 返回已接线 QMenu（供测试）
        """构造导出右键菜单并接线（exec 分离，便于离屏测试触发动作）。"""
        action = menu.addAction("导出K线图…")
        action.triggered.connect(lambda: self.exportRequested.emit())
        return menu

    def is_placeholder_visible(self) -> bool:
        """占位页是否可见（供测试）。"""
        return self._stack.currentWidget() is self._placeholder_page

    def placeholder_text(self) -> str:
        """当前占位文案（供测试）。"""
        return self._placeholder_label.text()

    # ------------------------------------------------------------------ #
    # 标注（P2-S4）
    # ------------------------------------------------------------------ #
    def set_annotations_enabled(self, enabled: bool) -> None:
        """开启 / 关闭标注层显示（关闭时隐藏标注层，但**保留**计算结果）。

        涨跌停蜡烛着色（金 / 深绿）随该开关联动：关闭后恢复普通红 / 绿。

        Args:
            enabled: ``True`` 显示（并按需重建），``False`` 隐藏。
        """
        self._annotations_enabled = bool(enabled)
        if not self._bars.empty:
            self._render()  # 重绘蜡烛（涨停 / 跌停着色随开关变化）
        if self._annotations_enabled:
            self._render_annotations()
        else:
            for item in self._annotation_items:
                try:
                    item.setVisible(False)
                except Exception:  # noqa: BLE001  # pragma: no cover - 防御
                    pass

    def annotations_enabled(self) -> bool:
        """当前是否显示标注层。"""
        return self._annotations_enabled

    def annotation_counts(self) -> dict:
        """各类标注事件计数（供测试复核）。"""
        return {kind: len(self._annotation_events.get(kind, [])) for kind in _ANNOTATION_KINDS}

    def annotation_events(self) -> dict:
        """各类标注事件明细（供测试复核；返回浅拷贝列表）。

        ``price`` / ``lower`` / ``upper`` 均已按「显示 / 判定」收盘比例换算到**显示刻度**
        （见 :meth:`_display_scale_map`），可直接与显示序列的价格比较。
        """
        return {kind: list(self._annotation_events.get(kind, [])) for kind in _ANNOTATION_KINDS}

    def annotation_item_count(self) -> int:
        """已渲染的标注图元数量（供测试）。"""
        return len(self._annotation_items)

    # ------------------------------------------------------------------ #
    # 交易价参考线（模拟训练自定义成交价）
    # ------------------------------------------------------------------ #
    def set_trade_price(self, price: Optional[float]) -> None:
        """设置 / 清除交易价参考线（黄色虚线，画在主图 ``price`` 处）。

        Args:
            price: 成交价（元）；``None`` / 非法 / 非有限值 → 隐藏参考线。
        """
        value: Optional[float] = None
        if price is not None:
            try:
                value = float(price)
            except (TypeError, ValueError):
                value = None
            if value is not None and not math.isfinite(value):
                value = None
        self._trade_price_value = value
        self._sync_trade_price_line()

    def trade_price_value(self) -> Optional[float]:
        """当前交易价参考线的价格（未设置为 ``None``；供测试）。"""
        return self._trade_price_value

    def _sync_trade_price_line(self) -> None:
        """按当前状态显示 / 隐藏交易价参考线（无数据或占位页时隐藏）。"""
        value = self._trade_price_value
        if value is None or self._bars.empty or self._stack.currentWidget() is not self._chart_page:
            self._trade_line.setVisible(False)
            return
        self._trade_line.setPos(float(value))
        self._trade_line.setVisible(True)

    def _compute_annotations(
        self,
        symbol: str,
        det_frame: pd.DataFrame,
        display_frame: pd.DataFrame,
    ) -> None:
        """用判定序列计算三类事件，并按 ``datetime`` 对齐到显示序列索引后缓存。"""
        self._annotation_events = {kind: [] for kind in _ANNOTATION_KINDS}
        if det_frame is None or det_frame.empty or display_frame is None or display_frame.empty:
            return
        try:
            limit_events = detect_limit_events(det_frame, symbol, include_touch=False)
            gap_events = detect_gaps(det_frame)
        except Exception as exc:  # noqa: BLE001 - 标注失败不影响主图
            logger.exception("检测涨跌停/缺口失败：%s", exc)
            return
        try:
            factors = self._db.get_adj_factors(symbol)
            ex_events = detect_ex_dividend(factors, det_frame)
        except Exception as exc:  # noqa: BLE001 - 读复权因子失败降级
            logger.warning("读取复权因子失败：%s", exc)
            ex_events = []

        dt_to_idx = self._display_index_map(display_frame)
        scale_map = self._display_scale_map(det_frame, display_frame)
        bucket: Dict[str, List[dict]] = {}
        for ev in self._align_events(limit_events, dt_to_idx):
            self._rescale_y(ev, scale_map)
            bucket.setdefault(str(ev["kind"]), []).append(ev)
        for ev in self._align_events(gap_events, dt_to_idx):
            self._rescale_y(ev, scale_map)
            bucket.setdefault(str(ev["kind"]), []).append(ev)
        for ev in self._align_events(ex_events, dt_to_idx):
            bucket.setdefault("ex_div", []).append(ev)
        self._annotation_events = {kind: bucket.get(kind, []) for kind in _ANNOTATION_KINDS}

    @staticmethod
    def _display_index_map(display_frame: pd.DataFrame) -> Dict[pd.Timestamp, int]:
        """构建显示序列 ``datetime -> index`` 映射（精确匹配用）。"""
        mapping: Dict[pd.Timestamp, int] = {}
        for i, value in enumerate(pd.to_datetime(display_frame["datetime"], errors="coerce")):
            if not pd.isna(value):
                mapping[pd.Timestamp(value)] = i
        return mapping

    @staticmethod
    def _align_events(events: List[dict], dt_to_idx: Dict[pd.Timestamp, int]) -> List[dict]:
        """把判定序列上的事件按 ``datetime`` 对齐到显示序列索引；不可对齐者丢弃。"""
        aligned: List[dict] = []
        for ev in events:
            dt = ev.get("datetime")
            if dt is None or pd.isna(dt):
                continue
            idx = dt_to_idx.get(pd.Timestamp(dt))
            if idx is None:
                logger.debug("标注事件 datetime=%s 不在显示序列，已跳过", dt)
                continue
            new_ev = dict(ev)
            new_ev["index"] = idx
            aligned.append(new_ev)
        return aligned

    @staticmethod
    def _display_scale_map(
        det_frame: pd.DataFrame, display_frame: pd.DataFrame
    ) -> Dict[pd.Timestamp, float]:
        """构建每个 ``datetime`` 的「显示收盘 / 判定收盘」比例（用于 y 刻度换算）。

        判定序列为未复权、显示序列（默认）为前复权，价格刻度不同；比例 = 当日复权系数，
        缺失或非法（<=0）的日期不进入映射。
        """
        det_close: Dict[pd.Timestamp, float] = {}
        for dt, close in zip(
            pd.to_datetime(det_frame["datetime"], errors="coerce"),
            pd.to_numeric(det_frame["close"], errors="coerce"),
        ):
            if pd.isna(dt) or pd.isna(close) or float(close) <= 0:
                continue
            det_close[pd.Timestamp(dt)] = float(close)
        mapping: Dict[pd.Timestamp, float] = {}
        for dt, close in zip(
            pd.to_datetime(display_frame["datetime"], errors="coerce"),
            pd.to_numeric(display_frame["close"], errors="coerce"),
        ):
            if pd.isna(dt) or pd.isna(close) or float(close) <= 0:
                continue
            key = pd.Timestamp(dt)
            base = det_close.get(key)
            if base is not None and base > 0:
                mapping[key] = float(close) / base
        return mapping

    @staticmethod
    def _rescale_y(ev: dict, scale_map: Dict[pd.Timestamp, float]) -> None:
        """把事件 y 值（``price`` / ``lower`` / ``upper``）由判定刻度换算到显示刻度（原地）。

        比例缺失或非法时保持原值（例如无未复权数据、判定序列回退为显示序列的情形）。
        """
        dt = ev.get("datetime")
        if dt is None:
            return
        factor = scale_map.get(pd.Timestamp(dt))
        if factor is None or pd.isna(factor) or factor <= 0:
            return
        for key in ("price", "lower", "upper"):
            value = ev.get(key)
            if value is None:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if not pd.isna(num):
                ev[key] = num * factor

    def _render_annotations(self) -> None:
        """把缓存的标注事件渲染到主图（缺口矩形、除权竖虚线）。

        涨停 / 跌停不再绘制文字标注，改为蜡烛着色（见 :meth:`_render`）。
        """
        self._clear_annotation_items()
        if not self._annotations_enabled:
            return
        if self._bars.empty or self._stack.currentWidget() is not self._chart_page:
            return

        # 缺口：用半透明矩形覆盖 [lower, upper]。
        for kind in ("gap_up", "gap_down"):
            for ev in self._annotation_events.get(kind, []):
                self._add_gap_rect(ev)

        # 除权：竖虚线 + 文字。
        for ev in self._annotation_events.get("ex_div", []):
            self._add_exdiv_line(ev)

    def _add_gap_rect(self, ev: dict) -> None:
        """在缺口 bar 的 x 宽度内绘制 ``[lower, upper]`` 半透明矩形。"""
        index = ev.get("index")
        lower = ev.get("lower")
        upper = ev.get("upper")
        if index is None or lower is None or upper is None:
            return
        height = max(float(upper) - float(lower), 1e-6)
        rect = QRectF(float(index) - _BAR_WIDTH / 2.0, float(lower), _BAR_WIDTH, height)
        item = QGraphicsRectItem(rect)
        fill = pg.mkColor(colors.LIMIT_COLOR)
        fill.setAlpha(_GAP_ALPHA)
        item.setBrush(pg.mkBrush(fill))
        item.setPen(pg.mkPen(colors.LIMIT_COLOR, width=1))
        self._main_plot.addItem(item, ignoreBounds=True)
        self._annotation_items.append(item)

    def _add_exdiv_line(self, ev: dict) -> None:
        """在除权日对应 x 处绘制竖虚线并标注「除权」。"""
        index = ev.get("index")
        if index is None:
            return
        line = pg.InfiniteLine(
            pos=float(index),
            angle=90,
            movable=False,
            pen=pg.mkPen(colors.DIVIDEND_COLOR, width=1, style=Qt.PenStyle.DashLine),
        )
        self._main_plot.addItem(line, ignoreBounds=True)
        self._annotation_items.append(line)

        ymax = float(pd.to_numeric(self._bars["high"], errors="coerce").max())
        text = pg.TextItem("除权", color=colors.DIVIDEND_COLOR, anchor=(0.5, 0.0))
        text.setPos(float(index), ymax)
        self._main_plot.addItem(text, ignoreBounds=True)
        self._annotation_items.append(text)

    def _clear_annotation_items(self) -> None:
        """移除全部已渲染的标注图元（保留事件缓存）。"""
        for item in self._annotation_items:
            try:
                self._main_plot.removeItem(item)
            except Exception:  # noqa: BLE001  # pragma: no cover - 防御
                pass
        self._annotation_items = []

    def _clear_annotations(self) -> None:
        """清空标注图元与事件缓存。"""
        self._clear_annotation_items()
        self._annotation_events = {kind: [] for kind in _ANNOTATION_KINDS}

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #
    def _render(self) -> None:
        """根据 ``self._bars`` 重绘蜡烛、量柱与坐标轴。"""
        frame = self._bars
        count = len(frame)
        xs = list(range(count))
        opens = frame["open"].to_numpy(dtype=float)
        highs = frame["high"].to_numpy(dtype=float)
        lows = frame["low"].to_numpy(dtype=float)
        closes = frame["close"].to_numpy(dtype=float)
        volumes = frame["volume"].to_numpy(dtype=float)

        # 蜡烛：常规红/绿；涨停 → 金色、跌停 → 深绿（随标注开关联动，
        # 替代原「涨停 / 跌停」黄色文字标注）。
        limit_up_idx: set = set()
        limit_down_idx: set = set()
        if self._annotations_enabled:
            limit_up_idx = {
                ev.get("index") for ev in self._annotation_events.get("limit_up", [])
            }
            limit_down_idx = {
                ev.get("index") for ev in self._annotation_events.get("limit_down", [])
            }
        candle_data = []
        for i in range(count):
            o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
            color = candle_color(o, c)
            if i in limit_up_idx:
                color = colors.LIMIT_UP_COLOR
            elif i in limit_down_idx:
                color = colors.LIMIT_DOWN_COLOR
            candle_data.append((xs[i], o, h, l, c, color))
        self._candle_item.set_data(candle_data)

        # 量柱
        brushes = [pg.mkBrush(volume_color(float(opens[i]), float(closes[i]))) for i in range(count)]
        self._volume_plot.removeItem(self._volume_item)
        self._volume_item = pg.BarGraphItem(
            x=xs, height=[float(v) for v in volumes], width=_BAR_WIDTH, brushes=brushes, pen=None
        )
        self._volume_plot.addItem(self._volume_item)

        # 轴标签
        labels = self._format_labels(frame["datetime"])
        self._main_axis.set_labels(labels)
        self._vol_axis.set_labels(labels)

        # 十字光标可用
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        self._vol_vline.setVisible(True)

        self._stack.setCurrentWidget(self._chart_page)
        # X 轴视窗：受限渲染（回放）用固定窗口，使数据逐根显现时图表宽度恒定；
        # 普通浏览按数据量铺满。
        if self._x_range is not None:
            lo, hi = float(self._x_range[0]), float(self._x_range[1])
        else:
            lo, hi = -1.0, float(count)
        self._main_plot.setXRange(lo, hi, padding=0.0)
        self._volume_plot.setXRange(lo, hi, padding=0.0)
        self._main_plot.enableAutoRange(axis="y", enable=True)

        # 数据已更新 → 立即重算 Y 轴自适应范围（pyqtgraph 只在图元增删时排队重算，
        # 自定义蜡烛项改数据不发信号；否则新 K 线会跳出纵向视野）。
        self._main_plot.getViewBox().updateAutoRange()
        self._volume_plot.getViewBox().updateAutoRange()
        self._sync_trade_price_line()

    def _format_labels(self, series: pd.Series) -> List[str]:
        """把 datetime 序列格式化为轴标签（分钟含时间）。"""
        fmt = "%Y-%m-%d %H:%M" if self._period in _MINUTE_PERIODS else "%Y-%m-%d"
        out: List[str] = []
        for value in pd.to_datetime(series, errors="coerce"):
            out.append(value.strftime(fmt) if not pd.isna(value) else "")
        return out

    def _update_title(self) -> None:
        """刷新标题为「代码 · 周期」。"""
        symbol = self._symbol or ""
        label = PERIOD_LABELS.get(self._period, self._period)
        self._title_label.setText(f"{symbol} · {label}" if symbol else label)

    def _show_placeholder(self, text: str) -> None:
        """切换到占位页并清空图表项。"""
        self._placeholder_label.setText(text)
        self._candle_item.set_data([])
        try:
            self._volume_plot.removeItem(self._volume_item)
            self._volume_item = pg.BarGraphItem(x=[], height=[], width=_BAR_WIDTH)
            self._volume_plot.addItem(self._volume_item)
        except Exception:  # noqa: BLE001  # pragma: no cover - 防御
            pass
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._vol_vline.setVisible(False)
        self._info_label.setText("")
        self._hide_hover()
        self._stack.setCurrentWidget(self._placeholder_page)
        self._sync_trade_price_line()

    # ------------------------------------------------------------------ #
    # 交互：十字光标
    # ------------------------------------------------------------------ #
    def _hide_hover(self) -> None:
        """隐藏悬停气泡（幂等）。"""
        self._hover_label.hide()

    def _on_mouse_moved(self, pos) -> None:  # noqa: ANN001 - pyqtgraph 场景坐标
        """鼠标移动：更新十字光标、角标文本与悬停气泡。"""
        if self._bars.empty or self._stack.currentWidget() is not self._chart_page:
            self._hide_hover()
            return
        if not self._main_plot.sceneBoundingRect().contains(pos):
            self._hide_hover()
            return
        try:
            point = self._main_plot.getViewBox().mapSceneToView(pos)
            index = int(round(point.x()))
            if index < 0 or index >= len(self._bars):
                self._hide_hover()
                return
            self._vline.setPos(index)
            self._hline.setPos(point.y())
            self._vol_vline.setPos(index)
            row = self._bars.iloc[index]
            dt = pd.to_datetime(row["datetime"])
            fmt = "%Y-%m-%d %H:%M" if self._period in _MINUTE_PERIODS else "%Y-%m-%d"
            self._info_label.setText(
                f"{dt.strftime(fmt)}  开{row['open']:.2f} 高{row['high']:.2f} "
                f"低{row['low']:.2f} 收{row['close']:.2f}  量{format_compact_number(row['volume'])}"
            )
            self._update_hover(index, dt.strftime(fmt), row, pos)
        except Exception as exc:  # noqa: BLE001 - 交互不外抛
            logger.debug("十字光标更新失败：%s", exc)
            self._hide_hover()

    def _update_hover(self, index: int, dt_text: str, row, pos) -> None:  # noqa: ANN001 - 交互内部
        """在鼠标**左上方**显示 / 更新悬停气泡（日期 / 开盘 / 收盘 / 涨跌幅）。

        越界时翻转到右下（横向/纵向分别判断）并夹紧在图表页内。
        """
        prev_close = float(self._bars.iloc[index - 1]["close"]) if index > 0 else None
        pct = hover_pct_change(prev_close, float(row["close"]))
        self._hover_label.setText(
            build_hover_html(dt_text, float(row["open"]), float(row["close"]), pct)
        )
        self._hover_label.adjustSize()

        # 定位：默认鼠标**左上方** (-14, -14)；左侧/上侧越界则翻转到右侧/下方，并夹紧在页内。
        # 注：普通 QWidget 无 mapFromScene（仅 QGraphicsView/QGraphicsItem 具备），故先经主图
        # viewport 把场景坐标转为全局坐标，再映射回图表页局部坐标。
        viewport = self._main_plot.viewport()
        scene_point = self._main_plot.mapFromScene(pos)
        local = self._chart_page.mapFromGlobal(viewport.mapToGlobal(scene_point))
        width = self._hover_label.width()
        height = self._hover_label.height()
        x = local.x() - 14 - width
        y = local.y() - 14 - height
        if x < 0:
            x = local.x() + 14
        if y < 0:
            y = local.y() + 14
        x = min(max(0, int(x)), max(0, self._chart_page.width() - width))
        y = min(max(0, int(y)), max(0, self._chart_page.height() - height))
        self._hover_label.move(x, y)
        self._hover_label.show()
        self._hover_label.raise_()
