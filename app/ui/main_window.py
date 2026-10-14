"""主窗口。

P1-S1：可启动的空主窗口骨架。
P1-S6：新增「数据 → 更新数据」菜单与状态栏「最后更新时间」，更新在 **后台线程** 执行，
      通过事件总线驱动进度/完成文案，绝不阻塞 UI 线程。
P2-S1：升级为经典三栏布局（左=自选股栏、右=主区占位）+ 顶部工具栏 + 深色红涨绿跌主题。
P2-S2：右主区替换为 K 线主图 + 成交量副图（:class:`KLineView`）。
P2-S3：右主区升级为 ``QTabWidget``（页1「K线」= :class:`KLineView`、页2「分时」= :class:`TickView`）；
      选中自选股时同时加载两页；周期切到 ``1m`` 时自动切到「分时」页。
P6 优化（2026-10-02 用户要求）：
    * 模拟训练改为**非模态多窗口**（``show()`` 而非 ``exec()``，主界面可继续操作，
      可同时多开互不干扰）；
    * 「数据 → 股票代码与名称…」名录对话框（:class:`StockListDialog`）：搜索全 A 股
      代码 / 名称，选中即加载该股本地数据；「更新名录」复用 :class:`_UpdateWorker`
      在后台线程执行，**不额外开线程**。
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QSystemTrayIcon,
    QTabWidget,
    QWidget,
)

from app.core.config import get_config
from app.core.event_bus import get_bus
from app.core.backup import BackupError, create_backup, restore_backup
from app.data.datasource.base import get_source
from app.export.image_exporter import ImageExportError, export_kline_image
from app.data import update_service as us
from app.data.store.db import Database, get_db
from app.data.update_service import UpdateReport, UpdateService, friendly_error
from app.engine.alert import EVT_ALERT_TRIGGERED
from app.ui.shortcuts import ShortcutManager
from app.ui.themes import apply_theme
from app.ui.widgets.watchlist_panel import WATCHLIST_CHANGED
from app.ui.widgets import (
    AlertManagerDialog,
    FactorAnalysisDialog,
    IndicatorEditorDialog,
    KLineView,
    LogViewerDialog,
    MainToolBar,
    MemberSelectionDialog,
    NoteDetailDialog,
    PortfolioBacktestDialog,
    QualityAuditDialog,
    ScoreBacktestDialog,
    ScreenerDialog,
    ShortcutHelpDialog,
    SimWindow,
    StockListDialog,
    StrategyAnalysisDialog,
    TickView,
    WatchlistPanel,
)
from app.ui.widgets.alert_dialog import AlertNotification, build_alert_lines
from app.ui.widgets.tray import SystemTray

logger = logging.getLogger(__name__)

# 窗口标题（《细化设计》P1-S1 验收要求）。
WINDOW_TITLE: str = "A股行情分析"

# 最后更新时间占位符。
_NO_TIME: str = "—"

# 右主区页签：索引与标题。
TAB_KLINE_INDEX: int = 0
TAB_TICK_INDEX: int = 1
TAB_KLINE_TITLE: str = "K线"
TAB_TICK_TITLE: str = "分时"

# 内置提示音路径（P4-S2；由 scripts/gen_alert_wav.py 生成）。
ALERT_WAV_PATH: Path = Path(__file__).resolve().parents[2] / "assets" / "alert.wav"


class QSound:
    """QSound 兼容层。

    Qt6 / PySide6 6.x 已移除原生 ``QSound``，这里用 ``QSoundEffect`` 提供同名的
    ``QSound(path).play()`` 语义（异步播放 wav 短提示音）。播放失败由调用侧 try/except
    兜底，保证无音频设备时不崩溃。
    """

    def __init__(self, path: str) -> None:
        self._effect = QSoundEffect()
        self._effect.setSource(QUrl.fromLocalFile(str(path)))

    def play(self) -> None:
        """开始播放（异步）。"""
        self._effect.play()


class _BusBridge(QObject):
    """把事件总线回调（可能来自后台线程）转成 Qt 信号，交由 UI 线程处理。"""

    progress = Signal(dict)
    failed = Signal(dict)
    completed = Signal(dict)
    alert_triggered = Signal(dict)


class _UpdateWorker(QThread):
    """在后台线程执行一次数据更新（日线 / 分钟线）。"""

    done = Signal(object)  # UpdateReport

    def __init__(
        self,
        service: UpdateService,
        parent: Optional[QObject] = None,
        symbols: Optional[List[str]] = None,
        kind: str = "daily",
    ) -> None:
        """构造后台更新线程。

        Args:
            service: 更新服务实例。
            parent: 父对象。
            symbols: 限定更新标的列表；``None`` 表示按自选股清单全量。
            kind: 更新种类；``'daily'`` 走 ``update_daily``，``'minutes'`` 走
                ``update_minutes``（5/15/30/60m 一次拉全，供模拟训练回放），
                ``'symbols'`` 走 ``update_symbols``（全 A 股代码与名称名录）。
        """
        super().__init__(parent)
        self._service = service
        self._symbols: Optional[List[str]] = symbols
        self._kind: str = kind

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：执行更新并把报告通过信号回传（异常已由服务内部消化）。"""
        try:
            if self._kind == "minutes":
                report = self._service.update_minutes(symbols=self._symbols)
            elif self._kind == "symbols":
                report = self._service.update_symbols()
            else:
                report = self._service.update_daily(symbols=self._symbols)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台更新线程异常")
            report = UpdateReport(
                symbols_count=0, bars_written=0, failed=[("", friendly_error(exc))]
            )
        self.done.emit(report)


class MainWindow(QMainWindow):
    """应用主窗口。"""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        db: Optional[Database] = None,
        service: Optional[UpdateService] = None,
    ) -> None:
        """构造主窗口。

        Args:
            parent: 父控件，通常为 ``None``。
            db: 存储实例（用于读取最后更新时间）；默认 :func:`get_db`。
            service: 更新服务；默认基于 ``db`` 构造。
        """
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 800)

        self._db: Database = db if db is not None else get_db()
        self._service: UpdateService = (
            service if service is not None else UpdateService(db=self._db)
        )
        self._update_thread: Optional[_UpdateWorker] = None
        #: 最近一次后台更新的种类（``daily`` / ``minutes`` / ``symbols``），用于完成文案。
        self._update_kind: str = "daily"
        #: 非模态「模拟训练」窗口引用（可多开；关闭即从列表移除，见 _open_sim_window）。
        self._sim_windows: List[SimWindow] = []
        #: 非模态「股票代码与名称」对话框（单实例；关闭即置空，见 _open_stock_list）。
        self._stock_list_dialog: Optional[StockListDialog] = None
        self._bus = get_bus()
        self._bridge = _BusBridge(self)

        # P4-S2：托盘 / 退出 / 提示音状态。
        self._really_quit: bool = False
        self._alert_sound: Optional[object] = None
        self._tray: Optional[SystemTray] = None

        self._setup_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._wire_events()
        self._apply_theme()
        self._init_tray()
        self._refresh_last_update_time()
        self._install_shortcuts()
        self._load_initial_data()
        logger.info("主窗口初始化完成：%s", WINDOW_TITLE)

    # ------------------------------------------------------------------ #
    # UI 骨架
    # ------------------------------------------------------------------ #
    def _setup_ui(self) -> None:
        """搭建三栏骨架：左=自选股栏，右=主区 ``QTabWidget``（K线 + 分时）。"""
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setChildrenCollapsible(False)

        # 左：自选股栏。
        self.watchlist_panel = WatchlistPanel(db=self._db, parent=self._splitter)
        self.watchlist_panel.symbolSelected.connect(self._on_symbol_selected)
        self.watchlist_panel.detailRequested.connect(self._open_note_detail)
        self.watchlist_panel.alertRequested.connect(self._open_alert_manager)

        # 右：QTabWidget（页1 K线 / 页2 分时）。
        self.right_tabs = QTabWidget(self._splitter)
        self.right_tabs.setDocumentMode(True)

        self.kline_view = KLineView(db=self._db, parent=self.right_tabs)
        self.kline_view.exportRequested.connect(self._on_export_kline)
        self.tick_view = TickView(db=self._db, parent=self.right_tabs)
        self.right_tabs.addTab(self.kline_view, TAB_KLINE_TITLE)
        self.right_tabs.addTab(self.tick_view, TAB_TICK_TITLE)
        self.right_tabs.setCurrentIndex(TAB_KLINE_INDEX)

        # 兼容旧引用（P2-S2 曾以 self._main_area 指代右主区）。
        self._main_area = self.right_tabs

        self._splitter.addWidget(self.watchlist_panel)
        self._splitter.addWidget(self.right_tabs)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([240, 960])
        self.setCentralWidget(self._splitter)

    def _build_toolbar(self) -> None:
        """顶部工具栏：更新数据 + 周期切换。"""
        self.toolbar = MainToolBar(self)
        self.toolbar.updateRequested.connect(self._on_update_daily)
        self.toolbar.periodChanged.connect(self._on_period_changed)
        self.addToolBar(self.toolbar)

    def _install_shortcuts(self) -> None:
        """注册快捷键（``1..8`` 切周期、``↑/↓`` 切自选），列表存 ``self.shortcuts``。"""
        self.shortcut_manager = ShortcutManager(self)
        self.shortcut_manager.install()
        self.shortcuts = self.shortcut_manager.qshortcuts()
        logger.info("已注册快捷键 %d 个", len(self.shortcuts))

    def _load_initial_data(self) -> None:
        """启动加载：若自选股非空，自动选中第一只并加载其本地 K 线（**只读本地库，绝不联网**）。

        复用 :meth:`WatchlistPanel.select_next` 从「无选中」落到第一只的既有语义，
        经 ``symbolSelected`` → :meth:`_on_symbol_selected` 触发 K 线/分时加载。
        无自选股则保持占位、不做任何事；任何异常都不得让主窗口起不来。
        """
        try:
            if self.watchlist_panel.symbols():
                self.watchlist_panel.select_next()
                logger.info("启动已加载首只自选股：%s", self.watchlist_panel.current_symbol())
        except Exception:  # noqa: BLE001 - 启动加载失败不影响主窗口
            logger.exception("启动加载本地自选股失败")

    # ------------------------------------------------------------------ #
    # 槽：快捷键（P2-S5）
    # ------------------------------------------------------------------ #
    def _on_period_shortcut(self, code: str) -> None:
        """数字键 ``1..8``：切换周期（复用 ``periodChanged`` → K 线重载；``1m`` 自动切分时页）。"""
        self.toolbar.set_period(code)

    def _on_next_symbol(self) -> None:
        """``↓``：选中下一只自选股（跨分组连续、端点不回绕）。

        说明：方向键**不再走 QShortcut**（会抢占 QTreeWidget 原生方向键）。焦点在树上时由
        :class:`~app.ui.widgets.watchlist_panel.WatchlistPanel` 的事件过滤器处理；焦点在别处
        （工具栏/状态栏等）时由本窗口的 :meth:`keyPressEvent` 兜底调用本方法。
        """
        try:
            self.watchlist_panel.select_next()
        except Exception:  # noqa: BLE001 - 交互不外抛
            logger.exception("切换到下一只自选股失败")

    def _on_prev_symbol(self) -> None:
        """``↑``：选中上一只自选股（处理分工见 :meth:`_on_next_symbol`）。"""
        try:
            self.watchlist_panel.select_previous()
        except Exception:  # noqa: BLE001 - 交互不外抛
            logger.exception("切换到上一只自选股失败")

    def keyPressEvent(self, event) -> None:  # noqa: ANN001 - Qt 约定
        """窗口级方向键兜底：``↓`` / ``↑`` 切换自选股。

        仅在**焦点不在自选股树上**时才会走到这里（树上按键由面板事件过滤器处理），
        以覆盖「焦点在工具栏/状态栏等不消费方向键的控件上」的场景。控件自行消费方向键时
        （如周期下拉框）事件不会到达本方法，属正常。
        """
        key = event.key()
        if key == Qt.Key.Key_Down:
            self._on_next_symbol()
            event.accept()
            return
        if key == Qt.Key.Key_Up:
            self._on_prev_symbol()
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_theme(self) -> None:
        """应用深色主题（app 为 None 时 apply_theme 内部跳过并记日志）。"""
        theme = str(get_config().get("ui.theme", "dark_red_green"))
        apply_theme(QApplication.instance(), theme)

    def _build_menu(self) -> None:
        """菜单栏：数据 / 指标 / 选股 / 导出（导出K线图）。"""
        menu_bar = self.menuBar()
        self._data_menu = menu_bar.addMenu("数据(&D)")

        self._action_update = QAction("更新数据(&U)", self)
        self._action_update.setStatusTip("从数据源拉取自选股日线并落库（后台执行）")
        self._action_update.triggered.connect(self._on_update_daily)
        self._data_menu.addAction(self._action_update)

        # P6-S4 缺陷修复：分钟线更新（baostock 分钟接口实测可用，供模拟训练回放）。
        self._action_update_minute = QAction("更新分钟数据(&M)", self)
        self._action_update_minute.setStatusTip(
            "拉取自选股近 10 日 5/15/30/60 分钟线并落库（后台执行）"
        )
        self._action_update_minute.triggered.connect(self._on_update_minutes)
        self._data_menu.addAction(self._action_update_minute)

        # P6 优化：股票代码与名称名录（搜索全 A 股 → 选中加载本地数据）。
        self._action_stock_list = QAction("股票代码与名称(&N)…", self)
        self._action_stock_list.setStatusTip(
            "搜索全 A 股代码 / 名称名录；选中即加载该股本地数据（名录可在对话框内更新）"
        )
        self._action_stock_list.triggered.connect(self._open_stock_list)
        self._data_menu.addAction(self._action_stock_list)

        # P7-S4：数据质量审计（六项检查：覆盖度 / OHLC / 日历 / 异常波动 / 复权对账 / 幸存者偏差）。
        self._action_quality_audit = QAction("数据质量审计(&A)…", self)
        self._action_quality_audit.setStatusTip(
            "审计本地日线质量：覆盖度与停牌、OHLC 逻辑、日历一致性、异常波动、"
            "复权自洽、幸存者偏差量化（只读，不修改数据）"
        )
        self._action_quality_audit.triggered.connect(self._open_quality_audit)
        self._data_menu.addAction(self._action_quality_audit)

        self._data_menu.addSeparator()
        # P6-S1：备份 / 恢复（数据管理，放在「退出」之前）。
        self._action_backup = QAction("备份数据(&B)", self)
        self._action_backup.setStatusTip("将数据库+配置+缓存+导出产物打包为时间戳 zip")
        self._action_backup.triggered.connect(self._on_backup)
        self._data_menu.addAction(self._action_backup)
        self._action_restore = QAction("恢复数据(&R)", self)
        self._action_restore.setStatusTip("从备份 zip 恢复；恢复前自动快照当前数据")
        self._action_restore.triggered.connect(self._on_restore)
        self._data_menu.addAction(self._action_restore)

        self._data_menu.addSeparator()
        self._action_quit = QAction("退出(&Q)", self)
        self._action_quit.triggered.connect(self._quit_application)
        self._data_menu.addAction(self._action_quit)

        self._indicator_menu = menu_bar.addMenu("指标(&I)")
        self._action_indicator_editor = QAction("自定义指标编辑器(&E)", self)
        self._action_indicator_editor.setStatusTip("编写自定义指标表达式并保存为模板")
        self._action_indicator_editor.triggered.connect(self._open_indicator_editor)
        self._indicator_menu.addAction(self._action_indicator_editor)

        self._screener_menu = menu_bar.addMenu("选股(&S)")
        self._action_screener = QAction("选股(&S)", self)
        self._action_screener.setStatusTip("按结构化条件在自选股范围内选股")
        self._action_screener.triggered.connect(self._open_screener)
        self._screener_menu.addAction(self._action_screener)

        # P7-S2：策略分析（回测 + 绩效归因，独立模态对话框）。
        self._strategy_menu = menu_bar.addMenu("策略(&P)")
        self._action_strategy_analysis = QAction("策略分析(&A)…", self)
        self._action_strategy_analysis.setStatusTip(
            "配置策略条件并回测，查看净值 / 回撤曲线与逐年、牛熊、α·β 归因"
        )
        self._action_strategy_analysis.triggered.connect(self._open_strategy_analysis)
        self._strategy_menu.addAction(self._action_strategy_analysis)

        # P7-S3：因子分析（多因子有效性：IC / 分层 / 共线性，独立模态对话框）。
        self._action_factor_analysis = QAction("因子分析(&F)…", self)
        self._action_factor_analysis.setStatusTip(
            "对股票池做多因子有效性分析：Rank IC 与显著性、分层单调性、相关矩阵与 VIF"
        )
        self._action_factor_analysis.triggered.connect(self._open_factor_analysis)
        self._strategy_menu.addAction(self._action_factor_analysis)

        # P7-S5：多因子打分选股（ICIR 加权打分 Top-N → 逐日回放，独立模态对话框）。
        self._action_score_backtest = QAction("多因子打分(&E)…", self)
        self._action_score_backtest.setStatusTip(
            "多因子 ICIR 加权打分取 Top-N 并回测：净值 / 绩效 / 逐年 / 牛熊 / 权重与名单"
        )
        self._action_score_backtest.triggered.connect(self._open_score_backtest)
        self._strategy_menu.addAction(self._action_score_backtest)

        # P7-S7：多策略组合回测（条件式 + 打分「资金分账 · 等权」，独立模态对话框）。
        self._action_portfolio_backtest = QAction("多策略组合(&C)…", self)
        self._action_portfolio_backtest.setStatusTip(
            "两条基线策略按资金分账等权组合回测：成员净值对照 / 相关性矩阵 / 贡献归因"
        )
        self._action_portfolio_backtest.triggered.connect(self._open_portfolio_backtest)
        self._strategy_menu.addAction(self._action_portfolio_backtest)

        # P7-S8：策略成员筛选（样本内筛选 + 样本外验证，独立模态对话框）。
        self._action_member_selection = QAction("成员筛选(&M)…", self)
        self._action_member_selection.setStatusTip(
            "候选策略筛选与筛选后组合验证：绝对门槛 + 分年度稳定性，四线样本外对比"
        )
        self._action_member_selection.triggered.connect(self._open_member_selection)
        self._strategy_menu.addAction(self._action_member_selection)

        self._export_menu = menu_bar.addMenu("导出(&E)")
        self._action_export_kline = QAction("导出K线图(&K)", self)
        self._action_export_kline.setStatusTip("将当前K线图（主图+副图+标注）导出为高清 PNG")
        self._action_export_kline.triggered.connect(self._on_export_kline)
        self._export_menu.addAction(self._action_export_kline)

        # P5-S4：模拟训练（独立模态窗口）。
        self._sim_menu = menu_bar.addMenu("训练(&T)")
        self._action_sim = QAction("模拟训练(&S)", self)
        self._action_sim.setStatusTip("历史 1 分钟回放训练（T+1、做多、无费、屏蔽未来）")
        self._action_sim.triggered.connect(self._open_sim_window)
        self._sim_menu.addAction(self._action_sim)

        # P6-S2：帮助菜单（快捷键说明）。
        self._help_menu = menu_bar.addMenu("帮助(&H)")
        self._action_shortcuts = QAction("快捷键说明(&K)", self)
        self._action_shortcuts.triggered.connect(self._open_shortcut_help)
        self._help_menu.addAction(self._action_shortcuts)
        # P6-S3：帮助菜单（查看日志）。
        self._action_log = QAction("查看日志(&L)", self)
        self._action_log.setStatusTip("在界面内查看 data/app.log 末尾日志")
        self._action_log.triggered.connect(self._open_log_viewer)
        self._help_menu.addAction(self._action_log)

    def _build_statusbar(self) -> None:
        """状态栏：常驻标签显示「最后更新时间」。"""
        self._last_update_label = QLabel(self)
        self.statusBar().addPermanentWidget(self._last_update_label)
        self.statusBar().showMessage("就绪")

    def _wire_events(self) -> None:
        """订阅事件总线，经 Qt 信号在 UI 线程更新状态。"""
        # 订阅回调可能在后台线程触发；仅做 Qt 信号发射（线程安全，跨线程自动排队）。
        self._bus.subscribe(us.EVT_PROGRESS, lambda payload: self._bridge.progress.emit(payload))
        self._bus.subscribe(us.EVT_FAILED, lambda payload: self._bridge.failed.emit(payload))
        self._bus.subscribe(us.EVT_COMPLETED, lambda payload: self._bridge.completed.emit(payload))
        self._bus.subscribe(
            EVT_ALERT_TRIGGERED, lambda payload: self._bridge.alert_triggered.emit(payload)
        )

        self._bridge.progress.connect(self._on_progress)
        self._bridge.failed.connect(self._on_failed)
        self._bridge.completed.connect(self._on_completed)
        self._bridge.alert_triggered.connect(self._on_alert_triggered)

        # 自选股变更：新增标的时若本地无该股 K 线 → 自动触发一次更新（避免「加了看不到图」）。
        # WATCHLIST_CHANGED 由自选面板在 UI 线程发布，此处可直接调用槽（无需桥接线程）。
        self._bus.subscribe(WATCHLIST_CHANGED, self._on_watchlist_changed)

    # ------------------------------------------------------------------ #
    # 状态刷新
    # ------------------------------------------------------------------ #
    def _refresh_last_update_time(self) -> None:
        """从 ``db.get_meta`` 读取最后更新时间并刷新标签。"""
        value: Optional[str] = None
        try:
            value = self._db.get_meta(us.META_LAST_UPDATE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 last_update_time 失败：%s", exc)
        self._last_update_label.setText(f"最后更新时间：{value or _NO_TIME}")

    # ------------------------------------------------------------------ #
    # 槽：更新流程
    # ------------------------------------------------------------------ #
    def _on_update_daily(self, symbols: Optional[List[str]] = None) -> None:
        """菜单触发：在后台线程启动日线更新（可选限定标的列表）。"""
        if self._update_thread is not None and self._update_thread.isRunning():
            logger.info("更新已在进行中，忽略重复触发")
            return
        self._action_update.setEnabled(False)
        self.statusBar().showMessage("正在更新数据…")
        self._update_kind = "daily"
        self._update_thread = _UpdateWorker(self._service, self, symbols=symbols)
        self._update_thread.done.connect(self._on_update_finished)
        self._update_thread.finished.connect(self._on_thread_finished)
        self._update_thread.start()
        logger.info("已在后台线程启动日线更新")

    def _on_update_minutes(self) -> None:
        """菜单触发：在后台线程拉取自选股近 10 日 5/15/30/60 分钟线并落库（供模拟训练回放）。"""
        if self._update_thread is not None and self._update_thread.isRunning():
            logger.info("更新已在进行中，忽略重复触发")
            return
        self._action_update_minute.setEnabled(False)
        self.statusBar().showMessage("正在更新分钟数据…")
        self._update_kind = "minutes"
        self._update_thread = _UpdateWorker(self._service, self, kind="minutes")
        self._update_thread.done.connect(self._on_update_finished)
        self._update_thread.finished.connect(self._on_thread_finished)
        self._update_thread.start()
        logger.info("已在后台线程启动分钟线更新")

    def _on_update_symbols(self) -> None:
        """名录对话框「更新名录」请求：后台线程拉取全 A 股代码与名称（不阻塞 UI）。

        复用 :attr:`_update_thread` 单线程槽位（与日线 / 分钟线更新互斥）；
        完成后 :meth:`_on_update_finished` 出状态栏文案，``EVT_COMPLETED``
        （``scope == "symbols"``）驱动名录对话框自动刷新。
        """
        if self._update_thread is not None and self._update_thread.isRunning():
            logger.info("更新已在进行中，忽略重复触发")
            return
        self.statusBar().showMessage("正在更新股票名录…")
        self._update_kind = "symbols"
        self._update_thread = _UpdateWorker(self._service, self, kind="symbols")
        self._update_thread.done.connect(self._on_update_finished)
        self._update_thread.finished.connect(self._on_thread_finished)
        self._update_thread.start()
        logger.info("已在后台线程启动股票名录更新")

    def _on_update_finished(self, report: object) -> None:
        """更新线程结束：恢复菜单并刷新状态栏（按 :attr:`_update_kind` 出对应文案）。"""
        self._action_update.setEnabled(True)
        self._action_update_minute.setEnabled(True)
        self._refresh_last_update_time()
        if not isinstance(report, UpdateReport):  # pragma: no cover - 防御
            self.statusBar().showMessage("更新完成", 10000)
            return
        if self._update_kind == "symbols":
            if report.ok:
                self.statusBar().showMessage(
                    f"名录更新完成：共 {report.symbols_count} 只", 10000
                )
            else:
                reason = report.failed[0][1] if report.failed else "未知原因"
                self.statusBar().showMessage(f"名录更新失败：{reason}", 10000)
            return
        self.statusBar().showMessage(
            f"更新完成：{report.symbols_count} 只，写入 {report.bars_written} 行，"
            f"失败 {len(report.failed)} 只",
            10000,
        )

    def _on_thread_finished(self) -> None:
        """线程对象结束后释放引用。"""
        if self._update_thread is not None:
            self._update_thread.deleteLater()
            self._update_thread = None

    def _on_progress(self, payload: dict) -> None:
        """进度事件：更新状态栏文案。"""
        current = payload.get("current")
        total = payload.get("total")
        symbol = payload.get("symbol")
        self.statusBar().showMessage(f"更新中 {current}/{total}：{symbol}")

    def _on_failed(self, payload: dict) -> None:
        """失败事件：状态栏提示「标的 + 人话原因」（原始详情见 data/app.log）。"""
        symbol = payload.get("symbol")
        reason = payload.get("error") or ""
        self.statusBar().showMessage(f"更新失败：{symbol}（{reason}）" if reason else f"更新失败：{symbol}")

    def _on_completed(self, payload: dict) -> None:
        """完成事件：刷新最后更新时间。"""
        self._refresh_last_update_time()

    def _on_watchlist_changed(self, payload: dict) -> None:
        """自选股变更：新增标的且本地无其 K 线 → 自动触发一次更新。

        触发条件（严格）：
            * ``action == "add"``（排除 remove/group 事件）；
            * ``symbol`` 非空；
            * 本地库该股无任何 K 线（``count_bars(symbol) == 0``）——已有数据则
              （如刚点过更新）不重复拉，避免冗余网络请求。
        复用 ``_on_update_daily`` 的后台线程更新入口；更新目标为**单只**增量拉取。
        """
        try:
            action = payload.get("action")
            symbol = str(payload.get("symbol") or "")
            if action != "add" or not symbol:
                return
            if self._db.count_bars(symbol) > 0:
                return
            logger.info("新增自选股 %s 本地无数据，自动触发更新", symbol)
            self._on_update_daily([symbol])
        except Exception:  # noqa: BLE001 - 自动触发失败不得影响交互
            logger.exception("自动更新新增自选股失败")

    # ------------------------------------------------------------------ #
    # 槽：自选股 / 周期
    # ------------------------------------------------------------------ #
    def _on_symbol_selected(self, symbol: str) -> None:
        """自选股选中：同时加载本地 K 线与分时并提示。"""
        logger.info("收到自选股选择：%s", symbol)
        self.statusBar().showMessage(f"已选择：{symbol}")
        self.kline_view.load(symbol, self.toolbar.current_period())
        self.tick_view.load(symbol)

    def _on_period_changed(self, code: str) -> None:
        """周期切换：已有标的则按新周期重载本地 K 线；切到 ``1m`` 时打开「分时」页。"""
        logger.info("周期变更：%s", code)
        symbol = self.kline_view.current_symbol()
        if symbol:
            self.kline_view.load(symbol, code)
        if code == "1m":
            self.right_tabs.setCurrentIndex(TAB_TICK_INDEX)

    def _open_indicator_editor(self) -> None:
        """打开「自定义指标编辑器」模态对话框，样本股候选取当前自选股（空则兜底 ``000001``）。"""
        symbols = self.watchlist_panel.symbols() or ["000001"]
        dialog = IndicatorEditorDialog(symbols=symbols, db=self._db, parent=self)
        dialog.exec()

    def _open_screener(self) -> None:
        """打开「选股」模态对话框，注入自选股面板（用于加自选）与存储实例。

        模态关闭后 ``deleteLater()``（D2）：避免每次开关泄漏一份对话框实例；
        与 :meth:`ScreenerDialog.closeEvent` 的 ``wait()`` 守卫配套使用，安全无误。
        """
        dialog = ScreenerDialog(
            db=self._db,
            watchlist_panel=self.watchlist_panel,
            parent=self,
        )
        dialog.exec()
        dialog.deleteLater()

    def _open_strategy_analysis(self) -> None:
        """打开「策略分析」模态对话框（P7-S2），注入存储实例与当前自选股（作为股票池）。

        股票池取自选股（空则用本地库全部有日线的标的）；回测只读本地库，不写数据。
        模态关闭后 ``deleteLater()``（与 :meth:`StrategyAnalysisDialog.closeEvent`
        的 ``wait()`` 守卫配套，避免泄漏或销毁运行中的线程）。
        """
        symbols = self.watchlist_panel.symbols() or None
        dialog = StrategyAnalysisDialog(db=self._db, symbols=symbols, parent=self)
        dialog.exec()
        dialog.deleteLater()

    def _open_factor_analysis(self) -> None:
        """打开「因子分析」模态对话框（P7-S3），注入存储实例。

        股票池固定为「本地库全部有前复权日线的标的（自动排除指数）」——横截面
        因子检验需要足够的截面宽度，自选股通常只有几只，做不出有效的 IC / 分层；
        因子分析只读本地库，不写数据（缺数据请先「数据 → 更新数据」）。
        模态关闭后 ``deleteLater()``（与 :meth:`FactorAnalysisDialog.closeEvent`
        的 ``wait()`` 守卫配套，避免泄漏或销毁运行中的线程）。
        """
        dialog = FactorAnalysisDialog(db=self._db, parent=self)
        dialog.exec()
        dialog.deleteLater()

    def _open_score_backtest(self) -> None:
        """打开「多因子打分选股」模态对话框（P7-S5），注入存储实例。

        股票池口径由对话框内切换（默认 PIT 成分股，消除幸存者偏差）；打分与回测
        只读本地库，不写数据（缺数据请先「数据 → 更新数据」，PIT 口径还需成分股快照）。
        模态关闭后 ``deleteLater()``（与 :meth:`ScoreBacktestDialog.closeEvent`
        的 ``wait()`` 守卫配套，避免泄漏或销毁运行中的线程）。
        """
        dialog = ScoreBacktestDialog(db=self._db, parent=self)
        dialog.exec()
        dialog.deleteLater()

    def _open_portfolio_backtest(self) -> None:
        """打开「多策略组合回测」模态对话框（P7-S7），注入存储实例。

        成员 = 「站上 N 日线」+「多因子打分」两条已有基线（资金分账 · 等权）；
        组合回测只读本地库，不写数据（缺数据请先「数据 → 更新数据」，PIT 口径还需
        成分股快照）。模态关闭后 ``deleteLater()``（与
        :meth:`PortfolioBacktestDialog.closeEvent` 的 ``wait()`` 守卫配套，
        避免泄漏或销毁运行中的线程）。
        """
        dialog = PortfolioBacktestDialog(db=self._db, parent=self)
        dialog.exec()
        dialog.deleteLater()

    def _open_member_selection(self) -> None:
        """打开「策略成员筛选」模态对话框（P7-S8），注入存储实例。

        候选池默认 15 个（均线网格 8 个 + 打分因子非空子集 7 个）；筛选与组合
        回测只读本地库，不写数据（缺数据请先「数据 → 更新数据」，PIT 口径还需
        成分股快照）。模态关闭后 ``deleteLater()``（与
        :meth:`MemberSelectionDialog.closeEvent` 的 ``wait()`` 守卫配套，
        避免泄漏或销毁运行中的线程）。
        """
        dialog = MemberSelectionDialog(db=self._db, parent=self)
        dialog.exec()
        dialog.deleteLater()

    def _open_quality_audit(self) -> None:
        """打开「数据质量审计」模态对话框（P7-S4），注入存储实例。

        标的池固定为「本地库全部有前复权日线的标的（自动排除指数）」——审计需要
        看全库质量，自选股通常只有几只；审计只读本地库，不写数据（缺数据请先
        「数据 → 更新数据」，缺成分股名单请先「数据 → 更新数据」的成分股拉取）。
        模态关闭后 ``deleteLater()``（与 :meth:`QualityAuditDialog.closeEvent`
        的 ``wait()`` 守卫配套，避免泄漏或销毁运行中的线程）。
        """
        dialog = QualityAuditDialog(db=self._db, parent=self)
        dialog.exec()
        dialog.deleteLater()

    def _open_note_detail(self, group_id: int, symbol: str) -> None:
        """打开「个股详情（备注）」弹窗（右键菜单 → ``detailRequested``）。

        基本面默认走主数据源实时抓取（后台线程加载，失败降级占位）；备注正文
        保存 / 清除时联动自选股列表 ★ 标记（保存非空 → ★，清除 → 取消 ★）。

        Args:
            group_id: 条目所在分组 id。
            symbol: 标的代码。
        """

        def _default_loader(sym: str) -> dict:
            source = get_source(str(get_config().get("data.source.primary", "baostock")))
            return source.get_fundamental(sym)

        dialog = NoteDetailDialog(
            symbol=str(symbol),
            db=self._db,
            fundamental_loader=_default_loader,
            parent=self,
        )
        dialog.note_saved.connect(
            lambda sym: self.watchlist_panel.set_note_mark(int(group_id), str(sym), True)
        )
        dialog.note_cleared.connect(
            lambda sym: self.watchlist_panel.set_note_mark(int(group_id), str(sym), False)
        )
        dialog.exec()
        dialog.deleteLater()

    def _open_sim_window(self) -> None:
        """打开「模拟训练」窗口（**非模态多窗口**：主界面可继续操作，可同时多开）。

        2026-10-02 用户优化①：由 ``exec()`` 改 ``show()`` —— 训练窗口不再阻塞主界面
        （可边回放边看行情 / 切换自选股），且每次触发都新建一个**独立回放会话**的窗口，
        多窗口互不干扰（各自账户 / 持仓 / 倍速）。窗口经 ``WA_DeleteOnClose`` 自销毁，
        ``destroyed`` 回调从 :attr:`_sim_windows` 移除引用，避免悬挂。
        """
        symbols = self.watchlist_panel.symbols() or []
        current = self.kline_view.current_symbol()
        dialog = SimWindow(
            db=self._db,
            symbols=symbols,
            default_symbol=current,
            parent=None,
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowFlag(Qt.WindowType.Window, True)
        dialog.destroyed.connect(lambda _obj=None, w=dialog: self._forget_sim_window(w))
        self._sim_windows.append(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _forget_sim_window(self, window: object) -> None:
        """从多窗口引用列表移除已销毁的模拟训练窗口（``destroyed`` 回调）。"""
        try:
            self._sim_windows.remove(window)  # type: ignore[arg-type]
        except ValueError:
            logger.debug("模拟训练窗口已不在引用列表：%r", window)

    # ------------------------------------------------------------------ #
    # P6 优化：股票代码与名称名录（查找 → 选中加载）
    # ------------------------------------------------------------------ #
    def _open_stock_list(self) -> None:
        """打开「股票代码与名称」名录对话框（**非模态单实例**：已打开则前置）。

        * 对话框只做本地名录的搜索 / 展示，**不联网、不自己开线程**；
        * 「更新名录」经 ``updateRequested`` 交回本窗口 → :meth:`_on_update_symbols`
          在后台线程执行（复用 :class:`_UpdateWorker`）；
        * 选中代码经 ``symbolChosen`` 回本窗口 → :meth:`_on_stock_chosen` 加载数据；
        * 关闭即销毁（``WA_DeleteOnClose``），``destroyed`` 回调置空引用。
        """
        existing = self._stock_list_dialog
        if existing is not None:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        dialog = StockListDialog(db=self._db, parent=self)
        dialog.updateRequested.connect(self._on_update_symbols)
        dialog.symbolChosen.connect(self._on_stock_chosen)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda _obj=None: self._forget_stock_list_dialog())
        self._stock_list_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _forget_stock_list_dialog(self) -> None:
        """名录对话框销毁后置空引用（``destroyed`` 回调）。"""
        self._stock_list_dialog = None

    def _on_stock_chosen(self, symbol: str) -> None:
        """名录对话框选定代码：加载其本地 K 线 / 分时；本地无 K 线时自动触发单股更新。

        「加载」复用自选股选中路径（:meth:`_on_symbol_selected`，只读本地库）；
        自动更新复用 :meth:`_on_update_daily`（与「新增自选股自动更新」同款：已有数据
        不重复拉，避免冗余网络请求）。
        """
        code = str(symbol).strip()
        if not code:
            return
        self._on_symbol_selected(code)
        try:
            if self._db.count_bars(code) <= 0:
                logger.info("名录选中 %s 本地无数据，自动触发更新", code)
                self._on_update_daily([code])
        except Exception:  # noqa: BLE001 - 自动触发失败不影响加载
            logger.exception("自动更新名录选中股票失败")

    def _open_alert_manager(self, group_id: int, symbol: str) -> None:
        """打开「预警管理」对话框（右键自选股 → 设置预警…）。"""
        dialog = AlertManagerDialog(symbol=str(symbol), db=self._db, parent=self)
        dialog.exec()
        dialog.deleteLater()

    def _open_shortcut_help(self) -> None:
        """打开「快捷键说明」对话框。"""
        dialog = ShortcutHelpDialog(
            shortcuts=self.shortcut_manager.shortcuts(), parent=self
        )
        dialog.exec()
        dialog.deleteLater()

    def _open_log_viewer(self) -> None:
        """打开「日志查看」对话框（P6-S3）。"""
        dialog = LogViewerDialog(parent=self)
        dialog.exec()
        dialog.deleteLater()

    # ------------------------------------------------------------------ #
    # P6-S1：一键备份 / 恢复
    # ------------------------------------------------------------------ #
    def _on_backup(self) -> None:
        """菜单「数据 → 备份数据」：一键备份（zip 写入 data/backups/，带时间戳）。"""
        try:
            saved = create_backup()
        except BackupError as exc:
            QMessageBox.warning(self, "备份失败", str(exc))
            return
        self.statusBar().showMessage(f"备份完成：{saved}", 10000)

    def _on_restore(self) -> None:
        """菜单「数据 → 恢复数据」：选择备份 zip 恢复（恢复前自动快照当前数据）。"""
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "选择备份文件",
            str(Path.home()),
            "Backup (*.zip)",
        )
        if not path:
            return
        answer = QMessageBox.question(
            self,
            "恢复数据",
            "恢复将用备份内容覆盖当前数据（恢复前会自动快照当前数据）。确认继续？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            snapshot = restore_backup(path)
        except BackupError as exc:
            QMessageBox.warning(self, "恢复失败", str(exc))
            return
        self.statusBar().showMessage(
            f"恢复完成，请重启程序生效（当前数据已快照至 {snapshot}）", 10000
        )

    # ------------------------------------------------------------------ #
    # P4-S4：K 线高清图导出
    # ------------------------------------------------------------------ #
    @staticmethod
    def _export_dpi() -> int:
        """读取配置 ``export.image_dpi``，非法 / 缺失回退 300。"""
        raw = get_config().get("export.image_dpi", 300)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 300

    def _on_export_kline(self) -> None:
        """菜单「导出 → 导出K线图」：抓取当前 K 线图并按 DPI 导出高清 PNG。

        无当前标的 / 无 K 线数据时友好提示；「选路径」（离屏阻塞的 ``QFileDialog``）
        与「写盘」分离，便于测试 monkeypatch 文件对话框与导出函数。
        """
        symbol = self.kline_view.current_symbol()
        if not symbol:
            self.statusBar().showMessage("请先选中一只股票并加载K线", 8000)
            return

        dpi = self._export_dpi()
        if self.kline_view.export_pixmap(dpi) is None:
            self.statusBar().showMessage("请先选中一只股票并加载K线", 8000)
            return

        period = self.kline_view.current_period()
        default_name = f"K线_{symbol}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "导出K线图",
            str(Path.home() / default_name),
            "PNG 图片 (*.png)",
        )
        if not path:
            return  # 用户取消，不提示、不写盘

        try:
            saved = export_kline_image(self.kline_view, path, dpi=dpi)
        except ImageExportError as exc:
            self.statusBar().showMessage(f"导出失败：{exc}", 10000)
            return
        self.statusBar().showMessage(f"已导出到 {saved}", 10000)

    # ------------------------------------------------------------------ #
    # P4-S2：托盘 / 预警通知 / 提示音
    # ------------------------------------------------------------------ #
    def _init_tray(self) -> None:
        """初始化系统托盘并保持引用（防 GC 导致图标消失）；失败降级为无托盘。"""
        self._tray: Optional[SystemTray] = None
        try:
            self._tray = SystemTray(self, parent=self)
            self._tray.quit_requested.connect(self._quit_application)
        except Exception:  # noqa: BLE001 - 托盘不可用不影响主流程
            logger.exception("初始化系统托盘失败，托盘功能不可用")
            self._tray = None

    def _quit_application(self) -> None:
        """真正退出（托盘「退出」/ 菜单「退出」走这里）：置位后退出事件循环。"""
        self._really_quit = True
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802 - Qt 约定
        """点 X 关闭 → 最小化到托盘；托盘不可用或真正退出时**真正关闭**。

        托盘不可用的两种情形：系统不支持系统托盘（``isSystemTrayAvailable()`` 为 False），
        或 :meth:`_init_tray` 初始化失败使 ``self._tray`` 为 ``None``。此时若仍
        ``ignore() + hide()``，窗口会被「藏丢」——进程仍在但既无托盘图标、也无任务栏入口，
        无法找回，故必须回退为 ``event.accept()`` 真退。
        """
        tray_unavailable = self._tray is None or not QSystemTrayIcon.isSystemTrayAvailable()
        if self._really_quit or tray_unavailable:
            event.accept()
            return
        event.ignore()
        self.hide()
        if self._tray is not None:
            try:
                self._tray.showMessage(
                    WINDOW_TITLE,
                    "已最小化到系统托盘",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000,
                )
            except Exception:  # noqa: BLE001
                logger.exception("托盘提示失败")

    def _on_alert_triggered(self, payload: dict) -> None:
        """预警触发：右下角非模态弹窗 + 提示音 + 托盘气泡。"""
        try:
            notification = AlertNotification(payload, anchor=self)
            notification.show()
        except Exception:  # noqa: BLE001 - 弹窗失败不影响提示音/托盘
            logger.exception("创建预警弹窗失败")

        self._play_alert_sound()

        if self._tray is not None:
            try:
                lines = build_alert_lines(payload)
                self._tray.showMessage(
                    lines["title"],
                    f"{lines['symbol']} {lines['kind_label']}：{lines['message']}",
                    QSystemTrayIcon.MessageIcon.Information,
                    5000,
                )
            except Exception:  # noqa: BLE001
                logger.exception("托盘预警气泡失败")

    def _play_alert_sound(self) -> None:
        """播放内置提示音；文件缺失 / 无音频设备一律静默降级，绝不抛异常。"""
        try:
            if not ALERT_WAV_PATH.exists():
                logger.warning("提示音文件缺失：%s", ALERT_WAV_PATH)
                return
            self._alert_sound = QSound(str(ALERT_WAV_PATH))
            self._alert_sound.play()
        except Exception:  # noqa: BLE001
            logger.exception("播放提示音失败（可能无音频设备），已静默降级")
