"""UI 控件子包。

对外出口：
    * :class:`WatchlistPanel`（左自选股栏）；
    * :class:`MainToolBar`（顶部工具栏）；
    * :class:`KLineView`（K 线主图 + 成交量副图 + P2-S4 辅助标注）；
    * :class:`TickView`（分时图独立页）；
    * :mod:`app.ui.widgets.kline_annotations` 的判定纯函数（涨跌停 / 缺口 / 除权）。
"""
from __future__ import annotations

from app.ui.widgets.alert_dialog import (
    AlertNotification,
    build_alert_lines,
    build_message,
    kind_label,
)
from app.ui.widgets.alert_manager import AlertManagerDialog
from app.ui.widgets.factor_analysis import FactorAnalysisDialog
from app.ui.widgets.indicator_editor import IndicatorEditorDialog
from app.ui.widgets.kline_annotations import (
    detect_ex_dividend,
    detect_gaps,
    detect_limit_events,
    limit_pct,
    limit_prices,
)
from app.ui.widgets.kline_view import KLineView
from app.ui.widgets.log_viewer import LogViewerDialog
from app.ui.widgets.member_selection import MemberSelectionDialog
from app.ui.widgets.note_editor import NoteDetailDialog, format_fundamental
from app.ui.widgets.portfolio_backtest import PortfolioBacktestDialog
from app.ui.widgets.quality_audit import QualityAuditDialog
from app.ui.widgets.score_backtest import ScoreBacktestDialog
from app.ui.widgets.screener_dialog import ScreenerDialog
from app.ui.widgets.shortcut_help import ShortcutHelpDialog
from app.ui.widgets.sim_window import SimWindow
from app.ui.widgets.stock_list_dialog import StockListDialog
from app.ui.widgets.strategy_analysis import StrategyAnalysisDialog
from app.ui.widgets.tick_view import TickView
from app.ui.widgets.toolbar import MainToolBar
from app.ui.widgets.tray import SystemTray
from app.ui.widgets.watchlist_panel import WatchlistPanel

__all__ = [
    "AlertNotification",
    "AlertManagerDialog",
    "SystemTray",
    "build_alert_lines",
    "build_message",
    "kind_label",
    "WatchlistPanel",
    "MainToolBar",
    "KLineView",
    "TickView",
    "IndicatorEditorDialog",
    "ScreenerDialog",
    "SimWindow",
    "StockListDialog",
    "StrategyAnalysisDialog",
    "FactorAnalysisDialog",
    "ScoreBacktestDialog",
    "PortfolioBacktestDialog",
    "MemberSelectionDialog",
    "QualityAuditDialog",
    "ShortcutHelpDialog",
    "LogViewerDialog",
    "NoteDetailDialog",
    "format_fundamental",
    "limit_pct",
    "limit_prices",
    "detect_limit_events",
    "detect_gaps",
    "detect_ex_dividend",
]
