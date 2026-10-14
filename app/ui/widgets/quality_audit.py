"""数据质量审计对话框（P7-S4）。

定位：**模态对话框**，把 P7-S4 数据质量审计引擎的产出落到界面：
    * 配置审计范围（起止日期 / 指数 / 复权对账抽样数）；
    * 运行审计（:class:`~app.engine.quality.audit.QualityAuditor`），后台线程执行；
    * 展示：审计概览（六项一行一句业务结论）+ 六项明细表 + 警告与状态。

架构约定（与 ``factor_analysis`` / ``strategy_analysis`` 同构）：
    * **业务动作 = 无弹窗的纯方法**（:meth:`current_spec` / :meth:`run_audit`），
      测试可直接调用并断言（注入 fake 审计器时无线程、无网络）；
    * **UI 触达层**（运行按钮）经 :class:`_QualityWorker`（``QThread``）后台执行，
      结果经 Qt 信号回 UI 线程刷新，**工作线程不碰任何 UI**；
    * **绝不向 UI 抛异常**：引擎 / 格式化异常一律捕获，状态标签显示可读原因；
    * 本对话框**只读**本地库（审计引擎不写库），不做任何数据更新。

展示口径（六项审计回答什么业务问题）：
    1. 覆盖度 / 停牌率——该有的交易日缺了多少、哪些标的停牌最多；
    2. OHLC 逻辑校验——单根 bar 内部（高 ≥ 开收 ≥ 低等）是否自洽；
    3. 日历一致性——有没有落在真实交易日历之外的「野生日期」；
    4. 异常波动——有没有超出物理涨跌停限制的收益（疑似复权错误）；
    5. 复权自洽对账——未复权与前复权两条序列是否对得上、除权缺口有没有抹平；
    6. 幸存者偏差量化——只用「今天的成分股」回溯，缺席了多少历史股票。
"""
from __future__ import annotations

import logging
import numbers
from typing import Any, Optional

import pandas as pd
from PySide6.QtCore import QDate, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
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
from app.data.universe import DEFAULT_INDEX_CODE
from app.engine.quality import (
    AUDIT_ITEM_LABELS,
    AUDIT_ITEMS,
    DEFAULT_SAMPLE_SIZE,
    QualityAuditor,
    QualityAuditReport,
    QualityAuditSpec,
)
from app.engine.quality.checks import (
    ADJUST_COLUMNS,
    CALENDAR_COLUMNS,
    COVERAGE_COLUMNS,
    OHLC_COLUMNS,
    OHLC_SAMPLE_COLUMNS,
    OUTLIER_COLUMNS,
    SURVIVORSHIP_MISSING_COLUMNS,
    SURVIVORSHIP_YEAR_COLUMNS,
)
from app.ui.themes import colors
from app.ui.widgets.strategy_analysis import format_count, format_num, format_pct

logger = logging.getLogger(__name__)

# 对话框标题。
WINDOW_TITLE: str = "数据质量审计"

# 概览表列头。
_OVERVIEW_HEADERS: tuple[str, ...] = ("审计项", "状态", "业务结论")

# 结果列 → 中文表头（引擎表契约列，UI 层负责中文化）。
_COLUMN_LABELS: dict[str, str] = {
    "symbol": "标的",
    "first_dt": "首日",
    "last_dt": "末日",
    "bars": "实际条数",
    "expected": "应有条数",
    "missing": "缺失",
    "suspension_rate": "停牌率",
    "adjust": "复权口径",
    "check": "检查项",
    "count": "次数",
    "dt": "日期",
    "detail": "详情",
    "reason": "原因",
    "ret": "收益",
    "prev_close": "前收",
    "close": "收盘",
    "gap_days": "间隔天数",
    "bars_raw": "未复权条数",
    "bars_qfq": "前复权条数",
    "date_mismatch": "日期不一致",
    "volume_mismatch": "成交量不一致",
    "dividend_events": "除权事件",
    "unsmoothed_events": "未抹平事件",
    "conclusion": "结论",
    "name": "名称",
    "last_effective_date": "最后生效日",
    "exit_year": "退出年份",
    "year": "年份",
    "pit_members": "PIT 成分数",
    "present": "本地可算数",
    "coverage_rate": "覆盖率",
}

# 比率列（展示为百分比）。
_PCT_COLUMNS: frozenset[str] = frozenset({"suspension_rate", "ret", "coverage_rate"})

# 计数列（展示为整数，不做定点补零）。
_COUNT_COLUMNS: frozenset[str] = frozenset(
    {
        "bars", "expected", "missing", "count", "gap_days", "bars_raw", "bars_qfq",
        "date_mismatch", "volume_mismatch", "dividend_events", "unsmoothed_events",
        "exit_year", "year", "pit_members", "present",
    }
)

# 六个标签页的表规格：{审计项: [(表格说明, 契约列)]}（按契约列固定表头与取值口径）。
_ITEM_TABLE_SPECS: dict[str, list[tuple[str, list[str]]]] = {
    "coverage": [("覆盖度明细（每标的一行；停牌率 = 缺失 / 应有）", COVERAGE_COLUMNS)],
    "ohlc": [
        ("检查项汇总（按复权口径分别统计）", OHLC_COLUMNS),
        (f"问题样本（最多 {200} 行）", OHLC_SAMPLE_COLUMNS),
    ],
    "calendar": [("野生日期样本（不在市场交易日历内的日期）", CALENDAR_COLUMNS)],
    "outliers": [("异常波动样本（按跌幅绝对值降序，最多 200 行）", OUTLIER_COLUMNS)],
    "adjust": [("复权自洽对账（每个抽样标的一行）", ADJUST_COLUMNS)],
    "survivorship": [
        ("逐年 PIT 覆盖率（名单成员里本地可算的比例）", SURVIVORSHIP_YEAR_COLUMNS),
        ("缺席标的清单（名单里有、本地日线没有）", SURVIVORSHIP_MISSING_COLUMNS),
    ],
}


# --------------------------------------------------------------------------- #
# 格式化纯函数（模块级，供测试直接对拍）
# --------------------------------------------------------------------------- #
def format_cell(column: str, value: Any) -> str:
    """结果单元格 → 展示文本（比率转百分比 / 计数转整数 / 数值定点 4 位）。"""
    if column in _PCT_COLUMNS:
        return format_pct(value)
    if column in _COUNT_COLUMNS:
        return format_count(value)
    if value is None:
        return "—"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return format_num(value, digits=4)
    return str(value)


def item_status(result: Any) -> str:
    """单项状态：结果缺失 → ``"未完成"``，否则 ``"完成"``。

    注意：「完成」不代表「没问题」——发现偏差本身就是审计的目的，
    问题写在业务结论列里。
    """
    return "未完成" if result is None else "完成"


def _summary_of(result: Any, key: str) -> Any:
    """安全取 summary 字段（缺失 / 非字典 → ``None``）。"""
    summary = getattr(result, "summary", None)
    if isinstance(summary, dict):
        return summary.get(key)
    return None


def describe_result(name: str, result: Any) -> str:
    """把单项结果编译成一句业务结论（缺失 / 异常时给可读文案，**不抛异常**）。"""
    if result is None:
        return "该项未能完成（详见警告）"
    try:
        if name == "coverage":
            return (
                f"标的 {format_count(_summary_of(result, 'n_symbols'))} 只；"
                f"整体停牌率 {format_pct(_summary_of(result, 'mean_suspension_rate'))}；"
                f"停牌最多 {_summary_of(result, 'worst_symbol') or '—'}"
                f"（{format_pct(_summary_of(result, 'worst_rate'))}）"
            )
        if name == "ohlc":
            total = _summary_of(result, "total_errors")
            return (
                "未发现 OHLC 逻辑错误"
                if not total
                else f"发现 {format_count(total)} 处 OHLC 逻辑错误（见明细）"
            )
        if name == "calendar":
            wild = _summary_of(result, "wild_dates")
            return (
                "所有日期均落在市场日历内"
                if not wild
                else f"发现 {format_count(wild)} 个野生日期（见明细）"
            )
        if name == "outliers":
            flagged = _summary_of(result, "flagged_count")
            return (
                f"阈值 {format_pct(_summary_of(result, 'threshold'))} 内无异常"
                if not flagged
                else f"命中 {format_count(flagged)} 次异常收益"
                f"（{format_count(_summary_of(result, 'flagged_symbols'))} 只标的，"
                f"占评估样本 {format_pct(_summary_of(result, 'flagged_ratio'))}）"
            )
        if name == "adjust":
            events = _summary_of(result, "unsmoothed_events")
            inconsistent = _summary_of(result, "inconsistent_symbols")
            return (
                f"抽样 {format_count(_summary_of(result, 'sampled'))} 只全部自洽"
                if not events and not inconsistent
                else f"抽样 {format_count(_summary_of(result, 'sampled'))} 只："
                f"未抹平除权 {format_count(events)} 处 / 不一致 "
                f"{format_count(len(inconsistent) if isinstance(inconsistent, (list, tuple)) else inconsistent)} 只"
            )
        if name == "survivorship":
            return (
                f"成分股名单并集 {format_count(_summary_of(result, 'union_size'))} 只；"
                f"本地可算 {format_count(_summary_of(result, 'present_size'))} 只；"
                f"缺席 {format_count(_summary_of(result, 'missing_size'))} 只"
                f"（{format_pct(_summary_of(result, 'missing_ratio'))}）"
            )
    except Exception:  # noqa: BLE001 - 结论文案生成失败不影响表格展示
        logger.exception("生成审计结论失败：%s", name)
        return "（结论生成失败，见明细表）"
    return ""


# --------------------------------------------------------------------------- #
# 后台线程
# --------------------------------------------------------------------------- #
class _QualityWorker(QThread):
    """在后台线程执行一次 :meth:`QualityAuditor.run`（结果经信号回传，不碰 UI）。"""

    done = Signal(object)  # QualityAuditReport 或 Exception

    def __init__(
        self,
        auditor: QualityAuditor,
        spec: QualityAuditSpec,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造后台审计线程（parent 默认 None，生命周期由 closeEvent 管理）。"""
        super().__init__(parent)
        self._auditor = auditor
        self._spec = spec

    def run(self) -> None:  # noqa: D401 - QThread 约定入口
        """线程体：执行审计，成功 / 失败均经 ``done`` 信号回传。"""
        try:
            report = self._auditor.run(self._spec)
        except Exception as exc:  # noqa: BLE001 - 兜底，绝不抛出到 Qt 事件循环
            logger.exception("后台数据质量审计线程异常")
            self.done.emit(exc)
            return
        self.done.emit(report)


# --------------------------------------------------------------------------- #
# 对话框
# --------------------------------------------------------------------------- #
class QualityAuditDialog(QDialog):
    """数据质量审计对话框（配置 → 审计 → 概览 + 六项明细）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        auditor: Optional[QualityAuditor] = None,
        symbols: Optional[list[str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """构造对话框。

        Args:
            db: 存储实例；默认 :func:`get_db`（只读）。
            auditor: 审计器；默认用 ``db`` 构造 :class:`QualityAuditor`。
            symbols: 审计标的；``None`` = 本地库全部有前复权日线的标的（排除指数）。
            parent: 父控件。
        """
        super().__init__(parent)
        self._db: Database = db if db is not None else get_db()
        self._auditor: QualityAuditor = (
            auditor if auditor is not None else QualityAuditor(db=self._db)
        )
        self._symbols: Optional[list[str]] = list(symbols) if symbols else None
        self._last_report: Optional[QualityAuditReport] = None
        self._worker: Optional[_QualityWorker] = None
        # {审计项: [(说明标签, 说明文本, 表格, 契约列)]}
        self._item_tables: dict[str, list[tuple[QLabel, str, QTableWidget, list[str]]]] = {}

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1120, 820)
        self._setup_ui()
        self._set_status("设置审计范围后点击『运行审计』开始（只读本地库，不修改数据）", ok=True)

    # ------------------------------------------------------------------ #
    # 线程生命周期（与 FactorAnalysisDialog 同约定）
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭前等待后台审计线程自然结束，避免析构连带销毁运行中的 QThread。"""
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
        """搭建「审计配置 / 概览 / 明细标签页 / 警告 / 状态」五段式布局。"""
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        root.addWidget(self._build_config_box())
        root.addWidget(self._build_overview_box())
        root.addWidget(self._build_detail_tabs(), 1)

        self.warnings_label = QLabel("", self)
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setVisible(False)
        root.addWidget(self.warnings_label)

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    def _build_config_box(self) -> QGroupBox:
        """审计配置区：起止日期 / 指数 / 抽样数 / 运行。"""
        box = QGroupBox("审计配置", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        hint = QLabel(
            "审计六项：覆盖度 / 停牌率、OHLC 逻辑校验、日历一致性、异常波动（>±21%）、"
            "复权自洽对账（抽样）、幸存者偏差量化（指数成分股名单 vs 本地可算）。"
            "审计只读本地库；缺失的股票请先「数据 → 更新数据」补齐后再审计。",
            box,
        )
        hint.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("起始日：", box))
        self.start_date_edit = QDateEdit(QDate(2015, 1, 1), box)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        row.addWidget(self.start_date_edit)
        row.addWidget(QLabel("结束日：", box))
        self.end_date_edit = QDateEdit(QDate.currentDate(), box)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        row.addWidget(self.end_date_edit)
        row.addWidget(QLabel("指数：", box))
        self.index_edit = QLineEdit(DEFAULT_INDEX_CODE, box)
        self.index_edit.setMaximumWidth(120)
        self.index_edit.setToolTip("用于幸存者偏差量化的指数成分股名单（库内已有快照的指数）")
        row.addWidget(self.index_edit)
        row.addWidget(QLabel("复权对账抽样标的数：", box))
        self.sample_spin = QSpinBox(box)
        self.sample_spin.setRange(1, 500)
        self.sample_spin.setValue(DEFAULT_SAMPLE_SIZE)
        row.addWidget(self.sample_spin)
        self.run_button = QPushButton("运行审计", box)
        self.run_button.clicked.connect(self._on_run)
        row.addWidget(self.run_button)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _build_overview_box(self) -> QGroupBox:
        """审计概览：六项一行（状态 + 一句业务结论）。"""
        box = QGroupBox("审计概览", self)
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        self.overview_label = QLabel("尚未运行审计", box)
        self.overview_label.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
        self.overview_label.setWordWrap(True)
        layout.addWidget(self.overview_label)
        self.overview_table = self._make_table(list(_OVERVIEW_HEADERS), box)
        layout.addWidget(self.overview_table)
        return box

    def _build_detail_tabs(self) -> QTabWidget:
        """明细标签页：六项审计各一页（按表规格生成，含说明标签）。"""
        tabs = QTabWidget(self)
        for name in AUDIT_ITEMS:
            tab = QWidget(tabs)
            layout = QVBoxLayout(tab)
            layout.setContentsMargins(0, 0, 0, 0)
            entries: list[tuple[QLabel, str, QTableWidget, list[str]]] = []
            for caption_text, columns in _ITEM_TABLE_SPECS.get(name, []):
                caption = QLabel(caption_text, tab)
                caption.setStyleSheet(f"color: {colors.SUBTEXT_COLOR};")
                layout.addWidget(caption)
                table = self._make_table(
                    [_COLUMN_LABELS.get(column, column) for column in columns], tab
                )
                layout.addWidget(table, 1)
                entries.append((caption, caption_text, table, list(columns)))
            self._item_tables[name] = entries
            tabs.addTab(tab, AUDIT_ITEM_LABELS.get(name, name))
        self.detail_tabs = tabs
        return tabs

    @staticmethod
    def _make_table(headers: list[str], parent: QWidget) -> QTableWidget:
        """构造统一风格的只读表格（首列自适应、其余拉伸）。"""
        table = QTableWidget(0, max(1, len(headers)), parent)
        table.setHorizontalHeaderLabels(headers or ["无数据"])
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
    def current_dates(self) -> tuple[str, str]:
        """返回起止日期（``YYYY-MM-DD``）。"""
        return (
            self.start_date_edit.date().toString("yyyy-MM-dd"),
            self.end_date_edit.date().toString("yyyy-MM-dd"),
        )

    def last_report(self) -> Optional[QualityAuditReport]:
        """返回最近一次审计结果；尚未运行返回 ``None``。"""
        return self._last_report

    def detail_table(self, name: str, index: int = 0) -> Optional[QTableWidget]:
        """取某项审计的第 ``index`` 张明细表（供测试 / 上层；越界返回 ``None``）。"""
        entries = self._item_tables.get(name, [])
        if 0 <= index < len(entries):
            return entries[index][2]
        return None

    # ------------------------------------------------------------------ #
    # 规格构建（无弹窗纯方法）
    # ------------------------------------------------------------------ #
    def current_spec(self) -> Optional[QualityAuditSpec]:
        """用当前界面参数构建 :class:`QualityAuditSpec`；起止日期倒置返回 ``None``。"""
        start, end = self.current_dates()
        if start > end:
            self._set_status(f"起始日晚于结束日：{start} > {end}", ok=False)
            return None
        return QualityAuditSpec(
            symbols=self._symbols,
            index_code=self.index_edit.text().strip() or DEFAULT_INDEX_CODE,
            start=start,
            end=end,
            sample_size=int(self.sample_spin.value()),
        )

    # ------------------------------------------------------------------ #
    # 运行审计（同步纯方法 + 后台线程触达层）
    # ------------------------------------------------------------------ #
    def run_audit(self) -> Optional[QualityAuditReport]:
        """同步执行审计并把结果刷新到概览 / 明细 / 状态标签（**不抛异常**）。

        供测试直接调用（注入 fake 审计器时无线程、无网络）。
        """
        spec = self.current_spec()
        if spec is None:
            return None
        try:
            report = self._auditor.run(spec)
        except Exception as exc:  # noqa: BLE001 - 引擎异常不外抛
            logger.exception("数据质量审计执行失败")
            self._last_report = None
            self._clear_result_views()
            self._set_status(f"审计失败：{exc}", ok=False)
            return None
        self._last_report = report
        self._apply_result(report)
        return report

    def _on_run(self) -> None:
        """「运行审计」按钮：在后台线程执行，避免全库装载阻塞 UI。"""
        spec = self.current_spec()
        if spec is None:
            return
        if self._worker is not None and self._worker.isRunning():
            logger.info("数据质量审计已在进行中，忽略重复触发")
            return
        self.run_button.setEnabled(False)
        self._set_status("正在审计…（全库逐标的装载校验可能需要一些时间）", ok=True)
        # worker 的 parent 置为 None：对话框析构不连带销毁运行中的 QThread，
        # 生命周期统一交由 closeEvent 的 wait() + _on_worker_finished 管理。
        self._worker = _QualityWorker(self._auditor, spec)
        self._worker.done.connect(self._on_audit_finished)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_audit_finished(self, payload: object) -> None:
        """后台审计结束：恢复按钮并把结果刷新到 UI（本槽运行在 UI 线程）。"""
        self.run_button.setEnabled(True)
        if isinstance(payload, QualityAuditReport):
            self._last_report = payload
            self._apply_result(payload)
        else:
            self._last_report = None
            self._clear_result_views()
            self._set_status(f"审计失败：{payload}", ok=False)

    def _on_worker_finished(self) -> None:
        """线程结束后释放引用。"""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------ #
    # 结果刷新
    # ------------------------------------------------------------------ #
    def _apply_result(self, report: QualityAuditReport) -> None:
        """把审计结果填充到概览 / 明细表 / 状态标签（UI 线程调用）。"""
        self._fill_overview(report)
        self._fill_items(report)
        self._show_warnings(getattr(report, "warnings", None))
        if getattr(report, "ok", False):
            self._set_status(
                f"审计完成：标的 {len(report.universe)} 只 / "
                f"日历来源 {report.calendar_source or '—'}"
                "（「完成」指六项检查都已执行；发现的问题见各项结论）",
                ok=True,
            )
        else:
            self._set_status(
                f"审计未全部完成：{getattr(report, 'error', '') or '未知原因'}"
                "（已完成项仍可查看）",
                ok=False,
            )

    def _fill_overview(self, report: QualityAuditReport) -> None:
        """概览表：六项一行（状态 + 业务结论）。"""
        rows: list[list[str]] = []
        for name in AUDIT_ITEMS:
            result = getattr(report, name, None)
            rows.append(
                [
                    AUDIT_ITEM_LABELS.get(name, name),
                    item_status(result),
                    describe_result(name, result),
                ]
            )
        self._fill_table(self.overview_table, rows)
        self.overview_label.setText(
            f"审计标的 {len(report.universe)} 只｜数据窗口 "
            f"{self.start_date_edit.date().toString('yyyy-MM-dd')} ~ "
            f"{self.end_date_edit.date().toString('yyyy-MM-dd')}｜"
            f"日历来源 {report.calendar_source or '—'}"
        )

    def _fill_items(self, report: QualityAuditReport) -> None:
        """六项明细表 + 说明标签（未完成项显示「未完成」并清空）。"""
        for name in AUDIT_ITEMS:
            result = getattr(report, name, None)
            entries = self._item_tables.get(name, [])
            if result is None:
                for caption, base, table, _columns in entries:
                    caption.setText(f"{base}（未完成）")
                    table.setRowCount(0)
                continue
            frames = self._frames_for(name, result)
            for (caption, base, table, columns), frame in zip(entries, frames):
                self._fill_frame(table, frame, columns)
                count = len(frame) if isinstance(frame, pd.DataFrame) else 0
                caption.setText(f"{base}；共 {count} 行")
                if isinstance(frame, pd.DataFrame) and frame.empty:
                    caption.setText(f"{base}；无记录")

    @staticmethod
    def _frames_for(name: str, result: Any) -> list[pd.DataFrame]:
        """单项结果 → 与表规格一一对应的 DataFrame 列表。"""
        if name == "ohlc":
            return [getattr(result, "table", pd.DataFrame()), getattr(result, "samples", pd.DataFrame())]
        if name == "survivorship":
            return [
                getattr(result, "year_table", pd.DataFrame()),
                getattr(result, "missing_table", pd.DataFrame()),
            ]
        return [getattr(result, "table", pd.DataFrame())]

    def _fill_frame(self, table: QTableWidget, frame: Any, columns: list[str]) -> None:
        """按契约列填充表格（结果非 DataFrame / 缺列时安全降级为 ``"—"``）。"""
        rows: list[list[str]] = []
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for _, row in frame.iterrows():
                rows.append([format_cell(column, row.get(column)) for column in columns])
        self._fill_table(table, rows)

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
        """清空概览 / 全部明细表 / 说明标签 / 警告。"""
        self.overview_table.setRowCount(0)
        self.overview_label.setText("尚未运行审计")
        for entries in self._item_tables.values():
            for caption, base, table, _columns in entries:
                caption.setText(base)
                table.setRowCount(0)
        self.warnings_label.setText("")
        self.warnings_label.setVisible(False)

    # ------------------------------------------------------------------ #
    # 警告 / 状态
    # ------------------------------------------------------------------ #
    def _show_warnings(self, warnings: Any) -> None:
        """展示审计警告（无警告则隐藏）。"""
        items = [str(item) for item in (warnings or [])]
        if not items:
            self.warnings_label.setText("")
            self.warnings_label.setVisible(False)
            return
        self.warnings_label.setStyleSheet(f"color: {colors.LIMIT_COLOR};")
        self.warnings_label.setText("审计警告：" + "；".join(items))
        self.warnings_label.setVisible(True)

    def _set_status(self, text: str, ok: bool = True) -> None:
        """更新状态标签（成功绿色 / 失败红色，语义为状态）。"""
        color = colors.DOWN_COLOR if ok else colors.UP_COLOR
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)


__all__ = [
    "QualityAuditDialog",
    "WINDOW_TITLE",
    "format_cell",
    "item_status",
    "describe_result",
]