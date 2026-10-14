"""P7-S4 数据质量审计对话框测试（离屏 ``offscreen``）。

覆盖：
    1. 构造 + 默认参数（起始日 2015-01-01 / 指数 sh.000300 / 抽样 50 /
       概览表为空 / 六个明细标签页齐全）；
    2. ``current_spec()``：symbols / 指数 / 起止日期 / 抽样数透传；
       起止日期倒置 → ``None`` 且状态提示；
    3. ``run_audit()``（注入 fake 审计器）：概览 6 行、六项明细表填充、
       说明标签含行数、状态「审计完成」、审计器收到正确 spec；
    4. ``run_audit()``：``ok=False``（某项缺失）→ 该项状态「未完成」+ 未完成提示；
    5. ``run_audit()``：审计器抛异常 → 不崩溃 + 失败提示 + 视图清空；
    6. 纯函数（``format_cell`` / ``item_status`` / ``describe_result``）。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.engine.quality import (  # noqa: E402
    AUDIT_ITEM_LABELS,
    QualityAuditReport,
    QualityAuditSpec,
)
from app.engine.quality.checks import (  # noqa: E402
    AdjustResult,
    CalendarResult,
    CoverageResult,
    OhlcResult,
    OutlierResult,
    SurvivorshipResult,
)
from app.ui.widgets.quality_audit import (  # noqa: E402
    QualityAuditDialog,
    describe_result,
    format_cell,
    item_status,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
class FakeAuditor:
    """注入用 fake 审计器（返回固定结果或抛指定异常，并记录收到的 spec）。"""

    def __init__(self, report=None, error: Exception | None = None) -> None:
        self._report = report
        self._error = error
        self.calls: list = []

    def run(self, spec):
        self.calls.append(spec)
        if self._error is not None:
            raise self._error
        return self._report


def _coverage() -> CoverageResult:
    """合成覆盖度结果：2 只标的，1 只带停牌。"""
    return CoverageResult(
        ok=True,
        table=pd.DataFrame(
            [
                {
                    "symbol": "000001", "first_dt": "2015-01-05", "last_dt": "2026-10-09",
                    "bars": 2850, "expected": 2857, "missing": 7, "suspension_rate": 0.0025,
                },
                {
                    "symbol": "600000", "first_dt": "2015-01-05", "last_dt": "2026-10-09",
                    "bars": 2800, "expected": 2857, "missing": 57, "suspension_rate": 0.02,
                },
            ]
        ),
        summary={
            "n_symbols": 2,
            "total_expected": 5714,
            "total_missing": 64,
            "mean_suspension_rate": 0.0112,
            "median_suspension_rate": 0.0112,
            "worst_symbol": "600000",
            "worst_rate": 0.02,
        },
    )


def _ohlc() -> OhlcResult:
    """合成 OHLC 结果：汇总 2 行、问题样本空表。"""
    return OhlcResult(
        ok=True,
        table=pd.DataFrame(
            [
                {"adjust": "", "check": "high>=low", "count": 0},
                {"adjust": "qfq", "check": "high>=low", "count": 1},
            ]
        ),
        samples=pd.DataFrame(
            [
                {
                    "adjust": "qfq", "symbol": "000001", "dt": "2019-05-06",
                    "check": "high>=low", "detail": "high 38.20 < low 39.00",
                }
            ]
        ),
        summary={"total_errors": 1, "by_check": {"high>=low": 1}},
    )


def _calendar() -> CalendarResult:
    """合成日历结果：无野生日期。"""
    return CalendarResult(
        ok=True,
        table=pd.DataFrame(columns=["symbol", "dt", "reason"]),
        summary={
            "calendar_source": "sh.000300", "calendar_days": 2857,
            "calendar_first": "2015-01-05", "calendar_last": "2026-10-09", "wild_dates": 0,
        },
    )


def _outliers() -> OutlierResult:
    """合成异常波动结果：1 次命中。"""
    return OutlierResult(
        ok=False,
        table=pd.DataFrame(
            [
                {
                    "symbol": "600000", "dt": "2015-07-09", "ret": -0.31,
                    "prev_close": 16.2, "close": 11.18, "gap_days": 12,
                }
            ]
        ),
        summary={
            "threshold": 0.21, "evaluated_returns": 1_800_000, "flagged_count": 1,
            "flagged_symbols": 1, "flagged_ratio": 0.0000006,
        },
    )


def _adjust() -> AdjustResult:
    """合成复权对账结果：1 只标的自洽。"""
    return AdjustResult(
        ok=True,
        table=pd.DataFrame(
            [
                {
                    "symbol": "000001", "bars_raw": 2850, "bars_qfq": 2850,
                    "date_mismatch": 0, "volume_mismatch": 0, "dividend_events": 15,
                    "unsmoothed_events": 0, "conclusion": "自洽",
                }
            ]
        ),
        summary={
            "sampled": 1, "dividend_events": 15, "unsmoothed_events": 0,
            "inconsistent_symbols": [],
        },
    )


def _survivorship() -> SurvivorshipResult:
    """合成幸存者偏差结果：名单并集 3 只、缺席 2 只。"""
    return SurvivorshipResult(
        ok=True,
        missing_table=pd.DataFrame(
            [
                {"symbol": "000024", "name": "招商地产", "last_effective_date": "2015-06-12", "exit_year": 2015},
                {"symbol": "600005", "name": "武钢股份", "last_effective_date": "2016-03-11", "exit_year": 2016},
            ]
        ),
        year_table=pd.DataFrame(
            [
                {"year": 2015, "pit_members": 300, "present": 298, "missing": 2, "coverage_rate": 0.9933},
                {"year": 2016, "pit_members": 300, "present": 300, "missing": 0, "coverage_rate": 1.0},
            ]
        ),
        summary={
            "union_size": 3, "present_size": 1, "missing_size": 2,
            "missing_ratio": 0.6667, "current_members": 300,
        },
    )


def _report(ok: bool = True) -> QualityAuditReport:
    """合成完整审计报告（六项齐全）。"""
    return QualityAuditReport(
        ok=ok,
        error="" if ok else "以下审计项未能完成：异常波动",
        coverage=_coverage(),
        ohlc=_ohlc(),
        calendar=_calendar(),
        outliers=_outliers(),
        adjust=_adjust(),
        survivorship=_survivorship(),
        universe=["000001", "600000"],
        calendar_source="sh.000300",
        warnings=["示例警告：停牌 57 个交易日（600000）"],
    )


@pytest.fixture(scope="module")
def app():
    """模块级 QApplication（offscreen）。"""
    instance = QApplication.instance() or QApplication([])
    yield instance


# --------------------------------------------------------------------------- #
# 构造与配置
# --------------------------------------------------------------------------- #
def test_defaults_and_tabs(app):
    dialog = QualityAuditDialog()
    assert dialog.windowTitle() == "数据质量审计"
    start, end = dialog.current_dates()
    assert start == "2015-01-01"
    assert dialog.index_edit.text() == "sh.000300"
    assert dialog.sample_spin.value() == 50
    assert dialog.overview_table.rowCount() == 0
    # 六个明细标签页齐全，标题取自引擎的中文名映射。
    assert dialog.detail_tabs.count() == 6
    titles = [dialog.detail_tabs.tabText(i) for i in range(6)]
    assert titles == [AUDIT_ITEM_LABELS[name] for name in
                      ("coverage", "ohlc", "calendar", "outliers", "adjust", "survivorship")]
    dialog.deleteLater()


def test_current_spec_passthrough_and_invalid(app):
    dialog = QualityAuditDialog(symbols=["000001"])
    spec = dialog.current_spec()
    assert isinstance(spec, QualityAuditSpec)
    assert spec.symbols == ["000001"]
    assert spec.index_code == "sh.000300"
    assert spec.start == "2015-01-01"
    assert spec.sample_size == 50

    dialog.index_edit.setText("")           # 空指数 → 回退默认
    dialog.sample_spin.setValue(20)
    assert dialog.current_spec().index_code == "sh.000300"
    assert dialog.current_spec().sample_size == 20

    # 起止日期倒置 → None + 状态提示。
    dialog.start_date_edit.setDate(dialog.end_date_edit.date().addDays(1))
    assert dialog.current_spec() is None
    assert "起始日晚于结束日" in dialog.status_label.text()
    dialog.deleteLater()


# --------------------------------------------------------------------------- #
# 运行审计（注入 fake）
# --------------------------------------------------------------------------- #
def test_run_audit_fills_views(app):
    fake = FakeAuditor(_report())
    dialog = QualityAuditDialog(auditor=fake)
    report = dialog.run_audit()
    assert report is dialog.last_report()
    assert isinstance(fake.calls[0], QualityAuditSpec)

    # 概览：六行（状态 + 结论）。
    assert dialog.overview_table.rowCount() == 6
    assert dialog.overview_table.item(0, 0).text() == "覆盖度 / 停牌率"
    assert dialog.overview_table.item(0, 1).text() == "完成"
    assert "停牌率" in dialog.overview_table.item(0, 2).text()
    assert "缺席 2 只" in dialog.overview_table.item(5, 2).text()

    # 明细：覆盖度 2 行、OHLC 汇总 2 行 + 样本 1 行、日历空表、异常 1 行。
    assert dialog.detail_table("coverage").rowCount() == 2
    assert dialog.detail_table("ohlc", 0).rowCount() == 2
    assert dialog.detail_table("ohlc", 1).rowCount() == 1
    assert dialog.detail_table("calendar").rowCount() == 0
    assert dialog.detail_table("outliers").rowCount() == 1
    assert dialog.detail_table("adjust").rowCount() == 1
    assert dialog.detail_table("survivorship", 0).rowCount() == 2
    assert dialog.detail_table("survivorship", 1).rowCount() == 2

    # 表头中文化 + 比率列百分比化。
    assert dialog.detail_table("coverage").horizontalHeaderItem(0).text() == "标的"
    assert dialog.detail_table("survivorship", 0).horizontalHeaderItem(1).text() == "PIT 成分数"
    assert dialog.detail_table("coverage").item(0, 6).text() == "0.25%"

    # 状态与警告。
    assert "审计完成" in dialog.status_label.text()
    assert "示例警告" in dialog.warnings_label.text()
    dialog.deleteLater()


def test_run_audit_partial_result(app):
    report = _report(ok=False)
    report.outliers = None                       # 模拟单项失败
    dialog = QualityAuditDialog(auditor=FakeAuditor(report))
    dialog.run_audit()
    assert dialog.overview_table.item(3, 1).text() == "未完成"
    assert "该项未能完成" in dialog.overview_table.item(3, 2).text()
    assert dialog.detail_table("outliers").rowCount() == 0
    assert "审计未全部完成" in dialog.status_label.text()
    dialog.deleteLater()


def test_run_audit_engine_error(app):
    dialog = QualityAuditDialog(auditor=FakeAuditor(error=RuntimeError("库连接中断")))
    assert dialog.run_audit() is None
    assert dialog.last_report() is None
    assert dialog.overview_table.rowCount() == 0
    assert "库连接中断" in dialog.status_label.text()
    dialog.deleteLater()


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #
def test_format_cell():
    assert format_cell("suspension_rate", 0.0123) == "1.23%"
    assert format_cell("bars", 2850) == "2850"
    assert format_cell("ret", -0.31) == "-31.00%"
    assert format_cell("conclusion", "自洽") == "自洽"
    assert format_cell("prev_close", None) == "—"
    assert format_cell("bars", float("nan")) == "—"


def test_item_status_and_describe():
    assert item_status(None) == "未完成"
    assert item_status(_coverage()) == "完成"
    # 结论文案：缺失项给可读提示；正常项含关键指标。
    assert "未能完成" in describe_result("coverage", None)
    text = describe_result("coverage", _coverage())
    assert "600000" in text and "2.00%" in text
    assert "未发现 OHLC 逻辑错误" not in describe_result("ohlc", _ohlc())
    assert "1 处 OHLC 逻辑错误" in describe_result("ohlc", _ohlc())
    assert "野生日期" not in describe_result("calendar", _calendar())
    assert "命中 1 次异常收益" in describe_result("outliers", _outliers())
    assert "全部自洽" in describe_result("adjust", _adjust())
    assert "缺席 2 只" in describe_result("survivorship", _survivorship())