"""数据质量审计引擎（P7-S4）。

对外出口：
    * :class:`QualityAuditor` / :class:`QualityAuditSpec` / :class:`QualityAuditReport` /
      :data:`AUDIT_ITEMS` / :data:`AUDIT_ITEM_LABELS` / :data:`CALENDAR_INDEX_CANDIDATES`
      / :data:`DEFAULT_SAMPLE_SIZE`：编排入口与结果容器（:mod:`app.engine.quality.audit`）；
    * 六项检查（纯函数，可单独调用与单测，:mod:`app.engine.quality.checks`）：
      :func:`check_coverage`（覆盖度 / 停牌率）、:func:`check_ohlc`（OHLC 逻辑校验）、
      :func:`check_calendar`（日历一致性）、:func:`check_outliers`（异常波动）、
      :func:`check_adjust`（复权自洽对账）、:func:`check_survivorship`（幸存者偏差量化）；
    * 各结果容器与表契约列常量（如 :data:`COVERAGE_COLUMNS` / :data:`OUTLIER_COLUMNS`）。

六项检查回答的业务问题：
    1. **数据在不在**（覆盖度）：评估窗口内该有 N 天、实际只有 M 天，缺的哪去了？
    2. **数据对不对**（OHLC / 日历）：单根 bar 内部逻辑是否自洽、日期是否落在
       真实交易日历内；
    3. **数据怪不怪**（异常波动）：有没有超出物理涨跌停限制的收益（疑似复权错误）；
    4. **口径乱不乱**（复权自洽）：同一只股票的未复权 / 前复权两条序列是否自洽，
       前复权有没有真的把除权缺口抹平；
    5. **样本全不全**（幸存者偏差）：只用「今天的成分股」回溯，缺席了多少历史股票。

典型用法::

    from app.engine.quality import QualityAuditor, QualityAuditSpec

    report = QualityAuditor().run(QualityAuditSpec(start="2015-01-01"))
    print(report.survivorship.summary["missing_size"])  # 缺席股票数

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）。
"""
from __future__ import annotations

from app.engine.quality.audit import (
    AUDIT_ITEM_LABELS,
    AUDIT_ITEMS,
    CALENDAR_INDEX_CANDIDATES,
    DEFAULT_SAMPLE_SIZE,
    QualityAuditReport,
    QualityAuditor,
    QualityAuditSpec,
)
from app.engine.quality.checks import (
    ADJUST_COLUMNS,
    ADJUST_IDENTITY_TOL,
    ADJUST_STEP_TOL,
    CALENDAR_COLUMNS,
    CONCLUSION_DATE_MISMATCH,
    CONCLUSION_SELF_CONSISTENT,
    CONCLUSION_UNSMOOTHED,
    COVERAGE_COLUMNS,
    DEFAULT_LISTING_SKIP_BARS,
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SUSPENSION_GAP,
    MAX_SAMPLE_ROWS,
    OHLC_COLUMNS,
    OHLC_SAMPLE_COLUMNS,
    OHLC_TOLERANCE,
    OUTLIER_COLUMNS,
    SURVIVORSHIP_MISSING_COLUMNS,
    SURVIVORSHIP_YEAR_COLUMNS,
    AdjustResult,
    CalendarResult,
    CoverageResult,
    OhlcResult,
    OutlierResult,
    SurvivorshipResult,
    check_adjust,
    check_calendar,
    check_coverage,
    check_ohlc,
    check_outliers,
    check_survivorship,
)

__all__ = [
    # 编排
    "QualityAuditor",
    "QualityAuditSpec",
    "QualityAuditReport",
    "AUDIT_ITEMS",
    "AUDIT_ITEM_LABELS",
    "CALENDAR_INDEX_CANDIDATES",
    "DEFAULT_SAMPLE_SIZE",
    # 检查函数
    "check_coverage",
    "check_ohlc",
    "check_calendar",
    "check_outliers",
    "check_adjust",
    "check_survivorship",
    # 结果容器
    "CoverageResult",
    "OhlcResult",
    "CalendarResult",
    "OutlierResult",
    "AdjustResult",
    "SurvivorshipResult",
    # 表契约列
    "COVERAGE_COLUMNS",
    "OHLC_COLUMNS",
    "OHLC_SAMPLE_COLUMNS",
    "CALENDAR_COLUMNS",
    "OUTLIER_COLUMNS",
    "ADJUST_COLUMNS",
    "SURVIVORSHIP_MISSING_COLUMNS",
    "SURVIVORSHIP_YEAR_COLUMNS",
    # 阈值常量
    "OHLC_TOLERANCE",
    "DEFAULT_OUTLIER_THRESHOLD",
    "DEFAULT_SUSPENSION_GAP",
    "DEFAULT_LISTING_SKIP_BARS",
    "ADJUST_STEP_TOL",
    "ADJUST_IDENTITY_TOL",
    "MAX_SAMPLE_ROWS",
    "CONCLUSION_SELF_CONSISTENT",
    "CONCLUSION_DATE_MISMATCH",
    "CONCLUSION_UNSMOOTHED",
]