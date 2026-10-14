"""P7-S4 数据质量审计引擎单测。

覆盖六项检查的行为对拍与编排层 E2E：
    1. ``check_coverage``     —— 覆盖度 / 停牌率（窗口裁剪、上市晚、停牌缺口）；
    2. ``check_ohlc``         —— OHLC 逻辑校验（六类错误、容差边界）；
    3. ``check_calendar``     —— 日历一致性（野生日期）；
    4. ``check_outliers``     —— 异常波动（上市初期豁免、停牌填充 bar 不评估、
       复牌跳空豁免、阈值）；
    5. ``check_adjust``       —— 复权自洽对账（比值跳变判定 / 未抹平 / 日期集不一致）；
    6. ``check_survivorship`` —— 幸存者偏差量化（并集、缺席清单、逐年覆盖率）；
    7. ``QualityAuditor``     —— 编排（空库降级、六项齐全、单项失败不中断、
       抽样确定性、TypeError）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.models import IndexConstituent
from app.data.store.db import Database
from app.engine.quality import (
    ADJUST_COLUMNS,
    CALENDAR_COLUMNS,
    CONCLUSION_DATE_MISMATCH,
    CONCLUSION_SELF_CONSISTENT,
    CONCLUSION_UNSMOOTHED,
    COVERAGE_COLUMNS,
    OHLC_SAMPLE_COLUMNS,
    OHLC_TOLERANCE,
    OUTLIER_COLUMNS,
    SURVIVORSHIP_MISSING_COLUMNS,
    SURVIVORSHIP_YEAR_COLUMNS,
    QualityAuditor,
    QualityAuditSpec,
    check_adjust,
    check_calendar,
    check_coverage,
    check_ohlc,
    check_outliers,
    check_survivorship,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _bars(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    """``[(日期, o, h, l, c, volume)]`` → bars DataFrame（amount = close × volume）。"""
    return pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(day),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume,
                "amount": c * volume,
            }
            for day, o, h, l, c, volume in rows
        ]
    )


def _days(count: int, start: str = "2024-01-01") -> list[pd.Timestamp]:
    """生成 count 个「交易日」（工作日口径，与真实日历同构）。"""
    return list(pd.bdate_range(start, periods=count))


def _flat(days: list[pd.Timestamp], price: float = 10.0, volume: float = 1000.0) -> pd.DataFrame:
    """价格恒定的 bars（无异常收益）。"""
    return _bars([(d.strftime("%Y-%m-%d"), price, price, price, price, volume) for d in days])


def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


# --------------------------------------------------------------------------- #
# 1) check_coverage
# --------------------------------------------------------------------------- #
def test_coverage_basic_hand_calc() -> None:
    """全勤：expected = 区间交易日数，missing = 0，停牌率 0。"""
    calendar = _days(10)
    bars = {"000001": _flat(calendar)}
    result = check_coverage(bars, calendar)

    assert result.ok
    row = result.table.iloc[0]
    assert row["symbol"] == "000001"
    assert int(row["expected"]) == 10
    assert int(row["bars"]) == 10
    assert int(row["missing"]) == 0
    assert float(row["suspension_rate"]) == 0.0


def test_coverage_counts_suspension_gap() -> None:
    """停牌 3 天（日历有、bar 缺）⇒ missing = 3、停牌率 = 30%。"""
    calendar = _days(10)
    bars = {"000001": _flat([d for i, d in enumerate(calendar) if i not in (2, 3, 4)])}
    result = check_coverage(bars, calendar)

    row = result.table.iloc[0]
    assert int(row["bars"]) == 7
    assert int(row["missing"]) == 3
    assert float(row["suspension_rate"]) == pytest.approx(0.3)


def test_coverage_late_listing_not_counted_as_suspension() -> None:
    """上市晚于窗口起点：只统计自身首末区间（未上市 ≠ 停牌）。"""
    calendar = _days(20)
    bars = {"000001": _flat(calendar[10:])}  # 第 11 个交易日起才有数据
    result = check_coverage(bars, calendar)

    row = result.table.iloc[0]
    assert int(row["expected"]) == 10
    assert int(row["missing"]) == 0
    assert row["first_dt"] == calendar[10].strftime("%Y-%m-%d")


def test_coverage_window_clip() -> None:
    """评估窗口只裁剪统计区间，不影响标的自身覆盖判断。"""
    calendar = _days(20)
    bars = {"000001": _flat(calendar)}
    result = check_coverage(bars, calendar, start=calendar[5], end=calendar[9])

    row = result.table.iloc[0]
    assert int(row["expected"]) == 5
    assert int(row["bars"]) == 5


def test_coverage_empty_calendar_degrades() -> None:
    """空日历：ok=False + 警告（不抛异常）。"""
    result = check_coverage({"000001": _flat(_days(3))}, [])
    assert result.ok is False
    assert result.warnings


def test_coverage_no_overlap_skipped() -> None:
    """标的与窗口无交集 ⇒ 跳过并记警告。"""
    calendar = _days(10)
    bars = {"000001": _flat(calendar[:3])}
    result = check_coverage(bars, calendar, start=calendar[5])
    assert result.ok is False
    assert any("无交集" in message for message in result.warnings)


def test_coverage_summary_weighted_rate() -> None:
    """汇总口径：加权平均停牌率 = 总缺失 / 总应有；给出最差标的。"""
    calendar = _days(10)
    full = _flat(calendar)
    # 000002 中段停牌 5 天（首末都在，属「区间内缺失」而非「数据截止」）。
    gapped = _flat([d for i, d in enumerate(calendar) if i not in (2, 3, 4, 5, 6)])
    result = check_coverage({"000001": full, "000002": gapped}, calendar)

    assert int(result.summary["total_expected"]) == 20
    assert int(result.summary["total_missing"]) == 5
    assert float(result.summary["mean_suspension_rate"]) == pytest.approx(0.25)
    assert result.summary["worst_symbol"] == "000002"


def test_coverage_table_contract() -> None:
    """结果表契约列固定。"""
    result = check_coverage({"000001": _flat(_days(5))}, _days(5))
    assert list(result.table.columns) == COVERAGE_COLUMNS


# --------------------------------------------------------------------------- #
# 2) check_ohlc
# --------------------------------------------------------------------------- #
def test_ohlc_clean_passes() -> None:
    """合法数据：无任何错误，ok=True。"""
    bars = {"000001": _bars([("2024-01-01", 10.0, 10.5, 9.8, 10.2, 1000.0)])}
    result = check_ohlc(bars, adjust="qfq")
    assert result.ok
    assert result.summary["total_errors"] == 0
    assert list(result.table.columns) == ["adjust", "check", "count"]


def test_ohlc_high_lt_low_detected() -> None:
    """high < low 被检出。"""
    bars = {"000001": _bars([("2024-01-01", 10.0, 9.0, 10.5, 10.2, 1000.0)])}
    result = check_ohlc(bars)
    assert not result.ok
    counts = dict(zip(result.table["check"], result.table["count"]))
    assert counts["high_lt_low"] == 1


def test_ohlc_high_lt_body_detected() -> None:
    """high < max(open, close) 被检出。"""
    bars = {"000001": _bars([("2024-01-01", 10.0, 10.1, 9.9, 10.6, 1000.0)])}
    result = check_ohlc(bars)
    counts = dict(zip(result.table["check"], result.table["count"]))
    assert counts["high_lt_body"] == 1


def test_ohlc_low_gt_body_detected() -> None:
    """low > min(open, close) 被检出。"""
    bars = {"000001": _bars([("2024-01-01", 10.0, 10.6, 10.1, 9.9, 1000.0)])}
    result = check_ohlc(bars)
    counts = dict(zip(result.table["check"], result.table["count"]))
    assert counts["low_gt_body"] == 1


def test_ohlc_nonpositive_and_nan_price() -> None:
    """非正价格与缺失价格分别计数。"""
    bars = {
        "000001": _bars([("2024-01-01", 0.0, 10.5, 9.8, 10.2, 1000.0)]),
        "000002": _bars([("2024-01-01", 10.0, 10.5, 9.8, float("nan"), 1000.0)]),
    }
    result = check_ohlc(bars)
    counts = dict(zip(result.table["check"], result.table["count"]))
    assert counts["nonpositive_price"] == 1
    assert counts["nan_price"] == 1


def test_ohlc_negative_flow_detected() -> None:
    """负成交量被检出（amount 为负同样命中）。"""
    bars = {"000001": _bars([("2024-01-01", 10.0, 10.5, 9.8, 10.2, -1.0)])}
    result = check_ohlc(bars)
    counts = dict(zip(result.table["check"], result.table["count"]))
    assert counts["negative_flow"] == 1


def test_ohlc_tolerance_boundary() -> None:
    """复权舍入的微小越界（< 容差）不判定为错误。"""
    tiny = 10.5 * (1.0 - OHLC_TOLERANCE / 2)
    bars = {"000001": _bars([("2024-01-01", 10.0, tiny, 9.8, 10.5, 1000.0)])}
    result = check_ohlc(bars)
    assert result.ok


def test_ohlc_samples_contract() -> None:
    """问题样本表契约列与内容。"""
    bars = {"000001": _bars([("2024-01-02", 10.0, 9.0, 10.5, 10.2, 1000.0)])}
    result = check_ohlc(bars, adjust="qfq")
    assert list(result.samples.columns) == OHLC_SAMPLE_COLUMNS
    assert result.samples.iloc[0]["symbol"] == "000001"
    assert result.samples.iloc[0]["dt"] == "2024-01-02"


# --------------------------------------------------------------------------- #
# 3) check_calendar
# --------------------------------------------------------------------------- #
def test_calendar_all_in_calendar() -> None:
    """所有日期都落在市场日历内 ⇒ ok=True。"""
    calendar = _days(5)
    result = check_calendar({"000001": _flat(calendar)}, calendar, calendar_source="sh.000300")
    assert result.ok
    assert result.summary["calendar_days"] == 5
    assert result.summary["calendar_source"] == "sh.000300"


def test_calendar_wild_date_detected() -> None:
    """不在日历内的日期（周末 / 脏数据）被记为野生日期。"""
    calendar = _days(5)
    bars = {"000001": _flat([*calendar, pd.Timestamp("2024-01-06")])}  # 周六
    result = check_calendar(bars, calendar)
    assert not result.ok
    assert result.summary["wild_dates"] == 1
    assert list(result.table.columns) == CALENDAR_COLUMNS
    assert any("野生日期" in message for message in result.warnings)


def test_calendar_empty_calendar_degrades() -> None:
    """空日历 ⇒ ok=False + 警告。"""
    result = check_calendar({"000001": _flat(_days(3))}, [])
    assert result.ok is False


# --------------------------------------------------------------------------- #
# 4) check_outliers
# --------------------------------------------------------------------------- #
def test_outliers_normal_moves_not_flagged() -> None:
    """涨跌幅在限制内的正常波动不命中。"""
    days = _days(5)
    bars = {"000001": _bars(
        [(day.strftime("%Y-%m-%d"), 10.0, 10.5, 9.8, 10.0 * (1 + 0.03 * i), 1000.0)
         for i, day in enumerate(days)]
    )}
    result = check_outliers(bars, days, listing_skip=0)
    assert result.ok
    assert result.summary["evaluated_returns"] == 4  # 5 根 bar ⇒ 4 个收益
    assert result.summary["flagged_count"] == 0


def test_outliers_big_jump_flagged() -> None:
    """次日 +50% 跳空（无停牌）⇒ 命中，且字段口径正确。"""
    days = _days(3)
    bars = {"000001": _bars(
        [("2024-01-01", 10.0, 10.0, 10.0, 10.0, 1000.0),
         ("2024-01-02", 15.0, 15.0, 15.0, 15.0, 1000.0),
         ("2024-01-03", 15.0, 15.0, 15.0, 15.0, 1000.0)]
    )}
    result = check_outliers(bars, days, listing_skip=0)
    assert not result.ok
    assert result.summary["flagged_count"] == 1
    row = result.table.iloc[0]
    assert row["symbol"] == "000001"
    assert row["dt"] == "2024-01-02"
    assert float(row["ret"]) == pytest.approx(0.5)
    assert float(row["prev_close"]) == pytest.approx(10.0)
    assert list(result.table.columns) == OUTLIER_COLUMNS


def test_outliers_resume_gap_exempt() -> None:
    """停牌复牌（间隔 > 阈值）的跳空豁免；同样跳空若间隔正常则命中。"""
    days = _days(15)
    gapped = {"000001": _bars(
        [(days[0].strftime("%Y-%m-%d"), 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[11].strftime("%Y-%m-%d"), 15.0, 15.0, 15.0, 15.0, 1000.0)]
    )}
    result_gap = check_outliers(gapped, days, listing_skip=0)
    assert result_gap.ok  # 复牌跳空被豁免

    normal = {"000001": _bars(
        [(days[0].strftime("%Y-%m-%d"), 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[1].strftime("%Y-%m-%d"), 15.0, 15.0, 15.0, 15.0, 1000.0)]
    )}
    result_normal = check_outliers(normal, days, listing_skip=0)
    assert not result_normal.ok


def test_outliers_first_bar_no_return() -> None:
    """标的自身第一根 bar 不产生收益（上市首日无前收盘约束）。"""
    days = _days(2)
    bars = {"000001": _bars(
        [("2024-01-01", 50.0, 50.0, 50.0, 50.0, 1000.0),
         ("2024-01-02", 50.0, 50.0, 50.0, 50.0, 1000.0)]
    )}
    result = check_outliers(bars, days, listing_skip=0)
    assert result.summary["evaluated_returns"] == 1  # 只有 1 个收益可评估


def test_outliers_threshold_parametrized() -> None:
    """阈值可调：+8% 在默认阈值下不命中、在 5% 阈值下命中。"""
    days = _days(2)
    bars = {"000001": _bars(
        [("2024-01-01", 10.0, 10.0, 10.0, 10.0, 1000.0),
         ("2024-01-02", 10.8, 10.8, 10.8, 10.8, 1000.0)]
    )}
    assert check_outliers(bars, days, listing_skip=0).ok
    assert not check_outliers(bars, days, listing_skip=0, threshold=0.05).ok


def test_outliers_listing_window_exempt() -> None:
    """上市初期豁免：前 5 个交易日的跳空不评估，之后的跳空正常命中。"""
    days = _days(7)
    stamps = [d.strftime("%Y-%m-%d") for d in days]
    bars = {"000001": _bars(
        [(stamps[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (stamps[1], 20.0, 20.0, 20.0, 20.0, 1000.0),  # 上市第 2 日 +100%（豁免）
         (stamps[2], 20.0, 20.0, 20.0, 20.0, 1000.0),
         (stamps[3], 20.0, 20.0, 20.0, 20.0, 1000.0),
         (stamps[4], 20.0, 20.0, 20.0, 20.0, 1000.0),
         (stamps[5], 20.0, 20.0, 20.0, 20.0, 1000.0),
         (stamps[6], 40.0, 40.0, 40.0, 40.0, 1000.0)]  # 上市第 7 日 +100%（命中）
    )}
    result = check_outliers(bars, days)
    assert result.summary["evaluated_returns"] == 2  # 第 6、7 个交易日
    assert result.summary["flagged_count"] == 1
    assert result.table.iloc[0]["dt"] == stamps[6]


def test_outliers_suspension_zero_volume_bars_skipped() -> None:
    """停牌填充 bar（volume=0）不评估收益：间隔按最近可交易 bar 计、长停牌复牌豁免。"""
    days = _days(12)
    stamps = [d.strftime("%Y-%m-%d") for d in days]

    short = {"000001": _bars(
        [(stamps[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (stamps[1], 10.0, 10.0, 10.0, 10.0, 0.0),    # 停牌填充 bar
         (stamps[2], 10.0, 10.0, 10.0, 10.0, 0.0),    # 停牌填充 bar
         (stamps[3], 15.0, 15.0, 15.0, 15.0, 1000.0)]  # 复牌 +50%
    )}
    result_short = check_outliers(short, days, listing_skip=0)
    assert not result_short.ok
    row = result_short.table.iloc[0]
    assert float(row["ret"]) == pytest.approx(0.5)
    assert int(row["gap_days"]) == 3  # 与最近可交易 bar 的真实间隔（不是 1）

    long_gap = {"000001": _bars(
        [(stamps[0], 10.0, 10.0, 10.0, 10.0, 1000.0)]
        + [(d, 10.0, 10.0, 10.0, 10.0, 0.0) for d in stamps[1:11]]
        + [(stamps[11], 15.0, 15.0, 15.0, 15.0, 1000.0)]  # 停牌 10 天后复牌 +50%
    )}
    result_long = check_outliers(long_gap, days, listing_skip=0)
    assert result_long.ok  # 间隔 11 > 5 ⇒ 复牌跳空豁免


def test_outliers_suspension_nan_volume_bars_skipped() -> None:
    """停牌填充 bar（volume=NaN，BaoStock 实际形态）同样不可交易：复牌跳空按真实间隔豁免。"""
    days = _days(12)
    stamps = [d.strftime("%Y-%m-%d") for d in days]
    nan = float("nan")

    short = {"000001": _bars(
        [(stamps[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (stamps[1], 10.0, 10.0, 10.0, 10.0, nan),    # 停牌填充 bar（NaN）
         (stamps[2], 10.0, 10.0, 10.0, 10.0, nan),    # 停牌填充 bar（NaN）
         (stamps[3], 15.0, 15.0, 15.0, 15.0, 1000.0)]  # 复牌 +50%
    )}
    result_short = check_outliers(short, days, listing_skip=0)
    assert not result_short.ok
    row = result_short.table.iloc[0]
    assert float(row["ret"]) == pytest.approx(0.5)
    assert int(row["gap_days"]) == 3  # 与最近可交易 bar 的真实间隔（不是 1）

    long_gap = {"000001": _bars(
        [(stamps[0], 10.0, 10.0, 10.0, 10.0, 1000.0)]
        + [(d, 10.0, 10.0, 10.0, 10.0, nan) for d in stamps[1:11]]
        + [(stamps[11], 15.0, 15.0, 15.0, 15.0, 1000.0)]  # 停牌 10 天后复牌 +50%
    )}
    assert check_outliers(long_gap, days, listing_skip=0).ok  # 间隔 11 > 5 ⇒ 复牌跳空豁免


# --------------------------------------------------------------------------- #
# 5) check_adjust
# --------------------------------------------------------------------------- #
def _raw_qfq_pair(collect: bool) -> tuple[dict, dict]:
    """构造「未复权有 3% 除权跳空、前复权抹平 / 未抹平」一对序列。"""
    days = [d.strftime("%Y-%m-%d") for d in _days(4)]
    raw = _bars(
        [(days[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[1], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[2], 9.7, 9.7, 9.7, 9.7, 1000.0),   # 除权日：-3%
         (days[3], 9.7, 9.7, 9.7, 9.7, 1000.0)]
    )
    if collect:
        qfq = _bars(
            [(days[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
             (days[1], 10.0, 10.0, 10.0, 10.0, 1000.0),
             (days[2], 10.0, 10.0, 10.0, 10.0, 1000.0),  # 抹平（收益差 ≈ +3%）
             (days[3], 10.0, 10.0, 10.0, 10.0, 1000.0)]
        )
    else:
        qfq = raw.copy()  # 前复权与未复权相同 ⇒ 未抹平
    return {"000001": raw}, {"000001": qfq}


def test_adjust_smoothed_is_self_consistent() -> None:
    """未复权除权跳空被前复权抹平 ⇒ 判定自洽，事件计数 ≥1。"""
    raw, qfq = _raw_qfq_pair(collect=True)
    result = check_adjust(raw, qfq)
    assert result.ok
    row = result.table.iloc[0]
    assert row["conclusion"] == CONCLUSION_SELF_CONSISTENT
    assert int(row["dividend_events"]) >= 1
    assert int(row["unsmoothed_events"]) == 0


def test_adjust_unsmoothed_detected() -> None:
    """前复权与未复权完全相同（比值 k ≡ 1）⇒ 判定疑似未抹平（口径混用）。"""
    raw, qfq = _raw_qfq_pair(collect=False)
    result = check_adjust(raw, qfq)
    assert not result.ok
    row = result.table.iloc[0]
    assert row["conclusion"] == CONCLUSION_UNSMOOTHED
    assert int(row["unsmoothed_events"]) >= 1
    assert result.summary["inconsistent_symbols"] == ["000001"]


def test_adjust_tiny_step_is_real_event_self_consistent() -> None:
    """微幅比值跳变（0.1%）是真实小额分红 ⇒ 计入除权事件、判自洽（不误报未抹平）。"""
    days = [d.strftime("%Y-%m-%d") for d in _days(4)]
    raw = _bars(
        [(days[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[1], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[2], 9.99, 9.99, 9.99, 9.99, 1000.0),  # 0.1% 微幅跳变（> 容差）
         (days[3], 9.99, 9.99, 9.99, 9.99, 1000.0)]
    )
    qfq = _bars([(day, 10.0, 10.0, 10.0, 10.0, 1000.0) for day in days])
    result = check_adjust({"000001": raw}, {"000001": qfq})
    assert result.ok
    row = result.table.iloc[0]
    assert int(row["dividend_events"]) == 1
    assert int(row["unsmoothed_events"]) == 0
    assert row["conclusion"] == CONCLUSION_SELF_CONSISTENT


def test_adjust_date_mismatch_detected() -> None:
    """两条序列日期集不一致 ⇒ 判定日期集不一致（优先级最高）。"""
    days = [d.strftime("%Y-%m-%d") for d in _days(4)]
    raw = _bars(
        [(days[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[1], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[2], 9.7, 9.7, 9.7, 9.7, 1000.0),  # 真实除权跳变（比值 k 变化）
         (days[3], 9.7, 9.7, 9.7, 9.7, 1000.0)]
    )
    qfq = _bars([(day, 10.0, 10.0, 10.0, 10.0, 1000.0) for day in days[:3]])
    result = check_adjust({"000001": raw}, {"000001": qfq})
    assert not result.ok
    row = result.table.iloc[0]
    assert row["conclusion"] == CONCLUSION_DATE_MISMATCH
    assert int(row["date_mismatch"]) == 1  # 对称差：仅 raw 多出的最后一天
    assert int(row["unsmoothed_events"]) == 0  # 结论来自日期集（而非未抹平）


def test_adjust_volume_mismatch_is_informational() -> None:
    """成交量差异只作信息统计，不影响自洽结论。"""
    days = [d.strftime("%Y-%m-%d") for d in _days(3)]
    raw = _bars(
        [(days[0], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[1], 10.0, 10.0, 10.0, 10.0, 1000.0),
         (days[2], 9.7, 9.7, 9.7, 9.7, 1000.0)]  # 真实除权跳变（序列确有差异）
    )
    qfq = _bars([(day, 10.0, 10.0, 10.0, 10.0, 2000.0) for day in days])
    result = check_adjust({"000001": raw}, {"000001": qfq})
    row = result.table.iloc[0]
    assert int(row["volume_mismatch"]) == 3
    assert row["conclusion"] == CONCLUSION_SELF_CONSISTENT


def test_adjust_empty_sample_degrades() -> None:
    """没有可对账的标的 ⇒ 警告 + 表为空。"""
    result = check_adjust({}, {})
    assert result.warnings
    assert result.table.empty
    assert list(result.table.columns) == ADJUST_COLUMNS


# --------------------------------------------------------------------------- #
# 6) check_survivorship
# --------------------------------------------------------------------------- #
def _records() -> list[IndexConstituent]:
    """三份快照：A 全程在；B 只在首份；C 最后一份；D 中段（已退出）。"""
    return [
        IndexConstituent(index_code="sh.000300", symbol="000001", name="甲", effective_date=pd.Timestamp("2015-03-30")),
        IndexConstituent(index_code="sh.000300", symbol="000002", name="乙", effective_date=pd.Timestamp("2015-03-30")),
        IndexConstituent(index_code="sh.000300", symbol="000004", name="丁", effective_date=pd.Timestamp("2015-03-30")),
        IndexConstituent(index_code="sh.000300", symbol="000001", name="甲", effective_date=pd.Timestamp("2020-06-15")),
        IndexConstituent(index_code="sh.000300", symbol="000004", name="丁", effective_date=pd.Timestamp("2020-06-15")),
        IndexConstituent(index_code="sh.000300", symbol="000001", name="甲", effective_date=pd.Timestamp("2024-12-16")),
        IndexConstituent(index_code="sh.000300", symbol="000003", name="丙", effective_date=pd.Timestamp("2024-12-16")),
    ]


def test_survivorship_missing_quantified() -> None:
    """并集 4 只、本地 2 只 ⇒ 缺席 2 只、比例 50%。"""
    result = check_survivorship(
        _records(), ["000001", "000003"], index_code="sh.000300", start="2015-01-01"
    )
    assert result.ok
    assert result.summary["union_size"] == 4
    assert result.summary["present_size"] == 2
    assert result.summary["missing_size"] == 2
    assert result.summary["missing_ratio"] == pytest.approx(0.5)
    assert list(result.missing_table.columns) == SURVIVORSHIP_MISSING_COLUMNS
    assert set(result.missing_table["symbol"]) == {"000002", "000004"}


def test_survivorship_missing_last_seen_info() -> None:
    """缺席标的的记录：最后生效日 / 名称 / 退出年份取最后一次出现。"""
    result = check_survivorship(_records(), ["000001"], index_code="sh.000300")
    row = result.missing_table.set_index("symbol").loc["000004"]
    assert row["last_effective_date"] == "2020-06-15"
    assert row["name"] == "丁"
    assert int(row["exit_year"]) == 2020


def test_survivorship_year_table_coverage() -> None:
    """逐年覆盖率：以每年 12-31 的 PIT 名单为准。"""
    result = check_survivorship(
        _records(), ["000001", "000003"], index_code="sh.000300",
        years=[2015, 2020, 2024],
    )
    assert list(result.year_table.columns) == SURVIVORSHIP_YEAR_COLUMNS
    table = result.year_table.set_index("year")
    assert int(table.loc[2015, "pit_members"]) == 3
    assert int(table.loc[2015, "present"]) == 1
    assert float(table.loc[2015, "coverage_rate"]) == pytest.approx(1 / 3)
    assert int(table.loc[2024, "present"]) == 2  # 000001 + 000003


def test_survivorship_empty_records_degrades() -> None:
    """无名单快照 ⇒ ok=False + 警告。"""
    result = check_survivorship([], ["000001"])
    assert result.ok is False
    assert result.warnings


# --------------------------------------------------------------------------- #
# 7) QualityAuditor 编排 E2E
# --------------------------------------------------------------------------- #
def _seed_engine_db(path, with_index: bool = True) -> Database:
    """播一个小型可用库：3 只股票 + 指数日历 + 成分股快照。

    股票：000001 / 000002 全程在，000003 只在最后一份名单（模拟新纳入）。
    """
    db = _open(path)
    days = _days(60)
    for symbol, price in (("000001", 10.0), ("000002", 20.0)):
        frame = _flat(days, price=price)
        frame["symbol"] = symbol
        frame["period"] = "d"
        frame["adjust"] = "qfq"
        db.upsert_bars(frame)
        # 未复权：第 41 个交易日除权（价格阶跃 -3%），前复权已抹平（保持恒定）
        raw = _bars(
            [
                (d.strftime("%Y-%m-%d"), price, price, price, price, 1000.0)
                if i < 40
                else (d.strftime("%Y-%m-%d"), price * 0.97, price * 0.97, price * 0.97, price * 0.97, 1000.0)
                for i, d in enumerate(days)
            ]
        )
        raw["symbol"] = symbol
        raw["period"] = "d"
        raw["adjust"] = ""
        db.upsert_bars(raw)
    if with_index:
        frame = _flat(days, price=3000.0)
        frame["symbol"] = "sh.000300"
        frame["period"] = "d"
        frame["adjust"] = ""
        db.upsert_bars(frame)
    db.upsert_index_constituents(_records())
    return db


def test_auditor_empty_db_degrades(tmp_path) -> None:
    """空库：ok=False + 可读错误（不抛异常）。"""
    db = _open(tmp_path / "empty.db")
    report = QualityAuditor(db).run(QualityAuditSpec())
    assert report.ok is False
    assert report.error


def test_auditor_full_run_all_items(tmp_path) -> None:
    """正常库：六项齐全、ok=True、日历来源为指数代码。"""
    db = _seed_engine_db(tmp_path / "audit.db")
    report = QualityAuditor(db).run(QualityAuditSpec())

    assert report.ok, report.error
    assert report.coverage is not None and report.coverage.ok
    assert report.ohlc is not None and report.ohlc.ok
    assert report.calendar is not None and report.calendar.ok
    assert report.outliers is not None and report.outliers.ok
    assert report.adjust is not None and report.adjust.ok
    assert report.survivorship is not None and report.survivorship.ok
    assert report.calendar_source == "sh.000300"
    # 默认池排除指数。
    assert "sh.000300" not in report.universe
    # 幸存者偏差：并集 4 只、本地 2 只 ⇒ 缺席 2 只。
    assert report.survivorship.summary["union_size"] == 4
    assert report.survivorship.summary["missing_size"] == 2


def test_auditor_survivorship_respects_window(tmp_path) -> None:
    """幸存者偏差的并集按评估窗口起算（窗口内只看窗口内的名单）。"""
    db = _seed_engine_db(tmp_path / "audit_window.db")
    report = QualityAuditor(db).run(QualityAuditSpec(start="2024-01-01"))
    assert report.ok, report.error
    # 2024-01-01 起只剩 2024-12-16 一份名单（000001 / 000003）⇒ 并集 2 只、缺 1 只。
    assert report.survivorship.summary["union_size"] == 2
    assert report.survivorship.summary["missing_size"] == 1


def test_auditor_explicit_symbols_pool(tmp_path) -> None:
    """显式标的池只审计给定标的。"""
    db = _seed_engine_db(tmp_path / "audit2.db")
    report = QualityAuditor(db).run(QualityAuditSpec(symbols=["000001"]))
    assert report.ok, report.error
    assert report.universe == ["000001"]


def test_auditor_item_failure_does_not_break_others(tmp_path, monkeypatch) -> None:
    """单项检查抛异常：该项为 None、其余照常、ok=False 且 error 指出失败项。"""
    db = _seed_engine_db(tmp_path / "audit3.db")

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("模拟单项崩溃")

    monkeypatch.setattr("app.engine.quality.audit.check_outliers", _boom)
    report = QualityAuditor(db).run(QualityAuditSpec())

    assert report.ok is False
    assert "异常波动" in report.error
    assert report.outliers is None
    assert report.coverage is not None
    assert report.survivorship is not None


def test_auditor_spec_type_error() -> None:
    """spec 类型错误应抛 TypeError（调用方编程错误）。"""
    with pytest.raises(TypeError):
        QualityAuditor().run(None)  # type: ignore[arg-type]


def test_sample_symbols_deterministic_and_dedup() -> None:
    """抽样：样本数 ≤ pool 时全取；否则等距、去重、保序。"""
    pool = [f"{i:06d}" for i in range(100)]
    assert QualityAuditor._sample_symbols(pool, 100) == pool
    picked = QualityAuditor._sample_symbols(pool, 10)
    assert len(picked) == 10
    assert picked == sorted(picked)
    assert len(set(picked)) == len(picked)
    assert picked[0] == "000000" and picked[-1] == "000090"


def test_auditor_ohlc_checks_both_adjusts(tmp_path) -> None:
    """OHLC 校验同时覆盖前复权（全池）与未复权（抽样）。"""
    db = _seed_engine_db(tmp_path / "audit4.db")
    report = QualityAuditor(db).run(QualityAuditSpec(symbols=["000001", "000002"]))
    assert report.ok, report.error
    adjusts = set(report.ohlc.table["adjust"])
    assert adjusts == {"", "qfq"}