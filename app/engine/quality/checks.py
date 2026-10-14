"""数据质量审计的**六项检查**（P7-S4，纯函数 / 零 Qt / 零 DB）。

定位：
    :mod:`app.engine.quality.audit` 负责从本地库装载数据并编排；本模块只做
    「给定 DataFrame 算出问题」的计算，便于单测与复用。六项检查分别回答：

    1. :func:`check_coverage`      —— **覆盖度 / 停牌率**：区间内应有 N 个交易日，
       实际只有 M 根 bar 的标的有哪些、停牌比例多高；
    2. :func:`check_ohlc`          —— **OHLC 逻辑校验**：``high >= low``、
       ``high >= max(open, close)``、``low <= min(open, close)``、价格为正、
       成交量/额非负、无 NaN；
    3. :func:`check_calendar`      —— **日历一致性**：每根 bar 的日期都要落在
       市场交易日历（默认取沪深300 指数日线）内，出现「野生日期」即为数据问题；
    4. :func:`check_outliers`      —— **异常波动**：个股日收益超出物理涨跌停限制
       （主板 10% / 创业板·科创板 20% + 容差）的样本；停牌填充 bar（成交量为
       NaN / 0）不评估、间隔按最近可交易 bar 计，剔除上市初期与复牌跳空；
    5. :func:`check_adjust`        —— **复权自洽抽样对账**：同一标的「未复权 vs 前复权」
       两条序列应对齐；以「未复权 ÷ 前复权」比值 k 的跳变判定真实复权日，
       若整段 k ≡ 1（毫无差异）说明 qfq 列疑似实为未复权（口径混用 → 假动量）；
    6. :func:`check_survivorship`  —— **幸存者偏差量化**：PIT 并集（历史全部成分股）
       与本地现有标的的差距，以及逐年覆盖率。

数据约定（输入）：
    * ``bars_by_symbol``：``{symbol: DataFrame}``，列至少含
      ``datetime / open / high / low / close / volume / amount``；
    * 所有检查**不抛业务异常**：非法 / 空输入降级为空结果，问题写进
      ``warnings`` 或结果表的 ``issues`` 字段。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from app.data.universe import to_date_str

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 阈值常量（集中定义，UI 与测试引用同一份口径）
# --------------------------------------------------------------------------- #
#: OHLC 逻辑校验的相对容差（千分之 0.5）：复权换算的浮点舍入会带来微小越界，
#: 只有超过该比例才判定为真实的数据错误。
OHLC_TOLERANCE: float = 5e-4

#: 异常波动阈值：创业板 / 科创板涨跌停为 ±20%，加 1 个百分点覆盖四舍五入。
DEFAULT_OUTLIER_THRESHOLD: float = 0.21

#: 复牌跳空的日历间隔阈值（交易日）：**相邻两个可交易 bar** 的日历间隔超过该
#: 天数视为停牌复牌，跳过其跳空（停牌填充 bar 成交量为 NaN / 0，不参与）。
DEFAULT_SUSPENSION_GAP: int = 5

#: 上市初期豁免（交易日）：标的自身前 5 个交易日不评估收益——主板首日无跌停
#: 限制（±44%）、创业板 / 科创板新股前 5 日无涨跌幅限制，跳空属正常现象。
DEFAULT_LISTING_SKIP_BARS: int = 5

#: 复权跳变判定容差：未复权 ÷ 前复权 比值序列相邻两日的相对变化超过该值，视为
#: 该日发生了一次复权调整（真实除权 / 送转日；比值只在调整日跳变）。
ADJUST_STEP_TOL: float = 5e-4

#: 口径混用判定容差：整段「未复权 ÷ 前复权」比值 k 与 1 的最大偏差不超过该值，
#: 视为两条序列恒等（qfq 列疑似实为未复权）。注：k 的**跳变日**即真实复权日，
#: 但「跳变日两条收益差 ≈ 跳变幅度」是数学恒等，无法据此检测未抹平。
ADJUST_IDENTITY_TOL: float = 1e-6

#: 各结果表最多保留的问题样本行数（防止 UI 表格被几十万行拖垮）。
MAX_SAMPLE_ROWS: int = 200


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class CoverageResult:
    """覆盖度 / 停牌率结果。

    Attributes:
        ok: 是否至少有一个标的进入统计。
        table: 契约列 :data:`COVERAGE_COLUMNS`（每个标的一行）。
        summary: 汇总指标（标的数、加权平均停牌率、最长停牌标的等）。
        warnings: 过程警告。
    """

    ok: bool = False
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COVERAGE_COLUMNS))
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class OhlcResult:
    """OHLC 逻辑校验结果。

    Attributes:
        ok: 是否未发现任何逻辑错误。
        table: 契约列 :data:`OHLC_COLUMNS`（「检查项 × 复权口径」各一行）。
        samples: 问题样本（最多 :data:`MAX_SAMPLE_ROWS` 行，契约列 :data:`OHLC_SAMPLE_COLUMNS`）。
        summary: 汇总（总错误数、各口径错误数）。
        warnings: 过程警告。
    """

    ok: bool = True
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=OHLC_COLUMNS))
    samples: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=OHLC_SAMPLE_COLUMNS)
    )
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CalendarResult:
    """日历一致性结果。

    Attributes:
        ok: 是否未发现野生日期。
        table: 契约列 :data:`CALENDAR_COLUMNS`（野生日期样本）。
        summary: 汇总（日历长度、来源、野生日期数）。
        warnings: 过程警告。
    """

    ok: bool = True
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CALENDAR_COLUMNS))
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class OutlierResult:
    """异常波动结果。

    Attributes:
        ok: 是否未发现超过阈值的异常收益。
        table: 契约列 :data:`OUTLIER_COLUMNS`（异常样本，按跌幅绝对值降序）。
        summary: 汇总（阈值、样本数、命中率、命中标的数）。
        warnings: 过程警告。
    """

    ok: bool = True
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=OUTLIER_COLUMNS))
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AdjustResult:
    """复权自洽抽样对账结果。

    Attributes:
        ok: 抽样标的全部自洽（日期集一致且无未抹平除权）。
        table: 契约列 :data:`ADJUST_COLUMNS`（每个抽样标的一行）。
        summary: 汇总（抽样数、除权事件数、未抹平行数、不一致标的清单）。
        warnings: 过程警告。
    """

    ok: bool = True
    table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ADJUST_COLUMNS))
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SurvivorshipResult:
    """幸存者偏差量化结果。

    Attributes:
        ok: 是否成功完成量化（与「偏差是否存在」无关——偏差本身就是结论）。
        missing_table: 契约列 :data:`SURVIVORSHIP_MISSING_COLUMNS`（缺席标的清单）。
        year_table: 契约列 :data:`SURVIVORSHIP_YEAR_COLUMNS`（逐年 PIT 覆盖率）。
        summary: 汇总（并集规模、当前可用数、缺席数与比例）。
        warnings: 过程警告。
    """

    ok: bool = False
    missing_table: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=SURVIVORSHIP_MISSING_COLUMNS)
    )
    year_table: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=SURVIVORSHIP_YEAR_COLUMNS)
    )
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 结果表契约列
# --------------------------------------------------------------------------- #
COVERAGE_COLUMNS: list[str] = [
    "symbol", "first_dt", "last_dt", "bars", "expected", "missing", "suspension_rate",
]
OHLC_COLUMNS: list[str] = ["adjust", "check", "count"]
OHLC_SAMPLE_COLUMNS: list[str] = ["adjust", "symbol", "dt", "check", "detail"]
CALENDAR_COLUMNS: list[str] = ["symbol", "dt", "reason"]
OUTLIER_COLUMNS: list[str] = ["symbol", "dt", "ret", "prev_close", "close", "gap_days"]
ADJUST_COLUMNS: list[str] = [
    "symbol", "bars_raw", "bars_qfq", "date_mismatch", "volume_mismatch",
    "dividend_events", "unsmoothed_events", "conclusion",
]
SURVIVORSHIP_MISSING_COLUMNS: list[str] = [
    "symbol", "name", "last_effective_date", "exit_year",
]
SURVIVORSHIP_YEAR_COLUMNS: list[str] = [
    "year", "pit_members", "present", "missing", "coverage_rate",
]

#: 结论文案（表内取值，UI 可原样展示）。
CONCLUSION_SELF_CONSISTENT: str = "自洽"
CONCLUSION_DATE_MISMATCH: str = "日期集不一致"
CONCLUSION_UNSMOOTHED: str = "疑似未抹平除权"


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _sorted_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """取 bar 的 ``datetime`` 列并转为升序去重的 ``DatetimeIndex``。"""
    if frame is None or frame.empty or "datetime" not in frame.columns:
        return pd.DatetimeIndex([])
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="coerce"))
    stamps = stamps.dropna()
    return stamps.sort_values().unique()


def _ordered_bars(frame: pd.DataFrame) -> tuple[Optional[pd.DataFrame], pd.DatetimeIndex]:
    """按日期升序、剔除无效日期后的 ``(bars, 日期索引)``。

    与 :func:`_sorted_dates` 的差别：**不去重**，保证与数据列长度一一对应
    （逐行收益 / 向量化检查都依赖这个对齐；主键约束下真实数据不会重复，
    但合成输入可能有）。
    """
    if frame is None or frame.empty or "datetime" not in frame.columns:
        return None, pd.DatetimeIndex([])
    ordered = (
        frame.assign(_dt=pd.to_datetime(frame["datetime"], errors="coerce"))
        .dropna(subset=["_dt"])
        .sort_values("_dt")
        .reset_index(drop=True)
    )
    if ordered.empty:
        return None, pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(ordered["_dt"]).normalize()
    return ordered, dates


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """把某一列转成 float ndarray（缺失为 NaN）；列不存在时全 NaN。"""
    if frame is None or column not in frame.columns:
        return np.full(len(frame) if frame is not None else 0, np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _normalize_symbol_set(symbols: Sequence[Any]) -> list[str]:
    """把标的序列转成去重升序字符串列表。"""
    out: set[str] = set()
    for item in symbols:
        text = str(item or "").strip()
        if text:
            out.add(text)
    return sorted(out)


# --------------------------------------------------------------------------- #
# 1) 覆盖度 / 停牌率
# --------------------------------------------------------------------------- #
def check_coverage(
    bars_by_symbol: dict[str, pd.DataFrame],
    calendar: Sequence[Any],
    start: Any = None,
    end: Any = None,
) -> CoverageResult:
    """统计每个标的在评估窗口内的**应有交易日 vs 实际 bar 数**（停牌率）。

    口径：只考察标的自身首末 bar 与评估窗口的交集区间——一只 2018 年才上市的
    股票在 2015~2017 年没有 bar 是**正常**的（未上市），不应当成停牌。

    Args:
        bars_by_symbol: ``{symbol: bars DataFrame}``。
        calendar: 市场交易日历（升序日期序列；通常取指数日线）。
        start: 评估窗口起始（含）；``None`` 不限。
        end: 评估窗口结束（含）；``None`` 不限。

    Returns:
        :class:`CoverageResult`。
    """
    result = CoverageResult()
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce")).dropna()
    cal = cal.sort_values().unique()
    if len(cal) == 0:
        result.warnings.append("市场日历为空，无法统计覆盖度")
        return result
    if not bars_by_symbol:
        result.warnings.append("没有可统计的标的")
        return result

    cal_set = cal
    window_start = pd.to_datetime(start, errors="coerce") if start is not None else None
    window_end = pd.to_datetime(end, errors="coerce") if end is not None else None
    if start is not None and pd.isna(window_start):
        result.warnings.append(f"起始日期无法解析，已忽略：{start!r}")
        window_start = None
    if end is not None and pd.isna(window_end):
        result.warnings.append(f"结束日期无法解析，已忽略：{end!r}")
        window_end = None

    rows: list[dict[str, Any]] = []
    for symbol in sorted(bars_by_symbol):
        stamps = _sorted_dates(bars_by_symbol[symbol])
        if len(stamps) == 0:
            result.warnings.append(f"{symbol} 无有效日期，跳过覆盖度统计")
            continue
        first = pd.Timestamp(stamps[0])
        last = pd.Timestamp(stamps[-1])
        span_start = max(first, window_start) if window_start is not None else first
        span_end = min(last, window_end) if window_end is not None else last
        if span_start > span_end:
            result.warnings.append(f"{symbol} 与评估窗口无交集，跳过覆盖度统计")
            continue
        expected_mask = (cal_set >= span_start) & (cal_set <= span_end)
        expected = int(expected_mask.sum())
        bars = int(((stamps >= span_start) & (stamps <= span_end)).sum())
        missing = max(expected - bars, 0)
        rate = (missing / expected) if expected else 0.0
        rows.append(
            {
                "symbol": symbol,
                "first_dt": first.strftime("%Y-%m-%d"),
                "last_dt": last.strftime("%Y-%m-%d"),
                "bars": bars,
                "expected": expected,
                "missing": missing,
                "suspension_rate": rate,
            }
        )

    if not rows:
        result.warnings.append("所有标的都与评估窗口无交集")
        return result

    table = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    total_expected = int(table["expected"].sum())
    total_missing = int(table["missing"].sum())
    worst = table.loc[table["suspension_rate"].idxmax()]
    result.table = table
    result.ok = True
    result.summary = {
        "n_symbols": int(len(table)),
        "total_expected": total_expected,
        "total_missing": total_missing,
        "mean_suspension_rate": (total_missing / total_expected) if total_expected else 0.0,
        "median_suspension_rate": float(table["suspension_rate"].median()),
        "worst_symbol": str(worst["symbol"]),
        "worst_rate": float(worst["suspension_rate"]),
    }
    return result


# --------------------------------------------------------------------------- #
# 2) OHLC 逻辑校验
# --------------------------------------------------------------------------- #
def check_ohlc(
    bars_by_symbol: dict[str, pd.DataFrame],
    adjust: str = "qfq",
) -> OhlcResult:
    """逐标的校验 OHLC 逻辑自洽性（``high`` 最大、``low`` 最小、价格为正等）。

    检查项（表内 ``check`` 取值）：
        * ``high_lt_low``：``high < low``；
        * ``high_lt_body``：``high < max(open, close)``；
        * ``low_gt_body``：``low > min(open, close)``；
        * ``nonpositive_price``：任一价格 ≤ 0；
        * ``nan_price``：任一价格缺失；
        * ``negative_flow``：``volume < 0`` 或 ``amount < 0``。

    容差：相对 :data:`OHLC_TOLERANCE`（复权换算舍入），超出才判定为错误。

    Args:
        bars_by_symbol: ``{symbol: bars DataFrame}``。
        adjust: 这批数据的复权口径标签（用于结果表区分 ``""`` / ``"qfq"``）。

    Returns:
        :class:`OhlcResult`。
    """
    result = OhlcResult()
    if not bars_by_symbol:
        result.warnings.append(f"[{adjust or '未复权'}] 没有可校验的标的")
        return result

    counters: dict[str, int] = {}
    samples: list[dict[str, Any]] = []

    def _record(symbol: str, stamps: pd.DatetimeIndex, check: str, mask: np.ndarray, detail: str) -> int:
        hits = int(mask.sum())
        if hits:
            counters[check] = counters.get(check, 0) + hits
            if len(samples) < MAX_SAMPLE_ROWS:
                idx = np.flatnonzero(mask)[:5]
                for pos in idx:
                    samples.append(
                        {
                            "adjust": adjust or "",
                            "symbol": symbol,
                            "dt": pd.Timestamp(stamps[pos]).strftime("%Y-%m-%d"),
                            "check": check,
                            "detail": detail,
                        }
                    )
        return hits

    for symbol in sorted(bars_by_symbol):
        frame = bars_by_symbol[symbol]
        ordered, stamps = _ordered_bars(frame)
        if ordered is None:
            result.warnings.append(f"[{adjust or '未复权'}] {symbol} 无有效日期，跳过")
            continue
        open_ = _numeric(ordered, "open")
        high = _numeric(ordered, "high")
        low = _numeric(ordered, "low")
        close = _numeric(ordered, "close")
        volume = _numeric(ordered, "volume")
        amount = _numeric(ordered, "amount")

        with np.errstate(invalid="ignore"):
            body_hi = np.fmax(open_, close)
            body_lo = np.fmin(open_, close)
            _record(
                symbol, stamps, "high_lt_low",
                high < low * (1.0 - OHLC_TOLERANCE), "high < low",
            )
            _record(
                symbol, stamps, "high_lt_body",
                high < body_hi * (1.0 - OHLC_TOLERANCE), "high < max(open, close)",
            )
            _record(
                symbol, stamps, "low_gt_body",
                low > body_lo * (1.0 + OHLC_TOLERANCE), "low > min(open, close)",
            )
            prices = np.vstack([open_, high, low, close])
            nonpositive = ((prices <= 0) & ~np.isnan(prices)).any(axis=0)
            _record(symbol, stamps, "nonpositive_price", nonpositive, "存在 ≤ 0 的价格")
            nan_mask = np.isnan(prices).any(axis=0)
            _record(symbol, stamps, "nan_price", nan_mask, "存在缺失价格")
            negative = np.where(np.isnan(volume), False, volume < 0) | np.where(
                np.isnan(amount), False, amount < 0
            )
            _record(symbol, stamps, "negative_flow", negative, "成交量 / 成交额为负")

    rows = [
        {"adjust": adjust or "", "check": name, "count": counters.get(name, 0)}
        for name in (
            "high_lt_low", "high_lt_body", "low_gt_body",
            "nonpositive_price", "nan_price", "negative_flow",
        )
    ]
    result.table = pd.DataFrame(rows, columns=OHLC_COLUMNS)
    result.samples = pd.DataFrame(samples, columns=OHLC_SAMPLE_COLUMNS)
    total = int(sum(counters.values()))
    result.ok = total == 0
    result.summary = {
        "adjust": adjust or "",
        "total_errors": total,
        "by_check": dict(counters),
    }
    if not result.ok:
        result.warnings.append(
            f"[{adjust or '未复权'}] OHLC 逻辑错误 {total} 处（详见 samples）"
        )
    return result


# --------------------------------------------------------------------------- #
# 3) 日历一致性
# --------------------------------------------------------------------------- #
def check_calendar(
    bars_by_symbol: dict[str, pd.DataFrame],
    calendar: Sequence[Any],
    calendar_source: str = "",
) -> CalendarResult:
    """校验每根 bar 的日期是否落在市场交易日历内（发现「野生日期」）。

    Args:
        bars_by_symbol: ``{symbol: bars DataFrame}``。
        calendar: 市场交易日历（升序）。
        calendar_source: 日历来源描述（如 ``"sh.000300"``），仅用于报告。

    Returns:
        :class:`CalendarResult`。
    """
    result = CalendarResult()
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce")).dropna()
    cal = cal.sort_values().unique()
    if len(cal) == 0:
        result.ok = False
        result.warnings.append("市场日历为空，无法校验日期一致性")
        return result

    cal_index = pd.DatetimeIndex(cal).normalize()
    rows: list[dict[str, Any]] = []
    wild_total = 0
    for symbol in sorted(bars_by_symbol):
        stamps = _sorted_dates(bars_by_symbol[symbol])
        if len(stamps) == 0:
            continue
        normalized = pd.DatetimeIndex(stamps).normalize()
        wild_mask = ~normalized.isin(cal_index)
        hits = int(wild_mask.sum())
        if hits:
            wild_total += hits
            for stamp in normalized[wild_mask][:5]:
                if len(rows) < MAX_SAMPLE_ROWS:
                    rows.append(
                        {
                            "symbol": symbol,
                            "dt": stamp.strftime("%Y-%m-%d"),
                            "reason": "该日期不在市场交易日历内",
                        }
                    )

    result.table = pd.DataFrame(rows, columns=CALENDAR_COLUMNS)
    result.ok = wild_total == 0
    result.summary = {
        "calendar_source": calendar_source,
        "calendar_days": int(len(cal_index)),
        "calendar_first": cal_index[0].strftime("%Y-%m-%d"),
        "calendar_last": cal_index[-1].strftime("%Y-%m-%d"),
        "wild_dates": wild_total,
    }
    if wild_total:
        result.warnings.append(f"发现 {wild_total} 个不在市场日历内的野生日期")
    return result


# --------------------------------------------------------------------------- #
# 4) 异常波动
# --------------------------------------------------------------------------- #
def check_outliers(
    bars_by_symbol: dict[str, pd.DataFrame],
    calendar: Sequence[Any],
    threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    suspension_gap: int = DEFAULT_SUSPENSION_GAP,
    listing_skip: int = DEFAULT_LISTING_SKIP_BARS,
) -> OutlierResult:
    """找出日收益超过物理涨跌停限制的样本（疑似复权错误 / 脏数据）。

    判定口径（修正版）：
        * **停牌填充 bar 不评估**：BaoStock 等数据源在停牌日仍返回 bar
          （成交量为 NaN / 0、开高低收 = 前收），这类 bar 不产生收益、也不作为
          基准；收益一律按「最近一根可交易 bar → 当前 bar」计算；
        * **复牌跳空豁免**：与最近可交易 bar 的日历间隔 > ``suspension_gap``
          个交易日（停牌期间累积的消息面变动）；
        * **上市初期豁免**：标的自身前 ``listing_skip`` 个交易日不评估
          （主板首日无跌停限制、创业板 / 科创板新股前 5 日无涨跌幅限制）。

    Args:
        bars_by_symbol: ``{symbol: bars DataFrame}``（建议用前复权口径，
            未复权口径的除权跳空会全部误报为异常）。
        calendar: 市场交易日历（用于计算复牌间隔）。
        threshold: 异常阈值（默认 :data:`DEFAULT_OUTLIER_THRESHOLD`）。
        suspension_gap: 复牌跳空的间隔阈值（交易日）。
        listing_skip: 上市初期豁免的交易日根数（默认 :data:`DEFAULT_LISTING_SKIP_BARS`）。

    Returns:
        :class:`OutlierResult`。
    """
    result = OutlierResult()
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce")).dropna()
    cal = cal.sort_values().unique()
    if len(cal) == 0:
        result.warnings.append("市场日历为空，无法计算复牌间隔（按不剔除处理）")
    pos_map = {pd.Timestamp(day).normalize(): i for i, day in enumerate(cal)}

    rows: list[dict[str, Any]] = []
    flagged_symbols: set[str] = set()
    evaluated = 0
    for symbol in sorted(bars_by_symbol):
        frame = bars_by_symbol[symbol]
        ordered, stamps = _ordered_bars(frame)
        if ordered is None or len(stamps) < 2:
            continue
        close = _numeric(ordered, "close")
        volume = _numeric(ordered, "volume")
        # 可交易 bar：成交量 > 0。停牌填充 bar 的成交量为 NaN（BaoStock）或 0，
        # 两类都不可交易：不产生收益、也不作基准。
        active = np.isfinite(volume) & (volume > 0)
        last_active: Optional[int] = None
        for i in range(len(stamps)):
            if not bool(active[i]):
                continue  # 停牌填充 bar：不产生收益、不作基准
            value: Optional[float] = None
            if last_active is not None and i >= listing_skip:
                prev_close = close[last_active]
                with np.errstate(invalid="ignore", divide="ignore"):
                    candidate = close[i] / prev_close - 1.0
                if np.isfinite(candidate):
                    value = float(candidate)
            if value is not None:
                prev_pos = pos_map.get(pd.Timestamp(stamps[last_active]))
                cur_pos = pos_map.get(pd.Timestamp(stamps[i]))
                gap_days = (
                    (cur_pos - prev_pos)
                    if (prev_pos is not None and cur_pos is not None)
                    else int(i - last_active)
                )
                if gap_days <= suspension_gap:
                    evaluated += 1
                    if abs(value) > threshold:
                        flagged_symbols.add(symbol)
                        rows.append(
                            {
                                "symbol": symbol,
                                "dt": pd.Timestamp(stamps[i]).strftime("%Y-%m-%d"),
                                "ret": value,
                                "prev_close": float(prev_close),
                                "close": float(close[i]),
                                "gap_days": int(gap_days),
                            }
                        )
            last_active = i

    table = pd.DataFrame(rows, columns=OUTLIER_COLUMNS)
    if not table.empty:
        table["_abs"] = table["ret"].abs()
        table = (
            table.sort_values("_abs", ascending=False)
            .drop(columns=["_abs"])
            .head(MAX_SAMPLE_ROWS)
            .reset_index(drop=True)
        )
    result.table = table
    result.ok = len(rows) == 0
    result.summary = {
        "threshold": float(threshold),
        "evaluated_returns": int(evaluated),
        "flagged_count": int(len(rows)),
        "flagged_symbols": int(len(flagged_symbols)),
        "flagged_ratio": (len(rows) / evaluated) if evaluated else 0.0,
    }
    if rows:
        result.warnings.append(
            f"发现 {len(rows)} 个超过 ±{threshold:.0%} 的异常收益（疑似复权错误或脏数据）"
        )
    return result


# --------------------------------------------------------------------------- #
# 5) 复权自洽抽样对账
# --------------------------------------------------------------------------- #
def check_adjust(
    raw_by_symbol: dict[str, pd.DataFrame],
    qfq_by_symbol: dict[str, pd.DataFrame],
    sample_symbols: Optional[Sequence[str]] = None,
) -> AdjustResult:
    """对抽样标的做「未复权 vs 前复权」自洽对账。

    三项检验：
        * **日期集一致**：两条序列的交易日必须完全相同（同源同标的）；
        * **成交量一致**：成交量（手）不应随复权口径变化（信息性指标）；
        * **复权跳变**：未复权 ÷ 前复权 的比值 ``k`` 只在真实复权日跳变
          （除权 / 送转 ⇒ 复权因子阶跃），故以跳变日作为「除权事件」——不再
          用「未复权跌幅」猜除权（会把普通大跌日全部误判为除权）。唯一的
          未抹平签名是整段 ``k ≡ 1``（两条序列毫无差异）⇒ qfq 列疑似实为
          未复权（口径混用 → 假动量），记为 1 个未抹平事件（无法量化到日期）。
          ⚠️ 「跳变日两条收益差 ≈ 跳变幅度」是数学恒等，不能据此检测未抹平；
          其他未抹平形态（如部分区间 qfq 不随除权更新）由异常波动检查兜底。

    Args:
        raw_by_symbol: ``{symbol: 未复权 bars}``。
        qfq_by_symbol: ``{symbol: 前复权 bars}``。
        sample_symbols: 抽样标的；``None`` = 两表键的交集。

    Returns:
        :class:`AdjustResult`。
    """
    result = AdjustResult()
    if sample_symbols is None:
        candidates = sorted(set(raw_by_symbol) & set(qfq_by_symbol))
    else:
        candidates = sorted(
            {str(item) for item in sample_symbols if str(item) in raw_by_symbol and str(item) in qfq_by_symbol}
        )
    if not candidates:
        result.warnings.append("抽样标的为空（未复权 / 前复权数据都没有）")
        return result

    rows: list[dict[str, Any]] = []
    total_events = 0
    total_unsmoothed = 0
    inconsistent: list[str] = []
    for symbol in candidates:
        raw = raw_by_symbol[symbol]
        qfq = qfq_by_symbol[symbol]
        ordered_raw, raw_dates = _ordered_bars(raw)
        ordered_qfq, qfq_dates = _ordered_bars(qfq)
        if ordered_raw is None or ordered_qfq is None:
            result.warnings.append(f"{symbol} 缺少有效日期，跳过对账")
            continue
        raw_index = pd.DatetimeIndex(_sorted_dates(raw)).normalize()
        qfq_index = pd.DatetimeIndex(_sorted_dates(qfq)).normalize()
        date_mismatch = int(len(raw_index.symmetric_difference(qfq_index)))

        common = raw_index.intersection(qfq_index)
        vol_mismatch = 0
        events = 0
        unsmoothed = 0
        if len(common) >= 2:
            raw_vol_map = pd.Series(_numeric(ordered_raw, "volume"), index=raw_dates)
            qfq_vol_map = pd.Series(_numeric(ordered_qfq, "volume"), index=qfq_dates)
            both = (
                pd.DataFrame({"raw": raw_vol_map.reindex(common), "qfq": qfq_vol_map.reindex(common)})
                .dropna()
            )
            if len(both):
                diff = (both["raw"] - both["qfq"]).abs()
                vol_mismatch = int((diff > 1e-6).sum())

            # 复权因子 k = 未复权收盘 / 前复权收盘：只在真实复权（除权 / 送转）日跳变。
            pair = (
                pd.DataFrame(
                    {
                        "raw": pd.Series(_numeric(ordered_raw, "close"), index=raw_dates).reindex(common),
                        "qfq": pd.Series(_numeric(ordered_qfq, "close"), index=qfq_dates).reindex(common),
                    }
                )
                .dropna()
            )
            pair = pair[(pair["raw"] > 0) & (pair["qfq"] > 0)]
            if len(pair) >= 2:
                ratio = pair["raw"] / pair["qfq"]
                rel_step = (ratio / ratio.shift(1) - 1.0).abs().dropna()
                step_dates = rel_step[rel_step > ADJUST_STEP_TOL].index
                events = int(len(step_dates))
                if events == 0 and float((ratio - 1.0).abs().max()) <= ADJUST_IDENTITY_TOL:
                    # 整段 k ≡ 1（两条序列毫无差异）⇒ qfq 列疑似实为未复权（口径混用）
                    unsmoothed = 1
        if date_mismatch > 0:
            conclusion = CONCLUSION_DATE_MISMATCH
        elif unsmoothed > 0:
            conclusion = CONCLUSION_UNSMOOTHED
        else:
            conclusion = CONCLUSION_SELF_CONSISTENT
        if conclusion != CONCLUSION_SELF_CONSISTENT:
            inconsistent.append(symbol)
        total_events += events
        total_unsmoothed += unsmoothed
        rows.append(
            {
                "symbol": symbol,
                "bars_raw": int(len(raw_index)),
                "bars_qfq": int(len(qfq_index)),
                "date_mismatch": date_mismatch,
                "volume_mismatch": vol_mismatch,
                "dividend_events": events,
                "unsmoothed_events": unsmoothed,
                "conclusion": conclusion,
            }
        )

    result.table = pd.DataFrame(rows, columns=ADJUST_COLUMNS)
    result.ok = len(inconsistent) == 0
    result.summary = {
        "sampled": len(rows),
        "dividend_events": total_events,
        "unsmoothed_events": total_unsmoothed,
        "inconsistent_symbols": inconsistent,
    }
    if inconsistent:
        result.warnings.append(
            f"{len(inconsistent)} 只抽样标的复权口径不自洽（{'、'.join(inconsistent[:5])}"
            + ("…" if len(inconsistent) > 5 else "")
            + "）"
        )
    return result


# --------------------------------------------------------------------------- #
# 6) 幸存者偏差量化
# --------------------------------------------------------------------------- #
def check_survivorship(
    records: Any,
    local_symbols: Sequence[Any],
    index_code: Optional[str] = None,
    start: Any = None,
    end: Any = None,
    years: Optional[Sequence[int]] = None,
) -> SurvivorshipResult:
    """量化「研究池 = 当前成分股」造成的幸存者偏差。

    方法：
        * **并集口径**：PIT 并集 = 区间内所有快照出现过的标的（历史全部成分股）；
        * **缺席规模**：``缺席 = 并集 − 本地可用``——这些退市 / 被剔除 / 被吸并的
          股票在「当前成分股」口径下**完全不存在**，其负收益被系统性剔除；
        * **逐年覆盖率**：每年末交易日（12-31）的 PIT 名单里，本地有多少可用，
          用于观察偏差随时间的变化（越早的年份偏差越大——退市股大多在早期）。

    Args:
        records: 成分股名单记录（见 :mod:`app.data.universe` 输入约定）。
        local_symbols: 本地可用标的（通常为本地库 qfq 日线标的）。
        index_code: 指数代码；``None`` 表示不区分。
        start: 并集起始（生效日 ≥ start）；``None`` 不限。
        end: 并集结束；``None`` 不限。
        years: 逐年覆盖率的年份列表；``None`` = 从（start 年或首份名单年）到今年。

    Returns:
        :class:`SurvivorshipResult`。
    """
    from app.data.universe import pit_members, universe_union

    result = SurvivorshipResult()
    local = set(_normalize_symbol_set(local_symbols))
    union = universe_union(records, index_code=index_code, start=start, end=end)
    if not union:
        result.warnings.append("没有可用的成分股名单快照（请先更新指数成分股）")
        return result

    # 缺席标的：取「最后一次出现在名单里的生效日 + 名称」作为退出时点。
    last_seen: dict[str, tuple[str, str]] = {}
    try:
        for item in records:
            code = str(getattr(item, "index_code", "") or "").strip()
            symbol = str(getattr(item, "symbol", "") or "").strip()
            eff = to_date_str(getattr(item, "effective_date", None))
            name = str(getattr(item, "name", "") or "").strip()
            if index_code and code != index_code:
                continue
            if symbol and eff and (symbol not in last_seen or eff > last_seen[symbol][0]):
                last_seen[symbol] = (eff, name)
    except TypeError:  # 传入的可能是 DataFrame：转 records 后重试
        for item in pd.DataFrame(records).to_dict("records"):
            code = str(item.get("index_code", "") or "").strip()
            symbol = str(item.get("symbol", "") or "").strip()
            eff = to_date_str(item.get("effective_date"))
            name = str(item.get("name", "") or "").strip()
            if index_code and code != index_code:
                continue
            if symbol and eff and (symbol not in last_seen or eff > last_seen[symbol][0]):
                last_seen[symbol] = (eff, name)

    missing = [symbol for symbol in union if symbol not in local]
    missing_rows: list[dict[str, Any]] = []
    for symbol in missing:
        eff, name = last_seen.get(symbol, ("", ""))
        missing_rows.append(
            {
                "symbol": symbol,
                "name": name,
                "last_effective_date": eff,
                "exit_year": int(eff[:4]) if len(eff) >= 4 and eff[:4].isdigit() else 0,
            }
        )

    # 逐年覆盖率（以每年 12-31 的 PIT 名单为准）。
    start_year = None
    if start is not None:
        text = to_date_str(start)
        if len(text) >= 4 and text[:4].isdigit():
            start_year = int(text[:4])
    if start_year is None and last_seen:
        early = [eff for eff, _ in last_seen.values() if eff]
        if early:
            start_year = min(int(value[:4]) for value in early)
    if start_year is None:
        start_year = pd.Timestamp.today().year
    if years is None:
        years = list(range(start_year, pd.Timestamp.today().year + 1))

    year_rows: list[dict[str, Any]] = []
    for year in sorted({int(item) for item in years}):
        members = pit_members(records, f"{year}-12-31", index_code=index_code)
        if not members:
            continue
        present = [symbol for symbol in members if symbol in local]
        year_rows.append(
            {
                "year": year,
                "pit_members": len(members),
                "present": len(present),
                "missing": len(members) - len(present),
                "coverage_rate": (len(present) / len(members)) if members else 0.0,
            }
        )

    result.ok = True
    result.missing_table = pd.DataFrame(missing_rows, columns=SURVIVORSHIP_MISSING_COLUMNS)
    if not result.missing_table.empty:
        result.missing_table = result.missing_table.sort_values(
            ["exit_year", "symbol"], ascending=[True, True]
        ).reset_index(drop=True)
    result.year_table = pd.DataFrame(year_rows, columns=SURVIVORSHIP_YEAR_COLUMNS)
    result.summary = {
        "union_size": len(union),
        "present_size": len(union) - len(missing),
        "missing_size": len(missing),
        "missing_ratio": (len(missing) / len(union)) if union else 0.0,
        "current_members": (
            int(result.year_table["pit_members"].iloc[-1]) if not result.year_table.empty else 0
        ),
    }
    if missing:
        result.warnings.append(
            f"幸存者偏差规模：历史成分股并集 {len(union)} 只，本地缺席 {len(missing)} 只"
            f"（{len(missing) / len(union):.1%}）"
        )
    return result