"""策略成员筛选与筛选后组合验证（P7-S8）。

定位：
    阶段 7 已证明**资金分账·等权组合是「诚实的容器」、自己不制造 α**——组合的
    好坏完全取决于「成员个体质量 × 成员间低相关」。本模块回答阶段 8 的业务问题：
    **候选策略一大堆，挑哪些进组合？以及「怎么挑」才不算偷看未来？**

    直觉做法（把候选全部回测、挑历史表现最好的几个组队）会踩中最经典的
    **数据窥探 / 选择偏差**：用全样本挑出的「好策略」混着大量运气成分，再用
    同一段历史评估组合，等于开卷考试给自己打分——组合绩效被系统性高估。
    本模块的处理口径（用户拍板）：

    * **样本内筛选 + 样本外验证**：默认 2015-01-01 ~ 2020-12-31 为样本内
      （筛选依据），其后为样本外（**验证用，不参与筛选判定**）；同时按全样本
      口径再筛一遍作为「上界参考」，量化选择偏差有多大；
    * **绝对门槛 + 分年度稳定性**：成员在样本内段独立回测须同时满足
      「跑赢基准 + 回撤不深于基准 + 夏普达阈值 + 正收益年份过半」，
      每条门槛可调、每个候选的通过 / 淘汰原因逐条记录（:class:`SelectionCriteria`）。

关键口径（为什么这样切段）：
    每个候选**只跑一次全期回测**，指标按「样本内 / 样本外 / 全样本」三段切净值
    重算——而不是把候选分别在两个窗口各跑一次。原因：样本外段的信号依赖历史
    数据预热（如打分的 ICIR 扩窗权重需要足够 IC 样本），若从切分点重新起跑，
    样本外前段会「空仓热身」，不等于实盘中「已运行多年的策略」的真实状态。
    全期回测 + 分段评估才是对「样本外验证」的正确模拟，且总回测次数减半。

    分段评估把段内首日净值归一为 1.0（段内收益 = 段末 / 段首 − 1），成交明细
    按日期过滤（胜率 / 盈亏比只统计段内已平仓卖出，与全局口径一致）。

对外 API::

    from app.engine.backtest import (
        MemberCandidate, SampleSplit, SelectionCriteria, SelectionRunner,
        build_candidate_pool,
    )

    pool = build_candidate_pool(symbols=pit_symbols)     # 15 个候选（8 均线 + 7 打分）
    runner = SelectionRunner()
    report = runner.run(pool)                            # 筛选（样本内判定 + 全样本参考）
    print(report.passed_in, report.to_frame())
    comparison = runner.compare(report)                  # 筛选前 / 筛选后 / 上界 / 最优单策略
    print(comparison.to_frame())                         # 各组合的样本外段表现

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field, replace
from itertools import combinations
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from app.data.store.db import Database, get_db
from app.data.universe import DEFAULT_INDEX_CODE
from app.engine.backtest.attribution import yearly_summary
from app.engine.backtest.engine import (
    EQUITY_COLUMNS,
    BacktestEngine,
    BacktestResult,
)
from app.engine.backtest.metrics import (
    PerformanceMetrics,
    compute_metrics,
    drawdown_series,
)
from app.engine.backtest.portfolio import (
    PortfolioMember,
    PortfolioResult,
    PortfolioRunner,
    PortfolioSpec,
)
from app.engine.backtest.scored import (
    DEFAULT_SCORED_REBALANCE,
    DEFAULT_SCORED_TOP_N,
    DEFAULT_SCORED_UNIVERSE,
    UNIVERSE_CHOICES,
    ScoredBacktestRunner,
    ScoredBacktestSpec,
)
from app.engine.backtest.strategy import (
    DEFAULT_BENCHMARK,
    DEFAULT_INITIAL_CASH,
    REBALANCE_CHOICES,
    REBALANCE_DAILY,
    REBALANCE_MONTHLY,
    REBALANCE_WEEKLY,
    StrategySpec,
)
from app.engine.factor.scoring import DEFAULT_SCORING_FACTORS
from app.engine.screener import Condition

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 默认口径常量（集中定义，UI 与测试引用同一份）
# --------------------------------------------------------------------------- #
#: 默认样本内区间（筛选依据）：2015-01-01 ~ 2020-12-31（约 6 年，覆盖一轮牛熊）。
DEFAULT_IN_START: str = "2015-01-01"
DEFAULT_IN_END: str = "2020-12-31"

#: 均线族候选的窗口参数（站上 N 日均线）。
DEFAULT_MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

#: 均线族候选的调仓频率（周频 / 月频，与阶段 7 两条基线策略同口径）。
DEFAULT_MA_REBALANCES: tuple[str, ...] = (REBALANCE_WEEKLY, REBALANCE_MONTHLY)

#: 均线族候选最多持有只数（沿 P7-S1/S2/S7「周频 top3」口径）。
DEFAULT_MA_TOP_N: int = 3

# 候选来源标记（报告 / UI 展示用）。
CANDIDATE_FAMILY_MA: str = "ma"
CANDIDATE_FAMILY_SCORED: str = "scored"

#: 候选来源中文标签。
CANDIDATE_FAMILY_LABELS: dict[str, str] = {
    CANDIDATE_FAMILY_MA: "均线族",
    CANDIDATE_FAMILY_SCORED: "打分族",
}

#: 调仓频率中文标签（候选命名用）。
REBALANCE_LABELS: dict[str, str] = {
    REBALANCE_DAILY: "日频",
    REBALANCE_WEEKLY: "周频",
    REBALANCE_MONTHLY: "月频",
}

# 组合对比变体键（顺序固定：筛选前 → 筛选后 → 上界参考 → 最优单策略对照）。
VARIANT_ALL: str = "all"
VARIANT_SELECTED_IN: str = "selected_in"
VARIANT_SELECTED_FULL: str = "selected_full"
VARIANT_BEST_SINGLE: str = "best_single"

#: 变体中文名（UI 展示用）。
VARIANT_NAMES: dict[str, str] = {
    VARIANT_ALL: "筛选前·全候选等权",
    VARIANT_SELECTED_IN: "筛选后·样本内合格等权",
    VARIANT_SELECTED_FULL: "上界参考·全样本合格等权",
    VARIANT_BEST_SINGLE: "对照·最优单策略",
}

#: 筛选报告表契约列（顺序固定，便于 UI 逐列展示）。
SELECTION_TABLE_COLUMNS: list[str] = [
    "candidate",
    "family",
    "in_return",
    "in_benchmark_return",
    "in_max_drawdown",
    "in_sharpe",
    "in_win_year_ratio",
    "in_passed",
    "out_return",
    "out_max_drawdown",
    "out_sharpe",
    "full_passed",
    "reasons",
]

#: 组合对比表契约列（顺序固定）。
SELECTION_COMPARISON_COLUMNS: list[str] = [
    "variant",
    "name",
    "members",
    "member_count",
    "out_return",
    "out_benchmark_return",
    "out_max_drawdown",
    "out_sharpe",
    "note",
]

# 浮点比较容差（门槛判定用）。
_EPS: float = 1e-9


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _parse_bound(value: Any) -> Optional[pd.Timestamp]:
    """把日期解析为当日零点 ``Timestamp``；``None`` 原样返回（口径与回测引擎一致）。"""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法解析日期：{value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"无法解析日期：{value!r}")
    return stamp.normalize()


def _format_pct(value: Optional[float]) -> str:
    """百分比格式化（缺失 / 非有限值显示「—」，供淘汰原因文案使用）。"""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.2%}" if np.isfinite(number) else "—"


# --------------------------------------------------------------------------- #
# 输入规格
# --------------------------------------------------------------------------- #
@dataclass
class MemberCandidate:
    """候选成员：一个「还没有被验证过」的策略规格 + 来源标记。

    Attributes:
        name: 候选名称（筛选报告内**唯一**；留空时回退为策略规格自身的名称）。
        spec: 候选策略规格，:class:`StrategySpec`（条件式）或
            :class:`ScoredBacktestSpec`（多因子打分）。
        family: 来源标记（``"ma"`` / ``"scored"``）；留空按 ``spec`` 类型自动推断。

    Raises:
        ValueError: ``spec`` 类型不支持 / ``family`` 非法取值。
    """

    name: str
    spec: Union[StrategySpec, ScoredBacktestSpec]
    family: str = ""

    def __post_init__(self) -> None:
        """构造后校验（属调用方输入错误 → 抛 :class:`ValueError`）。"""
        if not isinstance(self.spec, (StrategySpec, ScoredBacktestSpec)):
            raise ValueError(
                "候选 spec 必须是 StrategySpec 或 ScoredBacktestSpec，"
                f"实际：{type(self.spec).__name__}"
            )
        self.name = str(self.name).strip() or str(self.spec.name).strip() or "未命名候选"
        family = str(self.family).strip()
        if not family:
            family = (
                CANDIDATE_FAMILY_SCORED
                if isinstance(self.spec, ScoredBacktestSpec)
                else CANDIDATE_FAMILY_MA
            )
        if family not in CANDIDATE_FAMILY_LABELS:
            raise ValueError(
                f"候选来源标记非法：{family!r}（可选：{sorted(CANDIDATE_FAMILY_LABELS)}）"
            )
        self.family = family


@dataclass
class SampleSplit:
    """样本内 / 样本外切分口径（防数据窥探的时间闸门）。

    Attributes:
        in_start: 样本内起点（含）；``None`` = 不设下限（回测从数据起点开始）。
        in_end: 样本内截止（含）——**必填**（切分点）；筛选判定只用
            ``[in_start, in_end]`` 段，样本外从其次一自然日起。
        out_end: 样本外截止（含）；``None`` = 到数据末端。

    Raises:
        ValueError: 日期无法解析 / ``in_end`` 为空 / 起止顺序非法。
    """

    in_start: Any = DEFAULT_IN_START
    in_end: Any = DEFAULT_IN_END
    out_end: Any = None

    def __post_init__(self) -> None:
        """构造后解析与校验（属调用方输入错误 → 抛 :class:`ValueError`）。"""
        self.in_start = _parse_bound(self.in_start)
        self.in_end = _parse_bound(self.in_end)
        self.out_end = _parse_bound(self.out_end)
        if self.in_end is None:
            raise ValueError("样本内截止日（in_end）不能为空——它是筛选与验证的切分点")
        if self.in_start is not None and self.in_start >= self.in_end:
            raise ValueError(
                f"样本内起点必须早于截止日：{self.in_start.date()} >= {self.in_end.date()}"
            )
        if self.out_end is not None and self.out_end <= self.in_end:
            raise ValueError(
                f"样本外截止必须晚于样本内截止：{self.out_end.date()} <= {self.in_end.date()}"
            )

    @property
    def out_start(self) -> pd.Timestamp:
        """样本外起点 = 样本内截止的次一自然日（交易日历在自然日粒度上离散）。"""
        return self.in_end + pd.Timedelta(days=1)


@dataclass
class SelectionCriteria:
    """「合格成员」的绝对门槛（每条可调、可单独关闭）。

    口径（全部满足才算合格；`None` 表示不启用该条）：**不依赖与其他候选的排名
    比较**——门槛写死，避免「相对排序 Top-K」被个别年份的运气带偏。

    Attributes:
        beat_benchmark: 段内总收益须 ≥ 基准总收益（跑赢基准）。
        drawdown_not_worse_than_benchmark: 段内最大回撤须不深于基准回撤。
        max_drawdown_limit: 段内最大回撤的绝对上限（如 ``0.5`` = 回撤不超过 50%）；
            ``None``（默认）不启用。
        min_sharpe: 段内夏普比率下限；``None`` 不启用（默认 0.5）。
        min_win_year_ratio: 段内正收益年份占比下限（分年度稳定性）；
            ``None`` 不启用（默认 0.5，即「过半」）。

    Raises:
        ValueError: 门槛取值非法（回撤上限超出 (0, 1] / 占比超出 [0, 1] / 非有限数）。
    """

    beat_benchmark: bool = True
    drawdown_not_worse_than_benchmark: bool = True
    max_drawdown_limit: Optional[float] = None
    min_sharpe: Optional[float] = 0.5
    min_win_year_ratio: Optional[float] = 0.5

    def __post_init__(self) -> None:
        """构造后校验（属调用方输入错误 → 抛 :class:`ValueError`）。"""
        self.beat_benchmark = bool(self.beat_benchmark)
        self.drawdown_not_worse_than_benchmark = bool(
            self.drawdown_not_worse_than_benchmark
        )

        if self.max_drawdown_limit is not None:
            try:
                limit = float(self.max_drawdown_limit)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"回撤上限必须是数值：{self.max_drawdown_limit!r}"
                ) from exc
            if not np.isfinite(limit) or not (0.0 < limit <= 1.0):
                raise ValueError(f"回撤上限必须在 (0, 1] 区间：{self.max_drawdown_limit!r}")
            self.max_drawdown_limit = limit

        if self.min_sharpe is not None:
            try:
                sharpe = float(self.min_sharpe)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"夏普门槛必须是数值：{self.min_sharpe!r}") from exc
            if not np.isfinite(sharpe):
                raise ValueError(f"夏普门槛必须是有限数值：{self.min_sharpe!r}")
            self.min_sharpe = sharpe

        if self.min_win_year_ratio is not None:
            try:
                ratio = float(self.min_win_year_ratio)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"分年度胜率门槛必须是数值：{self.min_win_year_ratio!r}"
                ) from exc
            if not np.isfinite(ratio) or not (0.0 <= ratio <= 1.0):
                raise ValueError(
                    f"分年度胜率门槛必须在 [0, 1] 区间：{self.min_win_year_ratio!r}"
                )
            self.min_win_year_ratio = ratio


# --------------------------------------------------------------------------- #
# 候选池构造
# --------------------------------------------------------------------------- #
def _ma_condition(window: int) -> Condition:
    """均线族选股条件：收盘价站上 N 日均线（差值表达式 vs 常数 0）。"""
    return Condition(f"close - close.rolling({window}).mean()", ">", 0)


def _ma_rank(window: int) -> str:
    """均线族排序表达式：偏离均线更小者优先（「刚站上均线」排在前面）。"""
    return f"close.rolling({window}).mean() / close"


def build_candidate_pool(
    symbols: Optional[Sequence[str]] = None,
    index_code: str = DEFAULT_INDEX_CODE,
    benchmark: str = DEFAULT_BENCHMARK,
    ma_windows: Sequence[int] = DEFAULT_MA_WINDOWS,
    ma_rebalances: Sequence[str] = DEFAULT_MA_REBALANCES,
    ma_top_n: int = DEFAULT_MA_TOP_N,
    factors: Sequence[str] = DEFAULT_SCORING_FACTORS,
    scored_top_n: int = DEFAULT_SCORED_TOP_N,
    scored_rebalance: str = DEFAULT_SCORED_REBALANCE,
    scored_universe: str = DEFAULT_SCORED_UNIVERSE,
) -> list[MemberCandidate]:
    """按默认口径生成候选池：均线族参数网格 + 打分因子子集（**不抛业务异常**）。

    候选构成（默认 4×2 + 7 = **15 个**）：
        * **均线族**：站上 N 日均线（``N ∈ ma_windows``）× 调仓频率
          （``ma_rebalances``），top_n / 排序表达式固定（沿阶段 7 基线口径）；
        * **打分族**：给定因子列表的**全部非空子集**（默认核心三因子 → 7 个子集），
          每个子集一个候选（ICIR 动态加权由打分引擎内部完成）。

    Args:
        symbols: 均线族候选的股票池（如 PIT 成分股并集）；``None`` = 用本地库
            全部有日线标的（默认口径，注意幸存者偏差，回测层如实登记）。
        index_code: 打分族 PIT 口径使用的指数代码（默认沪深300）。
        benchmark: 两个族统一的基准代码（默认沪深300）。
        ma_windows: 均线窗口清单（每个 ≥2）。
        ma_rebalances: 均线族调仓频率清单（取值同 ``REBALANCE_CHOICES``）。
        ma_top_n: 均线族最多持有只数（≥1）。
        factors: 打分族因子名清单（非空；子集按输入顺序组合）。
        scored_top_n: 打分族每日持有只数（≥1）。
        scored_rebalance: 打分族调仓频率。
        scored_universe: 打分族股票池口径（``"pit"`` / ``"current"``）。

    Returns:
        候选清单（名称唯一：如「站上20日线·周频」「打分·log_amt_20+ret_5」）。

    Raises:
        ValueError: 任一参数非法（窗口 <2 / 频率非法 / 因子为空 / 池口径非法）。
    """
    windows: list[int] = []
    for value in ma_windows or ():
        try:
            window = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"均线窗口必须是整数：{value!r}") from exc
        if window < 2:
            raise ValueError(f"均线窗口必须 ≥2：{value!r}")
        if window not in windows:
            windows.append(window)
    if not windows:
        raise ValueError("均线窗口不能为空")

    rebalances: list[str] = []
    for value in ma_rebalances or ():
        text = str(value).strip()
        if text not in REBALANCE_CHOICES:
            raise ValueError(f"调仓频率非法：{value!r}（可选：{REBALANCE_CHOICES}）")
        if text not in rebalances:
            rebalances.append(text)
    if not rebalances:
        raise ValueError("均线族调仓频率不能为空")

    try:
        ma_count = int(ma_top_n)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"均线族最多持有只数必须是整数：{ma_top_n!r}") from exc
    if ma_count < 1:
        raise ValueError(f"均线族最多持有只数必须 ≥1：{ma_top_n!r}")

    factor_list: list[str] = []
    for value in factors or ():
        name = str(value).strip()
        if not name:
            continue
        if name not in factor_list:
            factor_list.append(name)
    if not factor_list:
        raise ValueError("打分族因子清单不能为空")

    try:
        scored_count = int(scored_top_n)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"打分族最多持有只数必须是整数：{scored_top_n!r}") from exc
    if scored_count < 1:
        raise ValueError(f"打分族最多持有只数必须 ≥1：{scored_top_n!r}")

    if scored_rebalance not in REBALANCE_CHOICES:
        raise ValueError(
            f"打分族调仓频率非法：{scored_rebalance!r}（可选：{REBALANCE_CHOICES}）"
        )
    if scored_universe not in UNIVERSE_CHOICES:
        raise ValueError(
            f"打分族股票池口径非法：{scored_universe!r}（可选：{UNIVERSE_CHOICES}）"
        )

    symbol_list = [str(symbol) for symbol in symbols] if symbols is not None else None
    candidates: list[MemberCandidate] = []

    # 1) 均线族：窗口 × 调仓频率。
    for window in windows:
        for rebalance in rebalances:
            label = REBALANCE_LABELS.get(rebalance, rebalance)
            name = f"站上{window}日线·{label}"
            spec = StrategySpec(
                name=name,
                conditions=[_ma_condition(window)],
                symbols=symbol_list,
                rebalance=rebalance,
                top_n=ma_count,
                rank_by=_ma_rank(window),
                benchmark=benchmark,
            )
            candidates.append(
                MemberCandidate(name=name, spec=spec, family=CANDIDATE_FAMILY_MA)
            )

    # 2) 打分族：因子的全部非空子集。
    for size in range(1, len(factor_list) + 1):
        for combo in combinations(factor_list, size):
            name = "打分·" + "+".join(combo)
            spec = ScoredBacktestSpec(
                name=name,
                factors=list(combo),
                top_n=scored_count,
                rebalance=scored_rebalance,
                universe=scored_universe,
                index_code=index_code,
                benchmark=benchmark,
            )
            candidates.append(
                MemberCandidate(name=name, spec=spec, family=CANDIDATE_FAMILY_SCORED)
            )

    return candidates


# --------------------------------------------------------------------------- #
# 分段评估工具（公开：供 UI / 测试与筛选编排共用）
# --------------------------------------------------------------------------- #
def slice_equity(
    equity: Any,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """切出净值表的 ``[start, end]`` 段并把两列净值**归一为段首 1.0**（不抛异常）。

    口径：
        * 边界双闭区间（样本外段请传 ``split.out_start``，即样本内截止次日）；
        * 段内收益 = 段末净值 / 段首净值 − 1（与整段口径完全一致）；
        * 非法输入 / 空段返回空表（保持 :data:`EQUITY_COLUMNS` 列）。

    Args:
        equity: 净值表（``date/nav[/benchmark_nav]``；来自回测 / 组合结果）。
        start: 段起点（含）；``None`` = 不设下限。
        end: 段终点（含）；``None`` = 不设上限。

    Returns:
        归一化后的净值段（按日期升序）。
    """
    if not isinstance(equity, pd.DataFrame) or equity.empty:
        return pd.DataFrame(columns=EQUITY_COLUMNS)
    if "date" not in equity.columns or "nav" not in equity.columns:
        return pd.DataFrame(columns=EQUITY_COLUMNS)
    try:
        start_ts = _parse_bound(start)
        end_ts = _parse_bound(end)
    except ValueError:
        return pd.DataFrame(columns=EQUITY_COLUMNS)

    frame = equity.copy()
    stamps = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[stamps.notna()].copy()
    frame["date"] = stamps[stamps.notna()]
    frame = frame.sort_values("date").reset_index(drop=True)
    if start_ts is not None:
        frame = frame[frame["date"] >= start_ts]
    if end_ts is not None:
        frame = frame[frame["date"] <= end_ts]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=EQUITY_COLUMNS)

    nav = pd.to_numeric(frame["nav"], errors="coerce")
    base = nav.iloc[0] if len(nav) else None
    if base is not None and np.isfinite(base) and base > 0:
        frame["nav"] = nav / float(base)
    else:
        frame["nav"] = nav
    if "benchmark_nav" in frame.columns:
        bench = pd.to_numeric(frame["benchmark_nav"], errors="coerce")
        bench_base = bench.iloc[0] if len(bench) else None
        if bench_base is not None and np.isfinite(bench_base) and bench_base > 0:
            frame["benchmark_nav"] = bench / float(bench_base)
        else:
            frame["benchmark_nav"] = bench
    return frame


def _filter_trades(
    trades: Any,
    start: Any = None,
    end: Any = None,
) -> Optional[pd.DataFrame]:
    """按日期过滤成交明细；无有效成交返回 ``None``（= 无交易口径）。"""
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return None
    if "date" not in trades.columns:
        return None
    stamps = pd.to_datetime(trades["date"], errors="coerce")
    mask = stamps.notna()
    start_ts = _parse_bound(start)
    end_ts = _parse_bound(end)
    if start_ts is not None:
        mask = mask & (stamps >= start_ts)
    if end_ts is not None:
        mask = mask & (stamps <= end_ts)
    segment = trades[mask]
    return segment if not segment.empty else None


def _metrics_of_frame(
    frame: pd.DataFrame,
    trades_segment: Optional[pd.DataFrame],
) -> Optional[PerformanceMetrics]:
    """段内净值 + 段内成交 → 绩效指标（空段返回 ``None``）。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    nav = pd.Series(
        pd.to_numeric(frame["nav"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame["date"])),
    )
    bench: Any = None
    if "benchmark_nav" in frame.columns:
        bench = pd.Series(
            pd.to_numeric(frame["benchmark_nav"], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(frame["date"])),
        )
    return compute_metrics(nav, bench, trades_segment)


def segment_metrics(
    source: Any,
    start: Any = None,
    end: Any = None,
) -> Optional[PerformanceMetrics]:
    """对回测（或组合）结果切段重算绩效指标（**不抛异常**，空段返回 ``None``）。

    适配对象按鸭子类型：只要具备 ``equity``（``date/nav/benchmark_nav``）与
    ``trades`` 属性即可——:class:`~app.engine.backtest.engine.BacktestResult`
    与 :class:`~app.engine.backtest.portfolio.PortfolioResult` 均适用。
    成交明细按段过滤（胜率 / 盈亏比只统计段内已平仓卖出，口径与全局一致）。

    Args:
        source: 回测 / 组合结果对象。
        start: 段起点（含）；``None`` = 不设下限。
        end: 段终点（含）；``None`` = 不设上限。

    Returns:
        段内 :class:`PerformanceMetrics`；无有效数据返回 ``None``。
    """
    equity = getattr(source, "equity", None)
    frame = slice_equity(equity, start, end)
    if frame.empty:
        return None
    return _metrics_of_frame(frame, _filter_trades(getattr(source, "trades", None), start, end))


def _year_returns(frame: pd.DataFrame) -> dict[int, float]:
    """段内逐年收益（口径复用 :func:`~app.engine.backtest.attribution.yearly_summary`）。

    Returns:
        ``{年份: 收益}``；首年以段首为基（口径与归因模块一致），无数据返回空 dict。
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    summary = yearly_summary(frame)
    if summary.empty:
        return {}
    result: dict[int, float] = {}
    for _, row in summary.iterrows():
        try:
            year = int(row["year"])
        except (KeyError, TypeError, ValueError):
            continue
        value = row.get("strategy_return")
        if value is None or pd.isna(value):
            continue
        number = float(value)
        if np.isfinite(number):
            result[year] = number
    return result


def _positive_ratio(year_returns: dict[int, float]) -> Optional[float]:
    """正收益年份占比；无年度数据返回 ``None``。"""
    values = [float(v) for v in (year_returns or {}).values() if np.isfinite(float(v))]
    if not values:
        return None
    return float(sum(1 for value in values if value > 0) / len(values))


def _benchmark_max_dd(frame: pd.DataFrame) -> Optional[float]:
    """段内基准最大回撤（≤0）；无基准列 / 无数据返回 ``None``。"""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if "benchmark_nav" not in frame.columns:
        return None
    series = pd.to_numeric(frame["benchmark_nav"], errors="coerce").dropna()
    if series.empty:
        return None
    dd = drawdown_series(series)
    dd = dd.dropna()
    if dd.empty:
        return None
    return float(dd.min())


def judge_selection(
    metrics: Optional[PerformanceMetrics],
    benchmark_max_drawdown: Optional[float],
    win_year_ratio: Optional[float],
    criteria: SelectionCriteria,
    label: str = "样本内",
) -> tuple[bool, list[str]]:
    """按绝对门槛判定单个候选（**不抛异常**），返回 ``(是否合格, 淘汰原因列表)``。

    判定顺序：跑赢基准 → 回撤上限 → 回撤不深于基准 → 夏普 → 分年度稳定性；
    任一数据缺失（如无净值 / 无逐年数据）按「无法判定」处理，**不通过并记录原因**。
    """
    reasons: list[str] = []
    if metrics is None:
        return False, [f"{label}指标缺失（无有效净值数据）"]

    if criteria.beat_benchmark:
        total = metrics.total_return
        bench_total = metrics.benchmark_total_return
        if total is None or bench_total is None:
            reasons.append(f"{label}收益或基准收益缺失，无法判定「跑赢基准」")
        elif total < bench_total - _EPS:
            reasons.append(
                f"{label}未跑赢基准（{_format_pct(total)} < {_format_pct(bench_total)}）"
            )

    drawdown = metrics.max_drawdown
    if criteria.max_drawdown_limit is not None:
        if drawdown is None:
            reasons.append(f"{label}回撤缺失，无法判定回撤上限")
        elif abs(drawdown) > criteria.max_drawdown_limit + _EPS:
            reasons.append(
                f"{label}回撤 {_format_pct(drawdown)} 超过上限 "
                f"{criteria.max_drawdown_limit:.0%}"
            )

    if criteria.drawdown_not_worse_than_benchmark:
        if drawdown is None or benchmark_max_drawdown is None:
            reasons.append(f"{label}回撤或基准回撤缺失，无法判定「回撤不深于基准」")
        elif abs(drawdown) > abs(benchmark_max_drawdown) + _EPS:
            reasons.append(
                f"{label}回撤 {_format_pct(drawdown)} 深于基准 "
                f"{_format_pct(benchmark_max_drawdown)}"
            )

    if criteria.min_sharpe is not None:
        sharpe = metrics.sharpe
        if sharpe is None:
            reasons.append(f"{label}夏普缺失，无法判定夏普门槛")
        elif sharpe < criteria.min_sharpe - _EPS:
            reasons.append(
                f"{label}夏普 {sharpe:.2f} < 门槛 {criteria.min_sharpe:.2f}"
            )

    if criteria.min_win_year_ratio is not None:
        if win_year_ratio is None:
            reasons.append(f"{label}无逐年收益数据，无法判定分年度稳定性")
        elif win_year_ratio < criteria.min_win_year_ratio - _EPS:
            reasons.append(
                f"{label}正收益年份占比 {_format_pct(win_year_ratio)} < 门槛 "
                f"{criteria.min_win_year_ratio:.0%}"
            )

    return (not reasons), reasons


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class MemberEvaluation:
    """单个候选的筛选评估产出（**不抛业务异常**，问题封装为 ``ok=False``）。

    Attributes:
        candidate: 候选本身（含规格，组合对比阶段直接复用）。
        ok: 该候选是否完成回测与分段评估。
        error: 失败原因（成功时为空串）。
        result: 全期回测结果（失败时保留现场，可能为 ``None``）。
        metrics_in: 样本内段绩效（筛选依据）。
        metrics_out: 样本外段绩效（**只展示、不参与筛选判定**）。
        metrics_full: 全样本段绩效（上界参考筛选依据）。
        year_returns_in: 样本内逐年收益（``{年份: 收益}``）。
        win_year_ratio_in: 样本内正收益年份占比。
        passed_in: 样本内段是否合格（**筛选结论**）。
        passed_full: 全样本段是否合格（上界参考）。
        reasons_in: 样本内淘汰原因清单（合格时为空）。
        reasons_full: 全样本淘汰原因清单（合格时为空）。
        warnings: 该候选回测的原始警告。
    """

    candidate: MemberCandidate
    ok: bool = False
    error: str = ""
    result: Optional[BacktestResult] = None
    metrics_in: Optional[PerformanceMetrics] = None
    metrics_out: Optional[PerformanceMetrics] = None
    metrics_full: Optional[PerformanceMetrics] = None
    year_returns_in: dict[int, float] = dc_field(default_factory=dict)
    win_year_ratio_in: Optional[float] = None
    passed_in: bool = False
    passed_full: bool = False
    reasons_in: list[str] = dc_field(default_factory=list)
    reasons_full: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)


@dataclass
class SelectionReport:
    """一次成员筛选的完整产出（**不抛业务异常**）。

    Attributes:
        ok: 是否至少有一个候选完成评估（个别候选失败不阻断报告）。
        error: 整体失败原因（如全部候选回测失败）。
        split: 样本内 / 样本外切分口径。
        criteria: 合格门槛。
        evaluations: 候选评估清单（与输入池同序）。
        passed_in: 样本内合格成员名（**筛选结论**，供组合对比使用）。
        passed_full: 全样本合格成员名（上界参考）。
        warnings: 汇总警告（失败候选原因带 ``[候选·名]`` 前缀；成功候选的
            过程警告保留在各 ``evaluations[*].warnings`` 现场，不重复汇总）。
    """

    ok: bool
    error: str = ""
    split: SampleSplit = dc_field(default_factory=SampleSplit)
    criteria: SelectionCriteria = dc_field(default_factory=SelectionCriteria)
    evaluations: list[MemberEvaluation] = dc_field(default_factory=list)
    passed_in: list[str] = dc_field(default_factory=list)
    passed_full: list[str] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """逐候选明细表（契约列 :data:`SELECTION_TABLE_COLUMNS`，供 UI 展示）。"""
        rows: list[dict] = []
        for evaluation in self.evaluations:
            metrics_in = evaluation.metrics_in
            metrics_out = evaluation.metrics_out
            if not evaluation.ok:
                reasons = f"回测失败：{evaluation.error}"
            else:
                reasons = "；".join(evaluation.reasons_in)
            rows.append(
                {
                    "candidate": evaluation.candidate.name,
                    "family": CANDIDATE_FAMILY_LABELS.get(
                        evaluation.candidate.family, evaluation.candidate.family
                    ),
                    "in_return": metrics_in.total_return if metrics_in else None,
                    "in_benchmark_return": metrics_in.benchmark_total_return
                    if metrics_in
                    else None,
                    "in_max_drawdown": metrics_in.max_drawdown if metrics_in else None,
                    "in_sharpe": metrics_in.sharpe if metrics_in else None,
                    "in_win_year_ratio": evaluation.win_year_ratio_in,
                    "in_passed": evaluation.passed_in,
                    "out_return": metrics_out.total_return if metrics_out else None,
                    "out_max_drawdown": metrics_out.max_drawdown if metrics_out else None,
                    "out_sharpe": metrics_out.sharpe if metrics_out else None,
                    "full_passed": evaluation.passed_full,
                    "reasons": reasons,
                }
            )
        return pd.DataFrame(rows, columns=SELECTION_TABLE_COLUMNS)


@dataclass
class PortfolioVariant:
    """组合对比中的一个变体（**不抛业务异常**，问题封装为 ``ok=False``）。

    Attributes:
        key: 变体键（:data:`VARIANT_ALL` / :data:`VARIANT_SELECTED_IN` /
            :data:`VARIANT_SELECTED_FULL` / :data:`VARIANT_BEST_SINGLE`）。
        name: 变体中文名。
        members: 成员名清单。
        ok: 该变体是否构建并回测成功。
        note: 口径说明（如「样本内夏普最高」的挑选依据）。
        error: 失败原因（成功时为空串）。
        result: 组合回测结果（失败时保留现场）。
        metrics_out: **样本外段**绩效（组合全期回测切样本外段重算）——
            这才是可比口径的「验证结论」。
    """

    key: str
    name: str
    members: list[str] = dc_field(default_factory=list)
    ok: bool = False
    note: str = ""
    error: str = ""
    result: Optional[PortfolioResult] = None
    metrics_out: Optional[PerformanceMetrics] = None


@dataclass
class SelectionComparison:
    """筛选前 / 筛选后 / 上界参考 / 最优单策略 的四线对比产出。

    Attributes:
        ok: 是否至少有一个变体构建成功。
        error: 整体失败原因。
        split: 样本内 / 样本外切分口径（组合从样本内起点跑到样本外终点，
            评估只取样本外段——成员带着完整历史预热，才是对实盘状态的正确模拟）。
        variants: 四个变体（顺序固定：全候选 / 样本内合格 / 全样本合格 / 最优单策略）。
        warnings: 构建级警告（被剔除的失败候选、空成员集等；成员级过程警告
            保留在各 ``variants[*].result.warnings`` 现场）。
    """

    ok: bool
    error: str = ""
    split: SampleSplit = dc_field(default_factory=SampleSplit)
    variants: list[PortfolioVariant] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """四线对比表（契约列 :data:`SELECTION_COMPARISON_COLUMNS`，供 UI 展示）。"""
        rows: list[dict] = []
        for variant in self.variants:
            metrics = variant.metrics_out
            note = variant.error if not variant.ok else variant.note
            rows.append(
                {
                    "variant": variant.key,
                    "name": variant.name,
                    "members": "、".join(variant.members),
                    "member_count": len(variant.members),
                    "out_return": metrics.total_return if metrics else None,
                    "out_benchmark_return": metrics.benchmark_total_return
                    if metrics
                    else None,
                    "out_max_drawdown": metrics.max_drawdown if metrics else None,
                    "out_sharpe": metrics.sharpe if metrics else None,
                    "note": note,
                }
            )
        return pd.DataFrame(rows, columns=SELECTION_COMPARISON_COLUMNS)


# --------------------------------------------------------------------------- #
# 筛选编排器
# --------------------------------------------------------------------------- #
class SelectionRunner:
    """成员筛选与筛选后组合验证编排器（只读本地库，不依赖 Qt）。

    两段式 API：
        * :meth:`run` —— 逐候选全期回测 + 分段评估 + 门槛判定（**筛选**）；
        * :meth:`compare` —— 复用筛选结论构建四个组合变体（**验证**）。
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        engine: Optional[BacktestEngine] = None,
        scored_runner: Optional[ScoredBacktestRunner] = None,
        portfolio_runner: Optional[PortfolioRunner] = None,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        benchmark: str = DEFAULT_BENCHMARK,
    ) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`~app.data.store.db.get_db`。
            engine: 条件式候选的回测引擎；默认用同一个 ``db`` 构造。
            scored_runner: 打分候选的回测编排器；默认用同一个 ``db`` / ``engine`` 构造。
            portfolio_runner: 组合对比编排器；默认复用以上依赖构造。
            initial_cash: 组合对比各变体的总资金（元，默认 100 万）。
            benchmark: 组合对比的基准代码（默认沪深300）。
        """
        self._db: Database = db if db is not None else get_db()
        self._engine: BacktestEngine = (
            engine if engine is not None else BacktestEngine(self._db)
        )
        self._scored: ScoredBacktestRunner = (
            scored_runner
            if scored_runner is not None
            else ScoredBacktestRunner(db=self._db, engine=self._engine)
        )
        self._portfolio: PortfolioRunner = (
            portfolio_runner
            if portfolio_runner is not None
            else PortfolioRunner(
                db=self._db, engine=self._engine, scored_runner=self._scored
            )
        )
        self._initial_cash = float(initial_cash)
        self._benchmark = str(benchmark).strip() or DEFAULT_BENCHMARK

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def run(
        self,
        pool: Sequence[MemberCandidate],
        split: Optional[SampleSplit] = None,
        criteria: Optional[SelectionCriteria] = None,
    ) -> SelectionReport:
        """逐候选回测 → 分段评估 → 按门槛筛选（**不抛业务异常**）。

        每个候选只跑**一次全期回测**（``in_start → out_end``），绩效按
        样本内 / 样本外 / 全样本三段切净值重算（见模块 docstring 的口径说明）。

        Args:
            pool: 候选清单（:class:`MemberCandidate`，名称唯一、至少一个）。
            split: 切分口径；默认 :class:`SampleSplit`（2015~2020 样本内）。
            criteria: 合格门槛；默认 :class:`SelectionCriteria`。

        Returns:
            :class:`SelectionReport`。

        Raises:
            TypeError: ``pool`` / ``split`` / ``criteria`` 类型错误（调用方编程错误）。
            ValueError: 候选为空 / 元素非法 / 名称重复（调用方输入错误）。
        """
        if isinstance(pool, (str, bytes)) or not isinstance(pool, (list, tuple)):
            raise TypeError(
                "候选池必须是列表 / 元组（MemberCandidate 序列），"
                f"实际：{type(pool).__name__}"
            )
        candidates = list(pool)
        if not candidates:
            raise ValueError("候选池不能为空")
        for candidate in candidates:
            if not isinstance(candidate, MemberCandidate):
                raise ValueError(
                    f"候选必须是 MemberCandidate，实际：{type(candidate).__name__}"
                )
        names = [candidate.name for candidate in candidates]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f"候选名称重复（筛选报告需要唯一标识）：{duplicated}")

        split_ = split if split is not None else SampleSplit()
        if not isinstance(split_, SampleSplit):
            raise TypeError(f"split 必须是 SampleSplit，实际：{type(split_).__name__}")
        criteria_ = criteria if criteria is not None else SelectionCriteria()
        if not isinstance(criteria_, SelectionCriteria):
            raise TypeError(
                f"criteria 必须是 SelectionCriteria，实际：{type(criteria_).__name__}"
            )

        evaluations = [
            self._evaluate(candidate, split_, criteria_) for candidate in candidates
        ]
        passed_in = [e.candidate.name for e in evaluations if e.ok and e.passed_in]
        passed_full = [e.candidate.name for e in evaluations if e.ok and e.passed_full]

        warnings: list[str] = []
        for evaluation in evaluations:
            if not evaluation.ok:
                warnings.append(
                    f"[候选·{evaluation.candidate.name}] 回测失败：{evaluation.error}"
                )
                warnings.extend(
                    f"[候选·{evaluation.candidate.name}] {text}"
                    for text in evaluation.warnings
                )
            elif evaluation.metrics_out is None:
                warnings.append(
                    f"[候选·{evaluation.candidate.name}] 样本外段无有效数据，"
                    "样本外验证不可用"
                )

        ok = any(evaluation.ok for evaluation in evaluations)
        error = "" if ok else "全部候选回测失败，筛选报告无效"
        return SelectionReport(
            ok=ok,
            error=error,
            split=split_,
            criteria=criteria_,
            evaluations=evaluations,
            passed_in=passed_in,
            passed_full=passed_full,
            warnings=warnings,
        )

    def compare(
        self,
        report: SelectionReport,
        split: Optional[SampleSplit] = None,
    ) -> SelectionComparison:
        """按筛选结论构建四个组合变体并做样本外验证（**不抛业务异常**）。

        变体（全部等权、同一窗口与基准，从 ``in_start`` 跑到 ``out_end``，
        评估只取**样本外段**）：

        * ``all``：筛选前——全部成功评估的候选；
        * ``selected_in``：筛选后——样本内合格成员（**主结论**）；
        * ``selected_full``：上界参考——全样本合格成员（量化选择偏差）；
        * ``best_single``：对照——样本内夏普最高的单个候选（「押宝最亮的」）。

        Args:
            report: :meth:`run` 的筛选报告。
            split: 切分口径；默认沿用报告。

        Returns:
            :class:`SelectionComparison`。

        Raises:
            TypeError: ``report`` / ``split`` 类型错误（调用方编程错误）。
        """
        if not isinstance(report, SelectionReport):
            raise TypeError(
                f"report 必须是 SelectionReport，实际：{type(report).__name__}"
            )
        split_ = split if split is not None else report.split
        if not isinstance(split_, SampleSplit):
            raise TypeError(f"split 必须是 SampleSplit，实际：{type(split_).__name__}")
        if not report.ok:
            return SelectionComparison(
                ok=False,
                error=f"筛选报告无效（{report.error}），无法进行组合对比",
                split=split_,
            )

        ok_evaluations = [e for e in report.evaluations if e.ok]
        warnings = [
            f"候选「{e.candidate.name}」回测失败，未纳入组合对比：{e.error}"
            for e in report.evaluations
            if not e.ok
        ]
        by_name = {e.candidate.name: e for e in ok_evaluations}
        in_members = [by_name[name] for name in report.passed_in if name in by_name]
        full_members = [by_name[name] for name in report.passed_full if name in by_name]

        # 最优单策略 = 样本内夏普最高的成功候选（全候选范围，不限于合格成员）。
        sharpe_pool = [
            e
            for e in ok_evaluations
            if e.metrics_in is not None and e.metrics_in.sharpe is not None
        ]
        best = (
            max(sharpe_pool, key=lambda e: float(e.metrics_in.sharpe))
            if sharpe_pool
            else None
        )

        variants: list[PortfolioVariant] = [
            self._run_variant(VARIANT_ALL, VARIANT_NAMES[VARIANT_ALL], ok_evaluations, split_),
            self._run_variant(
                VARIANT_SELECTED_IN, VARIANT_NAMES[VARIANT_SELECTED_IN], in_members, split_
            ),
            self._run_variant(
                VARIANT_SELECTED_FULL,
                VARIANT_NAMES[VARIANT_SELECTED_FULL],
                full_members,
                split_,
            ),
        ]
        if best is not None:
            note = (
                f"样本内夏普最高：{best.candidate.name}"
                f"（{float(best.metrics_in.sharpe):.2f}）"
            )
            variants.append(
                self._run_variant(
                    VARIANT_BEST_SINGLE, VARIANT_NAMES[VARIANT_BEST_SINGLE], [best], split_, note
                )
            )
        else:
            variants.append(
                PortfolioVariant(
                    key=VARIANT_BEST_SINGLE,
                    name=VARIANT_NAMES[VARIANT_BEST_SINGLE],
                    ok=False,
                    error="无候选具备样本内夏普，无法构建对照",
                )
            )
            warnings.append("无候选具备样本内夏普，跳过「最优单策略」对照")

        if not in_members:
            warnings.append("样本内无合格成员，「筛选后」组合无法构建")
        if not full_members:
            warnings.append("全样本无合格成员，「上界参考」组合无法构建")

        ok = any(variant.ok for variant in variants)
        return SelectionComparison(
            ok=ok,
            error="" if ok else "全部组合变体构建失败",
            split=split_,
            variants=variants,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # 内部：单候选评估
    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        candidate: MemberCandidate,
        split: SampleSplit,
        criteria: SelectionCriteria,
    ) -> MemberEvaluation:
        """单个候选：全期回测 → 三段切净值评估 → 门槛判定。"""
        spec = candidate.spec
        if isinstance(spec, ScoredBacktestSpec):
            outcome = self._scored.run(
                replace(spec, start=split.in_start, end=split.out_end)
            )
            backtest = outcome.backtest
            ok = bool(
                outcome.ok
                and backtest is not None
                and backtest.ok
                and not backtest.equity.empty
            )
            error = outcome.error if not ok else ""
            warnings = list(outcome.warnings)
        else:
            backtest = self._engine.run(
                spec, start=split.in_start, end=split.out_end
            )
            ok = bool(backtest.ok and not backtest.equity.empty)
            error = backtest.error if not ok else ""
            warnings = list(backtest.warnings)

        if not ok:
            return MemberEvaluation(
                candidate=candidate,
                ok=False,
                error=str(error).strip() or "回测失败",
                result=backtest,
                warnings=warnings,
            )

        trades = backtest.trades
        frame_in = slice_equity(backtest.equity, split.in_start, split.in_end)
        frame_out = slice_equity(backtest.equity, split.out_start, split.out_end)
        frame_full = slice_equity(backtest.equity, split.in_start, split.out_end)

        metrics_in = _metrics_of_frame(
            frame_in, _filter_trades(trades, split.in_start, split.in_end)
        )
        metrics_out = _metrics_of_frame(
            frame_out, _filter_trades(trades, split.out_start, split.out_end)
        )
        metrics_full = _metrics_of_frame(
            frame_full, _filter_trades(trades, split.in_start, split.out_end)
        )

        year_returns_in = _year_returns(frame_in)
        year_returns_full = _year_returns(frame_full)
        ratio_in = _positive_ratio(year_returns_in)
        ratio_full = _positive_ratio(year_returns_full)

        passed_in, reasons_in = judge_selection(
            metrics_in, _benchmark_max_dd(frame_in), ratio_in, criteria, label="样本内"
        )
        passed_full, reasons_full = judge_selection(
            metrics_full, _benchmark_max_dd(frame_full), ratio_full, criteria, label="全样本"
        )

        return MemberEvaluation(
            candidate=candidate,
            ok=True,
            result=backtest,
            metrics_in=metrics_in,
            metrics_out=metrics_out,
            metrics_full=metrics_full,
            year_returns_in=year_returns_in,
            win_year_ratio_in=ratio_in,
            passed_in=passed_in,
            passed_full=passed_full,
            reasons_in=reasons_in,
            reasons_full=reasons_full,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # 内部：组合变体
    # ------------------------------------------------------------------ #
    def _run_variant(
        self,
        key: str,
        name: str,
        member_evaluations: Sequence[MemberEvaluation],
        split: SampleSplit,
        note: str = "",
    ) -> PortfolioVariant:
        """构建并回测一个组合变体（成员等权；窗口 = 样本内起点 → 样本外终点）。"""
        members = [
            PortfolioMember(name=e.candidate.name, spec=e.candidate.spec)
            for e in member_evaluations
        ]
        member_names = [member.name for member in members]
        if not members:
            return PortfolioVariant(
                key=key,
                name=name,
                members=[],
                ok=False,
                note=note,
                error="无可用成员（筛选结果为空）",
            )

        spec = PortfolioSpec(
            name=name,
            members=members,
            start=split.in_start,
            end=split.out_end,
            initial_cash=self._initial_cash,
            benchmark=self._benchmark,
        )
        result = self._portfolio.run(spec)
        metrics_out = (
            segment_metrics(result, split.out_start, split.out_end) if result.ok else None
        )
        return PortfolioVariant(
            key=key,
            name=name,
            members=member_names,
            ok=result.ok,
            note=note,
            error=result.error,
            result=result,
            metrics_out=metrics_out,
        )


__all__ = [
    "DEFAULT_IN_START",
    "DEFAULT_IN_END",
    "DEFAULT_MA_WINDOWS",
    "DEFAULT_MA_REBALANCES",
    "DEFAULT_MA_TOP_N",
    "CANDIDATE_FAMILY_MA",
    "CANDIDATE_FAMILY_SCORED",
    "CANDIDATE_FAMILY_LABELS",
    "REBALANCE_LABELS",
    "VARIANT_ALL",
    "VARIANT_SELECTED_IN",
    "VARIANT_SELECTED_FULL",
    "VARIANT_BEST_SINGLE",
    "VARIANT_NAMES",
    "SELECTION_TABLE_COLUMNS",
    "SELECTION_COMPARISON_COLUMNS",
    "MemberCandidate",
    "SampleSplit",
    "SelectionCriteria",
    "MemberEvaluation",
    "SelectionReport",
    "PortfolioVariant",
    "SelectionComparison",
    "SelectionRunner",
    "build_candidate_pool",
    "slice_equity",
    "segment_metrics",
    "judge_selection",
]