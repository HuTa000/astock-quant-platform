"""分层回测（分位数组合，P7-S3）。

定位：
    IC 回答「因子值与未来收益的排名是否一致」（相关性视角），分层回测回答
    「按因子分档买，收益是不是一档比一档高」（组合视角）——
    把股票按因子值从低到高分成 N 层（默认 5 层：Q1 最低 … Q5 最高），
    每层等权持有，看：
    ① 各层净值是否**单调**（好因子：Q5>Q4>…>Q1，或反向）；
    ② **多空组合**（Q5 − Q1）是否稳定向上（因子多空收益）。

交易口径（与 P7-S1 回测完全一致，避免前视偏差）：
    * 信号日 T 收盘按因子值分组 → **T+1 开盘等权买入 → T+2 开盘轮动换仓**
      （每日再平衡），所以分层组合的日收益 = 组内成员
      ``open[T+2] / open[T+1] − 1`` 的等权均值；
    * 停牌 / 无 bar 的个股当日不可成交，自动不参与该日的组合；
    * 当日有效标的数不足（``< min_stocks``）或因子值全部相同（无区分度）
      → 该日不建仓（各层收益记 NaN，净值持平）。

净值口径：净值为「每日收益链乘」，首个有效交易日的净值 = ``1 + 当日收益``
（没有可成交日之前的占位 1.0）。

统计口径：各层年化收益 / 波动 / 夏普 / 最大回撤复用 P7-S1
:mod:`app.engine.backtest.metrics`（同口径）；**单调性** = 层序号（1..N）与
各层区间总收益的秩相关（+1 完全单调递增、−1 完全单调递减、0 无规律）。

对外函数**不抛业务异常**：无有效数据 → ``LayeredResult(ok=False, error=...)``。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.engine.backtest.metrics import (
    annual_return,
    annual_volatility,
    max_drawdown,
    sharpe_ratio,
    total_return,
)

logger = logging.getLogger(__name__)

# 分层统计表契约列。
GROUP_STATS_COLUMNS: list[str] = [
    "group",
    "days",
    "mean_daily_return",
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "max_drawdown",
]


@dataclass
class LayeredResult:
    """一次分层回测的产出（**不抛异常**，问题封装为 ``ok=False`` + ``error``）。

    Attributes:
        ok: 是否有足够数据完成分层。
        error: 失败原因（成功时为空串）。
        n_groups: 层数 N。
        group_names: 层名（``["Q1", ..., "QN"]``，Q1 = 因子值最低层）。
        group_nav: 各层净值表（dates × 层名，每日收益链乘）。
        group_stats: 各层统计表，契约列 :data:`GROUP_STATS_COLUMNS`。
        long_short_nav: 多空组合净值（``QN − Q1`` 每日收益链乘）。
        long_short_annual: 多空组合年化收益。
        long_short_sharpe: 多空组合夏普比率。
        monotonicity: 单调性（层序号 vs 区间总收益的秩相关，∈[−1, 1]）。
        days: 有效建仓交易日数（所有层共用的日数）。
        warnings: 过程警告（如大量交易日因样本不足被跳过）。
    """

    ok: bool
    error: str = ""
    n_groups: int = 5
    group_names: list[str] = field(default_factory=list)
    group_nav: pd.DataFrame = field(default_factory=pd.DataFrame)
    group_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    long_short_nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    long_short_annual: float | None = None
    long_short_sharpe: float | None = None
    monotonicity: float | None = None
    days: int = 0
    warnings: list[str] = field(default_factory=list)


def spearman_rank_corr(left: pd.Series, right: pd.Series) -> float | None:
    """两条等长序列的秩相关（排名后算皮尔逊相关；**不依赖 scipy**）。

    Args:
        left: 序列一。
        right: 序列二。

    Returns:
        秩相关系数 ∈ [−1, 1]；样本 < 2 / 方差为 0 / 非法输入返回 ``None``。
    """
    try:
        pair = pd.DataFrame(
            {
                "left": pd.to_numeric(pd.Series(left), errors="coerce"),
                "right": pd.to_numeric(pd.Series(right), errors="coerce"),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
    except (TypeError, ValueError):
        return None
    if len(pair) < 2:
        return None
    left_rank = pair["left"].rank().to_numpy(dtype=float)
    right_rank = pair["right"].rank().to_numpy(dtype=float)
    left_std = float(left_rank.std(ddof=0))
    right_std = float(right_rank.std(ddof=0))
    if left_std <= 0 or right_std <= 0:
        return None
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    corr = float((left_centered @ right_centered) / (len(pair) * left_std * right_std))
    return corr if np.isfinite(corr) else None


def run_layered(
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    n_groups: int = 5,
    min_stocks: int = 10,
) -> LayeredResult:
    """执行分层回测（**不抛业务异常**）。

    Args:
        factor_panel: 因子面板（dates × symbols）。
        forward_panel: **1 日前瞻收益**面板（``open[T+2]/open[T+1] − 1``，
            即 :meth:`~app.engine.factor.panel.FactorPanel.forward` 取 ``1``）。
        n_groups: 层数（≥2，默认 5）。
        min_stocks: 当日最少有效标的数（低于则该日不建仓）。

    Returns:
        :class:`LayeredResult`。
    """
    groups = int(n_groups) if isinstance(n_groups, (int, float)) else 5
    if groups < 2:
        return LayeredResult(ok=False, error=f"层数必须 ≥2，实际 {n_groups}")
    if not isinstance(factor_panel, pd.DataFrame) or not isinstance(forward_panel, pd.DataFrame):
        return LayeredResult(ok=False, error="输入面板非法", n_groups=groups)
    if factor_panel.empty or forward_panel.empty:
        return LayeredResult(ok=False, error="输入面板为空", n_groups=groups)
    common_dates = factor_panel.index.intersection(forward_panel.index)
    common_symbols = factor_panel.columns.intersection(forward_panel.columns)
    if len(common_dates) == 0 or len(common_symbols) == 0:
        return LayeredResult(ok=False, error="因子与收益面板没有公共日期或标的", n_groups=groups)
    factor = factor_panel.reindex(index=common_dates, columns=common_symbols)
    forward = forward_panel.reindex(index=common_dates, columns=common_symbols)

    threshold = max(int(min_stocks), groups)
    valid = factor.notna() & forward.notna()
    counts = valid.sum(axis=1)
    dispersion = factor.where(valid).std(axis=1, ddof=0)
    eligible = (counts >= threshold) & (dispersion > 0)

    # 等数量分档：按当日排名切 N 段（rank 的 method="first" 打散同分值，
    # 保证每层股数尽量相等），层序号 0..N-1 对应 Q1..QN。
    ranks = factor.where(valid).rank(axis=1, method="first")
    labels = np.ceil(ranks.mul(groups).div(counts.astype(float), axis=0)) - 1.0
    labels = labels.where(valid).clip(upper=groups - 1.0)

    group_names = [f"Q{index + 1}" for index in range(groups)]
    returns: dict[str, pd.Series] = {}
    for index, name in enumerate(group_names):
        member = (labels == float(index)) & valid
        member_count = member.sum(axis=1)
        summed = forward.where(member).sum(axis=1, skipna=True)
        averaged = summed / member_count.where(member_count > 0)
        returns[name] = averaged.where(eligible)
    return_frame = pd.DataFrame(returns, index=common_dates)
    if return_frame.dropna(how="all").empty:
        return LayeredResult(ok=False, error="无任何有效建仓日（样本不足）", n_groups=groups)

    active = return_frame.dropna(how="all")
    group_nav = (1.0 + active.fillna(0.0)).cumprod()

    stats_rows: list[dict] = []
    for name in group_names:
        nav = group_nav[name]
        nav_returns = active[name].dropna()
        stats_rows.append(
            {
                "group": name,
                "days": int(len(nav_returns)),
                "mean_daily_return": float(nav_returns.mean()) if len(nav_returns) else None,
                "total_return": total_return(nav),
                "annual_return": annual_return(nav),
                "annual_vol": annual_volatility(nav_returns),
                "sharpe": sharpe_ratio(nav_returns),
                "max_drawdown": max_drawdown(nav),
            }
        )
    group_stats = pd.DataFrame(stats_rows, columns=GROUP_STATS_COLUMNS)

    top, bottom = group_names[-1], group_names[0]
    long_short_returns = (active[top] - active[bottom]).dropna()
    long_short_nav = (1.0 + long_short_returns).cumprod()
    long_short_annual = annual_return(long_short_nav)
    long_short_sharpe = sharpe_ratio(long_short_returns)

    monotonicity = spearman_rank_corr(
        pd.Series(range(1, groups + 1), dtype=float),
        group_stats["total_return"],
    )

    warnings: list[str] = []
    skipped = int((~eligible).sum())
    if skipped > 0:
        warnings.append(
            f"{skipped} 个交易日因样本不足或无区分度未建仓（净值持平）"
        )
    return LayeredResult(
        ok=True,
        n_groups=groups,
        group_names=group_names,
        group_nav=group_nav,
        group_stats=group_stats,
        long_short_nav=long_short_nav,
        long_short_annual=long_short_annual,
        long_short_sharpe=long_short_sharpe,
        monotonicity=monotonicity,
        days=int(len(active)),
        warnings=warnings,
    )