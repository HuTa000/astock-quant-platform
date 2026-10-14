"""策略回测引擎（P7-S1）。

对外出口：
    * :class:`StrategySpec` / :data:`REBALANCE_CHOICES` / :data:`DEFAULT_BENCHMARK`：
      策略规格（选股条件 + 调仓频率 + 成本与基准口径，:mod:`app.engine.backtest.strategy`）；
    * :class:`CostModel` / :data:`DEFAULT_COST_MODEL` / :data:`ZERO_COST_MODEL` /
      :data:`BUY` / :data:`SELL`：A 股成本模型（:mod:`app.engine.backtest.costs`）；
    * :class:`PerformanceMetrics` / :func:`compute_metrics` 及全部单指标函数：
      绩效指标纯函数库（:mod:`app.engine.backtest.metrics`）；
    * :func:`load_benchmark` / :data:`BENCHMARK_KIND_INDEX` /
      :data:`BENCHMARK_KIND_PROXY`：基准净值（真实指数 / 等权代理，
      :mod:`app.engine.backtest.benchmark`）；
    * :func:`classify_regime` / :func:`regime_summary` / :data:`REGIME_LABELS`：
      牛熊环境拆分（:mod:`app.engine.backtest.regime`）；
    * :class:`RiskSpec` / :func:`evaluate_exit` / :data:`RISK_REASON_LABELS` 及全部
      风控常量：策略风控与交易规则（止损 / 固定止盈 / 移动止盈 / 冷却期 /
      牛熊环境仓位开关，P7-S6，:mod:`app.engine.backtest.risk`）；
    * :func:`yearly_summary` / :func:`drawdown_episodes` / :func:`jensen_decomposition`：
      绩效归因（逐年拆解 / 回撤区间明细 / α·β 收益来源分解，
      :mod:`app.engine.backtest.attribution`）；
    * :class:`BacktestEngine` / :class:`BacktestResult`：回测主引擎
      （T+1 开盘成交 / 整手 / 全额再平衡 / 停牌与一字板跳过，
      :mod:`app.engine.backtest.engine`）；
    * :class:`ScoredBacktestRunner` / :class:`ScoredBacktestSpec` /
      :class:`ScoredBacktestResult`：多因子打分选股回测编排
      （ICIR 加权打分 Top-N → 预计算名单 → 引擎回放，P7-S5，
      :mod:`app.engine.backtest.scored`）；
    * :class:`PortfolioRunner` / :class:`PortfolioSpec` / :class:`PortfolioResult` /
      :class:`PortfolioMember` / :class:`PortfolioMemberResult`：多策略组合回测
      （资金分账·等权、子策略净值合并、相关矩阵、贡献归因，P7-S7，
      :mod:`app.engine.backtest.portfolio`）；
    * :class:`SelectionRunner` / :class:`MemberCandidate` / :class:`SampleSplit` /
      :class:`SelectionCriteria` / :class:`SelectionReport` / :func:`build_candidate_pool`：
      策略成员筛选与筛选后组合验证（样本内筛选 + 样本外验证、绝对门槛 + 分年度
      稳定性、四线组合对比，P7-S8，:mod:`app.engine.backtest.member_selection`）。

典型用法::

    from app.engine.backtest import BacktestEngine, StrategySpec
    from app.engine.screener import Condition

    spec = StrategySpec(
        name="站上20日线",
        conditions=[Condition("close - close.rolling(20).mean()", ">", 0)],
        rebalance="weekly",
        top_n=3,
    )
    result = BacktestEngine().run(spec, start="2020-01-01", end="2025-12-31")
    print(result.metrics.sharpe, result.metrics.max_drawdown)
    print(result.regime)          # 牛/熊拆分

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）。
"""
from __future__ import annotations

from app.engine.backtest.attribution import (
    DECOMPOSITION_KEYS,
    DRAWDOWN_COLUMNS,
    YEARLY_COLUMNS,
    drawdown_episodes,
    jensen_decomposition,
    yearly_summary,
)
from app.engine.backtest.benchmark import (
    BENCHMARK_KIND_INDEX,
    BENCHMARK_KIND_PROXY,
    equal_weight_nav,
    index_nav,
    load_benchmark,
)
from app.engine.backtest.costs import (
    BUY,
    DEFAULT_COST_MODEL,
    SELL,
    ZERO_COST_MODEL,
    CostModel,
)
from app.engine.backtest.engine import (
    EQUITY_COLUMNS,
    LOT_SIZE,
    BacktestEngine,
    BacktestResult,
)
from app.engine.backtest.metrics import (
    TRADE_COLUMNS,
    TRADING_DAYS_PER_YEAR,
    PerformanceMetrics,
    alpha_beta,
    annual_return,
    annual_volatility,
    calmar_ratio,
    closed_trades,
    compute_metrics,
    drawdown_series,
    information_ratio,
    max_drawdown,
    nav_returns,
    profit_loss_ratio,
    sharpe_ratio,
    total_return,
    tracking_error,
    trade_count,
    win_rate,
)
from app.engine.backtest.member_selection import (
    CANDIDATE_FAMILY_LABELS,
    CANDIDATE_FAMILY_MA,
    CANDIDATE_FAMILY_SCORED,
    DEFAULT_IN_END,
    DEFAULT_IN_START,
    DEFAULT_MA_REBALANCES,
    DEFAULT_MA_TOP_N,
    DEFAULT_MA_WINDOWS,
    REBALANCE_LABELS,
    SELECTION_COMPARISON_COLUMNS,
    SELECTION_TABLE_COLUMNS,
    VARIANT_ALL,
    VARIANT_BEST_SINGLE,
    VARIANT_NAMES,
    VARIANT_SELECTED_FULL,
    VARIANT_SELECTED_IN,
    MemberCandidate,
    MemberEvaluation,
    PortfolioVariant,
    SampleSplit,
    SelectionComparison,
    SelectionCriteria,
    SelectionReport,
    SelectionRunner,
    build_candidate_pool,
    judge_selection,
    segment_metrics,
    slice_equity,
)
from app.engine.backtest.portfolio import (
    CONTRIBUTION_COLUMNS,
    MEMBER_KIND_CONDITION,
    MEMBER_KIND_SCORED,
    PORTFOLIO_TRADE_COLUMNS,
    PortfolioMember,
    PortfolioMemberResult,
    PortfolioResult,
    PortfolioRunner,
    PortfolioSpec,
)
from app.engine.backtest.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_LABELS,
    classify_regime,
    regime_summary,
)
from app.engine.backtest.risk import (
    DEFAULT_COOLDOWN_DAYS,
    DEFAULT_REGIME_EXPOSURE,
    DEFAULT_REGIME_WINDOW,
    DEFAULT_STOP_LOSS_PCT,
    RISK_REASON_LABELS,
    RISK_STOP_LOSS,
    RISK_TAKE_PROFIT,
    RISK_TRAILING_STOP,
    RiskSpec,
    evaluate_exit,
)
from app.engine.backtest.scored import (
    DEFAULT_SCORED_REBALANCE,
    DEFAULT_SCORED_TOP_N,
    DEFAULT_SCORED_UNIVERSE,
    UNIVERSE_CHOICES,
    ScoredBacktestResult,
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

__all__ = [
    # 策略规格
    "StrategySpec",
    "REBALANCE_DAILY",
    "REBALANCE_WEEKLY",
    "REBALANCE_MONTHLY",
    "REBALANCE_CHOICES",
    "DEFAULT_BENCHMARK",
    "DEFAULT_INITIAL_CASH",
    # 成本模型
    "CostModel",
    "DEFAULT_COST_MODEL",
    "ZERO_COST_MODEL",
    "BUY",
    "SELL",
    # 绩效指标
    "TRADING_DAYS_PER_YEAR",
    "TRADE_COLUMNS",
    "PerformanceMetrics",
    "nav_returns",
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "drawdown_series",
    "max_drawdown",
    "calmar_ratio",
    "alpha_beta",
    "tracking_error",
    "information_ratio",
    "closed_trades",
    "trade_count",
    "win_rate",
    "profit_loss_ratio",
    "compute_metrics",
    # 基准
    "BENCHMARK_KIND_INDEX",
    "BENCHMARK_KIND_PROXY",
    "index_nav",
    "equal_weight_nav",
    "load_benchmark",
    # 牛熊环境
    "REGIME_BULL",
    "REGIME_BEAR",
    "REGIME_LABELS",
    "classify_regime",
    "regime_summary",
    # 风控与交易规则（P7-S6）
    "RiskSpec",
    "evaluate_exit",
    "RISK_STOP_LOSS",
    "RISK_TAKE_PROFIT",
    "RISK_TRAILING_STOP",
    "RISK_REASON_LABELS",
    "DEFAULT_STOP_LOSS_PCT",
    "DEFAULT_COOLDOWN_DAYS",
    "DEFAULT_REGIME_WINDOW",
    "DEFAULT_REGIME_EXPOSURE",
    # 绩效归因
    "YEARLY_COLUMNS",
    "DRAWDOWN_COLUMNS",
    "DECOMPOSITION_KEYS",
    "yearly_summary",
    "drawdown_episodes",
    "jensen_decomposition",
    # 回测引擎
    "LOT_SIZE",
    "EQUITY_COLUMNS",
    "BacktestEngine",
    "BacktestResult",
    # 打分选股回测（P7-S5）
    "DEFAULT_SCORED_TOP_N",
    "DEFAULT_SCORED_REBALANCE",
    "DEFAULT_SCORED_UNIVERSE",
    "UNIVERSE_CHOICES",
    "ScoredBacktestSpec",
    "ScoredBacktestResult",
    "ScoredBacktestRunner",
    # 多策略组合（P7-S7）
    "MEMBER_KIND_CONDITION",
    "MEMBER_KIND_SCORED",
    "PORTFOLIO_TRADE_COLUMNS",
    "CONTRIBUTION_COLUMNS",
    "PortfolioMember",
    "PortfolioSpec",
    "PortfolioMemberResult",
    "PortfolioResult",
    "PortfolioRunner",
    # 成员筛选与筛选后组合验证（P7-S8）
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