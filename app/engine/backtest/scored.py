"""多因子打分选股回测编排（P7-S5）。

定位：
    把「多因子打分选股」从研究结论变成**可验证的组合**——
    :class:`~app.engine.factor.scoring.FactorScorer` 逐日打分取 Top-N，
    交给 :class:`~app.engine.backtest.engine.BacktestEngine` 按「T 日收盘出信号、
    T+1 开盘成交」逐日回放，回答阶段 5 的业务问题：**照这个分数买，钱能不能赚、
    赚的是 α 还是 β、在不同市场环境里稳不稳**。

两条防坑设计（与《开发手册》统一口径一致）：
    * **无前视**：名单在信号日 T 收盘产生（因子与 ICIR 权重都只用 ≤T 数据，
      见 scoring 模块「三道闸门」），引擎在 T+1 开盘成交——不存在「用今日收盘
      选股、按今日收盘价买入」；权重预热期没有得分 → 没有名单 → 空仓等待；
    * **无幸存者偏**（默认口径）：``universe="pit"`` 用成分股历史快照并集 +
      逐日掩码，非成分日 / 非成分股不参与打分；库内无快照时给出明确失败原因，
      需显式切换 ``universe="current"`` 才按「当前可见标的」口径回测。

成本 / 基准 / 交易规则全部复用回测引擎：100 股整手、全额再平衡、默认现实费率、
``sh.000300`` 基准（库内无该指数时降级为股票池等权代理并记警告）。

对外 API::

    from app.engine.backtest import ScoredBacktestRunner, ScoredBacktestSpec

    spec = ScoredBacktestSpec(start="2018-01-01", end="2026-10-09", top_n=30)
    result = ScoredBacktestRunner().run(spec)
    print(result.backtest.metrics.total_return, result.backtest.metrics.max_drawdown)
    print(result.backtest.regime)          # 牛/熊拆分
    print(result.scoring.latest_weights)   # 三因子最新权重诊断

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from app.data.store.db import Database, get_db
from app.data.universe import DEFAULT_INDEX_CODE
from app.engine.backtest.costs import CostModel
from app.engine.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    is_rebalance_day,
)
from app.engine.backtest.risk import RiskSpec
from app.engine.backtest.strategy import (
    DEFAULT_BENCHMARK,
    DEFAULT_INITIAL_CASH,
    REBALANCE_CHOICES,
    REBALANCE_MONTHLY,
    StrategySpec,
)
from app.engine.factor.scoring import (
    DEFAULT_IC_HORIZON,
    DEFAULT_MIN_IC_DAYS,
    DEFAULT_Z_CLIP,
    FactorScorer,
    ScoreReport,
    ScoringSpec,
    top_picks,
)

logger = logging.getLogger(__name__)

#: 默认每日持有只数（打分选股 Top-N）。
DEFAULT_SCORED_TOP_N: int = 30

#: 默认调仓频率：月频（与 20 日前瞻期权重口径匹配，降低换手与成本）。
DEFAULT_SCORED_REBALANCE: str = REBALANCE_MONTHLY

#: 默认股票池口径：PIT 成分股（消除幸存者偏差；无快照时显式失败而非静默降级）。
DEFAULT_SCORED_UNIVERSE: str = "pit"

#: ``universe`` 合法取值。
UNIVERSE_CHOICES: tuple[str, ...] = ("current", "pit")


@dataclass
class ScoredBacktestSpec:
    """一次「打分选股回测」的输入参数（打分口径 + 交易口径）。

    Attributes:
        name: 策略名称（展示用）。
        factors: 参与合成的因子名；``None`` = 核心三因子
            （:data:`~app.engine.factor.scoring.DEFAULT_SCORING_FACTORS`）。
        top_n: 每日持有只数（≥1，默认 :data:`DEFAULT_SCORED_TOP_N`）。
        rebalance: 调仓频率 ``daily``/``weekly``/``monthly``（默认月频）。
        universe: 股票池口径——``"pit"``（默认，成分股历史快照，消除幸存者偏差）/
            ``"current"``（当前可见标的）。
        index_code: PIT 口径使用的指数代码（默认沪深300）。
        start: 回测起始日（含）；``None`` = 不设下限（含权重预热期，前期空仓）。
        end: 回测结束日（含）；``None`` = 不设上限。
        ic_horizon: 权重估计前瞻期（交易日，默认 20）。
        min_ic_days: 扩展窗口最少 IC 样本数（默认 120，预热期内不建仓）。
        z_clip: 横截面 z 值截断边界（默认 ±3σ）。
        min_stocks: 横截面最少有效标的数（默认 10）。
        costs: 交易成本模型（默认现实费率）。
        benchmark: 基准代码（默认沪深300）。
        initial_cash: 初始资金（元，默认 100 万）。
        risk: 风控规格（止损 / 止盈 / 移动止盈 / 冷却期 / 牛熊环境仓位开关，
            :class:`~app.engine.backtest.risk.RiskSpec`）；``None``（默认）表示
            不启用任何风控——回放行为与不带风控的打分回测完全一致。

    Raises:
        ValueError: ``top_n`` / ``rebalance`` / ``universe`` / ``factors`` /
            ``costs`` / ``initial_cash`` / ``risk`` 非法。
    """

    name: str = "多因子打分选股"
    factors: Optional[Sequence[str]] = None
    top_n: int = DEFAULT_SCORED_TOP_N
    rebalance: str = DEFAULT_SCORED_REBALANCE
    universe: str = DEFAULT_SCORED_UNIVERSE
    index_code: str = DEFAULT_INDEX_CODE
    start: Any = None
    end: Any = None
    ic_horizon: int = DEFAULT_IC_HORIZON
    min_ic_days: int = DEFAULT_MIN_IC_DAYS
    z_clip: float = DEFAULT_Z_CLIP
    min_stocks: int = 10
    costs: CostModel = field(default_factory=CostModel)
    benchmark: str = DEFAULT_BENCHMARK
    initial_cash: float = DEFAULT_INITIAL_CASH
    risk: Optional[RiskSpec] = None

    def __post_init__(self) -> None:
        """构造后严格校验（属调用方输入错误 → 抛 :class:`ValueError`）。"""
        self.name = str(self.name).strip() or "多因子打分选股"

        try:
            top = int(self.top_n)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"top_n 必须是正整数：{self.top_n!r}") from exc
        if top <= 0:
            raise ValueError(f"top_n 必须为正整数：{self.top_n!r}")
        self.top_n = top

        rebalance_text = str(self.rebalance).strip().lower()
        if rebalance_text not in REBALANCE_CHOICES:
            raise ValueError(
                f"rebalance 必须是 {list(REBALANCE_CHOICES)} 之一，实际：{self.rebalance!r}"
            )
        self.rebalance = rebalance_text

        mode = str(self.universe or "pit").strip().lower()
        if mode not in UNIVERSE_CHOICES:
            raise ValueError(
                f"universe 必须是 {list(UNIVERSE_CHOICES)} 之一，实际：{self.universe!r}"
            )
        self.universe = mode

        if self.factors is not None:
            cleaned = tuple(str(name).strip() for name in self.factors if str(name).strip())
            self.factors = cleaned or None

        self.index_code = str(self.index_code).strip() or DEFAULT_INDEX_CODE
        if not isinstance(self.costs, CostModel):
            raise ValueError(f"costs 必须是 CostModel，实际：{type(self.costs).__name__}")

        try:
            cash = float(self.initial_cash)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"initial_cash 必须是数值：{self.initial_cash!r}") from exc
        if cash <= 0:
            raise ValueError(f"initial_cash 必须为正数：{self.initial_cash!r}")
        self.initial_cash = cash

        # 风控规格：None = 不启用（默认）；类型错误属调用方输入错误。
        if self.risk is not None and not isinstance(self.risk, RiskSpec):
            raise ValueError(
                f"risk 必须是 RiskSpec 或 None，实际：{type(self.risk).__name__}"
            )


@dataclass
class ScoredBacktestResult:
    """一次打分选股回测的完整产出（**不抛业务异常**，问题封装为 ``ok=False``）。

    Attributes:
        ok: 是否成功完成（打分与回测都成功）。
        error: 失败原因（成功时为空串）。
        backtest: 回测结果（:class:`~app.engine.backtest.engine.BacktestResult`；
            打分失败时为 ``None``，回测失败时 ``ok=False`` 但保留现场）。
        scoring: 打分报告（:class:`~app.engine.factor.scoring.ScoreReport`）。
        picks: 实际用于回测的每日名单 ``{信号日: [标的, ...]}``。
        warnings: 打分 + 回测全过程警告汇总（口径降级、预热、停牌未成交等）。
    """

    ok: bool
    error: str = ""
    backtest: Optional[BacktestResult] = None
    scoring: Optional[ScoreReport] = None
    picks: dict[pd.Timestamp, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ScoredBacktestRunner:
    """打分选股回测编排器（只读本地库，不依赖 Qt）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        scorer: Optional[FactorScorer] = None,
        engine: Optional[BacktestEngine] = None,
    ) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`~app.data.store.db.get_db`。
            scorer: 打分器；默认用同一个 ``db`` 构造 :class:`FactorScorer`。
            engine: 回测引擎；默认用同一个 ``db`` 构造 :class:`BacktestEngine`。
        """
        self._db: Database = db if db is not None else get_db()
        self._scorer: FactorScorer = scorer if scorer is not None else FactorScorer(self._db)
        self._engine: BacktestEngine = (
            engine if engine is not None else BacktestEngine(self._db)
        )

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def run(self, spec: ScoredBacktestSpec) -> ScoredBacktestResult:
        """执行「打分 → Top-N 名单 → 回测」（**不抛业务异常**）。

        Args:
            spec: 打分选股回测参数（:class:`ScoredBacktestSpec`）。

        Returns:
            :class:`ScoredBacktestResult`。

        Raises:
            TypeError: ``spec`` 类型错误（调用方编程错误）。
            ValueError: 打分口径参数非法（由打分器抛出）。
        """
        if not isinstance(spec, ScoredBacktestSpec):
            raise TypeError(f"spec 必须是 ScoredBacktestSpec，实际：{type(spec).__name__}")

        # 1) 多因子打分（PIT 掩码 / 权重预热 / 缺失口径全部在打分层完成）。
        report = self._scorer.run(
            ScoringSpec(
                factor_names=spec.factors,
                ic_horizon=spec.ic_horizon,
                min_ic_days=spec.min_ic_days,
                z_clip=spec.z_clip,
                min_stocks=spec.min_stocks,
                start=spec.start,
                end=spec.end,
                universe=spec.universe,
                index_code=spec.index_code,
            )
        )
        if not report.ok:
            return ScoredBacktestResult(
                ok=False,
                error=report.error,
                scoring=report,
                warnings=list(report.warnings),
            )

        # 2) 逐日 Top-N 名单 → 回测规格（受限池 = 打分实际用到的标的）。
        picks = top_picks(report.score, spec.top_n)
        if not picks:
            return ScoredBacktestResult(
                ok=False,
                error=(
                    "打分窗口内没有任何有效得分，未产生调仓名单"
                    "（请扩大窗口或补充历史数据）"
                ),
                scoring=report,
                warnings=list(report.warnings),
            )
        strategy = StrategySpec(
            name=spec.name,
            conditions=[],
            symbols=list(report.universe),
            rebalance=spec.rebalance,
            daily_picks=picks,
            costs=spec.costs,
            benchmark=spec.benchmark,
            initial_cash=spec.initial_cash,
            risk=spec.risk,
        )

        # 3) 逐日回放（T 日名单 → T+1 开盘成交、整手、成本、停牌跳过，全在引擎内）。
        backtest = self._engine.run(strategy, start=spec.start, end=spec.end)
        warnings = list(report.warnings) + list(backtest.warnings)

        # 4) 诊断：调仓信号日无名单 = 当日空仓（预热期 / 数据缺失），如实告知。
        if backtest.ok and not backtest.equity.empty:
            calendar = pd.DatetimeIndex(pd.to_datetime(backtest.equity["date"]))
            missing = self._missing_signal_days(calendar, spec.rebalance, picks)
            if missing:
                warnings.append(
                    f"有 {missing} 个调仓信号日没有有效名单（权重预热 / 数据缺失），"
                    "这些调仓日按空仓处理"
                )

        return ScoredBacktestResult(
            ok=backtest.ok,
            error=backtest.error,
            backtest=backtest,
            scoring=report,
            picks=picks,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # 内部：诊断（不抛业务异常）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _missing_signal_days(
        calendar: pd.DatetimeIndex,
        rebalance: str,
        picks: dict[pd.Timestamp, list[str]],
    ) -> int:
        """统计「是调仓信号日但没有名单」的天数（口径与引擎逐日回放一致）。"""
        if len(calendar) < 2:
            return 0
        missing = 0
        for index in range(len(calendar) - 1):
            if not is_rebalance_day(index, calendar, rebalance):
                continue
            if calendar[index] not in picks:
                missing += 1
        return missing


__all__ = [
    "DEFAULT_SCORED_TOP_N",
    "DEFAULT_SCORED_REBALANCE",
    "DEFAULT_SCORED_UNIVERSE",
    "UNIVERSE_CHOICES",
    "ScoredBacktestSpec",
    "ScoredBacktestResult",
    "ScoredBacktestRunner",
]