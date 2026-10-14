"""多策略组合回测（P7-S7）。

定位：
    把多个**已能独立回测**的策略拼成一个组合，回答阶段 7 的核心业务问题：
    「单策略各自能赚，拼在一起是不是更稳？」——采用**资金分账·等权**口径回放：
    组合总资金按权重切成 N 份子账户，各策略在自己的子账户里独立调仓、独立记账，
    组合净值 = Σ 子账户资产 / 组合总资金。输出四张关键结果表：
    ① 子策略净值对照（成员各自的收益 / 回撤 / 夏普，组合层不改写）；
    ② 相关性矩阵（成员日收益两两相关，越低越能互相对冲、分散越有效）；
    ③ 贡献归因表（每个成员给组合赚了多少、在组合最大回撤里各"贡献"了多少跌幅）；
    ④ 组合合并净值与绩效（基准对比、牛熊拆分、成交明细）。

设计要点（口径与金融数据坑的对应处理）：
    * **资金分账 / 不再平衡**：成员初始资金 = 归一化权重 × 组合总资金，子账户之间
      **互不挪用资金、不做再平衡**——权重随各自净值自然漂移。这是「策略组合」最
      诚实的还原口径（区别于「单账户混合调仓」：后者会让强势策略被动吃掉更多资金）；
    * **无前视 / 无幸存者偏**：组合层只做**结果合并**，不重新选股——成员的信号
      因果性（T 日收盘出信号、T+1 开盘成交、打分成员默认 PIT 股票池）全部由成员
      自身回测保证（见 :mod:`app.engine.backtest.engine` / ``scored``）；
    * **日历对齐**：各成员回测日历不完全一致时，组合日历取**并集**：成员缺失日按
      「前值照延」估值为近似、成员首个净值日之前按**闲置现金**计入，并记警告
      （共同交易日越多结论越可靠）；**相关性用两两共同交易日的真实日收益**计算，
      不受照延影响；
    * **基准**：组合基准沿用**首个成员**的基准口径——同一 ``benchmark`` 下指数口径
      各成员完全一致；库内无指数时为首个成员股票池的「等权代理」（记警告）；
    * **成本**：各子账户按各自成员的成本模型独立扣费，组合层**不重复扣费**；
    * 成员各自的风控规格（``risk``）**保留生效**；``start`` / ``end`` / ``benchmark``
      一律被组合级参数覆盖（保证同窗口、同基准的可比性），子账户初始资金由权重决定。

成员规格（:class:`PortfolioMember`）支持两种现有入口：
    * :class:`~app.engine.backtest.strategy.StrategySpec`（条件式选股）→ 标记 ``kind="condition"``；
    * :class:`~app.engine.backtest.scored.ScoredBacktestSpec`（多因子打分选股）→ 标记 ``kind="scored"``。

对外 API::

    from app.engine.backtest import (
        PortfolioMember, PortfolioRunner, PortfolioSpec,
    )

    spec = PortfolioSpec(
        name="条件式 + 打分 等权组合",
        members=[
            PortfolioMember("站上20日线", condition_spec),
            PortfolioMember("多因子打分", scored_spec),
        ],                       # 权重缺省等权（1/N）
        start="2018-01-01", end="2026-10-09",
    )
    result = PortfolioRunner().run(spec)
    print(result.metrics.total_return, result.metrics.max_drawdown)
    print(result.correlation)    # 成员日收益相关矩阵
    print(result.contribution)   # 成员贡献归因（含最大回撤日分解）

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field, replace
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from app.data.store.db import Database, get_db
from app.engine.backtest.benchmark import BENCHMARK_KIND_PROXY
from app.engine.backtest.engine import (
    EQUITY_COLUMNS,
    BacktestEngine,
    BacktestResult,
)
from app.engine.backtest.metrics import (
    TRADE_COLUMNS,
    TRADING_DAYS_PER_YEAR,
    PerformanceMetrics,
    compute_metrics,
    drawdown_series,
    nav_returns,
)
from app.engine.backtest.regime import classify_regime, regime_summary
from app.engine.backtest.scored import (
    ScoredBacktestResult,
    ScoredBacktestRunner,
    ScoredBacktestSpec,
)
from app.engine.backtest.strategy import (
    DEFAULT_BENCHMARK,
    DEFAULT_INITIAL_CASH,
    StrategySpec,
)
from app.engine.factor.scoring import ScoreReport

logger = logging.getLogger(__name__)

# 成员类型标记（结果表 / UI 展示用：「条件式选股」/「打分选股」）。
MEMBER_KIND_CONDITION: str = "condition"
MEMBER_KIND_SCORED: str = "scored"

# 组合成交明细契约列 = 引擎成交契约列 + 成员名（最后一列，便于按成员拆账）。
PORTFOLIO_TRADE_COLUMNS: list[str] = TRADE_COLUMNS + ["member"]

# 贡献归因表契约列：
#   contribution   = 成员盈亏 / 组合总资金（与组合总收益同口径，Σ = 组合总收益）
#   dd_contribution= 组合「峰值日 → 最大回撤谷底日」区间内该成员的资产变化 / 峰值日组合资产
#                    （Σ = 组合最大回撤幅度，与 metrics.max_drawdown 同口径；无回撤时恒 0）
CONTRIBUTION_COLUMNS: list[str] = [
    "member",
    "weight",
    "initial_cash",
    "final_assets",
    "pnl",
    "member_return",
    "contribution",
    "dd_contribution",
]


# --------------------------------------------------------------------------- #
# 输入规格
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioMember:
    """组合成员：一个策略规格 + 其资金权重。

    Attributes:
        name: 成员名称（组合内**唯一**，用于归因表 / 成交明细标识；
            留空时回退为策略规格自身的名称）。
        spec: 成员策略规格，:class:`StrategySpec`（条件式）或
            :class:`ScoredBacktestSpec`（多因子打分）；其 ``start`` / ``end`` /
            ``benchmark`` / ``initial_cash`` 会被组合级参数覆盖（见 :class:`PortfolioSpec`）。
        weight: 资金权重（正数，默认 1.0）；组合构造时按全体权重之和**归一化**
            （如 3:1 → 75% / 25%），缺省全部 1.0 即等权。

    Raises:
        ValueError: ``spec`` 类型不支持 / ``weight`` 非正数或非有限值。
    """

    name: str
    spec: Union[StrategySpec, ScoredBacktestSpec]
    weight: float = 1.0

    def __post_init__(self) -> None:
        """构造后校验（属调用方输入错误 → 抛 :class:`ValueError`）。"""
        if not isinstance(self.spec, (StrategySpec, ScoredBacktestSpec)):
            raise ValueError(
                "成员 spec 必须是 StrategySpec 或 ScoredBacktestSpec，"
                f"实际：{type(self.spec).__name__}"
            )
        self.name = str(self.name).strip() or str(self.spec.name).strip() or "未命名成员"
        try:
            weight = float(self.weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"成员权重必须是数值：{self.weight!r}") from exc
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"成员权重必须是正数：{self.weight!r}")
        self.weight = weight

    @property
    def kind(self) -> str:
        """成员类型标记：``"scored"``（打分）或 ``"condition"``（条件式）。"""
        return MEMBER_KIND_SCORED if isinstance(self.spec, ScoredBacktestSpec) else MEMBER_KIND_CONDITION


@dataclass
class PortfolioSpec:
    """一次「多策略组合回测」的输入参数（组合级资金与窗口口径）。

    Attributes:
        name: 组合名称（展示用）。
        members: 成员列表（:class:`PortfolioMember`，至少一个、名称互不重复）；
            构造时权重自动归一化（总和 = 1.0）。
        start: 组合回测起始日（含）——**覆盖**全体成员的 ``start``；``None`` = 不设下限。
        end: 组合回测结束日（含）——**覆盖**全体成员的 ``end``；``None`` = 不设上限。
        initial_cash: 组合总资金（元，默认 100 万）；子账户初始资金 = 权重 × 该值。
        benchmark: 组合基准代码（默认沪深300）——**覆盖**全体成员的 ``benchmark``。

    Raises:
        ValueError: 成员为空 / 成员类型非法 / 成员名称重复 / 权重非法 / 总资金非正。
    """

    name: str = "多策略组合"
    members: list[PortfolioMember] = dc_field(default_factory=list)
    start: Any = None
    end: Any = None
    initial_cash: float = DEFAULT_INITIAL_CASH
    benchmark: str = DEFAULT_BENCHMARK

    def __post_init__(self) -> None:
        """构造后严格校验：成员契约 + 权重归一化（权重随净值漂移、子账户间不再平衡）。"""
        self.name = str(self.name).strip() or "多策略组合"

        raw = list(self.members or [])
        if not raw:
            raise ValueError("组合至少需要一个成员策略（members 不能为空）")
        for member in raw:
            if not isinstance(member, PortfolioMember):
                raise ValueError(
                    f"成员必须是 PortfolioMember，实际：{type(member).__name__}"
                )
        names = [member.name for member in raw]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f"组合成员名称重复（归因表需要唯一标识）：{duplicated}")

        weight_sum = float(sum(member.weight for member in raw))
        if not np.isfinite(weight_sum) or weight_sum <= 0:  # pragma: no cover - 成员已校验正数
            raise ValueError(f"成员权重之和必须为正数：{weight_sum!r}")
        self.members = [
            replace(member, weight=member.weight / weight_sum) for member in raw
        ]

        try:
            cash = float(self.initial_cash)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"initial_cash 必须是数值：{self.initial_cash!r}") from exc
        if not np.isfinite(cash) or cash <= 0:
            raise ValueError(f"initial_cash 必须为正数：{self.initial_cash!r}")
        self.initial_cash = cash

        self.benchmark = str(self.benchmark).strip() or DEFAULT_BENCHMARK


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioMemberResult:
    """单个成员在组合中的执行产出（**不抛业务异常**，问题封装为 ``ok=False``）。

    Attributes:
        name: 成员名称（与 :class:`PortfolioMember` 一致）。
        kind: 成员类型标记（``"condition"`` / ``"scored"``）。
        weight: 归一化后的资金权重。
        initial_cash: 子账户初始资金（= 权重 × 组合总资金）。
        ok: 该成员是否回测成功。
        error: 失败原因（成功时为空串）。
        backtest: 成员回测结果（失败时为 ``None``；打分成员在这里也是引擎层结果）。
        scoring: 打分报告（仅打分成员有；条件式成员为 ``None``）。
        warnings: 该成员回测的原始警告（组合层汇总时会加成员名前缀）。
    """

    name: str
    kind: str
    weight: float
    initial_cash: float
    ok: bool = False
    error: str = ""
    backtest: Optional[BacktestResult] = None
    scoring: Optional[ScoreReport] = None
    warnings: list[str] = dc_field(default_factory=list)


@dataclass
class PortfolioResult:
    """一次组合回测的完整产出（**不抛业务异常**，问题封装为 ``ok=False`` + ``error``）。

    Attributes:
        ok: 是否成功完成组合回测（**任一成员失败即整体失败**，error 指明成员）。
        error: 失败原因（成功时为空串）。
        members: 成员执行产出清单（:class:`PortfolioMemberResult`；失败时保留现场）。
        equity: 组合净值表，契约列 :data:`EQUITY_COLUMNS`（``nav`` 组合净值、
            ``benchmark_nav`` 基准、``drawdown`` 回撤；首日净值 1.0）。
        trades: 组合成交明细，契约列见 :data:`PORTFOLIO_TRADE_COLUMNS`
            （引擎契约列 + ``member`` 成员名，按日期排序）。
        metrics: 组合绩效指标（:class:`PerformanceMetrics`）。
        regime: 组合牛 / 熊拆分表（契约列见 :mod:`app.engine.backtest.regime`）。
        correlation: 成员日收益相关矩阵（行 / 列为成员名；两两用**共同交易日**的
            真实收益计算，样本不足时为 NaN；单成员时为 1×1 对角 1.0）。
        contribution: 成员贡献归因表，契约列 :data:`CONTRIBUTION_COLUMNS`
            （Σ ``contribution`` = 组合总收益；Σ ``dd_contribution`` = 组合最大回撤）。
        dd_window: 组合最大回撤的「峰值日 → 谷底日」；无回撤 / 样本不足时为 ``None``。
        benchmark_symbol: 组合基准代码。
        benchmark_kind: 组合基准口径（``INDEX`` / ``EQUAL_WEIGHT_PROXY``，
            沿用首个成员的判定结果）。
        warnings: 全过程警告汇总（成员警告带 ``[成员名]`` 前缀）。
    """

    ok: bool
    error: str = ""
    members: list[PortfolioMemberResult] = dc_field(default_factory=list)
    equity: pd.DataFrame = dc_field(
        default_factory=lambda: pd.DataFrame(columns=EQUITY_COLUMNS)
    )
    trades: pd.DataFrame = dc_field(
        default_factory=lambda: pd.DataFrame(columns=PORTFOLIO_TRADE_COLUMNS)
    )
    metrics: PerformanceMetrics = dc_field(default_factory=PerformanceMetrics)
    regime: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    correlation: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    contribution: pd.DataFrame = dc_field(
        default_factory=lambda: pd.DataFrame(columns=CONTRIBUTION_COLUMNS)
    )
    dd_window: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None
    benchmark_symbol: str = ""
    benchmark_kind: str = ""
    warnings: list[str] = dc_field(default_factory=list)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _parse_bound(value: Any) -> Optional[pd.Timestamp]:
    """把起始 / 结束日解析为当日零点 ``Timestamp``；``None`` 原样返回（口径与回测引擎一致）。"""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法解析日期：{value!r}") from exc
    if pd.isna(stamp):
        raise ValueError(f"无法解析日期：{value!r}")
    return stamp.normalize()


def _equity_nav(equity_frame: pd.DataFrame) -> pd.Series:
    """净值表 → 以日期为索引的 float 净值 Series。"""
    index = pd.DatetimeIndex(pd.to_datetime(equity_frame["date"]))
    values = pd.to_numeric(equity_frame["nav"], errors="coerce").to_numpy(dtype=float)
    return pd.Series(values, index=index)


# --------------------------------------------------------------------------- #
# 组合编排器
# --------------------------------------------------------------------------- #
class PortfolioRunner:
    """多策略组合回测编排器（只读本地库，不依赖 Qt）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        engine: Optional[BacktestEngine] = None,
        scored_runner: Optional[ScoredBacktestRunner] = None,
    ) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`~app.data.store.db.get_db`。
            engine: 条件式成员的回测引擎；默认用同一个 ``db`` 构造。
            scored_runner: 打分成员的回测编排器；默认用同一个 ``db`` / ``engine`` 构造。
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

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def run(self, spec: PortfolioSpec) -> PortfolioResult:
        """执行「分账 → 逐成员回测 → 合并净值 / 相关 / 归因」（**不抛业务异常**）。

        Args:
            spec: 组合参数（:class:`PortfolioSpec`）。

        Returns:
            :class:`PortfolioResult`。

        Raises:
            TypeError: ``spec`` 类型错误（调用方编程错误）。
            ValueError: 起止日期无法解析或起始晚于结束（调用方输入错误）。
        """
        if not isinstance(spec, PortfolioSpec):
            raise TypeError(f"spec 必须是 PortfolioSpec，实际：{type(spec).__name__}")
        start_ts = _parse_bound(spec.start)
        end_ts = _parse_bound(spec.end)
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError(f"起始日期晚于结束日期：{start_ts.date()} > {end_ts.date()}")

        total_cash = float(spec.initial_cash)
        warnings: list[str] = []
        # 非等权时显式说明分账口径（透明度；等权为默认口径无需重复）。
        weights = [member.weight for member in spec.members]
        if len(weights) > 1 and not np.allclose(weights, weights[0]):
            detail = "、".join(
                f"{member.name} {member.weight:.1%}" for member in spec.members
            )
            warnings.append(
                f"组合按输入权重归一化分账：{detail}"
                "（权重随各自净值漂移，子账户间不做再平衡）"
            )

        # 1) 逐成员回测（各自子账户；规格的窗口 / 基准被组合级参数覆盖）。
        outcomes: list[PortfolioMemberResult] = []
        for member in spec.members:
            outcome = self._run_member(member, spec, start_ts, end_ts, total_cash, warnings)
            outcomes.append(outcome)
            if not outcome.ok:
                warnings.extend(
                    f"[{outcome.name}] {text}" for text in outcome.warnings
                )
                return PortfolioResult(
                    ok=False,
                    error=f"成员「{outcome.name}」回测失败：{outcome.error}",
                    members=outcomes,
                    benchmark_symbol=spec.benchmark,
                    warnings=warnings,
                )

        # 2) 子账户净值 → 资产 → 组合净值（日历取并集，缺失日照延 / 期初现金）。
        equity_frame, assets_frame, nav_series, calendar = self._merge_nav(
            outcomes, total_cash, warnings
        )

        # 3) 基准（沿用首个成员的基准口径）与组合绩效 / 牛熊拆分。
        design = outcomes[0]
        benchmark_nav = self._member_benchmark(design, calendar)
        equity_frame.insert(2, "benchmark_nav", benchmark_nav.to_numpy(dtype=float))
        if design.backtest.benchmark_kind == BENCHMARK_KIND_PROXY:
            warnings.append(
                "组合基准为「首个成员股票池的等权代理」口径（库内无真实指数数据），"
                "代理池 ≠ 组合全池，结论须标注口径"
            )
        trades = self._merge_trades(outcomes)
        metrics = compute_metrics(
            nav_series, benchmark_nav, trades, TRADING_DAYS_PER_YEAR
        )
        regime_frame = regime_summary(
            nav_series, benchmark_nav, classify_regime(benchmark_nav)
        )

        # 4) 相关性矩阵（两两共同交易日的真实日收益）+ 贡献归因（含最大回撤日分解）。
        correlation = self._correlation(outcomes)
        contribution, dd_window = self._contribution(
            outcomes, assets_frame, nav_series, total_cash
        )

        # 5) 警告汇总（成员警告带成员名前缀）。
        for outcome in outcomes:
            warnings.extend(f"[{outcome.name}] {text}" for text in outcome.warnings)

        return PortfolioResult(
            ok=True,
            members=outcomes,
            equity=equity_frame,
            trades=trades,
            metrics=metrics,
            regime=regime_frame,
            correlation=correlation,
            contribution=contribution,
            dd_window=dd_window,
            benchmark_symbol=spec.benchmark,
            benchmark_kind=design.backtest.benchmark_kind,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # 内部：单成员
    # ------------------------------------------------------------------ #
    def _run_member(
        self,
        member: PortfolioMember,
        spec: PortfolioSpec,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
        total_cash: float,
        warnings: list[str],
    ) -> PortfolioMemberResult:
        """回测单个成员（组合级窗口 / 基准覆盖其规格；子账户资金 = 权重 × 总资金）。"""
        sub_cash = member.weight * total_cash
        if (
            str(member.spec.benchmark).strip()
            and str(member.spec.benchmark).strip() != spec.benchmark
        ):
            warnings.append(
                f"成员「{member.name}」的基准 {member.spec.benchmark} 已被组合基准 "
                f"{spec.benchmark} 覆盖（保证可比性）"
            )

        if member.kind == MEMBER_KIND_SCORED:
            member_spec = replace(
                member.spec,
                start=spec.start,
                end=spec.end,
                benchmark=spec.benchmark,
                initial_cash=sub_cash,
            )
            scored: ScoredBacktestResult = self._scored.run(member_spec)
            backtest = scored.backtest
            ok = scored.ok and backtest is not None
            return PortfolioMemberResult(
                name=member.name,
                kind=member.kind,
                weight=member.weight,
                initial_cash=sub_cash,
                ok=ok,
                error=scored.error,
                backtest=backtest,
                scoring=scored.scoring,
                warnings=list(scored.warnings),
            )

        member_spec = replace(
            member.spec, benchmark=spec.benchmark, initial_cash=sub_cash
        )
        backtest = self._engine.run(member_spec, start=start_ts, end=end_ts)
        return PortfolioMemberResult(
            name=member.name,
            kind=member.kind,
            weight=member.weight,
            initial_cash=sub_cash,
            ok=backtest.ok and not backtest.equity.empty,
            error=backtest.error,
            backtest=backtest,
            warnings=list(backtest.warnings),
        )

    # ------------------------------------------------------------------ #
    # 内部：合并
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_nav(
        outcomes: list[PortfolioMemberResult],
        total_cash: float,
        warnings: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """子账户净值 → 资产矩阵 → 组合净值（日历并集，缺失日照延 / 首日前计闲置现金）。

        Returns:
            ``(组合净值表, 资产矩阵, 组合净值序列, 组合日历)``。
        """
        assets: dict[str, pd.Series] = {}
        calendars: list[pd.DatetimeIndex] = []
        for outcome in outcomes:
            nav = _equity_nav(outcome.backtest.equity)
            assets[outcome.name] = nav * outcome.initial_cash
            calendars.append(nav.index)

        calendar = pd.DatetimeIndex(sorted(set().union(*calendars)))
        common = calendars[0]
        for index in calendars[1:]:
            common = common.intersection(index)
        if len(common) != len(calendar):
            warnings.append(
                f"各成员回测日历不完全一致（并集 {len(calendar)} 天 / 交集 {len(common)} 天）："
                "成员缺失日照「前值照延」估值、首日前按闲置现金计入，为近似口径"
            )

        aligned = pd.DataFrame(index=calendar)
        for outcome in outcomes:
            series = assets[outcome.name].reindex(calendar).ffill()
            aligned[outcome.name] = series.fillna(outcome.initial_cash)
        nav_series = aligned.sum(axis=1) / total_cash
        nav_series.name = "nav"

        drawdown = drawdown_series(nav_series)
        if drawdown.empty:  # pragma: no cover - 组合日历至少 2 天时不会触发
            drawdown = pd.Series(0.0, index=nav_series.index)
        equity_frame = pd.DataFrame(
            {
                "date": calendar,
                "nav": nav_series.to_numpy(dtype=float),
                "drawdown": drawdown.to_numpy(dtype=float),
            }
        )
        assets_frame = aligned
        return equity_frame, assets_frame, nav_series, calendar

    @staticmethod
    def _member_benchmark(
        design: PortfolioMemberResult, calendar: pd.DatetimeIndex
    ) -> pd.Series:
        """组合基准净值：取首个成员的 ``benchmark_nav`` 对齐到组合日历（前值照延）。"""
        equity = design.backtest.equity
        index = pd.DatetimeIndex(pd.to_datetime(equity["date"]))
        values = pd.to_numeric(equity["benchmark_nav"], errors="coerce").to_numpy(dtype=float)
        series = pd.Series(values, index=index).reindex(calendar).ffill().fillna(1.0)
        series.name = "benchmark_nav"
        return series

    @staticmethod
    def _merge_trades(outcomes: list[PortfolioMemberResult]) -> pd.DataFrame:
        """合并成员成交流水（引擎契约列 + ``member`` 列；按日期稳定排序）。"""
        frames: list[pd.DataFrame] = []
        for outcome in outcomes:
            frame = outcome.backtest.trades
            if frame is None or frame.empty:
                continue
            copy = frame.copy()
            copy["member"] = outcome.name
            frames.append(copy)
        if not frames:
            return pd.DataFrame(columns=PORTFOLIO_TRADE_COLUMNS)
        merged = pd.concat(frames, ignore_index=True)
        return merged.sort_values("date", kind="stable").reset_index(drop=True)

    @staticmethod
    def _correlation(outcomes: list[PortfolioMemberResult]) -> pd.DataFrame:
        """成员日收益相关矩阵（两两用**共同交易日**的真实收益；样本不足记 NaN）。

        Notes:
            与「并集照延」口径无关：直接取两成员各自净值序列的日收益，按日期索引
            取交集计算 Pearson 相关——成员缺失日不会污染相关系数；对角固定 1.0。
        """
        names = [outcome.name for outcome in outcomes]
        returns = {
            outcome.name: nav_returns(_equity_nav(outcome.backtest.equity))
            for outcome in outcomes
        }
        matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
        for i, name_a in enumerate(names):
            series_a = returns[name_a]
            matrix.iloc[i, i] = 1.0 if len(series_a) >= 2 else np.nan
            for j in range(i + 1, len(names)):
                name_b = names[j]
                series_b = returns[name_b]
                common = series_a.index.intersection(series_b.index)
                if len(common) < 2:
                    continue
                left = series_a.reindex(common)
                right = series_b.reindex(common)
                if float(left.std(ddof=1)) == 0.0 or float(right.std(ddof=1)) == 0.0:
                    continue  # 零波动（常数收益）无法定义相关，如实留 NaN
                value = left.corr(right)
                if value is not None and np.isfinite(value):
                    matrix.iloc[i, j] = matrix.iloc[j, i] = float(value)
        return matrix

    @staticmethod
    def _contribution(
        outcomes: list[PortfolioMemberResult],
        assets_frame: pd.DataFrame,
        nav_series: pd.Series,
        total_cash: float,
    ) -> tuple[pd.DataFrame, Optional[tuple[pd.Timestamp, pd.Timestamp]]]:
        """贡献归因表 + 组合最大回撤的「峰值日 → 谷底日」窗口。

        ``contribution`` = 成员盈亏 / 组合总资金（Σ = 组合总收益）；
        ``dd_contribution`` = 峰值日 → 谷底日内成员资产变化 / 峰值日组合资产
        （Σ = 组合最大回撤幅度）。
        """
        drawdown = drawdown_series(nav_series)
        dd_window: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None
        dd_base = 0.0
        if not drawdown.empty and float(drawdown.min()) < 0:
            trough = drawdown.idxmin()
            peak = nav_series.loc[:trough].idxmax()
            dd_window = (peak, trough)
            dd_base = float(nav_series.loc[peak]) * total_cash

        rows: list[dict] = []
        for outcome in outcomes:
            assets = assets_frame[outcome.name]
            final_assets = float(assets.iloc[-1])
            pnl = final_assets - outcome.initial_cash
            dd_contribution = 0.0
            if dd_window is not None and dd_base > 0:
                dd_contribution = (
                    float(assets.loc[dd_window[1]]) - float(assets.loc[dd_window[0]])
                ) / dd_base
            rows.append(
                {
                    "member": outcome.name,
                    "weight": outcome.weight,
                    "initial_cash": outcome.initial_cash,
                    "final_assets": final_assets,
                    "pnl": pnl,
                    "member_return": final_assets / outcome.initial_cash - 1.0,
                    "contribution": pnl / total_cash,
                    "dd_contribution": dd_contribution,
                }
            )
        return pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS), dd_window


__all__ = [
    "MEMBER_KIND_CONDITION",
    "MEMBER_KIND_SCORED",
    "PORTFOLIO_TRADE_COLUMNS",
    "CONTRIBUTION_COLUMNS",
    "PortfolioMember",
    "PortfolioSpec",
    "PortfolioMemberResult",
    "PortfolioResult",
    "PortfolioRunner",
]