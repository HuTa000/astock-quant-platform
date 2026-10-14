"""策略回测引擎（P7-S1）。

定位：
    把「选股条件」（复用 :class:`~app.engine.screener.Condition`）当作**历史每日信号**，
    在本地日线数据上逐日回放：**收盘出信号 → 次日开盘成交 → 收盘按市值计价**，
    输出净值曲线、成交明细、绩效指标与牛熊拆分。回答三个业务问题：
    ①这套选股条件历史上赚了多少、最坏亏多少（收益 / 回撤）；
    ②赚的是选股能力（α）还是大盘上涨（β）；
    ③在牛市 / 熊市不同环境里表现是否稳定。

设计要点（与金融数据四大坑的对应处理）：
    * **前视偏差**：信号只用「≤ 当日」数据（指标按全序列计算、取值一律 ``asof(t)``
      回溯，要求表达式为**因果型**——只引用当前及历史，``rolling/ewm/ta`` 均满足）；
      **T 日收盘出信号、T+1 开盘成交**，绝不「用当日收盘信号按当日收盘价成交」；
    * **幸存者偏差**：股票池 = 本地库现有 symbol（含停止更新的），已退市股票在库中
      缺失——结论只代表「当前仍可见的样本」，阶段 4「数据质量审计」会显式量化；
    * **复权口径**：价格统一用**前复权（qfq）**消除除权缺口造成的假摔 / 假涨；
      无 qfq 数据的标的全池回退未复权并记警告（结论需谨慎）；
    * **停牌 / 一字板（流动性坑）**：停牌（当日无 bar）**不可成交**，持仓估值用
      **前收照延（forward-fill）**；一字板（``high == low``）近似视为无法成交；
      两者均计入警告统计（不静默忽略）。

两种选股入口（与 :class:`~app.engine.backtest.strategy.StrategySpec` 一致）：
    * 条件表达式（``conditions``）——逐股按指标表达式求值选股；
    * **预计算名单**（``daily_picks``）——直接用外部（多因子打分）算好的
      「信号日 → 名单」，本引擎只负责撮合与记账，名单本身的因果性由生成方保证
      （见 :mod:`app.engine.factor.scoring`）；名单中标的不在数据池内的会被剔除。

交易规则（A 股现实约束）：
    * 100 股整手、只做多、不融资；
    * 调仓日**全额再平衡**：目标为等权持有入选标的（单只目标市值 =
      昨日收盘总资产 × 仓位上限 / 入选数量），**先卖后买**（卖出释放资金再买入，
      现金不足按序按整手尽力买入）；
    * T+1 自动满足（当日开盘买入的股票，最早次日开盘才可能被卖出）；
    * 费用按 :class:`~app.engine.backtest.costs.CostModel` 双边扣减（含最低佣金 5 元）。

风控（可选，``StrategySpec.risk``，见 :mod:`app.engine.backtest.risk`）：
    * **退出规则**（止损 / 固定止盈 / 移动止盈）：逐只持仓**收盘判定、次日开盘全清**；
      触发后 ``cooldown_days`` 个交易日内该标的**不再买回**（冷却期从触发日收盘起算）；
    * **环境开关**：基准转熊（收盘 vs 其 N 日均线）后总仓位降档，**转熊当日收盘判定、
      次日开盘执行**——无调仓挂单时做防御性等比例减仓，有挂单时随调仓一并降仓；
      转牛不即时加仓（等下一个调仓日恢复），避免高换手；
    * ``risk=None``（默认）时引擎行为与此前**完全一致**。

对外 API::

    from app.engine.backtest import BacktestEngine, StrategySpec
    from app.engine.screener import Condition

    spec = StrategySpec(
        name="站上20日线",
        conditions=[Condition("close - close.rolling(20).mean()", ">", 0)],
        rebalance="weekly",
        top_n=3,
        rank_by="close",
    )
    result = BacktestEngine().run(spec, start="2020-01-01", end="2025-12-31")
    print(result.metrics.sharpe, result.metrics.max_drawdown)
    print(result.regime)      # 牛/熊拆分
"""
from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.data.datasource.base import is_index_symbol
from app.data.store.db import BAR_COLUMNS, Database, get_db
from app.engine.backtest.benchmark import (
    BENCHMARK_KIND_PROXY,
    load_benchmark,
)
from app.engine.backtest.costs import BUY, SELL, CostModel
from app.engine.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    TRADE_COLUMNS,
    PerformanceMetrics,
    compute_metrics,
    drawdown_series,
)
from app.engine.backtest.regime import (
    REGIME_BEAR,
    REGIME_BULL,
    classify_regime,
    regime_summary,
)
from app.engine.backtest.risk import RISK_REASON_LABELS, RiskSpec, evaluate_exit
from app.engine.backtest.strategy import REBALANCE_DAILY, StrategySpec
from app.engine.indicator.engine import IndicatorEngine, get_engine

logger = logging.getLogger(__name__)

# 整手规模（A 股 100 股 / 手）。
LOT_SIZE: int = 100

# 浮点比较容差（资金校验用）。
_EPS: float = 1e-9

# 条件操作符 → 判定函数（与选股引擎同口径；``Condition`` 已把 ``=`` 归一化为 ``==``）。
_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

# equity 表契约列。
EQUITY_COLUMNS: list[str] = ["date", "nav", "benchmark_nav", "drawdown"]


def is_rebalance_day(index: int, calendar: pd.DatetimeIndex, rebalance: str) -> bool:
    """判断 ``calendar[index]`` 是否是调仓信号日（周/月频 = 本周/本月最后交易日）。

    Args:
        index: 日历下标（调用方需保证 ``index < len(calendar) - 1``）。
        calendar: 升序交易日历。
        rebalance: ``daily`` / ``weekly`` / ``monthly``。

    Returns:
        是否在该日出信号（信号在次一交易日开盘执行）。
    """
    if rebalance == REBALANCE_DAILY:
        return True
    day = calendar[index]
    next_day = calendar[index + 1]
    if rebalance == "weekly":
        return tuple(day.isocalendar())[:2] != tuple(next_day.isocalendar())[:2]
    return (day.year, day.month) != (next_day.year, next_day.month)


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """一次回测的完整产出（**不抛异常**，问题封装为 ``ok=False`` + ``error``）。

    Attributes:
        ok: 是否成功完成回测。
        error: 失败原因（成功时为空串）。
        equity: 净值表，契约列 :data:`EQUITY_COLUMNS`（``nav`` 策略净值、
            ``benchmark_nav`` 基准净值、``drawdown`` 回撤序列，首日净值均为 1.0）。
        trades: 成交明细，契约列见 :data:`~app.engine.backtest.metrics.TRADE_COLUMNS`
            （``pnl`` 仅卖出成交有值，为剔除费用后的实现盈亏）。
        metrics: 绩效指标汇总（:class:`PerformanceMetrics`）。
        regime: 牛 / 熊拆分表（契约列见 :mod:`app.engine.backtest.regime`）。
        universe: 实际参与回测的标的清单。
        benchmark_symbol: 基准代码（如 ``sh.000300``）。
        benchmark_kind: 基准口径，``INDEX``（真实指数）或
            ``EQUAL_WEIGHT_PROXY``（库内无指数时的股票池等权代理）。
        warnings: 回测过程中的警告（数据缺失、停牌未成交、代理基准口径等）。
    """

    ok: bool
    error: str = ""
    equity: pd.DataFrame = dc_field(
        default_factory=lambda: pd.DataFrame(columns=EQUITY_COLUMNS)
    )
    trades: pd.DataFrame = dc_field(
        default_factory=lambda: pd.DataFrame(columns=TRADE_COLUMNS)
    )
    metrics: PerformanceMetrics = dc_field(default_factory=PerformanceMetrics)
    regime: pd.DataFrame = dc_field(default_factory=pd.DataFrame)
    universe: list[str] = dc_field(default_factory=list)
    benchmark_symbol: str = ""
    benchmark_kind: str = ""
    warnings: list[str] = dc_field(default_factory=list)


# --------------------------------------------------------------------------- #
# 内部结构
# --------------------------------------------------------------------------- #
class _SymbolData:
    """单个标的的预计算数据（价格序列 + 各条件取值序列 + 排序序列）。"""

    __slots__ = ("symbol", "dates", "_pos", "open", "high", "low", "close", "cond_values", "rank")

    def __init__(
        self,
        symbol: str,
        bars: pd.DataFrame,
        cond_values: list[pd.Series],
        rank: Optional[pd.Series],
    ) -> None:
        self.symbol = symbol
        dates = pd.DatetimeIndex(pd.to_datetime(bars["datetime"], errors="coerce"))
        self.dates = dates
        self._pos: dict[pd.Timestamp, int] = {stamp: i for i, stamp in enumerate(dates)}
        self.open = self._column(bars["open"])
        self.high = self._column(bars["high"])
        self.low = self._column(bars["low"])
        self.close = self._column(bars["close"])
        self.cond_values = cond_values
        self.rank = rank

    def _column(self, column: pd.Series) -> pd.Series:
        """把价格列对齐到标的日期索引（float）。"""
        return pd.Series(
            pd.to_numeric(column, errors="coerce").to_numpy(dtype=float),
            index=self.dates,
        )

    @property
    def first_date(self) -> pd.Timestamp:
        """首根 bar 日期（此前视为未上市）。"""
        return self.dates[0]

    @property
    def last_date(self) -> pd.Timestamp:
        """末根 bar 日期（此后视为数据中断，不再参与选股）。"""
        return self.dates[-1]

    def has_bar(self, day: pd.Timestamp) -> bool:
        """当日是否有 bar（无 bar = 停牌 / 未上市 / 数据中断）。"""
        return day in self._pos

    def tradeable(self, day: pd.Timestamp) -> bool:
        """当日是否**可成交**：有 bar 且非一字板（``high == low`` 近似视为无法成交）。"""
        position = self._pos.get(day)
        if position is None:
            return False
        return float(self.high.iloc[position]) != float(self.low.iloc[position])

    def open_at(self, day: pd.Timestamp) -> Optional[float]:
        """当日开盘价；无 bar 返回 ``None``。"""
        position = self._pos.get(day)
        if position is None:
            return None
        value = float(self.open.iloc[position])
        return value if np.isfinite(value) and value > 0 else None

    def at(self, series: Optional[pd.Series], day: pd.Timestamp) -> Optional[float]:
        """取序列在 ``day``（含）之前**最后一个有效值**（``asof`` 回溯，防前视）。

        Args:
            series: 条件取值 / 排序序列；``None`` 返回 ``None``。
            day: 评估日。

        Returns:
            数值；无有效值 / 非有限值返回 ``None``（调用方按「条件不成立」处理）。
        """
        if series is None:
            return None
        try:
            value = series.asof(day)
        except (KeyError, ValueError):  # pragma: no cover - 索引异常兜底
            return None
        if value is None or not np.isfinite(value):
            return None
        return float(value)


@dataclass
class _Portfolio:
    """回测过程中的账户状态与成交暂存。"""

    cash: float
    last_equity: float
    positions: dict[str, int] = dc_field(default_factory=dict)
    basis: dict[str, float] = dc_field(default_factory=dict)
    trades: list[dict] = dc_field(default_factory=list)
    peak: dict[str, float] = dc_field(default_factory=dict)
    cooldown: dict[str, pd.Timestamp] = dc_field(default_factory=dict)
    risk_exits: dict[str, int] = dc_field(default_factory=dict)
    blocked_suspend: int = 0
    blocked_limit: int = 0
    defensive_trims: int = 0


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class BacktestEngine:
    """策略回测引擎（只读本地库，不依赖 Qt）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        indicator_engine: Optional[IndicatorEngine] = None,
    ) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`app.data.store.db.get_db`。
            indicator_engine: 指标引擎；默认 :func:`app.engine.indicator.engine.get_engine`。
        """
        self._db: Database = db if db is not None else get_db()
        self._engine: IndicatorEngine = (
            indicator_engine if indicator_engine is not None else get_engine()
        )

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def run(
        self,
        spec: StrategySpec,
        start: Any = None,
        end: Any = None,
    ) -> BacktestResult:
        """执行回测（**不抛业务异常**，数据问题封装为 ``ok=False``）。

        Args:
            spec: 策略规格（:class:`StrategySpec`）。
            start: 回测起始日（含）；``None`` 表示不设下限（取数据最早）。
            end: 回测结束日（含）；``None`` 表示不设上限（取数据最新）。

        Returns:
            :class:`BacktestResult`。

        Raises:
            TypeError: ``spec`` 类型错误。
            ValueError: 起止日期无法解析或起始晚于结束（调用方输入错误）。
        """
        if not isinstance(spec, StrategySpec):
            raise TypeError(f"spec 必须是 StrategySpec，实际：{type(spec).__name__}")
        start_ts = self._parse_bound(start)
        end_ts = self._parse_bound(end)
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError(f"起始日期晚于结束日期：{start_ts.date()} > {end_ts.date()}")

        warnings: list[str] = []

        # 1) 股票池与复权口径（优先前复权 qfq）。
        universe, adjust = self._resolve_universe(spec, warnings)
        if not universe:
            return self._fail("本地库内没有可用于回测的日线数据（请先更新数据）", warnings)

        # 2) 逐股读取 + 预计算信号序列（指标按全序列因果计算，取值时才 asof 回溯）。
        symbol_data: dict[str, _SymbolData] = {}
        for symbol in universe:
            try:
                bars = self._db.get_bars(symbol, "d", end=end_ts, adjust=adjust)
            except Exception as exc:  # noqa: BLE001 - 单股读库失败不中断
                warnings.append(f"{symbol}：读取日线失败（{exc}），已排除")
                continue
            if bars is None or bars.empty:
                warnings.append(f"{symbol}：回测区间内无日线数据，已排除")
                continue
            prepared = self._prepare_symbol(symbol, bars, spec, warnings)
            if prepared is not None:
                symbol_data[symbol] = prepared
        if not symbol_data:
            return self._fail("股票池内所有标的的信号计算均失败，无法回测", warnings)

        # 3) 交易日历 = 股票池日期 ∪ 基准指数日期（窗口内）。
        calendar = self._build_calendar(symbol_data, spec.benchmark, start_ts, end_ts)
        if len(calendar) < 2:
            return self._fail("回测区间内可用交易日不足 2 天，无法回测", warnings)

        # 4) 收盘价矩阵（前收照延到日历，供估值与代理基准）+ 基准净值。
        close_map = {
            symbol: data.close.reindex(calendar, method="ffill")
            for symbol, data in symbol_data.items()
        }
        benchmark_nav, benchmark_kind = load_benchmark(
            self._db, spec.benchmark, calendar, close_map
        )
        if benchmark_kind == BENCHMARK_KIND_PROXY:
            warnings.append(
                f"库内无基准指数 {spec.benchmark} 数据，改用「股票池等权」代理："
                "等权 ≠ 市值加权（沪深300），且不含已退市股票，结论须标注口径"
            )

        # 5) 逐日回放：先执行昨日挂单（今开盘），再收盘估值，最后收盘出信号。
        portfolio = _Portfolio(
            cash=float(spec.initial_cash), last_equity=float(spec.initial_cash)
        )
        # 风控规格：仅当启用任一规则时生效（全禁用等价于 None，行为与无风控一致）。
        risk: Optional[RiskSpec] = (
            spec.risk if spec.risk is not None and spec.risk.enabled else None
        )
        regime_labels: Optional[pd.Series] = (
            self._regime_series(spec.benchmark, risk, calendar, benchmark_nav, warnings)
            if risk is not None and risk.regime_filter
            else None
        )
        nav_rows: list[dict] = []
        pending: Optional[list[str]] = None
        pending_exposure: float = 1.0
        pending_exits: Optional[list[str]] = None
        pending_defensive: Optional[float] = None
        prev_regime: Optional[str] = None
        for index, day in enumerate(calendar):
            # ① 风控卖出：昨日收盘触发、今日开盘逐只全清（先腾出资金再调仓）。
            if pending_exits:
                for symbol in pending_exits:
                    qty = portfolio.positions.get(symbol, 0)
                    if qty > 0:
                        self._sell(symbol, day, qty, symbol_data, spec, portfolio)
                pending_exits = None
            # ② 调仓：昨日信号、今日开盘执行（熊市按降档比例建仓）。
            if pending is not None:
                self._rebalance(
                    day,
                    pending,
                    symbol_data,
                    close_map,
                    spec,
                    portfolio,
                    exposure=pending_exposure,
                )
                pending = None
                pending_exposure = 1.0
            # ③ 防御减仓：转熊次日开盘把总仓位压到上限（仅转熊日无调仓挂单时设置）。
            if pending_defensive is not None:
                self._defensive_trim(day, symbol_data, spec, portfolio, pending_defensive)
                pending_defensive = None
            # ④ 收盘估值（持仓按收盘 / 停牌前收照延）。
            equity = portfolio.cash + self._positions_value(portfolio, close_map, day)
            portfolio.last_equity = float(equity)
            nav_rows.append({"date": day, "nav": equity / float(spec.initial_cash)})
            # ⑤ 收盘段（信号只用到 ≤ 当日数据，一律次一交易日开盘执行）。
            if index < len(calendar) - 1:
                if risk is not None:
                    triggered = self._risk_exits(
                        index, calendar, day, close_map, spec, portfolio
                    )
                    pending_exits = triggered or None
                if self._is_rebalance_day(index, calendar, spec):
                    if spec.daily_picks is not None:
                        pending = self._picks_target(day, symbol_data, spec)
                    else:
                        pending = self._select_symbols(day, symbol_data, spec)
                    if risk is not None and portfolio.cooldown:
                        pending = self._filter_cooldown(pending, portfolio.cooldown, day)
                    pending_exposure = 1.0
                    if risk is not None and regime_labels is not None:
                        if regime_labels.get(day, REGIME_BULL) == REGIME_BEAR:
                            pending_exposure = risk.regime_exposure
                if risk is not None and regime_labels is not None:
                    regime_today = regime_labels.get(day, REGIME_BULL)
                    if (
                        prev_regime == REGIME_BULL
                        and regime_today == REGIME_BEAR
                        and pending is None
                    ):
                        pending_defensive = self._defensive_fraction(
                            portfolio, close_map, day, risk.regime_exposure
                        )
                    prev_regime = regime_today

        # 6) 结果装配（净值表 / 成交表 / 指标 / 牛熊拆分 + 警告汇总）。
        equity_frame = self._build_equity(nav_rows, calendar, benchmark_nav)
        trades_frame = pd.DataFrame(portfolio.trades, columns=TRADE_COLUMNS)
        metrics = compute_metrics(
            equity_frame["nav"],
            equity_frame["benchmark_nav"],
            trades_frame,
            TRADING_DAYS_PER_YEAR,
        )
        regime_labels = classify_regime(equity_frame["benchmark_nav"])
        regime_frame = regime_summary(
            equity_frame["nav"], equity_frame["benchmark_nav"], regime_labels
        )
        if portfolio.blocked_suspend:
            warnings.append(
                f"回放中有 {portfolio.blocked_suspend} 次买卖因「停牌无成交」被跳过"
                "（持仓按前收估值）"
            )
        if portfolio.blocked_limit:
            warnings.append(
                f"回放中有 {portfolio.blocked_limit} 次买卖因「一字板（high=low）」"
                "被近似视为无法成交"
            )
        if risk is not None:
            if portfolio.risk_exits:
                detail = "、".join(
                    f"{RISK_REASON_LABELS.get(reason, reason)} {count} 次"
                    for reason, count in sorted(portfolio.risk_exits.items())
                )
                cooldown_note = (
                    f"触发后 {risk.cooldown_days} 个交易日内不买回"
                    if risk.cooldown_days > 0
                    else "触发当日不再买回"
                )
                warnings.append(f"风控卖出：{detail}（{cooldown_note}）")
            if regime_labels is not None:
                bear_days = int((regime_labels == REGIME_BEAR).sum())
                warnings.append(
                    f"环境开关：熊市标签 {bear_days}/{len(calendar)} 个交易日，"
                    f"熊市总仓位上限 {risk.regime_exposure:.0%}，"
                    f"防御性减仓 {portfolio.defensive_trims} 次"
                )
        return BacktestResult(
            ok=True,
            equity=equity_frame,
            trades=trades_frame,
            metrics=metrics,
            regime=regime_frame,
            universe=list(symbol_data.keys()),
            benchmark_symbol=spec.benchmark,
            benchmark_kind=benchmark_kind,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # 数据准备
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_bound(value: Any) -> Optional[pd.Timestamp]:
        """把起始 / 结束日解析为当日零点 ``Timestamp``；``None`` 原样返回。"""
        if value is None:
            return None
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无法解析日期：{value!r}") from exc
        if pd.isna(stamp):
            raise ValueError(f"无法解析日期：{value!r}")
        return stamp.normalize()

    def _resolve_universe(
        self, spec: StrategySpec, warnings: list[str]
    ) -> tuple[list[str], str]:
        """确定股票池与复权口径：显式清单优先，否则取库内全部有日线的标的（qfq 优先）。"""
        if spec.symbols:
            symbols = list(spec.symbols)
            missing = [s for s in symbols if self._safe_bars(s, "qfq").empty]
            if missing:
                warnings.append(
                    f"{len(missing)} 只标的无前复权（qfq）数据，股票池整体回退未复权口径"
                    "（除权日会有假跌，结论需谨慎）"
                )
                return symbols, ""
            return symbols, "qfq"
        try:
            qfq_symbols = self._db.list_kline_symbols("d", "qfq")
        except Exception as exc:  # noqa: BLE001 - 读清单失败按空池处理
            warnings.append(f"读取库内标的清单失败：{exc}")
            qfq_symbols = []
        # 指数（如 sh.000300）只作基准、不作股票池标的（指数无法直接买卖）。
        qfq_symbols = [s for s in qfq_symbols if not is_index_symbol(s)]
        if qfq_symbols:
            return qfq_symbols, "qfq"
        raw_symbols: list[str] = []
        try:
            raw_symbols = self._db.list_kline_symbols("d", "")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"读取库内标的清单失败：{exc}")
        raw_symbols = [s for s in raw_symbols if not is_index_symbol(s)]
        if raw_symbols:
            warnings.append(
                "库内无前复权（qfq）数据，整个回测改用未复权口径（除权日会有假跌）"
            )
        return raw_symbols, ""

    def _safe_bars(self, symbol: str, adjust: str) -> pd.DataFrame:
        """读库（失败降级为空表，**不抛**）。"""
        try:
            return self._db.get_bars(symbol, "d", adjust=adjust)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 %s(%s) 失败：%s", symbol, adjust, exc)
            return pd.DataFrame(columns=BAR_COLUMNS)

    def _prepare_symbol(
        self,
        symbol: str,
        bars: pd.DataFrame,
        spec: StrategySpec,
        warnings: list[str],
    ) -> Optional[_SymbolData]:
        """预计算某标的的各条件取值序列与排序序列；任一条件求值失败 → 排除该标的。"""
        cond_values: list[pd.Series] = []
        for condition in spec.conditions:
            result = self._engine.calc(
                condition.field, bars, ctx={"symbol": symbol, "period": "d"}
            )
            if not result.ok:
                warnings.append(
                    f"{symbol}：条件表达式 {condition.field!r} 求值失败"
                    f"（{result.error}），该标的已排除"
                )
                return None
            series = self._as_series(result.data, bars)
            if series is None:
                warnings.append(
                    f"{symbol}：条件表达式 {condition.field!r} 结果不可用，该标的已排除"
                )
                return None
            cond_values.append(series)

        rank_series: Optional[pd.Series] = None
        if spec.rank_by and spec.top_n is not None:
            result = self._engine.calc(
                spec.rank_by, bars, ctx={"symbol": symbol, "period": "d"}
            )
            if result.ok:
                rank_series = self._as_series(result.data, bars)
            if rank_series is None:
                warnings.append(
                    f"{symbol}：排序表达式 {spec.rank_by!r} 求值失败，该标的无法参与排名"
                )
        return _SymbolData(symbol, bars, cond_values, rank_series)

    @staticmethod
    def _as_series(data: Any, bars: pd.DataFrame) -> Optional[pd.Series]:
        """把指标结果统一为与 bars 等长的 float Series（不支持的类型 → ``None``）。"""
        index = pd.DatetimeIndex(pd.to_datetime(bars["datetime"], errors="coerce"))
        if isinstance(data, pd.Series):
            series = pd.to_numeric(data, errors="coerce").astype(float)
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 0:
                return None
            series = pd.to_numeric(data.iloc[:, 0], errors="coerce").astype(float)
        elif isinstance(data, (int, float, np.number, np.bool_)):
            return pd.Series(float(data), index=index)
        else:
            return None
        if len(series) != len(index):
            return None
        series.index = index
        return series

    def _build_calendar(
        self,
        symbol_data: dict[str, _SymbolData],
        benchmark_symbol: str,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> pd.DatetimeIndex:
        """构造回测交易日历（股票池日期 ∪ 基准指数日期，限定在窗口内）。"""
        stamps: set[pd.Timestamp] = set()
        for data in symbol_data.values():
            stamps.update(self._window_slice(data.dates, start_ts, end_ts))
        benchmark_dates = self._benchmark_dates(benchmark_symbol)
        if len(benchmark_dates):
            stamps.update(self._window_slice(benchmark_dates, start_ts, end_ts))
        return pd.DatetimeIndex(sorted(stamps))

    def _benchmark_dates(self, symbol: str) -> pd.DatetimeIndex:
        """读基准指数日期（失败 / 无数据返回空索引，**不抛**）。"""
        if not symbol:
            return pd.DatetimeIndex([])
        bars = self._safe_bars(str(symbol), "")
        if bars.empty:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(pd.to_datetime(bars["datetime"], errors="coerce")).dropna()

    @staticmethod
    def _window_slice(
        dates: pd.DatetimeIndex,
        start_ts: Optional[pd.Timestamp],
        end_ts: Optional[pd.Timestamp],
    ) -> pd.DatetimeIndex:
        """按窗口裁剪日期索引。"""
        if len(dates) == 0:
            return dates
        sliced = dates
        if start_ts is not None:
            sliced = sliced[sliced >= start_ts]
        if end_ts is not None:
            sliced = sliced[sliced <= end_ts]
        return sliced

    # ------------------------------------------------------------------ #
    # 信号与调仓
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_rebalance_day(
        index: int, calendar: pd.DatetimeIndex, spec: StrategySpec
    ) -> bool:
        """判断 ``calendar[index]`` 是否是调仓信号日（逻辑见 :func:`is_rebalance_day`）。"""
        return is_rebalance_day(index, calendar, spec.rebalance)

    @staticmethod
    def _picks_target(
        day: pd.Timestamp,
        symbol_data: dict[str, _SymbolData],
        spec: StrategySpec,
    ) -> list[str]:
        """预计算名单口径的目标持仓：取当日名单，剔除数据池外的标的（保序）。

        * ``day`` 不在名单上（如权重预热期 / 数据缺失）→ 空名单（当日空仓）；
        * 名单中的标的**未被加载进数据池**（库内无数据 / 读库失败）→ 剔除；
        * 标的当日超出其数据区间（未上市 / 数据中断）→ 剔除（不可买）；
        * 保持名单原顺序（得分降序 → 买入优先级与名单一致）。
        """
        target: list[str] = []
        for symbol in spec.daily_picks.get(day, []):
            data = symbol_data.get(symbol)
            if data is None or day < data.first_date or day > data.last_date:
                continue
            target.append(symbol)
        return target

    def _select_symbols(
        self,
        day: pd.Timestamp,
        symbol_data: dict[str, _SymbolData],
        spec: StrategySpec,
    ) -> list[str]:
        """在 ``day`` 收盘评估全部条件，返回目标持仓清单（可能为空 = 清仓）。"""
        hits: list[str] = []
        for symbol, data in symbol_data.items():
            if day < data.first_date or day > data.last_date:
                continue  # 未上市 / 数据已中断：不可选（也就不可买）
            if self._passes(data, day, spec):
                hits.append(symbol)
        if spec.top_n is None:
            return sorted(hits)
        if spec.rank_by:
            ranked: list[tuple[float, str]] = []
            for symbol in hits:
                data = symbol_data[symbol]
                value = data.at(data.rank, day)
                if value is None:
                    continue  # 无法排名 → 剔除（避免把 NaN 当作最大值）
                ranked.append((-value, symbol))  # 值大者优先；同值按代码升序稳定
            ranked.sort()
            return [symbol for _, symbol in ranked[: spec.top_n]]
        return sorted(hits)[: spec.top_n]

    @staticmethod
    def _passes(data: _SymbolData, day: pd.Timestamp, spec: StrategySpec) -> bool:
        """评估某标的在 ``day`` 的多条件真值（条件值缺失视为不成立）。"""
        results: list[bool] = []
        for condition, series in zip(spec.conditions, data.cond_values):
            value = data.at(series, day)
            if value is None:
                results.append(False)
                continue
            results.append(bool(_OPERATORS[condition.op](value, condition.value)))
        return all(results) if spec.logic == "AND" else any(results)

    # ------------------------------------------------------------------ #
    # 风控（P7-S6）
    # ------------------------------------------------------------------ #
    def _risk_exits(
        self,
        index: int,
        calendar: pd.DatetimeIndex,
        day: pd.Timestamp,
        close_map: dict[str, pd.Series],
        spec: StrategySpec,
        portfolio: _Portfolio,
    ) -> list[str]:
        """收盘风控扫描：更新持仓峰值 → 逐只判定退出规则 → 返回次日开盘待卖清单。

        仅用「≤ 当日」数据：收盘价来自 ``close_map``（停牌前收照延）、峰值只升不降；
        触发者登记冷却期（截止日 = 触发日 + ``cooldown_days``，夹紧到日历末尾）。
        """
        risk = spec.risk
        if risk is None:
            return []
        exited: list[str] = []
        for symbol in sorted(portfolio.positions):
            quantity = portfolio.positions.get(symbol, 0)
            if quantity <= 0:
                continue
            series = close_map.get(symbol)
            if series is None:
                continue
            price = series.get(day, np.nan)
            if price is None or not np.isfinite(price) or float(price) <= 0:
                continue
            close = float(price)
            peak = max(portfolio.peak.get(symbol, close), close)
            portfolio.peak[symbol] = peak
            unit_cost = portfolio.basis.get(symbol, 0.0) / quantity
            reason = evaluate_exit(close, unit_cost, peak, risk)
            if reason is None:
                continue
            exited.append(symbol)
            portfolio.risk_exits[reason] = portfolio.risk_exits.get(reason, 0) + 1
            # 冷却截止日 = 触发日 + cooldown_days（夹紧到日历末尾）；冷却 0 也登记
            # （触发当日信号不再买回，避免「同一开盘价卖了又买回」的空转）。
            deadline_index = min(index + int(risk.cooldown_days), len(calendar) - 1)
            portfolio.cooldown[symbol] = calendar[deadline_index]
        return exited

    @staticmethod
    def _filter_cooldown(
        target: list[str],
        cooldown: dict[str, pd.Timestamp],
        day: pd.Timestamp,
    ) -> list[str]:
        """剔除仍在冷却期（``信号日 <= 触发日 + cooldown_days``）的标的（保序）。"""
        if not cooldown:
            return target
        return [
            symbol
            for symbol in target
            if not (symbol in cooldown and day <= cooldown[symbol])
        ]

    def _defensive_fraction(
        self,
        portfolio: _Portfolio,
        close_map: dict[str, pd.Series],
        day: pd.Timestamp,
        exposure: float,
    ) -> Optional[float]:
        """转熊日收盘计算「等比例减仓分数」= 目标仓位 / 当前仓位；无需减仓返回 ``None``。

        用转熊**判定日**收盘估值（仅 ≤ 当日数据），次日开盘按该分数逐只减仓。
        """
        current_total = self._positions_value(portfolio, close_map, day)
        target_total = float(exposure) * portfolio.last_equity
        if current_total <= target_total + _EPS:
            return None
        return target_total / current_total

    def _defensive_trim(
        self,
        day: pd.Timestamp,
        symbol_data: dict[str, _SymbolData],
        spec: StrategySpec,
        portfolio: _Portfolio,
        fraction: float,
    ) -> None:
        """转熊次日开盘的防御性减仓：按 ``fraction`` 等比例（整手向下取整）卖出超出部分。"""
        fraction = min(1.0, max(0.0, float(fraction)))
        for symbol in sorted(portfolio.positions):
            held = portfolio.positions.get(symbol, 0)
            if held <= 0:
                continue
            keep = int((held * fraction) // LOT_SIZE) * LOT_SIZE
            sell_qty = held - keep
            if sell_qty > 0:
                self._sell(symbol, day, sell_qty, symbol_data, spec, portfolio)
        portfolio.defensive_trims += 1

    def _regime_series(
        self,
        benchmark: str,
        risk: RiskSpec,
        calendar: pd.DatetimeIndex,
        benchmark_nav: pd.Series,
        warnings: list[str],
    ) -> pd.Series:
        """构造与日历对齐的牛/熊标签（环境开关用，标签只含 ≤ 当日数据、因果）。

        优先用**基准全历史收盘**打标签（均线窗口含预热，比从回测首日起重新累计
        更准确）；库内无指数数据时降级用「回测区间内基准净值」并记警告；
        对齐后仍缺标签的交易日按「牛市」处理（记警告，不静默）。
        """
        if benchmark:
            bars = self._safe_bars(str(benchmark), "")
            if not bars.empty:
                dates = pd.DatetimeIndex(pd.to_datetime(bars["datetime"], errors="coerce"))
                closes = pd.Series(
                    pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float),
                    index=dates,
                ).dropna()
                labels = classify_regime(closes, risk.regime_window)
                if not labels.empty:
                    aligned = labels.reindex(calendar, method="ffill")
                    missing = int(aligned.isna().sum())
                    if missing:
                        warnings.append(
                            f"环境开关：{missing} 个交易日无基准标签"
                            "（区间早于 / 晚于指数数据），已按「牛市」处理"
                        )
                    return aligned.fillna(REGIME_BULL)
        warnings.append(
            "环境开关：库内无基准指数数据，牛熊判定降级为「回测区间内基准净值」口径"
            "（均线预热从回测首日起算，结论需谨慎）"
        )
        fallback = classify_regime(pd.Series(benchmark_nav), risk.regime_window)
        if fallback.empty:
            warnings.append("环境开关：基准净值不可用，全部交易日按「牛市」处理")
            return pd.Series(REGIME_BULL, index=calendar, dtype=object)
        return fallback.reindex(calendar, method="ffill").fillna(REGIME_BULL)

    # ------------------------------------------------------------------ #
    # 调仓
    # ------------------------------------------------------------------ #
    def _rebalance(
        self,
        day: pd.Timestamp,
        target: list[str],
        symbol_data: dict[str, _SymbolData],
        close_map: dict[str, pd.Series],
        spec: StrategySpec,
        portfolio: _Portfolio,
        exposure: float = 1.0,
    ) -> None:
        """执行「次日开盘的全额再平衡」：先卖（清仓 + 减仓），再买（补足目标）。

        Args:
            exposure: 本次调仓的**总仓位上限**（占昨日收盘总资产比例，默认 1.0）；
                熊市环境开关启用时取 ``RiskSpec.regime_exposure``（默认 0.5）。
        """
        count = len(target)
        target_shares: dict[str, int] = {}
        if count > 0:
            value_each = portfolio.last_equity * float(exposure) / count
            for symbol in target:
                price = self._sizing_price(symbol_data[symbol], close_map, day, spec)
                if price is None or price <= 0:
                    continue
                lots = int(value_each // (price * LOT_SIZE))
                if lots > 0:
                    target_shares[symbol] = lots * LOT_SIZE

        # 1) 卖出：不在目标里的清仓、超配的减仓（先卖后买以释放资金）。
        for symbol in sorted(portfolio.positions):
            held = portfolio.positions.get(symbol, 0)
            if held <= 0:
                continue
            desired = target_shares.get(symbol, 0)
            if held - desired > 0:
                self._sell(symbol, day, held - desired, symbol_data, spec, portfolio)

        # 2) 买入：按目标顺序尽力补足（现金不足时按整手部分买入）。
        for symbol in target:
            desired = target_shares.get(symbol, 0)
            if desired <= 0:
                continue
            held = portfolio.positions.get(symbol, 0)
            if desired - held > 0:
                self._buy(symbol, day, desired - held, symbol_data, spec, portfolio)

    @staticmethod
    def _sizing_price(
        data: _SymbolData,
        close_map: dict[str, pd.Series],
        day: pd.Timestamp,
        spec: StrategySpec,
    ) -> Optional[float]:
        """计算目标仓位的定价基准：可成交用开盘价（含滑点），停牌 / 一字板用前收估值价。"""
        if data.tradeable(day):
            open_price = data.open_at(day)
            if open_price is not None:
                return float(spec.costs.fill_price(open_price, BUY))
        series = close_map.get(data.symbol)
        if series is None:
            return None
        price = series.get(day, np.nan)
        if price is None or not np.isfinite(price) or float(price) <= 0:
            return None
        return float(price)

    def _sell(
        self,
        symbol: str,
        day: pd.Timestamp,
        qty: int,
        symbol_data: dict[str, _SymbolData],
        spec: StrategySpec,
        portfolio: _Portfolio,
    ) -> None:
        """按开盘价（含滑点）卖出，费用双边扣减，记录实现盈亏（均价法）。"""
        data = symbol_data.get(symbol)
        if data is None or not data.tradeable(day):
            self._count_blocked(portfolio, data, day)
            return
        open_price = data.open_at(day)
        held = portfolio.positions.get(symbol, 0)
        quantity = min(int(qty), held)
        if open_price is None or quantity <= 0:
            self._count_blocked(portfolio, data, day)
            return
        price = float(spec.costs.fill_price(open_price, SELL))
        amount = price * quantity
        cost = float(spec.costs.trade_cost(amount, SELL))
        proceeds = amount - cost
        unit_basis = portfolio.basis.get(symbol, 0.0) / held if held > 0 else 0.0
        pnl = proceeds - unit_basis * quantity
        portfolio.cash += proceeds
        remaining = held - quantity
        if remaining > 0:
            portfolio.positions[symbol] = remaining
            portfolio.basis[symbol] = portfolio.basis.get(symbol, 0.0) - unit_basis * quantity
        else:
            portfolio.positions.pop(symbol, None)
            portfolio.basis.pop(symbol, None)
            portfolio.peak.pop(symbol, None)  # 清仓后重置持仓期峰值（下次买入重新累计）
        portfolio.trades.append(
            {
                "date": day,
                "symbol": symbol,
                "side": SELL,
                "price": price,
                "qty": quantity,
                "amount": amount,
                "cost": cost,
                "pnl": pnl,
            }
        )

    def _buy(
        self,
        symbol: str,
        day: pd.Timestamp,
        need: int,
        symbol_data: dict[str, _SymbolData],
        spec: StrategySpec,
        portfolio: _Portfolio,
    ) -> None:
        """按开盘价（含滑点）买入至目标股数；现金不足时按整手尽力买入。"""
        data = symbol_data.get(symbol)
        if data is None or not data.tradeable(day):
            self._count_blocked(portfolio, data, day)
            return
        open_price = data.open_at(day)
        if open_price is None or need <= 0:
            return
        price = float(spec.costs.fill_price(open_price, BUY))
        lots = min(
            int(need) // LOT_SIZE,
            self._max_affordable_lots(portfolio.cash, price, spec.costs),
        )
        if lots <= 0:
            return
        quantity = lots * LOT_SIZE
        amount = price * quantity
        cost = float(spec.costs.trade_cost(amount, BUY))
        portfolio.cash -= amount + cost
        portfolio.positions[symbol] = portfolio.positions.get(symbol, 0) + quantity
        portfolio.basis[symbol] = portfolio.basis.get(symbol, 0.0) + amount + cost
        # 记录持仓期峰值（含成交价的起点，供移动止盈判定）。
        portfolio.peak[symbol] = max(portfolio.peak.get(symbol, 0.0), price)
        portfolio.trades.append(
            {
                "date": day,
                "symbol": symbol,
                "side": BUY,
                "price": price,
                "qty": quantity,
                "amount": amount,
                "cost": cost,
                "pnl": None,
            }
        )

    @staticmethod
    def _max_affordable_lots(cash: float, price: float, costs: CostModel) -> int:
        """在含费用的前提下，现金最多能买多少手（100 股整手）。"""
        if price <= 0 or cash <= 0:
            return 0
        lots = int(cash // (price * LOT_SIZE))
        while lots > 0:
            amount = lots * LOT_SIZE * price
            if amount + costs.trade_cost(amount, BUY) <= cash + _EPS:
                return lots
            lots -= 1
        return 0

    @staticmethod
    def _count_blocked(
        portfolio: _Portfolio, data: Optional[_SymbolData], day: pd.Timestamp
    ) -> None:
        """统计未成交原因（停牌 / 一字板），供结果警告如实披露。"""
        if data is None or not data.has_bar(day):
            portfolio.blocked_suspend += 1
        else:
            portfolio.blocked_limit += 1

    @staticmethod
    def _positions_value(
        portfolio: _Portfolio, close_map: dict[str, pd.Series], day: pd.Timestamp
    ) -> float:
        """持仓按当日收盘（停牌用前收照延）估值。"""
        total = 0.0
        for symbol, qty in portfolio.positions.items():
            if qty <= 0:
                continue
            series = close_map.get(symbol)
            if series is None:
                continue
            price = series.get(day, np.nan)
            if price is None or not np.isfinite(price):
                continue
            total += qty * float(price)
        return total

    # ------------------------------------------------------------------ #
    # 结果装配
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_equity(
        nav_rows: list[dict],
        calendar: pd.DatetimeIndex,
        benchmark_nav: pd.Series,
    ) -> pd.DataFrame:
        """装配净值表（含基准净值与回撤序列）。"""
        frame = pd.DataFrame(nav_rows)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["benchmark_nav"] = pd.Series(benchmark_nav).reindex(calendar).to_numpy(dtype=float)
        frame["drawdown"] = drawdown_series(frame["nav"]).to_numpy(dtype=float)
        return frame[EQUITY_COLUMNS].reset_index(drop=True)

    @staticmethod
    def _fail(message: str, warnings: list[str]) -> BacktestResult:
        """构造失败结果（空表 + 原因 + 已积累的警告）。"""
        return BacktestResult(ok=False, error=str(message), warnings=list(warnings))


__all__ = ["LOT_SIZE", "EQUITY_COLUMNS", "BacktestResult", "BacktestEngine"]