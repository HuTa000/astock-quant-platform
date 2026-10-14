"""P7-S8 策略成员筛选与筛选后组合验证行为单测。

覆盖：
    1. 输入契约 —— MemberCandidate 类型 / 名称回退 / 来源标记推断与校验；
       SampleSplit 日期解析与顺序校验、样本外起点派生；SelectionCriteria
       各门槛取值校验；
    2. 候选池构造 —— 默认 15 个（均线族 8 + 打分族 7）、名称唯一、均线规格
       参数映射（条件表达式 / 排序 / top_n / 调仓 / 股票池）、打分因子子集与
       `combinations` 对拍；自定义与非法输入；
    3. 切段工具 —— slice_equity 边界双闭 + 段首归一（手算对拍）、空段 / 非法
       输入返回契约空表；segment_metrics 段内成交过滤（胜率只统计段内平仓）
       与鸭子类型（BacktestResult / PortfolioResult 均适配）；
    4. 门槛判定 —— judge_selection 五条规则矩阵（跑赢基准 / 回撤上限 / 回撤
       不深于基准 / 夏普 / 分年度稳定性）与数据缺失口径；
    5. 筛选编排 —— 全期回测单次 + 三段评估（样本内 / 样本外 / 全样本手算对拍）；
       样本内筛选与全样本参考筛选结论可不同；失败候选不阻断报告；全部失败
       报告无效；分年度稳定性闸门（1/3 正收益年份被淘汰）；
    6. 组合对比 —— 四变体结构（全候选 / 样本内合格 / 全样本合格 / 最优单策略）、
       组合窗口覆盖与等权归一、样本外段指标重算对拍、最优单策略挑选依据、
       空合格集与失败候选剔除警告、to_frame 契约；
    7. 真实库 E2E —— 动量池（条件式 + 打分混合候选）筛选 → 四线对比跑通。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.store.db import Database
from app.engine.backtest import (
    DEFAULT_BENCHMARK,
    EQUITY_COLUMNS,
    PORTFOLIO_TRADE_COLUMNS,
    SELECTION_COMPARISON_COLUMNS,
    SELECTION_TABLE_COLUMNS,
    TRADE_COLUMNS,
    VARIANT_ALL,
    VARIANT_BEST_SINGLE,
    VARIANT_SELECTED_FULL,
    VARIANT_SELECTED_IN,
    ZERO_COST_MODEL,
    BacktestResult,
    MemberCandidate,
    PortfolioResult,
    PortfolioSpec,
    SampleSplit,
    ScoredBacktestSpec,
    SelectionCriteria,
    SelectionReport,
    SelectionRunner,
    StrategySpec,
    build_candidate_pool,
    compute_metrics,
    drawdown_series,
    judge_selection,
    segment_metrics,
    slice_equity,
)
from app.engine.backtest.metrics import PerformanceMetrics
from app.engine.backtest.portfolio import PortfolioMember  # noqa: F401  (契约引用)
from app.engine.factor.scoring import DEFAULT_SCORING_FACTORS
from app.engine.screener import Condition


# --------------------------------------------------------------------------- #
# 辅助：内存库 / 行情播种（风格与 test_backtest_portfolio.py 一致）
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _biz_days(count: int, start: str = "2024-01-01") -> list[str]:
    """生成 count 个「交易日」日期串（工作日口径，便于周/月频调仓测试）。"""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=count)]


def _flat(days, price: float) -> dict:
    """常价行情：``{日期: (open, high, low, close)}``（high≠low，保证可成交）。"""
    return {day: (price, price + 0.5, price - 0.5, price) for day in days}


def _seed(db: Database, symbol: str, spec: dict, adjusts=("qfq",)) -> None:
    """按 ``{日期: (o, h, l, c[, volume])}`` 写入日线；默认仅前复权。"""
    for adjust in adjusts:
        frame = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp(day),
                    "open": float(values[0]),
                    "high": float(values[1]),
                    "low": float(values[2]),
                    "close": float(values[3]),
                    "volume": float(values[4]) if len(values) > 4 else 1_000_000.0,
                    "amount": float(values[3])
                    * (float(values[4]) if len(values) > 4 else 1_000_000.0),
                }
                for day, values in spec.items()
            ]
        )
        frame["symbol"] = symbol
        frame["period"] = "d"
        frame["adjust"] = adjust
        db.upsert_bars(frame)


def _momentum_spec(days: list[str], phase: int, start_price: float, seed: int) -> dict:
    """阶段性趋势行情 + 确定性噪声（60 日周期：前 30 日 +0.5%/日、后 30 日 −0.5%/日）。"""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.003, len(days))
    volume_noise = rng.uniform(0.8, 1.2, len(days))
    base_volume = 1_000_000.0 * (1.0 + 0.05 * phase / 60.0)
    spec: dict[str, tuple[float, float, float, float, float]] = {}
    price = start_price
    for index, day in enumerate(days):
        position = (index + phase) % 60
        drift = 0.005 if position < 30 else -0.005
        price *= 1.0 + drift + float(noise[index])
        band = 0.01 + abs(float(noise[index]))
        spec[day] = (
            price,
            price * (1.0 + band),
            price * (1.0 - band),
            price,
            base_volume * float(volume_noise[index]),
        )
    return spec


def _seed_momentum_universe(db: Database, count: int = 12, days_count: int = 320) -> list[str]:
    """播种 count 只相位错开的动量标的（qfq 日线），返回标的清单。"""
    days = _biz_days(days_count)
    symbols = [f"0000{i:02d}" for i in range(1, count + 1)]
    for index, symbol in enumerate(symbols):
        _seed(
            db,
            symbol,
            _momentum_spec(
                days, phase=index * 5, start_price=10.0 + index * 0.1, seed=20261012 + index
            ),
        )
    return symbols


# --------------------------------------------------------------------------- #
# 辅助：构造净值 / 结果对象（筛选编排的行为验证用确定性数据）
# --------------------------------------------------------------------------- #
def _equity_frame(days, nav_values, bench_values) -> pd.DataFrame:
    """按契约列构造净值表（回撤由 drawdown_series 计算）。"""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(list(days)),
            "nav": np.asarray(nav_values, dtype=float),
            "benchmark_nav": np.asarray(bench_values, dtype=float),
        }
    )
    frame["drawdown"] = drawdown_series(frame["nav"])
    return frame[EQUITY_COLUMNS]


def _make_result(days, nav_values, bench_values, trades=None) -> BacktestResult:
    """构造成功的 BacktestResult（净值 + 成交 + 指标）。"""
    frame = _equity_frame(days, nav_values, bench_values)
    if trades is None:
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
    return BacktestResult(
        ok=True,
        equity=frame,
        trades=trades,
        metrics=compute_metrics(frame["nav"], frame["benchmark_nav"], trades),
    )


def _make_portfolio_result(days, nav_values, bench_values) -> PortfolioResult:
    """构造成功的 PortfolioResult（仅净值，组合对比的段内指标只依赖它）。"""
    return PortfolioResult(
        ok=True,
        equity=_equity_frame(days, nav_values, bench_values),
        trades=pd.DataFrame(columns=PORTFOLIO_TRADE_COLUMNS),
    )


def _two_phase_nav(part1: tuple, part2: tuple) -> np.ndarray:
    """两段线性净值（各 100 点；段界点等值，便于「段内收益」手算对拍）。"""
    first = np.linspace(part1[0], part1[1], 100)
    second = np.linspace(part2[0], part2[1], 100)
    return np.concatenate([first, second])


def _yearly_nav(days, year_growth: dict[int, float]) -> np.ndarray:
    """按「自然年年收益目标」构造逐年线性净值（年内线性，各年末值手算可验）。"""
    stamps = [pd.Timestamp(day) for day in days]
    values = np.empty(len(stamps), dtype=float)
    cursor = 0
    level = 1.0
    for year in sorted(year_growth):
        count = sum(1 for stamp in stamps if stamp.year == year)
        end = level * (1.0 + year_growth[year])
        values[cursor : cursor + count] = np.linspace(level, end, count)
        cursor += count
        level = end
    return values


class _FakeEngine:
    """按规格名返回预置回测结果的假引擎（不读库，仅验证筛选编排的行为）。"""

    def __init__(self, results: dict) -> None:
        self.results = dict(results)
        self.calls: list[tuple] = []

    def run(self, spec, start=None, end=None) -> BacktestResult:
        self.calls.append((spec, start, end))
        result = self.results.get(spec.name)
        if result is None:
            return BacktestResult(ok=False, error=f"无数据：{spec.name}")
        return result


class _DummyScored:
    """占位打分编排器：条件式候选测试不会触发；误触发即失败。"""

    def run(self, spec):  # pragma: no cover - 触发即测试失败
        raise AssertionError("打分编排器不应在条件式候选测试中被调用")


class _FakePortfolio:
    """返回预置组合结果的假编排器（记录收到的规格，供窗口 / 权重断言）。"""

    def __init__(self, result: PortfolioResult) -> None:
        self.result = result
        self.specs: list[PortfolioSpec] = []

    def run(self, spec: PortfolioSpec) -> PortfolioResult:
        self.specs.append(spec)
        return self.result


def _cond_candidate(name: str, symbols=("000001",), rebalance: str = "weekly") -> MemberCandidate:
    """构造条件式候选（规格内容不影响假引擎行为，名称是关键）。"""
    spec = StrategySpec(
        name=name,
        conditions=[Condition("close - close.rolling(5).mean()", ">", 0)],
        symbols=list(symbols),
        rebalance=rebalance,
        top_n=3,
        rank_by="close.rolling(5).mean() / close",
        costs=ZERO_COST_MODEL,
        initial_cash=100_000.0,
    )
    return MemberCandidate(name=name, spec=spec)


def _make_runner(engine=None, portfolio_runner=None) -> SelectionRunner:
    """构造只依赖假组件的筛选编排器（db 不会被触碰）。"""
    return SelectionRunner(
        db=object(),
        engine=engine if engine is not None else _FakeEngine({}),
        scored_runner=_DummyScored(),
        portfolio_runner=portfolio_runner
        if portfolio_runner is not None
        else _FakePortfolio(PortfolioResult(ok=True)),
    )


def _loose_criteria(**overrides) -> SelectionCriteria:
    """宽松门槛（默认只保留「跑赢基准」——其余规则按需开启，便于定点测试）。"""
    kwargs = dict(
        beat_benchmark=True,
        drawdown_not_worse_than_benchmark=False,
        max_drawdown_limit=None,
        min_sharpe=None,
        min_win_year_ratio=None,
    )
    kwargs.update(overrides)
    return SelectionCriteria(**kwargs)


# --------------------------------------------------------------------------- #
# 1. 输入契约
# --------------------------------------------------------------------------- #
def test_member_candidate_validation() -> None:
    """候选契约：spec 类型校验 / 名称回退 / 来源标记自动推断与非法拒绝。"""
    spec = StrategySpec(name="规格名", conditions=[Condition("close", ">", 0)])
    assert MemberCandidate(name="", spec=spec).name == "规格名"
    assert MemberCandidate(name="候选", spec=spec).family == "ma"
    scored = ScoredBacktestSpec(name="打分名")
    assert MemberCandidate(name="", spec=scored).family == "scored"
    assert MemberCandidate(name="x", spec=scored, family="scored").family == "scored"

    with pytest.raises(ValueError, match="候选 spec"):
        MemberCandidate(name="x", spec=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="来源标记非法"):
        MemberCandidate(name="x", spec=spec, family="unknown")


def test_sample_split_validation_and_out_start() -> None:
    """切分口径：默认值、日期解析、顺序校验、样本外起点 = 截止次日。"""
    default = SampleSplit()
    assert default.in_start == pd.Timestamp("2015-01-01")
    assert default.in_end == pd.Timestamp("2020-12-31")
    assert default.out_end is None
    assert default.out_start == pd.Timestamp("2021-01-01")

    split = SampleSplit(in_start="2022-06-01", in_end=pd.Timestamp("2023-06-01"))
    assert split.in_start == pd.Timestamp("2022-06-01")
    assert split.out_start == pd.Timestamp("2023-06-02")

    with pytest.raises(ValueError, match="无法解析日期"):
        SampleSplit(in_end="不是日期")
    with pytest.raises(ValueError, match="不能为空"):
        SampleSplit(in_end=None)
    with pytest.raises(ValueError, match="早于"):
        SampleSplit(in_start="2024-01-01", in_end="2023-01-01")
    with pytest.raises(ValueError, match="晚于"):
        SampleSplit(in_end="2023-12-31", out_end="2023-01-01")


def test_criteria_validation() -> None:
    """门槛契约：默认值、回撤上限 / 夏普 / 占比校验、可整体关闭。"""
    criteria = SelectionCriteria()
    assert criteria.min_sharpe == pytest.approx(0.5)
    assert criteria.min_win_year_ratio == pytest.approx(0.5)
    assert criteria.max_drawdown_limit is None

    assert SelectionCriteria(max_drawdown_limit=0.6).max_drawdown_limit == pytest.approx(0.6)
    assert SelectionCriteria(min_sharpe=None, min_win_year_ratio=None).min_sharpe is None

    with pytest.raises(ValueError, match="回撤上限"):
        SelectionCriteria(max_drawdown_limit=1.5)
    with pytest.raises(ValueError, match="回撤上限"):
        SelectionCriteria(max_drawdown_limit=0)
    with pytest.raises(ValueError, match="夏普门槛"):
        SelectionCriteria(min_sharpe=float("nan"))
    with pytest.raises(ValueError, match="分年度胜率门槛"):
        SelectionCriteria(min_win_year_ratio=1.2)


# --------------------------------------------------------------------------- #
# 2. 候选池构造
# --------------------------------------------------------------------------- #
def test_build_candidate_pool_defaults() -> None:
    """默认候选池 = 均线族 8 + 打分族 7 = 15 个；名称唯一、规格参数映射正确。"""
    pool = build_candidate_pool(symbols=["000001", "000002"])

    assert len(pool) == 15
    assert len({candidate.name for candidate in pool}) == 15
    ma_candidates = [c for c in pool if c.family == "ma"]
    scored_candidates = [c for c in pool if c.family == "scored"]
    assert len(ma_candidates) == 8
    assert len(scored_candidates) == 7

    target = next(c for c in ma_candidates if c.name == "站上20日线·周频")
    condition = target.spec.conditions[0]
    assert condition.field == "close - close.rolling(20).mean()"
    assert condition.op == ">"
    assert condition.value == pytest.approx(0.0)
    assert target.spec.rebalance == "weekly"
    assert target.spec.top_n == 3
    assert target.spec.rank_by == "close.rolling(20).mean() / close"
    assert target.spec.symbols == ["000001", "000002"]
    assert target.spec.benchmark == DEFAULT_BENCHMARK

    combos = {
        tuple(candidate.spec.factors)
        for candidate in scored_candidates
    }
    expected: set[tuple[str, ...]] = set()
    for size in range(1, len(DEFAULT_SCORING_FACTORS) + 1):
        from itertools import combinations as _combinations

        expected.update(_combinations(DEFAULT_SCORING_FACTORS, size))
    assert combos == expected
    three = next(c for c in scored_candidates if len(c.spec.factors) == 3)
    assert three.name == "打分·" + "+".join(DEFAULT_SCORING_FACTORS)
    assert three.spec.universe == "pit"
    assert three.spec.top_n == 30


def test_build_candidate_pool_custom_and_invalid() -> None:
    """自定义窗口 / 因子 / 频率生效；非法入参抛 ValueError。"""
    pool = build_candidate_pool(
        ma_windows=[10, 10, 30],
        ma_rebalances=["monthly"],
        factors=["ret_5", "dist_high_250", "ret_5"],
        scored_top_n=5,
        scored_rebalance="weekly",
        scored_universe="current",
    )
    assert [c.name for c in pool] == [
        "站上10日线·月频",
        "站上30日线·月频",
        "打分·ret_5",
        "打分·dist_high_250",
        "打分·ret_5+dist_high_250",
    ]

    with pytest.raises(ValueError, match="均线窗口"):
        build_candidate_pool(ma_windows=[1])
    with pytest.raises(ValueError, match="均线窗口不能为空"):
        build_candidate_pool(ma_windows=[])
    with pytest.raises(ValueError, match="调仓频率非法"):
        build_candidate_pool(ma_rebalances=["yearly"])
    with pytest.raises(ValueError, match="因子清单不能为空"):
        build_candidate_pool(factors=[])
    with pytest.raises(ValueError, match="股票池口径非法"):
        build_candidate_pool(scored_universe="all")


# --------------------------------------------------------------------------- #
# 3. 切段工具
# --------------------------------------------------------------------------- #
def test_slice_equity_renormalizes_and_bounds() -> None:
    """切段：边界双闭、段首归一为 1.0、两列净值各自归一（手算对拍）。"""
    days = _biz_days(5)
    frame = _equity_frame(days, [1.0, 1.1, 1.2, 1.3, 1.4], [2.0, 2.2, 2.4, 2.6, 2.8])

    segment = slice_equity(frame, days[1], days[3])
    assert len(segment) == 3
    assert pd.Timestamp(segment.iloc[0]["date"]) == pd.Timestamp(days[1])
    assert pd.Timestamp(segment.iloc[-1]["date"]) == pd.Timestamp(days[3])
    assert float(segment.iloc[0]["nav"]) == pytest.approx(1.0)
    assert float(segment.iloc[-1]["nav"]) == pytest.approx(1.3 / 1.1)
    assert float(segment.iloc[0]["benchmark_nav"]) == pytest.approx(1.0)
    assert float(segment.iloc[-1]["benchmark_nav"]) == pytest.approx(2.6 / 2.2)

    # 无下界 / 无上界
    assert len(slice_equity(frame, None, days[1])) == 2
    assert len(slice_equity(frame, days[2], None)) == 3

    # 空段（起点晚于终点）保持契约列
    empty = slice_equity(frame, days[4], days[3])
    assert empty.empty
    assert list(empty.columns) == EQUITY_COLUMNS


def test_slice_equity_bad_input() -> None:
    """非法输入 / 缺列 / 坏日期一律降级为空契约表（不抛异常）。"""
    days = _biz_days(3)
    frame = _equity_frame(days, [1.0, 1.1, 1.2], [1.0, 1.0, 1.0])
    for bad in (None, pd.DataFrame(), frame.drop(columns=["nav"]), "x"):
        result = slice_equity(bad)
        assert result.empty
        assert list(result.columns) == EQUITY_COLUMNS
    result = slice_equity(frame, "不是日期")
    assert result.empty


def test_segment_metrics_filters_trades() -> None:
    """段内指标：收益 = 段末 / 段首 − 1；胜率 / 笔数只统计段内已平仓卖出。"""
    days = _biz_days(6)
    trades = pd.DataFrame(
        [
            {
                "date": days[1],
                "symbol": "X",
                "side": "buy",
                "price": 10.0,
                "qty": 100.0,
                "amount": 1000.0,
                "cost": 1.0,
                "pnl": None,
            },
            {
                "date": days[2],
                "symbol": "X",
                "side": "sell",
                "price": 11.0,
                "qty": 100.0,
                "amount": 1100.0,
                "cost": 1.0,
                "pnl": 98.0,
            },
            {
                "date": days[5],
                "symbol": "Y",
                "side": "sell",
                "price": 9.0,
                "qty": 100.0,
                "amount": 900.0,
                "cost": 1.0,
                "pnl": -100.0,
            },
        ],
        columns=TRADE_COLUMNS,
    )
    result = _make_result(days, np.linspace(1.0, 1.5, 6), np.linspace(1.0, 1.2, 6), trades)

    first = segment_metrics(result, days[0], days[2])
    assert first is not None
    assert first.total_return == pytest.approx(0.2)
    assert first.win_rate == pytest.approx(1.0)   # 段内仅 days[2] 一笔 +98
    assert first.trade_count == 1

    full = segment_metrics(result, days[0], days[5])
    assert full is not None
    assert full.win_rate == pytest.approx(0.5)    # 两笔：+98 / −100
    assert full.trade_count == 2

    assert segment_metrics(result, days[4], days[3]) is None  # 空段


def test_segment_metrics_accepts_portfolio_result() -> None:
    """鸭子类型：PortfolioResult（同契约 equity/trades）同样可切段评估。"""
    days = _biz_days(200)
    result = _make_portfolio_result(days, _two_phase_nav((1.0, 1.2), (1.2, 1.5)), np.ones(200))

    metrics = segment_metrics(result, days[100], days[199])
    assert metrics is not None
    assert metrics.total_return == pytest.approx(1.5 / 1.2 - 1.0)
    assert metrics.benchmark_total_return == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 4. 门槛判定
# --------------------------------------------------------------------------- #
def _metrics(**overrides) -> PerformanceMetrics:
    """构造判定用指标（默认全部达标）。"""
    base = dict(
        total_return=0.20,
        benchmark_total_return=0.10,
        max_drawdown=-0.30,
        sharpe=0.80,
    )
    base.update(overrides)
    return PerformanceMetrics(**base)


def test_judge_selection_rule_matrix() -> None:
    """五条规则的通过 / 淘汰原因逐条对拍。"""
    criteria = SelectionCriteria(
        beat_benchmark=True,
        drawdown_not_worse_than_benchmark=True,
        max_drawdown_limit=0.5,
        min_sharpe=0.5,
        min_win_year_ratio=0.5,
    )
    passed, reasons = judge_selection(_metrics(), -0.40, 0.667, criteria)
    assert passed is True
    assert reasons == []

    def _fail(reason_keyword: str, **metrics_overrides):
        merged = dict(benchmark_max_drawdown=-0.40)
        merged.update(metrics_overrides.pop("_bench", {}))
        passed_, reasons_ = judge_selection(
            _metrics(**metrics_overrides),
            merged["benchmark_max_drawdown"],
            metrics_overrides.pop("win_year_ratio", 0.667),
            criteria,
        )
        assert passed_ is False
        assert any(reason_keyword in reason for reason in reasons_), reasons_

    # 未跑赢基准
    passed_, reasons_ = judge_selection(
        _metrics(total_return=0.05), -0.40, 0.667, criteria
    )
    assert passed_ is False
    assert any("未跑赢基准" in reason for reason in reasons_)
    # 回撤超过绝对上限
    passed_, reasons_ = judge_selection(
        _metrics(max_drawdown=-0.60), -0.80, 0.667, criteria
    )
    assert passed_ is False
    assert any("超过上限" in reason for reason in reasons_)
    # 回撤深于基准
    passed_, reasons_ = judge_selection(
        _metrics(max_drawdown=-0.50), -0.40, 0.667, criteria
    )
    assert passed_ is False
    assert any("深于基准" in reason for reason in reasons_)
    # 夏普不达标
    passed_, reasons_ = judge_selection(
        _metrics(sharpe=0.30), -0.40, 0.667, criteria
    )
    assert passed_ is False
    assert any("夏普" in reason for reason in reasons_)
    # 分年度稳定性不足
    passed_, reasons_ = judge_selection(
        _metrics(), -0.40, 0.333, criteria
    )
    assert passed_ is False
    assert any("正收益年份占比" in reason for reason in reasons_)


def test_judge_selection_missing_data() -> None:
    """数据缺失按「无法判定」处理：不通过且原因说明；关闭的规则不参与。"""
    criteria = SelectionCriteria(
        beat_benchmark=True,
        drawdown_not_worse_than_benchmark=True,
        max_drawdown_limit=0.5,
        min_sharpe=0.5,
        min_win_year_ratio=0.5,
    )
    passed, reasons = judge_selection(None, -0.40, 0.667, criteria)
    assert passed is False
    assert any("指标缺失" in reason for reason in reasons)

    passed, reasons = judge_selection(_metrics(), None, None, criteria, label="全样本")
    assert passed is False
    assert any("全样本" in reason and "基准回撤缺失" in reason for reason in reasons)
    assert any("无逐年收益数据" in reason for reason in reasons)

    # 全部关闭：无指标也只在缺失时拒绝（这里指标齐全 → 通过）
    off = SelectionCriteria(
        beat_benchmark=False,
        drawdown_not_worse_than_benchmark=False,
        max_drawdown_limit=None,
        min_sharpe=None,
        min_win_year_ratio=None,
    )
    passed, reasons = judge_selection(_metrics(), None, None, off)
    assert passed is True and reasons == []


# --------------------------------------------------------------------------- #
# 5. 筛选编排（假引擎 - 确定性数据）
# --------------------------------------------------------------------------- #
def test_runner_rejects_bad_input() -> None:
    """入参契约：池类型 / 元素 / 重名 / split / criteria 类型校验。"""
    runner = _make_runner()
    spec = StrategySpec(name="t", conditions=[Condition("close", ">", 0)])
    good = MemberCandidate(name="A", spec=spec)

    with pytest.raises(TypeError, match="候选池"):
        runner.run("abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="不能为空"):
        runner.run([])
    with pytest.raises(ValueError, match="MemberCandidate"):
        runner.run([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="名称重复"):
        runner.run([good, MemberCandidate(name="A", spec=spec)])
    with pytest.raises(TypeError, match="SampleSplit"):
        runner.run([good], split="2024")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="SelectionCriteria"):
        runner.run([good], criteria="strict")  # type: ignore[arg-type]


def test_run_segments_pass_and_full_reference() -> None:
    """单次全期回测 + 三段评估手算对拍；样本内合格与全样本参考结论可不同。"""
    days = _biz_days(200)
    nav_a = _two_phase_nav((1.0, 1.2), (1.2, 0.96))     # 样本内 +20%，样本外 −20%
    nav_b = _two_phase_nav((1.0, 0.9), (0.9, 1.17))     # 样本内 −10%，样本外 +30%
    bench = _two_phase_nav((1.0, 1.1), (1.1, 1.155))    # 样本内 +10%，样本外 +5%
    engine = _FakeEngine(
        {"A": _make_result(days, nav_a, bench), "B": _make_result(days, nav_b, bench)}
    )
    split = SampleSplit(in_start=days[0], in_end=days[99], out_end=days[199])
    runner = _make_runner(engine=engine)

    report = runner.run([_cond_candidate("A"), _cond_candidate("B")], split=split, criteria=_loose_criteria())

    assert report.ok and report.error == ""
    assert len(report.evaluations) == 2
    eval_a, eval_b = report.evaluations

    # 三段指标（手算对拍）
    assert eval_a.metrics_in.total_return == pytest.approx(0.20, abs=1e-9)
    assert eval_a.metrics_in.benchmark_total_return == pytest.approx(0.10, abs=1e-9)
    assert eval_a.metrics_out.total_return == pytest.approx(-0.20, abs=1e-9)
    assert eval_a.metrics_full.total_return == pytest.approx(-0.04, abs=1e-9)
    assert eval_b.metrics_out.total_return == pytest.approx(0.30, abs=1e-9)

    # 样本内筛选：A 跑赢、B 没跑赢；全样本参考结论相反
    assert eval_a.passed_in is True and eval_a.reasons_in == []
    assert eval_b.passed_in is False
    assert any("未跑赢基准" in reason for reason in eval_b.reasons_in)
    assert eval_a.passed_full is False
    assert any("未跑赢基准" in reason for reason in eval_a.reasons_full)
    assert report.passed_in == ["A"]
    assert report.passed_full == ["B"]

    # 回测窗口 = 全期（in_start → out_end），单次回测
    spec, start, end = engine.calls[0]
    assert spec.name == "A"
    assert start == pd.Timestamp(days[0])
    assert end == pd.Timestamp(days[199])
    assert len(engine.calls) == 2


def test_run_records_failure_and_continues() -> None:
    """个别候选回测失败不阻断报告：成功候选照常筛选、失败原因进警告与表格。"""
    days = _biz_days(200)
    nav_a = _two_phase_nav((1.0, 1.2), (1.2, 1.2))
    bench = _two_phase_nav((1.0, 1.05), (1.05, 1.05))
    engine = _FakeEngine(
        {"A": _make_result(days, nav_a, bench), "B": BacktestResult(ok=False, error="本地无数据")}
    )
    split = SampleSplit(in_start=days[0], in_end=days[99], out_end=days[199])
    report = _make_runner(engine=engine).run(
        [_cond_candidate("A"), _cond_candidate("B")], split=split, criteria=_loose_criteria()
    )

    assert report.ok is True
    assert report.evaluations[1].ok is False
    assert report.evaluations[1].error == "本地无数据"
    assert report.passed_in == ["A"]
    assert any("候选·B" in warning and "本地无数据" in warning for warning in report.warnings)

    frame = report.to_frame()
    assert list(frame.columns) == SELECTION_TABLE_COLUMNS
    row_b = frame[frame["candidate"] == "B"].iloc[0]
    assert row_b["reasons"] == "回测失败：本地无数据"
    assert pd.isna(row_b["in_return"])


def test_run_all_fail_report_not_ok() -> None:
    """全部候选失败 ⇒ 报告无效（ok=False）且错误说明，输出空合格集。"""
    days = _biz_days(200)
    split = SampleSplit(in_start=days[0], in_end=days[99], out_end=days[199])
    report = _make_runner(engine=_FakeEngine({})).run(
        [_cond_candidate("A"), _cond_candidate("B")], split=split
    )

    assert report.ok is False
    assert "全部候选回测失败" in report.error
    assert report.passed_in == [] and report.passed_full == []
    assert len(report.warnings) >= 2
    assert len(report.to_frame()) == 2


def test_run_win_year_stability_gate() -> None:
    """分年度稳定性闸门：1/3 正收益年份被淘汰，全正年份通过；逐年收益可读。"""
    days = list(pd.bdate_range("2022-01-03", "2024-12-31"))
    nav_p = _yearly_nav(days, {2022: 0.10, 2023: -0.05, 2024: -0.08})   # 1/3 正
    nav_q = _yearly_nav(days, {2022: 0.05, 2023: 0.05, 2024: 0.05})     # 3/3 正
    flat = np.ones(len(days))
    engine = _FakeEngine(
        {"P": _make_result(days, nav_p, flat), "Q": _make_result(days, nav_q, flat)}
    )
    split = SampleSplit(in_start=days[0], in_end=days[-1], out_end=None)
    criteria = _loose_criteria(
        beat_benchmark=False,
        min_win_year_ratio=0.5,
    )
    report = _make_runner(engine=engine).run(
        [_cond_candidate("P"), _cond_candidate("Q")], split=split, criteria=criteria
    )

    eval_p, eval_q = report.evaluations
    assert eval_p.win_year_ratio_in == pytest.approx(1.0 / 3.0)
    assert eval_p.year_returns_in[2022] == pytest.approx(0.10)
    assert eval_p.year_returns_in[2023] == pytest.approx(-0.05)
    assert eval_p.year_returns_in[2024] == pytest.approx(-0.08)
    assert eval_p.passed_in is False
    assert any("正收益年份占比" in reason for reason in eval_p.reasons_in)
    assert eval_q.win_year_ratio_in == pytest.approx(1.0)
    assert eval_q.passed_in is True
    assert report.passed_in == ["Q"]
    # 样本外段为空（截止 = 数据末端）→ 记警告
    assert any("样本外段无有效数据" in warning for warning in report.warnings)


# --------------------------------------------------------------------------- #
# 6. 组合对比（假组合编排器）
# --------------------------------------------------------------------------- #
def _standard_report_and_engine(days):
    """构造「A 样本内合格 / B 样本内淘汰但全样本合格 / C 回测失败」的筛选报告素材。"""
    nav_a = _two_phase_nav((1.0, 1.2), (1.2, 0.96))
    nav_b = _two_phase_nav((1.0, 0.9), (0.9, 1.17))
    bench = _two_phase_nav((1.0, 1.1), (1.1, 1.155))
    engine = _FakeEngine(
        {"A": _make_result(days, nav_a, bench), "B": _make_result(days, nav_b, bench)}
    )
    return engine


def test_compare_builds_variants_and_selects_best_single() -> None:
    """四变体结构 / 成员选择 / 组合窗口与等权归一 / 样本外段指标重算对拍。"""
    days = _biz_days(200)
    engine = _standard_report_and_engine(days)
    split = SampleSplit(in_start=days[0], in_end=days[99], out_end=days[199])
    portfolio = _FakePortfolio(
        _make_portfolio_result(days, _two_phase_nav((1.0, 1.2), (1.2, 1.5)), np.ones(200))
    )
    runner = _make_runner(engine=engine, portfolio_runner=portfolio)
    report = runner.run(
        [_cond_candidate("A"), _cond_candidate("B")], split=split, criteria=_loose_criteria()
    )

    comparison = runner.compare(report)

    assert comparison.ok is True
    assert [v.key for v in comparison.variants] == [
        VARIANT_ALL,
        VARIANT_SELECTED_IN,
        VARIANT_SELECTED_FULL,
        VARIANT_BEST_SINGLE,
    ]
    variant_all, variant_in, variant_full, variant_best = comparison.variants
    assert variant_all.members == ["A", "B"]
    assert variant_in.members == ["A"]      # 样本内合格
    assert variant_full.members == ["B"]    # 全样本合格（上界参考）
    assert variant_full.note == ""

    # 最优单策略 = 样本内夏普最高的候选
    expected_best = max(
        [e for e in report.evaluations if e.metrics_in and e.metrics_in.sharpe is not None],
        key=lambda e: float(e.metrics_in.sharpe),
    ).candidate.name
    assert variant_best.members == [expected_best]
    assert "样本内夏普最高" in variant_best.note

    # 样本外段指标重算：+25% vs 基准 0%
    for variant in comparison.variants:
        assert variant.ok is True
        assert variant.metrics_out is not None
        assert variant.metrics_out.total_return == pytest.approx(0.25, abs=1e-9)
        assert variant.metrics_out.benchmark_total_return == pytest.approx(0.0)

    # 组合规格：窗口 = 样本内起点 → 样本外终点；等权归一（1/N）；基准透传
    spec = portfolio.specs[0]
    assert spec.start == pd.Timestamp(days[0])
    assert spec.end == pd.Timestamp(days[199])
    assert [m.name for m in spec.members] == ["A", "B"]
    assert all(m.weight == pytest.approx(0.5) for m in spec.members)
    assert spec.benchmark == DEFAULT_BENCHMARK
    # 变体逐个经过组合编排器（4 个变体 → 4 次）
    assert len(portfolio.specs) == 4


def test_compare_empty_pass_sets_and_excluded_failure() -> None:
    """空合格集：筛选后 / 上界变体失败并记警告；失败候选被剔除且留痕。"""
    days = _biz_days(200)
    nav_a = _two_phase_nav((1.0, 1.2), (1.2, 0.96))
    bench = _two_phase_nav((1.0, 1.1), (1.1, 1.155))
    engine = _FakeEngine(
        {
            "A": _make_result(days, nav_a, bench),
            "C": BacktestResult(ok=False, error="本地无数据"),
        }
    )
    split = SampleSplit(in_start=days[0], in_end=days[99], out_end=days[199])
    # 夏普门槛不可达 → 全部淘汰（但回测本身成功）
    criteria = _loose_criteria(min_sharpe=1000.0)
    runner = _make_runner(engine=engine)
    report = runner.run(
        [_cond_candidate("A"), _cond_candidate("C")], split=split, criteria=criteria
    )
    assert report.passed_in == [] and report.passed_full == []

    comparison = runner.compare(report)

    variants = {v.key: v for v in comparison.variants}
    assert variants[VARIANT_ALL].ok is True
    assert variants[VARIANT_ALL].members == ["A"]        # 失败的 C 被剔除
    assert variants[VARIANT_SELECTED_IN].ok is False
    assert "无可用成员" in variants[VARIANT_SELECTED_IN].error
    assert variants[VARIANT_SELECTED_FULL].ok is False
    assert variants[VARIANT_BEST_SINGLE].ok is True      # 单策略对照仍可构建
    assert comparison.ok is True

    assert any("样本内无合格成员" in warning for warning in comparison.warnings)
    assert any("全样本无合格成员" in warning for warning in comparison.warnings)
    assert any("候选「C」" in warning and "未纳入组合对比" in warning for warning in comparison.warnings)


def test_compare_rejects_bad_input() -> None:
    """report / split 类型校验；无效报告直接降级为 ok=False 不抛异常。"""
    runner = _make_runner()
    with pytest.raises(TypeError, match="SelectionReport"):
        runner.compare(None)  # type: ignore[arg-type]

    invalid = SelectionReport(ok=False, error="全部候选回测失败")
    comparison = runner.compare(invalid)
    assert comparison.ok is False
    assert "筛选报告无效" in comparison.error

    days = _biz_days(200)
    engine = _standard_report_and_engine(days)
    split = SampleSplit(in_start=days[0], in_end=days[99], out_end=days[199])
    runner2 = _make_runner(engine=engine)
    report = runner2.run([_cond_candidate("A"), _cond_candidate("B")], split=split, criteria=_loose_criteria())
    with pytest.raises(TypeError, match="SampleSplit"):
        runner2.compare(report, split="2024")  # type: ignore[arg-type]


def test_comparison_to_frame_contracts() -> None:
    """对比表契约列：成员拼接 / 数量 / 未构建变体的 note = 失败原因。"""
    days = _biz_days(200)
    engine = _FakeEngine({"A": BacktestResult(ok=False, error="无数据")})
    split = SampleSplit(in_start=days[0], in_end=days[99], out_end=days[199])
    runner = _make_runner(engine=engine)
    report = runner.run([_cond_candidate("A")], split=split, criteria=_loose_criteria())
    assert report.ok is False

    # 无效报告不产生变体；用「空合格集」场景验证表格
    nav = _two_phase_nav((1.0, 1.2), (1.2, 1.2))
    engine2 = _FakeEngine({"A": _make_result(days, nav, np.ones(200))})
    runner2 = _make_runner(engine=engine2)
    report2 = runner2.run(
        [_cond_candidate("A")], split=split, criteria=_loose_criteria(min_sharpe=1000.0)
    )
    comparison = runner2.compare(report2)
    frame = comparison.to_frame()

    assert list(frame.columns) == SELECTION_COMPARISON_COLUMNS
    assert len(frame) == 4
    row_in = frame[frame["variant"] == VARIANT_SELECTED_IN].iloc[0]
    assert row_in["member_count"] == 0
    assert row_in["members"] == ""
    assert "无可用成员" in row_in["note"]
    row_all = frame[frame["variant"] == VARIANT_ALL].iloc[0]
    assert row_all["member_count"] == 1
    assert row_all["members"] == "A"
    assert row_all["note"] == ""


# --------------------------------------------------------------------------- #
# 7. 真实库 E2E（条件式 + 打分混合候选）
# --------------------------------------------------------------------------- #
def test_real_db_selection_and_compare_e2e(tmp_path) -> None:
    """真实库端到端：混合候选筛选（宽松 / 严格两轮）+ 四线组合对照片。"""
    db = _open(tmp_path / "member_selection_e2e.db")
    days = _biz_days(320)
    symbols = _seed_momentum_universe(db)

    cond_candidate = _cond_candidate("站上10日线·周频", symbols=symbols, rebalance="weekly")
    scored_candidate = MemberCandidate(
        name="打分·ret_5+log_amt_20",
        spec=ScoredBacktestSpec(
            name="打分·ret_5+log_amt_20",
            factors=["ret_5", "log_amt_20"],
            ic_horizon=5,
            min_ic_days=30,
            min_stocks=5,
            top_n=3,
            rebalance="weekly",
            universe="current",
            costs=ZERO_COST_MODEL,
            initial_cash=100_000.0,
        ),
    )
    pool = [cond_candidate, scored_candidate]
    split = SampleSplit(in_start=days[0], in_end=days[219], out_end=days[299])
    runner = SelectionRunner(db=db)

    # ---- 第一轮：宽松门槛（只要求跑赢基准）----
    report = runner.run(pool, split=split, criteria=_loose_criteria())
    assert report.ok is True, report.error
    assert len(report.evaluations) == 2
    for evaluation in report.evaluations:
        assert evaluation.ok is True, evaluation.error
        assert evaluation.metrics_in is not None
        assert evaluation.metrics_out is not None
        assert evaluation.metrics_full is not None
        # 判定与原因列表自洽（通过 ⇔ 无淘汰原因）
        assert evaluation.passed_in == (not evaluation.reasons_in)

    frame = report.to_frame()
    assert list(frame.columns) == SELECTION_TABLE_COLUMNS
    assert len(frame) == 2
    assert set(frame["family"]) == {"均线族", "打分族"}

    comparison = runner.compare(report)
    assert comparison.ok is True, comparison.error
    variant_all = comparison.variants[0]
    assert variant_all.ok is True, variant_all.error
    assert variant_all.members == ["站上10日线·周频", "打分·ret_5+log_amt_20"]
    assert variant_all.metrics_out is not None
    assert len(comparison.to_frame()) == 4

    # ---- 第二轮：严格门槛（夏普门槛不可达）→ 空合格集正确降级 ----
    strict = runner.run(
        pool,
        split=split,
        criteria=_loose_criteria(beat_benchmark=False, min_sharpe=1000.0),
    )
    assert strict.ok is True
    assert strict.passed_in == [] and strict.passed_full == []
    strict_comparison = runner.compare(strict)
    variants = {v.key: v for v in strict_comparison.variants}
    assert variants[VARIANT_ALL].ok is True          # 筛选前组合仍可构建
    assert variants[VARIANT_SELECTED_IN].ok is False  # 筛选后为空
    assert variants[VARIANT_BEST_SINGLE].ok is True   # 单策略对照仍可构建
    assert any("无合格成员" in warning for warning in strict_comparison.warnings)

    db.close()