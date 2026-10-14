"""P7-S7 多策略组合回测（资金分账·等权）行为单测。

覆盖：
    1. 输入契约 —— PortfolioMember 类型 / 权重校验与 kind 标记；PortfolioSpec
       成员非空、名称唯一、权重归一化、总资金校验、基准回退；
    2. 分账口径 —— 子账户初始资金 = 归一化权重 × 组合总资金（成交手数与之一致）；
       非等权时权重比正确切分；3:1 权重的分账提示进入警告；
    3. 净值合并 —— 等权组合净值 = 成员资产加权平均（手算对拍：1.0 与 2.0 → 1.5）；
       日历不一致取并集、成员首日前按闲置现金计入 + 警告；
    4. 覆盖规则 —— 组合级 start/end/benchmark 覆盖成员规格（含警告）；
       成员独立回测结果与「单独跑同资金同窗口」逐字一致（无跨账户互相影响）；
    5. 相关性矩阵 —— 与「两两共同交易日收益」手算对拍、对称、对角 1.0、
       单成员 1×1；
    6. 贡献归因 —— Σ contribution = 组合总收益、Σ pnl = 总盈亏、
       member_return 与成员自身总收益一致；最大回撤日分解 Σ dd_contribution =
       组合最大回撤幅度（与 metrics.max_drawdown 同口径）；
    7. 失败与异常 —— 空库成员失败即整体失败（error 指明成员、保留现场）；
       runner 类型 / 日期非法抛异常；打分成员 PIT 无快照失败原因透传；
    8. 成员风控保留 —— 成员自带 RiskSpec 在组合内依然生效（警告带成员名前缀）；
    9. 打分成员分派 E2E —— 条件式 + 打分混合组合可跑通（scoring 报告、窗口覆盖、
       警告前缀、结果契约列）；
    10. 结果契约 —— equity / trades / contribution 契约列、首日净值 1.0、
        回撤 ≤ 0、牛熊拆分非空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.store.db import Database
from app.engine.backtest import (
    BENCHMARK_KIND_INDEX,
    CONTRIBUTION_COLUMNS,
    DEFAULT_BENCHMARK,
    EQUITY_COLUMNS,
    MEMBER_KIND_CONDITION,
    MEMBER_KIND_SCORED,
    PORTFOLIO_TRADE_COLUMNS,
    ZERO_COST_MODEL,
    BacktestEngine,
    PortfolioMember,
    PortfolioRunner,
    PortfolioSpec,
    RiskSpec,
    ScoredBacktestSpec,
    StrategySpec,
)
from app.engine.backtest.regime import REGIME_COLUMNS
from app.engine.screener import Condition


# --------------------------------------------------------------------------- #
# 辅助
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


def _cond_spec(symbols, **kw) -> StrategySpec:
    """构造测试用条件式规格（`收盘 > 0` 恒真 + 零成本 + 10 万初始资金）。"""
    kw.setdefault("costs", ZERO_COST_MODEL)
    kw.setdefault("initial_cash", 100_000.0)
    return StrategySpec(
        name=kw.pop("name", "条件式"),
        conditions=[Condition("close", ">", 0)],
        symbols=list(symbols),
        **kw,
    )


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


def _member(name: str, spec, weight: float = 1.0) -> PortfolioMember:
    """构造组合成员。"""
    return PortfolioMember(name=name, spec=spec, weight=weight)


# --------------------------------------------------------------------------- #
# 1~5. 输入契约
# --------------------------------------------------------------------------- #
def test_member_kind_and_name_fallback() -> None:
    """成员 kind 按规格类型标记；空名回退为规格自身名称。"""
    cond = _cond_spec(["000001"], name="站上20日线")
    scored = ScoredBacktestSpec(name="多因子打分")
    member = PortfolioMember(name="", spec=cond)
    assert member.name == "站上20日线"
    assert member.kind == MEMBER_KIND_CONDITION
    assert PortfolioMember(name="打分", spec=scored).kind == MEMBER_KIND_SCORED


def test_member_rejects_bad_spec_and_weight() -> None:
    """spec 类型不支持 / 权重非正数或 NaN ⇒ ValueError。"""
    with pytest.raises(ValueError, match="spec"):
        PortfolioMember(name="x", spec="not-a-spec")
    spec = _cond_spec(["000001"])
    for bad in (0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="正数"):
            PortfolioMember(name="x", spec=spec, weight=bad)
    with pytest.raises(ValueError, match="数值"):
        PortfolioMember(name="x", spec=spec, weight="heavy")


def test_spec_requires_members_and_unique_names() -> None:
    """成员为空 / 成员类型非法 / 名称重复 ⇒ ValueError。"""
    with pytest.raises(ValueError, match="至少"):
        PortfolioSpec(members=[])
    with pytest.raises(ValueError, match="PortfolioMember"):
        PortfolioSpec(members=[_cond_spec(["000001"])])
    dup = [_member("A", _cond_spec(["000001"])), _member("A", _cond_spec(["000002"]))]
    with pytest.raises(ValueError, match="重复"):
        PortfolioSpec(members=dup)


def test_spec_normalizes_weights() -> None:
    """权重按总和归一化：3:1 → 0.75/0.25；缺省全部 1.0 → 等权。"""
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"]), 3.0),
            _member("B", _cond_spec(["000002"]), 1.0),
        ]
    )
    assert [m.weight for m in spec.members] == pytest.approx([0.75, 0.25])
    equal = PortfolioSpec(
        members=[_member("A", _cond_spec(["000001"])), _member("B", _cond_spec(["000002"]))]
    )
    assert [m.weight for m in equal.members] == pytest.approx([0.5, 0.5])


def test_spec_rejects_bad_cash_and_defaults() -> None:
    """总资金非正 / 非数值 ⇒ ValueError；空名称与空基准回退默认值。"""
    with pytest.raises(ValueError, match="initial_cash"):
        PortfolioSpec(members=[_member("A", _cond_spec(["000001"]))], initial_cash=0)
    with pytest.raises(ValueError, match="initial_cash"):
        PortfolioSpec(members=[_member("A", _cond_spec(["000001"]))], initial_cash="abc")
    spec = PortfolioSpec(
        name="  ", members=[_member("A", _cond_spec(["000001"]))], benchmark=" "
    )
    assert spec.name == "多策略组合"
    assert spec.benchmark == DEFAULT_BENCHMARK


# --------------------------------------------------------------------------- #
# 6. runner 输入异常
# --------------------------------------------------------------------------- #
def test_runner_rejects_bad_input(tmp_path) -> None:
    """spec 类型错误抛 TypeError；日期无法解析 / 起晚于止抛 ValueError。"""
    runner = PortfolioRunner(db=_open(tmp_path / "bad_input.db"))
    with pytest.raises(TypeError):
        runner.run("not-a-spec")
    member = _member("A", _cond_spec(["000001"]))
    with pytest.raises(ValueError, match="晚于"):
        runner.run(PortfolioSpec(members=[member], start="2025-01-01", end="2024-01-01"))
    with pytest.raises(ValueError, match="无法解析"):
        runner.run(PortfolioSpec(members=[member], start="not-a-date"))


# --------------------------------------------------------------------------- #
# 7. 失败路径：成员失败即整体失败
# --------------------------------------------------------------------------- #
def test_empty_db_fails_fast_with_member_name(tmp_path) -> None:
    """空库：首个成员失败即整体失败，error 指明成员名，保留现场且不抛异常。"""
    db = _open(tmp_path / "empty.db")
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ]
    )
    result = PortfolioRunner(db=db).run(spec)

    assert not result.ok
    assert "成员「A」" in result.error
    assert len(result.members) == 1                 # fail-fast：后续成员不再执行
    assert result.members[0].ok is False
    assert result.equity.empty and result.trades.empty


# --------------------------------------------------------------------------- #
# 2~3. 分账与净值合并（手算对拍）
# --------------------------------------------------------------------------- #
def test_sub_account_cash_split(tmp_path) -> None:
    """子账户初始资金 = 权重 × 总资金，成交手数随之变化（50 万→5000 股 / 2500 股）。"""
    db = _open(tmp_path / "split.db")
    days = _biz_days(4)
    _seed(db, "000001", _flat(days, 10.0))
    _seed(db, "000002", _flat(days, 20.0))
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ],
        initial_cash=100_000.0,
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    member_a, member_b = result.members
    assert member_a.weight == pytest.approx(0.5)
    assert member_a.initial_cash == pytest.approx(50_000.0)
    assert member_b.initial_cash == pytest.approx(50_000.0)
    assert int(member_a.backtest.trades.iloc[0]["qty"]) == 5_000    # 50000 // (10×100)
    assert int(member_b.backtest.trades.iloc[0]["qty"]) == 2_500    # 50000 // (20×100)


def test_weight_ratio_splits_cash(tmp_path) -> None:
    """3:1 权重：子账户 75% / 25% 分账，且组合层给出归一化口径提示。"""
    db = _open(tmp_path / "ratio.db")
    days = _biz_days(3)
    _seed(db, "000001", _flat(days, 10.0))
    _seed(db, "000002", _flat(days, 20.0))
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"]), 3.0),
            _member("B", _cond_spec(["000002"]), 1.0),
        ],
        initial_cash=100_000.0,
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    member_a, member_b = result.members
    assert member_a.initial_cash == pytest.approx(75_000.0)
    assert member_b.initial_cash == pytest.approx(25_000.0)
    assert int(member_a.backtest.trades.iloc[0]["qty"]) == 7_500    # 75000 // (10×100)
    assert int(member_b.backtest.trades.iloc[0]["qty"]) == 1_200    # 25000 // (20×100)
    assert any("归一化分账" in w for w in result.warnings)


def test_combined_nav_equals_weighted_member_assets(tmp_path) -> None:
    """等权组合净值 = 成员净值加权平均：A 恒 1.0、B 翻倍 → 组合终值 1.5（手算对拍）。"""
    db = _open(tmp_path / "merge.db")
    days = _biz_days(5)
    _seed(db, "000001", _flat(days, 10.0))
    _seed(
        db,
        "000002",
        {
            days[0]: (10.0, 10.5, 9.5, 10.0),
            days[1]: (10.0, 10.5, 9.5, 10.0),
            days[2]: (20.0, 21.0, 19.0, 20.0),
            days[3]: (20.0, 21.0, 19.0, 20.0),
            days[4]: (20.0, 21.0, 19.0, 20.0),
        },
    )
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ],
        initial_cash=100_000.0,
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    nav_a = result.members[0].backtest.equity["nav"].to_numpy(dtype=float)
    nav_b = result.members[1].backtest.equity["nav"].to_numpy(dtype=float)
    assert list(nav_a) == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])
    assert list(nav_b) == pytest.approx([1.0, 1.0, 2.0, 2.0, 2.0])
    assert list(result.equity["nav"].to_numpy(dtype=float)) == pytest.approx(
        [1.0, 1.0, 1.5, 1.5, 1.5]
    )
    # 无回撤 ⇒ 不产生回撤窗口；贡献归因 Σ = 组合总收益 0.5。
    assert result.dd_window is None
    assert result.contribution["dd_contribution"].abs().max() == pytest.approx(0.0)
    contribution = result.contribution.set_index("member")
    assert float(contribution.loc["A", "contribution"]) == pytest.approx(0.0)
    assert float(contribution.loc["B", "contribution"]) == pytest.approx(0.5)
    assert float(result.contribution["contribution"].sum()) == pytest.approx(
        result.metrics.total_return
    )
    assert float(result.contribution["member_return"].iloc[1]) == pytest.approx(
        result.members[1].backtest.metrics.total_return
    )


def test_drawdown_decomposition_sums_to_max_drawdown(tmp_path) -> None:
    """最大回撤日分解：Σ dd_contribution = 组合最大回撤幅度，窗口 = 峰值日→谷底日。"""
    db = _open(tmp_path / "dd.db")
    days = _biz_days(5)
    _seed(db, "000001", _flat(days, 10.0))          # A 恒 1.0
    _seed(
        db,
        "000002",                                    # B：10 → 20 → 5（腰斩式回撤）
        {
            days[0]: (10.0, 10.5, 9.5, 10.0),
            days[1]: (10.0, 10.5, 9.5, 10.0),
            days[2]: (20.0, 21.0, 19.0, 20.0),
            days[3]: (20.0, 21.0, 19.0, 20.0),
            days[4]: (5.0, 5.5, 4.5, 5.0),
        },
    )
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ],
        initial_cash=100_000.0,
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    nav = result.equity["nav"].to_numpy(dtype=float)
    assert list(nav) == pytest.approx([1.0, 1.0, 1.5, 1.5, 0.75])
    assert result.metrics.max_drawdown == pytest.approx(-0.5)
    assert result.dd_window == (pd.Timestamp(days[2]), pd.Timestamp(days[4]))
    contribution = result.contribution.set_index("member")
    assert float(contribution.loc["A", "dd_contribution"]) == pytest.approx(0.0)
    assert float(contribution.loc["B", "dd_contribution"]) == pytest.approx(-0.5)
    assert float(result.contribution["dd_contribution"].sum()) == pytest.approx(
        result.metrics.max_drawdown
    )


def test_calendar_union_with_leading_idle_cash(tmp_path) -> None:
    """成员日历不一致：组合取并集，晚启动成员首日前按闲置现金计入 + 警告。"""
    db = _open(tmp_path / "union.db")
    days = _biz_days(5)
    _seed(db, "000001", _flat(days, 10.0))          # A：全窗口
    _seed(db, "000002", _flat(days[2:], 20.0))       # B：只有后 3 天数据
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ],
        initial_cash=100_000.0,
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    assert len(result.equity) == 5
    assert pd.Timestamp(result.members[1].backtest.equity.iloc[0]["date"]) == pd.Timestamp(
        days[2]
    )
    # B 首日前记为闲置现金（50000）+ A 净值 1.0 ⇒ 组合净值仍为 1.0。
    assert list(result.equity["nav"].to_numpy(dtype=float)) == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0]
    )
    assert any("日历不完全一致" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# 4. 覆盖规则与成员独立性
# --------------------------------------------------------------------------- #
def test_member_specs_overridden_by_portfolio(tmp_path) -> None:
    """组合级窗口 / 基准覆盖成员规格，且旧基准被覆盖时给出警告。"""
    db = _open(tmp_path / "override.db")
    days = _biz_days(6)
    _seed(db, "000001", _flat(days, 10.0))
    _seed(db, "000002", _flat(days, 20.0))
    _seed(db, "sh.000300", _flat(days, 3000.0), adjusts=("",))
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"], benchmark="sz.399001")),
            _member("B", _cond_spec(["000002"])),
        ],
        start=days[1],
        end=days[4],
        benchmark="sh.000300",
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    assert result.benchmark_symbol == "sh.000300"
    assert result.benchmark_kind == BENCHMARK_KIND_INDEX
    for outcome in result.members:
        dates = pd.DatetimeIndex(pd.to_datetime(outcome.backtest.equity["date"]))
        assert dates.min() >= pd.Timestamp(days[1])
        assert dates.max() <= pd.Timestamp(days[4])
    assert any("已被组合基准" in w for w in result.warnings)


def test_member_result_matches_standalone_run(tmp_path) -> None:
    """成员在组合内的回测结果与「单独跑同资金」完全一致（子账户互不挪用资金）。"""
    db = _open(tmp_path / "standalone.db")
    days = _biz_days(6)
    _seed(db, "000001", _flat(days, 10.0))
    _seed(db, "000002", _flat(days, 20.0))
    standalone = BacktestEngine(db=db).run(
        _cond_spec(["000001"], initial_cash=50_000.0)
    )
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ],
        initial_cash=100_000.0,
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    member = result.members[0].backtest
    pd.testing.assert_frame_equal(member.trades, standalone.trades)
    pd.testing.assert_frame_equal(member.equity, standalone.equity)


# --------------------------------------------------------------------------- #
# 5. 相关性矩阵
# --------------------------------------------------------------------------- #
def test_correlation_matches_pairwise_member_returns(tmp_path) -> None:
    """相关矩阵与「两两共同交易日日收益」手算一致：对称、对角 1.0、标签正确。"""
    db = _open(tmp_path / "corr.db")
    days = _biz_days(30)
    spec_a, spec_b = {}, {}
    price_a, price_b = 10.0, 10.0
    for index, day in enumerate(days):
        price_a *= 1.01 if index % 3 else 0.99
        price_b *= 0.99 if index % 2 else 1.02
        spec_a[day] = (price_a, price_a * 1.01, price_a * 0.99, price_a)
        spec_b[day] = (price_b, price_b * 1.01, price_b * 0.99, price_b)
    _seed(db, "000001", spec_a)
    _seed(db, "000002", spec_b)
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ]
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    returns_a = pd.Series(
        result.members[0].backtest.equity["nav"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(result.members[0].backtest.equity["date"])),
    ).pct_change().dropna()
    returns_b = pd.Series(
        result.members[1].backtest.equity["nav"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(result.members[1].backtest.equity["date"])),
    ).pct_change().dropna()
    common = returns_a.index.intersection(returns_b.index)
    expected = returns_a.reindex(common).corr(returns_b.reindex(common))

    assert list(result.correlation.index) == ["A", "B"]
    assert list(result.correlation.columns) == ["A", "B"]
    assert float(result.correlation.loc["A", "A"]) == pytest.approx(1.0)
    assert float(result.correlation.loc["A", "B"]) == pytest.approx(expected)
    assert float(result.correlation.loc["B", "A"]) == pytest.approx(expected)
    assert -1.0 <= float(result.correlation.loc["A", "B"]) <= 1.0


def test_single_member_correlation_and_warning_prefix(tmp_path) -> None:
    """单成员组合：相关矩阵 1×1 对角 1.0；成员警告带成员名前缀。"""
    db = _open(tmp_path / "single.db")
    days = _biz_days(4)
    _seed(db, "000001", _flat(days, 10.0))
    spec = PortfolioSpec(members=[_member("A", _cond_spec(["000001"]))])
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    assert result.correlation.shape == (1, 1)
    assert float(result.correlation.iloc[0, 0]) == pytest.approx(1.0)
    assert any(w.startswith("[A]") for w in result.warnings)   # 等权代理基准警告


# --------------------------------------------------------------------------- #
# 8. 成员风控保留
# --------------------------------------------------------------------------- #
def test_member_risk_still_applies_inside_portfolio(tmp_path) -> None:
    """成员自带 RiskSpec（止损 5%）在组合内依然生效，警告以成员名前缀汇总。"""
    db = _open(tmp_path / "risk.db")
    days = _biz_days(5)
    _seed(
        db,
        "000001",
        {
            days[0]: (10.0, 10.5, 9.5, 10.0),   # 信号日：次日开盘买入 @10
            days[1]: (10.0, 10.5, 9.5, 10.0),
            days[2]: (9.0, 9.5, 7.5, 8.0),      # 收盘 8.0：相对成本 −20% → 触发止损
            days[3]: (8.0, 8.5, 7.5, 8.0),      # 次日开盘 8.0 全清
            days[4]: (8.0, 8.5, 7.5, 8.0),
        },
    )
    _seed(db, "000002", _flat(days, 20.0))
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"], risk=RiskSpec(stop_loss_pct=0.05))),
            _member("B", _cond_spec(["000002"])),
        ]
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    member_a = result.members[0]
    assert any("风控卖出" in w for w in member_a.warnings)
    assert any(w.startswith("[A]") and "风控卖出" in w for w in result.warnings)
    sell = member_a.backtest.trades[member_a.backtest.trades["side"] == "sell"]
    assert pd.Timestamp(sell.iloc[0]["date"]) == pd.Timestamp(days[3])


# --------------------------------------------------------------------------- #
# 9. 打分成员分派 E2E
# --------------------------------------------------------------------------- #
def test_scored_member_dispatch_e2e(tmp_path) -> None:
    """条件式 + 打分混合组合：分派正确、窗口覆盖、scoring 报告透传、契约列完整。"""
    db = _open(tmp_path / "scored_member.db")
    days = _biz_days(320)
    symbols = _seed_momentum_universe(db)
    scored_spec = ScoredBacktestSpec(
        name="多因子打分",
        factors=["ret_5", "log_amt_20"],
        ic_horizon=5,
        min_ic_days=30,
        min_stocks=5,
        top_n=3,
        rebalance="weekly",
        universe="current",
        costs=ZERO_COST_MODEL,
        initial_cash=100_000.0,
    )
    spec = PortfolioSpec(
        members=[
            _member("条件式", _cond_spec(symbols[:1], name="条件式", rebalance="weekly")),
            _member("打分", scored_spec),
        ],
        start=days[100],
        end=days[300],
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    condition_member, scored_member = result.members
    assert condition_member.kind == MEMBER_KIND_CONDITION
    assert condition_member.scoring is None
    assert scored_member.kind == MEMBER_KIND_SCORED
    assert scored_member.scoring is not None
    assert scored_member.scoring.scored_days > 0
    # 窗口覆盖：两成员净值日期都落在组合窗口内。
    for outcome in result.members:
        dates = pd.DatetimeIndex(pd.to_datetime(outcome.backtest.equity["date"]))
        assert dates.min() >= pd.Timestamp(days[100])
        assert dates.max() <= pd.Timestamp(days[300])
    # 契约与相关矩阵：2×2、对角 1.0、off-diagonal 有限（两成员都有波动）。
    assert list(result.equity.columns) == EQUITY_COLUMNS
    assert result.correlation.shape == (2, 2)
    assert float(result.correlation.iloc[0, 0]) == pytest.approx(1.0)
    assert np.isfinite(float(result.correlation.iloc[0, 1]))
    assert any(w.startswith("[打分]") for w in result.warnings)


def test_scored_member_pit_failure_is_reported(tmp_path) -> None:
    """打分成员 PIT 口径无成分快照 ⇒ 整体失败且失败原因透传该成员。"""
    db = _open(tmp_path / "scored_fail.db")
    _seed_momentum_universe(db)                    # 有行情、无成分股快照
    scored_spec = ScoredBacktestSpec(
        name="多因子打分",
        factors=["ret_5", "log_amt_20"],
        ic_horizon=5,
        min_ic_days=30,
        min_stocks=5,
        top_n=3,
        universe="pit",
        costs=ZERO_COST_MODEL,
        initial_cash=100_000.0,
    )
    spec = PortfolioSpec(
        members=[
            _member("条件式", _cond_spec(["000001"])),
            _member("打分", scored_spec),
        ]
    )
    result = PortfolioRunner(db=db).run(spec)

    assert not result.ok
    assert "成员「打分」" in result.error
    failed = result.members[-1]
    assert failed.ok is False
    assert "快照" in failed.error


# --------------------------------------------------------------------------- #
# 10. 结果契约
# --------------------------------------------------------------------------- #
def test_result_contracts(tmp_path) -> None:
    """equity / trades / contribution 契约列、首日净值 1.0、回撤 ≤ 0、牛熊拆分非空。"""
    db = _open(tmp_path / "contracts.db")
    days = _biz_days(6)
    _seed(db, "000001", _flat(days, 10.0))
    _seed(db, "000002", _flat(days, 20.0))
    spec = PortfolioSpec(
        members=[
            _member("A", _cond_spec(["000001"])),
            _member("B", _cond_spec(["000002"])),
        ]
    )
    result = PortfolioRunner(db=db).run(spec)

    assert result.ok, result.error
    assert list(result.equity.columns) == EQUITY_COLUMNS
    assert float(result.equity.iloc[0]["nav"]) == pytest.approx(1.0)
    assert (result.equity["drawdown"] <= 0).all()
    assert list(result.trades.columns) == PORTFOLIO_TRADE_COLUMNS
    assert set(result.trades["member"]) == {"A", "B"}
    assert list(result.contribution.columns) == CONTRIBUTION_COLUMNS
    assert not result.regime.empty
    assert list(result.regime.columns) == REGIME_COLUMNS
    assert result.benchmark_symbol == DEFAULT_BENCHMARK
    assert sum(member.initial_cash for member in result.members) == pytest.approx(
        spec.initial_cash
    )