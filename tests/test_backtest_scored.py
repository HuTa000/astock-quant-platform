"""P7-S5 打分选股接入回测引擎行为单测。

覆盖：
    1. ``StrategySpec.daily_picks`` 契约 —— 日期键归一化、名单去重保序、
       受限池自动派生、与 conditions / top_n / rank_by 互斥、空名单与非法日期拒绝；
    2. 引擎名单口径 —— T 日名单 T+1 开盘成交、按名单换仓、名单缺失即清仓、
       数据池外标的剔除、停牌不成交进警告、基准 INDEX 口径；
    3. ``ScoredBacktestRunner`` E2E（current 口径）—— 打分 → Top-N → 逐日回放：
       买入标的必须来自前一交易日名单、周频只在「周末信号日」成交、预热警告、
       等权代理基准警告、净值契约列；
    4. 月频调仓 —— 成交日 = 「月末信号日」的次一交易日；
    5. PIT 口径 —— 名单只含「当时生效名单」成员；无快照显式失败；
       预热不足失败；空库失败；
    6. 规格校验与默认值 —— top_n / rebalance / universe / costs / initial_cash
       非法抛 ValueError，runner 类型错误抛 TypeError；
    7. 风控规格透传 —— ``ScoredBacktestSpec.risk`` 类型校验，以及 E2E 生效
       （止损触发、冷却口径与警告文案）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.models import IndexConstituent
from app.data.store.db import Database
from app.engine.backtest import (
    BENCHMARK_KIND_INDEX,
    BENCHMARK_KIND_PROXY,
    DEFAULT_SCORED_REBALANCE,
    DEFAULT_SCORED_TOP_N,
    DEFAULT_SCORED_UNIVERSE,
    EQUITY_COLUMNS,
    ZERO_COST_MODEL,
    BacktestEngine,
    RiskSpec,
    ScoredBacktestResult,
    ScoredBacktestRunner,
    ScoredBacktestSpec,
    StrategySpec,
)


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


def _seed_constituents(
    db: Database, snapshots: dict[str, list[str]], index_code: str = "sh.000300"
) -> None:
    """按 ``{生效日: [标的, ...]}`` 写入成分股快照（语义 = 该日生效的完整名单）。"""
    rows = [
        IndexConstituent(
            index_code=index_code,
            symbol=symbol,
            name="",
            effective_date=pd.Timestamp(day),
        )
        for day, members in snapshots.items()
        for symbol in members
    ]
    db.upsert_index_constituents(rows)


def _scored(db: Database, **kw) -> ScoredBacktestResult:
    """构造并运行一次打分选股回测（默认可跑的轻量口径 + 零成本）。"""
    kw.setdefault("factors", ["ret_5", "log_amt_20"])
    kw.setdefault("ic_horizon", 5)
    kw.setdefault("min_ic_days", 30)
    kw.setdefault("min_stocks", 5)
    kw.setdefault("top_n", 3)
    kw.setdefault("costs", ZERO_COST_MODEL)
    kw.setdefault("initial_cash", 100_000.0)
    return ScoredBacktestRunner(db=db).run(ScoredBacktestSpec(**kw))


# --------------------------------------------------------------------------- #
# 1) StrategySpec.daily_picks 契约
# --------------------------------------------------------------------------- #
def test_spec_derives_restricted_pool_from_picks() -> None:
    """名单日期键归一化为 Timestamp、名单去重保序、symbols 自动取名单并集。"""
    spec = StrategySpec(
        name="picks",
        daily_picks={"2024-01-02": ["000002", "000001", "000002"], "2024-01-01": ["000003"]},
    )
    assert spec.conditions == []
    assert set(spec.daily_picks) == {pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")}
    assert spec.daily_picks[pd.Timestamp("2024-01-02")] == ["000002", "000001"]
    assert spec.symbols == ["000001", "000002", "000003"]


def test_spec_rejects_picks_conflicts_and_bad_input() -> None:
    """名单与条件 / top_n / rank_by 互斥；空名单与非字典、非法日期 ⇒ ValueError。"""
    from app.engine.screener import Condition

    day = "2024-01-02"
    with pytest.raises(ValueError, match="互斥"):
        StrategySpec(
            name="x",
            conditions=[Condition("close", ">", 0)],
            daily_picks={day: ["000001"]},
        )
    with pytest.raises(ValueError, match="互斥"):
        StrategySpec(name="x", daily_picks={day: ["000001"]}, top_n=3)
    with pytest.raises(ValueError, match="互斥"):
        StrategySpec(name="x", daily_picks={day: ["000001"]}, rank_by="close")
    with pytest.raises(ValueError, match="为空"):
        StrategySpec(name="x", daily_picks={})
    with pytest.raises(ValueError, match="为空"):
        StrategySpec(name="x", daily_picks={day: []})
    with pytest.raises(ValueError, match="字典"):
        StrategySpec(name="x", daily_picks=["000001"])
    with pytest.raises(ValueError, match="无法解析"):
        StrategySpec(name="x", daily_picks={"not-a-date": ["000001"]})


# --------------------------------------------------------------------------- #
# 2) 引擎名单口径
# --------------------------------------------------------------------------- #
def test_engine_follows_daily_picks_with_next_day_open(tmp_path) -> None:
    """T 日名单 → T+1 开盘成交；按名单换仓（先卖后买），净值契约列完整。"""
    db = _open(tmp_path / "picks_flow.db")
    days = _biz_days(5)
    _seed(db, "000001", _flat(days, 10.0))
    _seed(db, "000002", _flat(days, 20.0))
    spec = StrategySpec(
        name="picks",
        daily_picks={days[0]: ["000001"], days[1]: ["000001"], days[2]: ["000002"], days[3]: ["000002"]},
        costs=ZERO_COST_MODEL,
        initial_cash=100_000.0,
    )
    result = BacktestEngine(db=db).run(spec)

    assert result.ok, result.error
    assert result.universe == ["000001", "000002"]  # 受限池 = 名单并集
    trades = result.trades
    assert list(trades["side"]) == ["buy", "sell", "buy"]
    assert [pd.Timestamp(d) for d in trades["date"]] == [
        pd.Timestamp(days[1]), pd.Timestamp(days[3]), pd.Timestamp(days[3])
    ]
    assert list(trades["symbol"]) == ["000001", "000001", "000002"]
    first = trades.iloc[0]
    assert float(first["price"]) == pytest.approx(10.0)        # T+1 开盘价
    assert int(first["qty"]) == 10_000                          # 100000 // (10×100) = 100 手
    third = trades.iloc[2]
    assert float(third["price"]) == pytest.approx(20.0)
    assert int(third["qty"]) == 5_000
    # 净值契约列 + 常价零成本 ⇒ 净值恒 1.0
    assert list(result.equity.columns) == EQUITY_COLUMNS
    assert float(result.equity.iloc[-1]["nav"]) == pytest.approx(1.0)


def test_engine_picks_ignore_symbols_outside_pool(tmp_path) -> None:
    """名单里数据池外的标的直接剔除（不报错、不成交）。"""
    db = _open(tmp_path / "picks_pool.db")
    days = _biz_days(3)
    _seed(db, "000001", _flat(days, 10.0))
    spec = StrategySpec(
        name="picks",
        daily_picks={days[0]: ["000001", "999999"]},
        symbols=["000001"],  # 显式指定池：999999 不在池内
        costs=ZERO_COST_MODEL,
        initial_cash=100_000.0,
    )
    result = BacktestEngine(db=db).run(spec)

    assert result.ok, result.error
    assert "999999" not in set(result.trades["symbol"])
    assert set(result.universe) == {"000001"}


def test_engine_missing_picks_day_clears_position(tmp_path) -> None:
    """名单缺失的调仓日按空仓处理：次日开盘清仓。"""
    db = _open(tmp_path / "picks_gap.db")
    days = _biz_days(4)
    _seed(db, "000001", _flat(days, 10.0))
    spec = StrategySpec(
        name="picks",
        daily_picks={days[0]: ["000001"]},  # 仅首日出名单，其余调仓日空仓
        costs=ZERO_COST_MODEL,
        initial_cash=100_000.0,
    )
    result = BacktestEngine(db=db).run(spec)

    assert result.ok, result.error
    assert list(result.trades["side"]) == ["buy", "sell"]
    assert pd.Timestamp(result.trades.iloc[1]["date"]) == pd.Timestamp(days[2])  # 次日开盘卖出


def test_engine_picks_suspend_blocks_and_warns(tmp_path) -> None:
    """名单标的执行日停牌（无 bar，基准补全日历）⇒ 不成交并进警告；基准为真实指数。"""
    db = _open(tmp_path / "picks_suspend.db")
    days = _biz_days(5)
    _seed(db, "000001", _flat([days[0], days[1], days[3], days[4]], 10.0))  # 缺 days[2]
    _seed(db, "sh.000300", _flat(days, 3000.0), adjusts=("",))
    spec = StrategySpec(
        name="picks",
        daily_picks={days[1]: ["000001"]},  # 信号在 days[1]，执行在 days[2]（停牌）
        costs=ZERO_COST_MODEL,
        initial_cash=100_000.0,
    )
    result = BacktestEngine(db=db).run(spec)

    assert result.ok, result.error
    assert result.trades.empty
    assert result.benchmark_kind == BENCHMARK_KIND_INDEX
    assert any("停牌" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# 3) ScoredBacktestRunner E2E（current 口径）
# --------------------------------------------------------------------------- #
def test_runner_current_mode_e2e_weekly(tmp_path) -> None:
    """打分 → 周频 Top-N → 回放：买入必须来自前一交易日名单、预热与代理基准进警告。"""
    db = _open(tmp_path / "runner_weekly.db")
    symbols = _seed_momentum_universe(db)
    result = _scored(db, rebalance="weekly", universe="current")

    assert result.ok, result.error
    backtest = result.backtest
    assert backtest.ok, backtest.error
    assert set(result.scoring.universe) == set(symbols)
    assert result.scoring.scored_days > 0

    # Top-N 数量约束：每天名单非空且 ≤ top_n，且存在正好 3 只的日子。
    assert result.picks
    sizes = [len(picked) for picked in result.picks.values()]
    assert max(sizes) <= 3 and min(sizes) >= 1

    # 买入时点与来源：成交日的前一交易日必须是「周末信号日」，且标在该日名单内。
    dates = pd.DatetimeIndex(pd.to_datetime(backtest.equity["date"]))
    buys = backtest.trades[backtest.trades["side"] == "buy"]
    assert not buys.empty
    for _, trade in buys.iterrows():
        position = int(dates.get_loc(pd.Timestamp(trade["date"])))
        assert position > 0
        signal_day = dates[position - 1]
        assert trade["symbol"] in result.picks.get(signal_day, [])
        # 周频：信号日 = 本周最后一个交易日（下一成交日已属下一周）
        assert tuple(signal_day.isocalendar())[:2] != tuple(
            dates[position].isocalendar()
        )[:2]

    # 预热与代理基准如实进警告；净值契约列、首日净值 1.0。
    assert any("预热" in w for w in result.warnings)
    assert any("没有有效名单" in w for w in result.warnings)
    assert any("代理" in w for w in result.warnings)
    assert backtest.benchmark_kind == BENCHMARK_KIND_PROXY
    assert list(backtest.equity.columns) == EQUITY_COLUMNS
    assert float(backtest.equity.iloc[0]["nav"]) == pytest.approx(1.0)


def test_runner_risk_spec_passes_through_and_takes_effect(tmp_path) -> None:
    """风控规格透传到引擎并生效：1% 宽松止损必然触发，警告如实记录原因与冷却口径。"""
    db = _open(tmp_path / "runner_risk.db")
    _seed_momentum_universe(db)
    baseline = _scored(db, rebalance="weekly", universe="current")
    guarded = _scored(
        db,
        rebalance="weekly",
        universe="current",
        risk=RiskSpec(stop_loss_pct=0.01, cooldown_days=20),
    )

    assert guarded.ok, guarded.error
    assert not guarded.backtest.trades.empty
    # 风控卖出警告：含原因（止损）与冷却口径（20 个交易日内不买回）。
    risk_warnings = [w for w in guarded.warnings if "风控卖出" in w]
    assert risk_warnings and "止损" in risk_warnings[0]
    assert "20 个交易日内不买回" in risk_warnings[0]
    # 基线未启用风控 ⇒ 不得出现风控警告（负向对照）。
    assert not any("风控卖出" in w for w in baseline.warnings)
    # 风控确实改变了回放（止损全清 + 冷却阻止买回，成交明细与基线不同）。
    assert not guarded.backtest.trades.equals(baseline.backtest.trades)


def test_runner_monthly_rebalance_trades_after_month_end(tmp_path) -> None:
    """月频：所有成交都发生在「月末信号日」的次一交易日。"""
    db = _open(tmp_path / "runner_monthly.db")
    _seed_momentum_universe(db)
    result = _scored(db, rebalance="monthly", universe="current")

    assert result.ok, result.error
    dates = pd.DatetimeIndex(pd.to_datetime(result.backtest.equity["date"]))
    assert not result.backtest.trades.empty
    for _, trade in result.backtest.trades.iterrows():
        position = int(dates.get_loc(pd.Timestamp(trade["date"])))
        assert position > 0
        signal_day = dates[position - 1]
        assert (signal_day.year, signal_day.month) != (
            dates[position].year,
            dates[position].month,
        )


# --------------------------------------------------------------------------- #
# 4) PIT 口径
# --------------------------------------------------------------------------- #
def test_runner_pit_mode_restricts_picks_to_members(tmp_path) -> None:
    """PIT：名单与成交只含「当时生效名单」的成员，换名单后旧成员不得再出现。"""
    db = _open(tmp_path / "runner_pit.db")
    days = _biz_days(320)
    symbols = _seed_momentum_universe(db)
    early, late = symbols[:8], symbols[8:]
    _seed_constituents(db, {days[0]: early, days[150]: late})

    result = _scored(
        db, rebalance="weekly", universe="pit", factors=["ret_5"], min_stocks=3
    )
    assert result.ok, result.error
    assert set(result.scoring.universe) == set(symbols)  # 研究池 = 快照并集

    # 名单成员约束：day < 换名单日 → 只含早名单；之后 → 只含晚名单。
    swap = pd.Timestamp(days[150])
    for day, picked in result.picks.items():
        expected = set(early) if day < swap else set(late)
        assert set(picked) <= expected, (day, picked)

    # 买入同样受约束（卖出不受约束：换名单后退出旧成员是正常清仓），
    # 且换名单前后都有成交（保证测试非空转）。
    trades = result.backtest.trades
    assert not trades.empty
    dates = pd.DatetimeIndex(pd.to_datetime(result.backtest.equity["date"]))
    buys = trades[trades["side"] == "buy"]
    assert not buys.empty
    for _, trade in buys.iterrows():
        position = int(dates.get_loc(pd.Timestamp(trade["date"])))
        signal_day = dates[position - 1]
        expected = set(early) if signal_day < swap else set(late)
        assert trade["symbol"] in expected
    assert (trades["date"] < swap).any() and (trades["date"] >= swap).any()


def test_runner_pit_without_snapshots_fails(tmp_path) -> None:
    """PIT 口径但库内无成分股快照 ⇒ ok=False（显式失败，不静默降级）。"""
    db = _open(tmp_path / "runner_pit_missing.db")
    _seed_momentum_universe(db, days_count=120)
    result = _scored(db, universe="pit", rebalance="weekly")

    assert result.ok is False
    assert "成分股快照" in result.error
    assert result.backtest is None


# --------------------------------------------------------------------------- #
# 5) 失败路径
# --------------------------------------------------------------------------- #
def test_runner_warmup_only_fails(tmp_path) -> None:
    """权重预热不足（打分全 NaN）⇒ ok=False 且错误含「预热」、无误产生名单。"""
    db = _open(tmp_path / "runner_warmup.db")
    _seed_momentum_universe(db, days_count=40)
    result = _scored(db, universe="current", min_ic_days=120)

    assert result.ok is False
    assert "预热" in result.error
    assert result.picks == {}


def test_runner_empty_db_fails(tmp_path) -> None:
    """空库 ⇒ ok=False（错误信息非空，不抛异常）。"""
    db = _open(tmp_path / "runner_empty.db")
    result = _scored(db, universe="current")
    assert result.ok is False
    assert result.error


# --------------------------------------------------------------------------- #
# 6) 规格校验与默认值
# --------------------------------------------------------------------------- #
def test_scored_spec_defaults_and_validation(tmp_path) -> None:
    """默认口径 = PIT / 月频 / Top-30；非法参数抛 ValueError，runner 类型错误抛 TypeError。"""
    spec = ScoredBacktestSpec()
    assert spec.top_n == DEFAULT_SCORED_TOP_N == 30
    assert spec.rebalance == DEFAULT_SCORED_REBALANCE == "monthly"
    assert spec.universe == DEFAULT_SCORED_UNIVERSE == "pit"
    assert spec.factors is None
    assert spec.name == "多因子打分选股"

    with pytest.raises(ValueError):
        ScoredBacktestSpec(top_n=0)
    with pytest.raises(ValueError):
        ScoredBacktestSpec(rebalance="hourly")
    with pytest.raises(ValueError):
        ScoredBacktestSpec(universe="foo")
    with pytest.raises(ValueError):
        ScoredBacktestSpec(costs="cheap")
    with pytest.raises(ValueError):
        ScoredBacktestSpec(initial_cash=0)
    with pytest.raises(ValueError, match="risk"):
        ScoredBacktestSpec(risk="stop-loss")

    db = _open(tmp_path / "runner_type.db")
    with pytest.raises(TypeError):
        ScoredBacktestRunner(db=db).run("not-a-spec")