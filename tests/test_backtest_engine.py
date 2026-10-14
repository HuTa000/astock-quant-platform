"""P7-S1 策略回测引擎行为单测。

覆盖：
    1. T+1 成交：T 日收盘出信号、T+1 开盘价成交（绝不用 T 日收盘价）；
    2. 100 股整手：不足一手的零头不成交；
    3. 成本扣减：默认费率（含最低佣金 5 元）从现金中扣除；
    4. 滑点：成交价按方向调整；
    5. 无前视：信号只用 ≤ 当日数据（条件 asof 回溯），信号日不成交；
    6. 停牌（当日无 bar）不可成交 + 警告；一字板（high=low）不可成交 + 警告；
    7. 指标表达式条件（rolling 预热期不产生信号）；
    8. 周频调仓：本周最后交易日出信号、次一交易日成交；
    9. top_n + rank_by：按排序值降序取前 N；
    10. 全额再平衡：先卖后买换仓、实现盈亏入 sell 成交；
    11. 基准：库内无指数 → 等权代理 + 警告；库内有 sh.000300 → INDEX 口径；
    12. 复权口径：显式股票池缺 qfq → 整体回退未复权 + 警告；自动池排除指数；
    13. 牛熊拆分：单调上涨基准 ⇒ 全「牛市」；
    14. 窗口裁剪：start/end 生效、首日净值 = 1.0、契约列完整；
    15. 失败路径：空库 / 全信号失败 ⇒ ok=False（不抛异常）；
    16. 规格校验：空条件 / 基本面条件（前视）/ 非法 rebalance / logic / top_n ⇒ ValueError。
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.store.db import Database
from app.engine.backtest import (
    BENCHMARK_KIND_INDEX,
    BENCHMARK_KIND_PROXY,
    DEFAULT_COST_MODEL,
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    ZERO_COST_MODEL,
    BacktestEngine,
    CostModel,
    StrategySpec,
)
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
    """生成 count 个「交易日」日期串（工作日口径，便于周频调仓测试）。"""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=count)]


def _flat(days, price: float) -> dict:
    """常价行情：``{日期: (open, high, low, close)}``（high≠low，保证可成交）。"""
    return {day: (price, price + 0.5, price - 0.5, price) for day in days}


def _seed(db: Database, symbol: str, spec: dict, adjusts=("", "qfq")) -> None:
    """按 ``{日期: (o, h, l, c)}`` 写入日线；默认同时落未复权与前复权两套。"""
    for adjust in adjusts:
        frame = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp(day),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": 100.0,
                    "amount": float(c) * 100.0,
                }
                for day, (o, h, l, c) in spec.items()
            ]
        )
        frame["symbol"] = symbol
        frame["period"] = "d"
        frame["adjust"] = adjust
        db.upsert_bars(frame)


def _spec(conditions, symbols, **kw) -> StrategySpec:
    """构造测试用策略规格（默认零成本 + 10 万初始资金）。"""
    kw.setdefault("costs", ZERO_COST_MODEL)
    kw.setdefault("initial_cash", 100_000.0)
    return StrategySpec(
        name=kw.pop("name", "t"), conditions=conditions, symbols=symbols, **kw
    )


# --------------------------------------------------------------------------- #
# 1~2. T+1 成交与整手
# --------------------------------------------------------------------------- #
def test_fill_at_next_day_open(tmp_path) -> None:
    """T 日收盘出信号、T+1 开盘成交；100 股整手向下取整。"""
    db = _open(tmp_path / "t1.db")
    _seed(db, "000001", {
        "2024-01-01": (9.0, 9.5, 8.5, 10.0),     # 信号日（close=10）
        "2024-01-02": (11.0, 11.5, 10.5, 10.0),  # 成交日：open=11 ≠ 收盘 10
        "2024-01-03": (10.0, 10.5, 9.5, 10.0),
    })
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"])
    )

    assert result.ok
    buy = result.trades.iloc[0]
    assert buy["side"] == "buy"
    assert buy["date"] == pd.Timestamp("2024-01-02")
    assert float(buy["price"]) == pytest.approx(11.0)   # 次日开盘价
    assert int(buy["qty"]) == 9000                      # 100000 // (11×100) = 90 手
    assert int(buy["qty"]) % 100 == 0
    # 净值表首日 1.0、契约列完整、回撤 ≤ 0
    assert list(result.equity.columns) == EQUITY_COLUMNS
    assert list(result.trades.columns) == TRADE_COLUMNS
    assert float(result.equity.iloc[0]["nav"]) == pytest.approx(1.0)
    assert (result.equity["drawdown"] <= 0).all()


def test_lot_size_round_down(tmp_path) -> None:
    """不足 100 股的零头不成交（10000 元 @ 33.33 → 3 手 = 300 股）。"""
    db = _open(tmp_path / "t2.db")
    _seed(db, "000001", _flat(_biz_days(3), 33.33), adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"], initial_cash=10_000.0)
    )

    buy = result.trades.iloc[0]
    assert int(buy["qty"]) == 300
    assert float(buy["amount"]) == pytest.approx(9_999.0)


# --------------------------------------------------------------------------- #
# 3~4. 成本与滑点
# --------------------------------------------------------------------------- #
def test_default_costs_deducted_with_min_commission(tmp_path) -> None:
    """默认费率：3000 元买 10 元股票 → 最低佣金生效，只能买 2 手（扣 5.02 元费用）。"""
    db = _open(tmp_path / "t3.db")
    _seed(db, "000001", _flat(_biz_days(3), 10.0), adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)], ["000001"],
            costs=DEFAULT_COST_MODEL, initial_cash=3_000.0,
        )
    )

    buy = result.trades.iloc[0]
    assert int(buy["qty"]) == 200                          # 3 手含费买不起（3005.03 > 3000）
    assert float(buy["amount"]) == pytest.approx(2_000.0)
    assert float(buy["cost"]) == pytest.approx(5.02)       # max(2000×0.00025, 5) + 0.02


def test_slippage_adjusts_fill_price(tmp_path) -> None:
    """100bp 滑点：买入成交价 = 开盘价 × 1.01。"""
    db = _open(tmp_path / "t4.db")
    _seed(db, "000001", _flat(_biz_days(3), 10.0), adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)], ["000001"],
            costs=CostModel(slippage_bps=100.0),
        )
    )

    assert float(result.trades.iloc[0]["price"]) == pytest.approx(10.1)


# --------------------------------------------------------------------------- #
# 5. 无前视
# --------------------------------------------------------------------------- #
def test_no_lookahead_signal_to_next_open(tmp_path) -> None:
    """信号只依赖 ≤ 当日数据：close>10 在 T 日收盘才成立时，T+1 开盘才成交。"""
    db = _open(tmp_path / "t5.db")
    _seed(db, "000001", {
        "2024-01-01": (5.0, 5.5, 4.5, 5.0),      # close=5 不满足
        "2024-01-02": (6.0, 20.5, 5.5, 20.0),    # 收盘才跳涨（close=20）
        "2024-01-03": (21.0, 21.5, 20.5, 21.0),
        "2024-01-04": (21.0, 21.5, 20.5, 21.0),
    })
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 10)], ["000001"])
    )

    trades = result.trades
    assert pd.Timestamp("2024-01-02") not in set(trades["date"])   # 信号日不成交
    assert trades.iloc[0]["date"] == pd.Timestamp("2024-01-03")
    assert float(trades.iloc[0]["price"]) == pytest.approx(21.0)   # 次日开盘
    assert float(trades.iloc[0]["price"]) != pytest.approx(20.0)   # 绝不是信号日收盘


# --------------------------------------------------------------------------- #
# 6. 停牌 / 一字板
# --------------------------------------------------------------------------- #
def test_suspended_stock_not_tradeable(tmp_path) -> None:
    """停牌（当日无 bar）不可成交 → 跳过 + 警告；复牌后补买。"""
    db = _open(tmp_path / "t6.db")
    _seed(db, "000001", {                         # 缺 01-02（停牌）
        "2024-01-01": (10.0, 10.5, 9.5, 10.0),
        "2024-01-03": (10.0, 10.5, 9.5, 10.0),
        "2024-01-04": (10.0, 10.5, 9.5, 10.0),
    })
    _seed(db, "000002", _flat(_biz_days(4), 10.0))   # 全勤，提供 01-02 日历日
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001", "000002"],
              initial_cash=200_000.0)
    )

    assert result.ok
    assert any("停牌" in w for w in result.warnings)
    a_trades = result.trades[result.trades["symbol"] == "000001"]
    b_trades = result.trades[result.trades["symbol"] == "000002"]
    assert b_trades.iloc[0]["date"] == pd.Timestamp("2024-01-02")   # B 正常成交
    assert a_trades.iloc[0]["date"] == pd.Timestamp("2024-01-03")   # A 复牌后才成交


def test_limit_one_word_board_not_tradeable(tmp_path) -> None:
    """一字板（high == low）近似不可成交 → 跳过 + 警告；正常日恢复。"""
    db = _open(tmp_path / "t7.db")
    _seed(db, "000001", {
        "2024-01-01": (10.0, 10.5, 9.5, 10.0),
        "2024-01-02": (11.0, 11.0, 11.0, 11.0),   # 一字涨停：买不进
        "2024-01-03": (12.0, 12.5, 11.5, 12.0),
        "2024-01-04": (12.0, 12.5, 11.5, 12.0),
    })
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"])
    )

    assert any("一字板" in w for w in result.warnings)
    assert result.trades.iloc[0]["date"] == pd.Timestamp("2024-01-03")


# --------------------------------------------------------------------------- #
# 7. 指标表达式条件（rolling 预热）
# --------------------------------------------------------------------------- #
def test_indicator_expression_condition_with_warmup(tmp_path) -> None:
    """表达式条件：rolling(20) 预热期（前 19 根）不产生信号，第 20 根收盘起成立。"""
    db = _open(tmp_path / "t8.db")
    days = _biz_days(25)
    rising = {day: (10.0 + i, 10.5 + i, 9.5 + i, 10.0 + i) for i, day in enumerate(days)}
    _seed(db, "000001", rising, adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close - close.rolling(20).mean()", ">", 0)], ["000001"])
    )

    assert result.ok
    # 第 20 根（index 19）收盘首次满足 → 第 21 根（index 20）开盘成交
    assert result.trades.iloc[0]["date"] == pd.Timestamp(days[20])


# --------------------------------------------------------------------------- #
# 8. 周频调仓
# --------------------------------------------------------------------------- #
def test_weekly_rebalance_signal_on_last_day_of_week(tmp_path) -> None:
    """周频：本周最后交易日收盘出信号、次周第一个交易日开盘成交。"""
    db = _open(tmp_path / "t9.db")
    days = _biz_days(7)   # 01-01(一)~01-05(五) + 01-08(一)~01-09(二)
    _seed(db, "000001", _flat(days, 10.0), adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"], rebalance="weekly")
    )

    assert len(result.trades) >= 1
    assert result.trades.iloc[0]["date"] == pd.Timestamp(days[5])   # 次周周一
    assert result.trades.iloc[0]["side"] == "buy"
    assert set(result.trades["date"]) == {pd.Timestamp(days[5])}    # 周内不重复调仓


# --------------------------------------------------------------------------- #
# 9. top_n + rank_by
# --------------------------------------------------------------------------- #
def test_top_n_rank_by_picks_highest(tmp_path) -> None:
    """top_n + rank_by：按排序值降序取前 N（close 大者优先）。"""
    db = _open(tmp_path / "t10.db")
    days = _biz_days(4)
    _seed(db, "000001", _flat(days, 30.0))
    _seed(db, "000002", _flat(days, 20.0))
    _seed(db, "000003", _flat(days, 10.0))
    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 5)], ["000001", "000002", "000003"],
            top_n=1, rank_by="close",
        )
    )

    assert set(result.universe) == {"000001", "000002", "000003"}
    assert set(result.trades["symbol"]) == {"000001"}   # 只买排序值最大的


# --------------------------------------------------------------------------- #
# 10. 全额再平衡换仓
# --------------------------------------------------------------------------- #
def test_full_rebalance_switch_symbols(tmp_path) -> None:
    """换仓：旧标的跌破条件后次日开盘清仓（实现盈亏入 sell），资金买入新标的。"""
    db = _open(tmp_path / "t11.db")
    days = _biz_days(5)
    _seed(db, "000001", {
        days[0]: (20.0, 20.5, 19.5, 20.0),
        days[1]: (20.0, 20.5, 19.5, 20.0),
        days[2]: (5.0, 5.5, 4.5, 5.0),    # 跌破 → 次日卖出
        days[3]: (5.0, 5.5, 4.5, 5.0),
        days[4]: (5.0, 5.5, 4.5, 5.0),
    })
    _seed(db, "000002", {
        days[0]: (5.0, 5.5, 4.5, 5.0),
        days[1]: (5.0, 5.5, 4.5, 5.0),
        days[2]: (20.0, 20.5, 19.5, 20.0),  # 站上 → 次日买入
        days[3]: (20.0, 20.5, 19.5, 20.0),
        days[4]: (20.0, 20.5, 19.5, 20.0),
    })
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 10)], ["000001", "000002"])
    )

    trades = result.trades
    assert len(trades) == 3
    buy_a, sell_a, buy_b = trades.iloc[0], trades.iloc[1], trades.iloc[2]
    # 01-02：买 A（100000 // 2000 = 50 手）
    assert (buy_a["symbol"], buy_a["side"], buy_a["date"].strftime("%m-%d")) == (
        "000001", "buy", "01-02",
    )
    assert int(buy_a["qty"]) == 5000
    # 01-04：先卖 A（100000 → 25000，亏 -75000），再买 B（25000 // 2000 = 12 手）
    assert (sell_a["symbol"], sell_a["side"]) == ("000001", "sell")
    assert sell_a["date"] == pd.Timestamp(days[3])
    assert int(sell_a["qty"]) == 5000
    assert float(sell_a["pnl"]) == pytest.approx(-75_000.0)
    assert (buy_b["symbol"], buy_b["side"]) == ("000002", "buy")
    assert buy_b["date"] == pd.Timestamp(days[3])
    assert int(buy_b["qty"]) == 1200
    assert result.metrics.trade_count == 1   # 仅 1 笔已平仓


# --------------------------------------------------------------------------- #
# 11. 基准：代理 / 真实指数
# --------------------------------------------------------------------------- #
def test_proxy_benchmark_when_index_missing(tmp_path) -> None:
    """库内无指数 ⇒ 等权代理口径 + 警告（等权 ≠ 市值加权）。"""
    db = _open(tmp_path / "t12.db")
    _seed(db, "000001", _flat(_biz_days(4), 10.0), adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"])
    )

    assert result.ok
    assert result.benchmark_kind == BENCHMARK_KIND_PROXY
    assert any("等权" in w for w in result.warnings)
    assert result.equity["benchmark_nav"].notna().all()
    assert float(result.equity["benchmark_nav"].iloc[0]) == pytest.approx(1.0)


def test_index_benchmark_when_seeded(tmp_path) -> None:
    """库内有 sh.000300 ⇒ INDEX 口径（首日归一为 1.0、无代理警告）。"""
    db = _open(tmp_path / "t13.db")
    _seed(db, "000001", _flat(_biz_days(4), 10.0), adjusts=("qfq",))
    _seed(db, "sh.000300", {                      # 指数只落未复权
        "2024-01-01": (3500.0, 3510.0, 3490.0, 3500.0),
        "2024-01-02": (3500.0, 3535.0, 3495.0, 3535.0),
        "2024-01-03": (3535.0, 3540.0, 3500.0, 3540.0),
    }, adjusts=("",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"])
    )

    assert result.benchmark_kind == BENCHMARK_KIND_INDEX
    assert result.benchmark_symbol == "sh.000300"
    assert not any("等权" in w for w in result.warnings)
    bench = result.equity["benchmark_nav"]
    assert float(bench.iloc[0]) == pytest.approx(1.0)
    assert float(bench.iloc[1]) == pytest.approx(3535.0 / 3500.0)


# --------------------------------------------------------------------------- #
# 12. 复权口径 / 自动股票池排除指数
# --------------------------------------------------------------------------- #
def test_missing_qfq_falls_back_to_raw(tmp_path) -> None:
    """显式股票池缺 qfq ⇒ 整体回退未复权 + 警告（除权日会有假跌）。"""
    db = _open(tmp_path / "t14.db")
    _seed(db, "000001", _flat(_biz_days(4), 10.0), adjusts=("",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"])
    )

    assert result.ok
    assert any("前复权" in w for w in result.warnings)


def test_auto_universe_from_db_excludes_index(tmp_path) -> None:
    """不指定 symbols：股票池 = 库内有 qfq 的标的；指数只作基准不入池。"""
    db = _open(tmp_path / "t15.db")
    _seed(db, "000001", _flat(_biz_days(4), 10.0))
    _seed(db, "000002", _flat(_biz_days(4), 10.0))
    _seed(db, "sh.000300", _flat(_biz_days(4), 3000.0), adjusts=("",))
    result = BacktestEngine(db=db).run(
        StrategySpec(name="auto", conditions=[Condition("close", ">", 0)])
    )

    assert result.ok
    assert set(result.universe) == {"000001", "000002"}


def test_index_only_database_not_backtestable(tmp_path) -> None:
    """库内仅有指数数据 ⇒ 股票池为空，ok=False（指数不能直接买卖）。"""
    db = _open(tmp_path / "t16.db")
    _seed(db, "sh.000300", _flat(_biz_days(4), 3000.0), adjusts=("",))
    result = BacktestEngine(db=db).run(
        StrategySpec(name="idxOnly", conditions=[Condition("close", ">", 0)])
    )

    assert not result.ok
    assert result.universe == []


# --------------------------------------------------------------------------- #
# 13. 牛熊拆分
# --------------------------------------------------------------------------- #
def test_regime_split_all_bull_for_rising_index(tmp_path) -> None:
    """单调上涨基准 ⇒ 全部交易日归入「牛市」（净值恒 ≥ 其均线）。"""
    db = _open(tmp_path / "t17.db")
    days = _biz_days(10)
    _seed(db, "000001", _flat(days, 10.0), adjusts=("qfq",))
    rising = {
        day: (3000.0 + 10 * i, 3010.0 + 10 * i, 2995.0 + 10 * i, 3000.0 + 10 * i)
        for i, day in enumerate(days)
    }
    _seed(db, "sh.000300", rising, adjusts=("",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"])
    )

    regime = result.regime
    assert list(regime["regime"]) == ["牛市"]
    assert int(regime.iloc[0]["days"]) == len(days)
    assert float(regime.iloc[0]["share"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 14. 窗口裁剪
# --------------------------------------------------------------------------- #
def test_window_slicing(tmp_path) -> None:
    """start/end 生效：净值表只含窗口内交易日。"""
    db = _open(tmp_path / "t18.db")
    _seed(db, "000001", _flat(_biz_days(10), 10.0), adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"]),
        start="2024-01-03", end="2024-01-08",
    )

    dates = result.equity["date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    assert float(result.equity.iloc[0]["nav"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 15. 失败路径（不抛异常）
# --------------------------------------------------------------------------- #
def test_empty_database_fails_gracefully(tmp_path) -> None:
    """空库 ⇒ ok=False + 可读原因。"""
    db = _open(tmp_path / "t19.db")
    result = BacktestEngine(db=db).run(
        StrategySpec(name="empty", conditions=[Condition("close", ">", 0)])
    )

    assert not result.ok
    assert "没有可用于回测" in result.error
    assert result.equity.empty
    assert list(result.equity.columns) == EQUITY_COLUMNS


def test_all_signals_failed_fails_gracefully(tmp_path) -> None:
    """全部标的的信号表达式求值失败 ⇒ ok=False + 警告。"""
    db = _open(tmp_path / "t20.db")
    _seed(db, "000001", _flat(_biz_days(4), 10.0), adjusts=("qfq",))
    result = BacktestEngine(db=db).run(
        _spec([Condition("no_such_name", ">", 1)], ["000001"])
    )

    assert not result.ok
    assert any("求值失败" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# 16. 规格校验 / 调用输入错误
# --------------------------------------------------------------------------- #
def test_spec_validation_errors() -> None:
    """规格层输入错误 ⇒ ValueError（含基本面条件的前视偏差拒绝）。"""
    with pytest.raises(ValueError):
        StrategySpec(name="x")                                       # 空条件
    with pytest.raises(ValueError):
        StrategySpec(name="x", conditions=[Condition("pe", "<", 30)])  # 基本面（前视）
    with pytest.raises(ValueError):
        StrategySpec(name="x", conditions=[Condition("close", ">", 0)], rebalance="hourly")
    with pytest.raises(ValueError):
        StrategySpec(name="x", conditions=[Condition("close", ">", 0)], top_n=0)
    with pytest.raises(ValueError):
        StrategySpec(name="x", conditions=[Condition("close", ">", 0)], logic="XOR")
    # dict 条件亦可（复用 Condition.from_dict）
    ok_spec = StrategySpec(
        name="ok", conditions=[{"field": "close", "op": ">", "value": 0}]
    )
    assert ok_spec.condition_fields == ["close"]


def test_run_input_errors_raise(tmp_path) -> None:
    """spec 类型错误 / 起止颠倒 ⇒ TypeError / ValueError（调用方输入错误不吞）。"""
    db = _open(tmp_path / "t21.db")
    engine = BacktestEngine(db=db)
    with pytest.raises(TypeError):
        engine.run({"conditions": []})  # type: ignore[arg-type]
    spec = StrategySpec(name="x", conditions=[Condition("close", ">", 0)])
    with pytest.raises(ValueError):
        engine.run(spec, start="2024-02-01", end="2024-01-01")