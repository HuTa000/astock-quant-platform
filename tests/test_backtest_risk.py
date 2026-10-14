"""P7-S6 策略风控与交易规则行为单测。

覆盖：
    1. ``RiskSpec`` 契约 —— 默认值、比例 / 冷却 / 窗口 / 仓位校验、``enabled`` 组合语义；
    2. ``evaluate_exit`` 纯函数 —— 止损 / 固定止盈 / 移动止盈边界、优先级（止损优先）、
       非法输入（缺失 / 非正 / NaN）一律不判定；
    3. ``StrategySpec.risk`` —— ``None`` 默认通行、非 ``RiskSpec`` 类型抛 ``ValueError``；
    4. 止损 —— 收盘触发、**次日开盘**全清（价格 = 次日开盘价）、pnl < 0；
       冷却 20 个交易日内不买回、第 21 个交易日信号起可买回；
    5. 冷却 0 —— 触发卖出后次日即可买回（无空转限制）；
    6. 止损卖出被停牌阻断 —— 不成交 + 警告，恢复交易日后重新触发卖出；
    7. 固定止盈 / 移动止盈 —— 各自触发时点（边界前不触发）与次日开盘成交；
    8. ``risk=None`` 与「全禁用 RiskSpec」净值 / 成交完全一致（不影响既有行为）；
    9. 环境开关 —— 熊市调仓仅建 50% 仓；转熊（当日收盘判定）**次日开盘防御减仓**
       （卖出价格取次日开盘价，验证无前视）；牛市保持满仓；无指数数据时降级 + 警告。
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.store.db import Database
from app.engine.backtest import (
    RISK_STOP_LOSS,
    RISK_TAKE_PROFIT,
    RISK_TRAILING_STOP,
    ZERO_COST_MODEL,
    BacktestEngine,
    RiskSpec,
    StrategySpec,
    evaluate_exit,
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
    """生成 count 个「交易日」日期串（工作日口径，便于周 / 月频调仓测试）。"""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=count)]


def _flat(days, price: float) -> dict:
    """常价行情：``{日期: (open, high, low, close)}``（high≠low，保证可成交）。"""
    return {day: (price, price + 0.5, price - 0.5, price) for day in days}


def _seed(db: Database, symbol: str, spec: dict, adjusts=("qfq",)) -> None:
    """按 ``{日期: (o, h, l, c)}`` 写入日线；默认仅前复权。"""
    for adjust in adjusts:
        frame = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp(day),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": 1_000_000.0,
                    "amount": float(c) * 1_000_000.0,
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


def _stoploss_market(days: list[str]) -> dict:
    """构造止损行情：day1 开盘 10 买入 → day2 收盘 8.9（−11%）触发 → day3 开盘 8.8 卖出。"""
    market: dict[str, tuple] = {}
    for index, day in enumerate(days):
        if index <= 1:
            market[day] = (10.0, 10.5, 9.5, 10.0)
        elif index == 2:
            market[day] = (9.6, 9.8, 8.5, 8.9)  # 收盘相对成本 −11% → 触发止损
        else:
            market[day] = (8.8, 9.3, 8.3, 8.8)  # day3 开盘 8.8 = 实际卖出价
    return market


def _stoploss_db(tmp_path, name: str = "risk.db") -> tuple[Database, list[str]]:
    """播种止损场景（单标的 30 个交易日），返回 ``(db, days)``。"""
    days = _biz_days(30)
    db = _open(tmp_path / name)
    _seed(db, "000001", _stoploss_market(days), adjusts=("qfq",))
    return db, days


def _declining_index(days: list[str], start_price: float = 3000.0) -> dict:
    """单调下跌的指数行情（用于制造熊市标签）。"""
    market: dict[str, tuple] = {}
    price = start_price
    for day in days:
        price *= 0.99
        market[day] = (price, price * 1.002, price * 0.998, price)
    return market


def _rising_index(days: list[str], start_price: float = 3000.0) -> dict:
    """单调上涨的指数行情（用于制造牛市标签）。"""
    market: dict[str, tuple] = {}
    price = start_price
    for day in days:
        price *= 1.01
        market[day] = (price, price * 1.002, price * 0.998, price)
    return market


# --------------------------------------------------------------------------- #
# 1. RiskSpec 契约
# --------------------------------------------------------------------------- #
def test_risk_spec_defaults_and_validation() -> None:
    """默认值 = 止损 10% + 冷却 20 日；非法比例 / 冷却 / 窗口 / 仓位抛 ValueError。"""
    spec = RiskSpec()
    assert spec.stop_loss_pct == pytest.approx(0.10)
    assert spec.take_profit_pct is None
    assert spec.trailing_stop_pct is None
    assert spec.cooldown_days == 20
    assert spec.regime_filter is False
    assert spec.regime_window == 200
    assert spec.regime_exposure == pytest.approx(0.5)
    assert spec.has_exit_rules is True
    assert spec.enabled is True

    for kwargs in (
        {"stop_loss_pct": 0.0},
        {"stop_loss_pct": 1.0},
        {"take_profit_pct": -0.1},
        {"trailing_stop_pct": 1.5},
        {"cooldown_days": -1},
        {"regime_window": 0},
        {"regime_exposure": 0.0},
        {"regime_exposure": 1.5},
    ):
        with pytest.raises(ValueError):
            RiskSpec(**kwargs)


def test_risk_spec_enabled_semantics() -> None:
    """全禁用（三个 None + 环境关）⇒ ``enabled=False``；仅环境开关 ⇒ ``enabled=True``。"""
    disabled = RiskSpec(
        stop_loss_pct=None,
        take_profit_pct=None,
        trailing_stop_pct=None,
        regime_filter=False,
        regime_exposure=1.0,  # 边界 (0,1] 上沿放行
    )
    assert disabled.has_exit_rules is False
    assert disabled.enabled is False

    regime_only = RiskSpec(stop_loss_pct=None, regime_filter=True)
    assert regime_only.has_exit_rules is False
    assert regime_only.enabled is True


# --------------------------------------------------------------------------- #
# 2. evaluate_exit 纯函数
# --------------------------------------------------------------------------- #
def test_evaluate_exit_stop_loss_boundary() -> None:
    """止损：相对成本 −10% 边界含（``<=``），未到边界不触发。"""
    spec = RiskSpec(stop_loss_pct=0.10, take_profit_pct=None)
    assert evaluate_exit(9.0, 10.0, 10.0, spec) == RISK_STOP_LOSS
    assert evaluate_exit(8.0, 10.0, 10.0, spec) == RISK_STOP_LOSS
    assert evaluate_exit(9.01, 10.0, 10.0, spec) is None


def test_evaluate_exit_take_profit_boundary() -> None:
    """固定止盈：相对成本 +20% 边界含（``>=``），未到边界不触发。"""
    spec = RiskSpec(stop_loss_pct=None, take_profit_pct=0.20)
    assert evaluate_exit(12.0, 10.0, 12.0, spec) == RISK_TAKE_PROFIT
    assert evaluate_exit(11.99, 10.0, 11.99, spec) is None


def test_evaluate_exit_trailing_stop_boundary() -> None:
    """移动止盈：峰值 13 → 回撤 10% 边界 = 11.7 触发；无峰值不判定。"""
    spec = RiskSpec(stop_loss_pct=None, trailing_stop_pct=0.10)
    assert evaluate_exit(11.7, 10.0, 13.0, spec) == RISK_TRAILING_STOP
    assert evaluate_exit(11.71, 10.0, 13.0, spec) is None
    assert evaluate_exit(12.0, 10.0, None, spec) is None


def test_evaluate_exit_priority_and_invalid_inputs() -> None:
    """同时满足止损与移动止盈 ⇒ 记「止损」（保本金优先）；非法输入一律不判定。"""
    both = RiskSpec(stop_loss_pct=0.10, trailing_stop_pct=0.10)
    # 成本 10、峰值 20、收盘 8：−20% 触发止损、自峰值回撤 60% 也触发移动止盈
    assert evaluate_exit(8.0, 10.0, 20.0, both) == RISK_STOP_LOSS
    # 收盘 9.5：未到止损（−5%）、已自峰值回撤 → 移动止盈
    assert evaluate_exit(9.5, 10.0, 20.0, both) == RISK_TRAILING_STOP

    disabled = RiskSpec(
        stop_loss_pct=None, take_profit_pct=None, trailing_stop_pct=None
    )
    assert evaluate_exit(10.0, 10.0, 10.0, disabled) is None
    assert evaluate_exit(10.0, 10.0, 10.0, None) is None
    assert evaluate_exit(None, 10.0, 10.0, both) is None
    assert evaluate_exit(10.0, 0.0, 10.0, both) is None  # 成本非正不可判定
    assert evaluate_exit(float("nan"), 10.0, 10.0, both) is None


# --------------------------------------------------------------------------- #
# 3. StrategySpec.risk 类型契约
# --------------------------------------------------------------------------- #
def test_strategy_spec_risk_type_check() -> None:
    """``risk`` 默认 ``None``；传 ``RiskSpec`` 通行；类型错误抛 ``ValueError``。"""
    conditions = [Condition("close", ">", 0)]
    assert StrategySpec(name="t", conditions=conditions).risk is None
    assert StrategySpec(name="t", conditions=conditions, risk=None).risk is None
    spec = StrategySpec(name="t", conditions=conditions, risk=RiskSpec())
    assert isinstance(spec.risk, RiskSpec)

    with pytest.raises(ValueError):
        StrategySpec(name="t", conditions=conditions, risk="stop_loss")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 4~5. 止损触发、次日开盘成交与冷却期
# --------------------------------------------------------------------------- #
def test_stop_loss_sells_next_open_with_loss(tmp_path) -> None:
    """止损收盘触发、**次日开盘**全清：成交价 = 次日开盘价（非触发日收盘）、pnl < 0。"""
    db, days = _stoploss_db(tmp_path)
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"], risk=RiskSpec(stop_loss_pct=0.10))
    )

    assert result.ok
    assert len(result.trades) == 3
    buy = result.trades.iloc[0]
    assert buy["side"] == "buy" and int(buy["qty"]) == 10000
    stop_sell = result.trades.iloc[1]
    assert stop_sell["side"] == "sell"
    assert stop_sell["date"] == pd.Timestamp(days[3])  # day2 收盘触发 → day3 开盘成交
    assert float(stop_sell["price"]) == pytest.approx(8.8)  # day3 开盘价（非 day2 收盘 8.9）
    assert int(stop_sell["qty"]) == 10000
    assert float(stop_sell["pnl"]) < 0
    assert any("止损 1 次" in w for w in result.warnings)


def test_stop_loss_cooldown_blocks_rebuy_for_20_days(tmp_path) -> None:
    """触发日（day2）后 20 个交易日不买回；第 21 个交易日信号起（day23）恢复可买。"""
    db, days = _stoploss_db(tmp_path, name="risk2.db")
    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"], risk=RiskSpec(stop_loss_pct=0.10))
    )

    trade_dates = set(pd.to_datetime(result.trades["date"]))
    # 买入（day1）→ 止损卖出（day3）→ 冷却期空白（day4~day23）→ 买回（day24 开盘）
    assert trade_dates == {
        pd.Timestamp(days[1]),
        pd.Timestamp(days[3]),
        pd.Timestamp(days[24]),
    }
    assert any("20 个交易日内不买回" in w for w in result.warnings)


def test_cooldown_zero_allows_immediate_rebuy(tmp_path) -> None:
    """冷却 0：触发当日信号不再买回（避免同价空转），次日信号即可再买（day4 开盘买回）。"""
    db, days = _stoploss_db(tmp_path, name="risk3.db")
    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)],
            ["000001"],
            risk=RiskSpec(stop_loss_pct=0.10, cooldown_days=0),
        )
    )

    trade_dates = list(pd.to_datetime(result.trades["date"]))
    assert trade_dates == [
        pd.Timestamp(days[1]),
        pd.Timestamp(days[3]),
        pd.Timestamp(days[4]),
    ]
    assert any("触发当日不再买回" in w for w in result.warnings)


def test_stop_loss_sell_blocked_by_suspension_retries(tmp_path) -> None:
    """止损卖出日停牌（无 bar）⇒ 不成交 + 警告；恢复交易日后重新触发卖出。"""
    days = _biz_days(30)
    db = _open(tmp_path / "risk4.db")
    market = _stoploss_market(days)
    market.pop(days[3])  # day3 停牌：日历仍含该日（基准指数有 bar）
    _seed(db, "000001", market, adjusts=("qfq",))
    _seed(db, "sh.000300", _flat(days, 3000.0), adjusts=("",))

    result = BacktestEngine(db=db).run(
        _spec([Condition("close", ">", 0)], ["000001"], risk=RiskSpec(stop_loss_pct=0.10))
    )

    assert result.ok
    sells = result.trades[result.trades["side"] == "sell"]
    assert len(sells) == 1
    assert sells.iloc[0]["date"] == pd.Timestamp(days[4])  # 次日复盘重触发 → day4 开盘卖出
    assert float(sells.iloc[0]["price"]) == pytest.approx(8.8)
    assert any("停牌无成交" in w for w in result.warnings)
    assert any("止损 2 次" in w for w in result.warnings)  # 同一持仓先后两次触发


# --------------------------------------------------------------------------- #
# 6. 固定止盈 / 移动止盈
# --------------------------------------------------------------------------- #
def test_take_profit_sells_next_open(tmp_path) -> None:
    """固定止盈：+20% 收盘触发（11.9 不触发、12.1 触发）→ 次日开盘卖出。

    行情「无跳空」（每日开盘 = 前收），避免满仓调仓的整手微调干扰成交断言。
    """
    days = _biz_days(10)
    db = _open(tmp_path / "risk5.db")
    market: dict[str, tuple] = {}
    for index, day in enumerate(days):
        if index <= 1:
            market[day] = (10.0, 10.5, 9.5, 10.0)
        elif index == 2:
            market[day] = (10.0, 12.2, 9.9, 11.9)  # +19% 未到边界
        elif index == 3:
            market[day] = (11.9, 12.4, 11.8, 12.1)  # +21% 触发
        elif index == 4:
            market[day] = (12.0, 12.5, 11.9, 12.0)  # 次日开盘 12.0 卖出
        else:
            market[day] = (12.0, 12.4, 11.8, 12.0)
    _seed(db, "000001", market, adjusts=("qfq",))

    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)],
            ["000001"],
            risk=RiskSpec(stop_loss_pct=None, take_profit_pct=0.20),
        )
    )

    assert result.ok
    assert len(result.trades) == 2
    sell = result.trades.iloc[1]
    assert sell["side"] == "sell"
    assert sell["date"] == pd.Timestamp(days[4])  # day3 收盘触发 → day4 开盘成交
    assert float(sell["price"]) == pytest.approx(12.0)
    assert float(sell["pnl"]) > 0
    assert any("固定止盈 1 次" in w for w in result.warnings)


def test_trailing_stop_locks_profit(tmp_path) -> None:
    """移动止盈：峰值 13 后回撤 10%（12.5 未到位、11.6 触发）→ 次日开盘落袋。"""
    days = _biz_days(10)
    db = _open(tmp_path / "risk6.db")
    market: dict[str, tuple] = {}
    for index, day in enumerate(days):
        if index <= 1:
            market[day] = (10.0, 10.5, 9.5, 10.0)
        elif index == 2:
            market[day] = (10.0, 11.2, 9.9, 11.0)  # 峰值 11
        elif index == 3:
            market[day] = (11.0, 13.2, 10.9, 13.0)  # 峰值 13
        elif index == 4:
            market[day] = (13.0, 13.1, 12.3, 12.5)  # 回撤 3.8%，未到 10%
        elif index == 5:
            market[day] = (12.5, 12.6, 11.4, 11.6)  # 自峰值回撤 10.8% → 触发
        else:
            market[day] = (11.6, 11.9, 11.3, 11.6)
    _seed(db, "000001", market, adjusts=("qfq",))

    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)],
            ["000001"],
            risk=RiskSpec(stop_loss_pct=None, trailing_stop_pct=0.10),
        )
    )

    assert result.ok
    assert len(result.trades) == 2
    sell = result.trades.iloc[1]
    assert sell["side"] == "sell"
    assert sell["date"] == pd.Timestamp(days[6])  # day5 收盘触发 → day6 开盘成交
    assert float(sell["price"]) == pytest.approx(11.6)
    assert float(sell["pnl"]) > 0  # 锁住利润（成本 10 → 卖 11.6）
    assert any("移动止盈 1 次" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# 7. 禁用等价性
# --------------------------------------------------------------------------- #
def test_risk_none_matches_disabled_risk_nav(tmp_path) -> None:
    """``risk=None`` 与「全禁用 RiskSpec」的净值 / 成交完全一致（不破坏既有行为）。"""
    db, _days = _stoploss_db(tmp_path, name="risk7.db")
    conditions = [Condition("close", ">", 0)]
    plain = BacktestEngine(db=db).run(_spec(conditions, ["000001"]))
    disabled = BacktestEngine(db=db).run(
        _spec(
            conditions,
            ["000001"],
            risk=RiskSpec(
                stop_loss_pct=None,
                take_profit_pct=None,
                trailing_stop_pct=None,
                regime_filter=False,
            ),
        )
    )

    assert plain.ok and disabled.ok
    pd.testing.assert_series_equal(plain.equity["nav"], disabled.equity["nav"])
    pd.testing.assert_frame_equal(plain.trades, disabled.trades)


# --------------------------------------------------------------------------- #
# 8. 环境开关（牛熊仓位）
# --------------------------------------------------------------------------- #
def test_bear_regime_builds_half_position(tmp_path) -> None:
    """熊市调仓仅建 50% 仓：day1 收盘转熊 → day2 开盘调仓降到 5000 股。"""
    days = _biz_days(8)
    db = _open(tmp_path / "risk8.db")
    _seed(db, "000001", _flat(days, 10.0), adjusts=("qfq",))
    _seed(db, "sh.000300", _declining_index(days), adjusts=("",))

    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)],
            ["000001"],
            risk=RiskSpec(
                stop_loss_pct=None, regime_filter=True, regime_window=3, regime_exposure=0.5
            ),
        )
    )

    assert result.ok
    assert len(result.trades) == 2
    buy = result.trades.iloc[0]
    assert buy["date"] == pd.Timestamp(days[1]) and int(buy["qty"]) == 10000  # day0 牛市满仓
    sell = result.trades.iloc[1]
    assert sell["side"] == "sell" and sell["date"] == pd.Timestamp(days[2])
    assert int(sell["qty"]) == 5000
    assert any("熊市标签 7/8" in w for w in result.warnings)
    assert any("防御性减仓 0 次" in w for w in result.warnings)  # 有调仓挂单，无需防御减仓


def test_bear_turn_defensive_trim_next_open_without_rebalance(tmp_path) -> None:
    """转熊日无调仓挂单 ⇒ **次日开盘**防御减仓：卖出价格 = 次日开盘价（无前视）。"""
    days = _biz_days(44)  # 2024-01-01 起：index 22 = 1月末、23 = 2月首日、43 = 2月末
    db = _open(tmp_path / "risk9.db")
    market = _flat(days, 10.0)
    market[days[25]] = (9.8, 10.3, 9.7, 10.0)  # day25 开盘 9.8：验证卖出用次日开盘价
    _seed(db, "000001", market, adjusts=("qfq",))
    index_market: dict[str, tuple] = {}
    for index, day in enumerate(days):
        if index <= 23:
            index_market[day] = (3000.0, 3010.0, 2990.0, 3000.0)  # 横盘 → 牛市
        else:
            index_market[day] = (2990.0, 3000.0, 2950.0, 2960.0)  # 跌破 3 日均线 → 熊市
    _seed(db, "sh.000300", index_market, adjusts=("",))

    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)],
            ["000001"],
            rebalance="monthly",
            risk=RiskSpec(
                stop_loss_pct=None, regime_filter=True, regime_window=3, regime_exposure=0.5
            ),
        )
    )

    assert result.ok
    assert len(result.trades) == 2
    buy = result.trades.iloc[0]
    assert buy["date"] == pd.Timestamp(days[23])  # 1 月末信号 → 2 月 1 日开盘买入
    assert int(buy["qty"]) == 10000
    sell = result.trades.iloc[1]
    assert sell["side"] == "sell"
    assert sell["date"] == pd.Timestamp(days[25])  # day24 收盘转熊 → day25 开盘减仓
    assert int(sell["qty"]) == 5000
    assert float(sell["price"]) == pytest.approx(9.8)  # 次日开盘价（非 day24 收盘 10.0）
    assert any("防御性减仓 1 次" in w for w in result.warnings)


def test_regime_bull_keeps_full_position(tmp_path) -> None:
    """牛市环境：全程满仓、不防御减仓。"""
    days = _biz_days(10)
    db = _open(tmp_path / "risk10.db")
    _seed(db, "000001", _flat(days, 10.0), adjusts=("qfq",))
    _seed(db, "sh.000300", _rising_index(days), adjusts=("",))

    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)],
            ["000001"],
            risk=RiskSpec(stop_loss_pct=None, regime_filter=True, regime_window=3),
        )
    )

    assert result.ok
    assert len(result.trades) == 1  # 只建仓一次，此后无买卖
    assert int(result.trades.iloc[0]["qty"]) == 10000
    assert any("熊市标签 0/10" in w for w in result.warnings)
    assert any("防御性减仓 0 次" in w for w in result.warnings)


def test_regime_fallback_without_index_data(tmp_path) -> None:
    """库内无指数数据 ⇒ 环境判定降级为「基准净值」口径 + 警告（不静默）。"""
    days = _biz_days(6)
    db = _open(tmp_path / "risk11.db")
    _seed(db, "000001", _flat(days, 10.0), adjusts=("qfq",))

    result = BacktestEngine(db=db).run(
        _spec(
            [Condition("close", ">", 0)],
            ["000001"],
            risk=RiskSpec(stop_loss_pct=None, regime_filter=True, regime_window=3),
        )
    )

    assert result.ok
    assert any("降级" in w for w in result.warnings)
    # 单只常价 ⇒ 等权代理净值平直 ⇒ 全程牛市标签
    assert any("熊市标签 0/6" in w for w in result.warnings)