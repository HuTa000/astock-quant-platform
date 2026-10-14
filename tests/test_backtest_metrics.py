"""P7-S1 绩效指标 + 成本模型单测（手算对拍）。

全部用例为纯函数测试（不依赖 Qt / 数据库 / 指标引擎）：
    1. 成本模型：默认费率（佣金 / 最低佣金 / 印花税 / 过户费 / 滑点）；
    2. 收益类：总收益 / 年化收益（几何）手算对拍；
    3. 风险类：年化波动率 / 夏普 / 回撤序列 / 最大回撤 / Calmar；
    4. 相对基准：β=2 构造对拍 / α 年化 / 跟踪误差 / 信息比率；
    5. 交易类：已平仓筛选 / 胜率 / 盈亏比（买入与未平仓不计入）；
    6. 集成：compute_metrics 全字段 + 样本不足 / 空输入返回 None（不抛异常）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engine.backtest import (
    DEFAULT_COST_MODEL,
    ZERO_COST_MODEL,
    CostModel,
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


# --------------------------------------------------------------------------- #
# 成本模型
# --------------------------------------------------------------------------- #
def test_default_cost_buy_and_sell() -> None:
    """默认费率：佣金 0.025%（双边，最低 5 元）、印花税 0.05%（仅卖出）、过户费 0.001% 双边。"""
    costs = DEFAULT_COST_MODEL
    # 买入 10 万：佣金 25 + 过户费 1 = 26（无印花税）
    assert costs.trade_cost(100_000.0, "buy") == pytest.approx(26.0)
    # 卖出 10 万：佣金 25 + 过户费 1 + 印花税 50 = 76
    assert costs.trade_cost(100_000.0, "sell") == pytest.approx(76.0)


def test_min_commission_applies_to_small_trade() -> None:
    """小额交易被最低佣金（5 元）显著惩罚（这正是现实）。"""
    costs = DEFAULT_COST_MODEL
    # 1000 元买入：按费率佣金 0.25 元 < 5 → 取 5；过户费 0.01 → 合计 5.01
    assert costs.trade_cost(1_000.0, "buy") == pytest.approx(5.01)
    assert costs.trade_cost(-1_000.0, "buy") == pytest.approx(5.01)  # 金额取绝对值


def test_cost_edge_cases() -> None:
    """金额为 0 返回 0；非法方向抛 ValueError（编程错误不吞）。"""
    costs = DEFAULT_COST_MODEL
    assert costs.trade_cost(0.0, "buy") == 0.0
    with pytest.raises(ValueError):
        costs.trade_cost(1_000.0, "hold")


def test_fill_price_slippage() -> None:
    """滑点：买入上浮、卖出下调（100bp = 1%）。"""
    costs = CostModel(slippage_bps=100.0)
    assert costs.fill_price(10.0, "buy") == pytest.approx(10.1)
    assert costs.fill_price(10.0, "sell") == pytest.approx(9.9)
    # 零成本模型：不改价、无费用（单测「价格/数量」逻辑用）
    assert ZERO_COST_MODEL.fill_price(10.0, "buy") == pytest.approx(10.0)
    assert ZERO_COST_MODEL.trade_cost(10_000.0, "sell") == 0.0
    with pytest.raises(ValueError):
        costs.fill_price(0.0, "buy")


# --------------------------------------------------------------------------- #
# 收益类
# --------------------------------------------------------------------------- #
def test_nav_returns_cleans_invalid() -> None:
    """净值转收益：去 NaN / inf，样本不足返回空。"""
    returns = nav_returns([1.0, 1.1, np.nan, 1.21])
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.1)
    assert returns.iloc[1] == pytest.approx(0.1)
    assert nav_returns([1.0]).empty
    assert nav_returns([]).empty


def test_total_return() -> None:
    """总收益 = 期末 / 期初 − 1。"""
    assert total_return([1.0, 1.5]) == pytest.approx(0.5)
    assert total_return([2.0, 1.0]) == pytest.approx(-0.5)
    assert total_return([1.0]) is None   # 样本不足
    assert total_return([]) is None


def test_annual_return_geometric() -> None:
    """几何年化：净值 1→2 跨 252 个收益周期 ⇒ 年化 = 2^(252/252) − 1 = 100%。"""
    nav_one_year = np.linspace(1.0, 2.0, 253)   # 252 个收益周期
    assert annual_return(nav_one_year, 252) == pytest.approx(1.0)
    # 跨 2 年（504 周期）翻倍 ⇒ 年化 = √2 − 1
    nav_two_years = np.linspace(1.0, 2.0, 505)
    assert annual_return(nav_two_years, 252) == pytest.approx(2 ** 0.5 - 1.0)
    assert annual_return([1.0], 252) is None


# --------------------------------------------------------------------------- #
# 风险类
# --------------------------------------------------------------------------- #
def test_annual_volatility_and_sharpe() -> None:
    """波动率 = 日收益标准差 × √252；对称收益（均值 0）夏普为 0。"""
    returns = [0.01, -0.01]
    sd = float(np.std(returns, ddof=1))
    assert annual_volatility(returns, 252) == pytest.approx(sd * np.sqrt(252))
    assert sharpe_ratio(returns, 0.0, 252) == pytest.approx(0.0)
    # 收益恒定 ⇒ 标准差 0，比率无意义 → None
    assert sharpe_ratio([0.001] * 5) is None
    assert annual_volatility([0.001]) is None   # 样本不足


def test_drawdown_and_max_drawdown() -> None:
    """回撤 = 净值 / 历史最高 − 1（≤0）；最大回撤取最深点。"""
    dd = drawdown_series([1.0, 1.2, 0.9, 1.1])
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.0)
    assert dd.iloc[2] == pytest.approx(-0.25)          # 从 1.2 高点跌到 0.9
    assert dd.iloc[3] == pytest.approx(1.1 / 1.2 - 1.0)
    assert max_drawdown([1.0, 1.2, 0.9, 1.1]) == pytest.approx(-0.25)
    assert max_drawdown([1.0]) is None


def test_calmar_ratio() -> None:
    """Calmar = 年化收益 / |最大回撤|；无回撤（分母 0）返回 None。"""
    nav = [1.0, 1.2, 0.9, 1.1]
    annual = annual_return(nav, 252)
    assert calmar_ratio(nav, 252) == pytest.approx(annual / 0.25)
    assert calmar_ratio([1.0, 1.1, 1.2], 252) is None   # 单调上升无回撤


# --------------------------------------------------------------------------- #
# 相对基准
# --------------------------------------------------------------------------- #
def test_alpha_beta_constructed() -> None:
    """构造 rs = 0.001 + 2 × rb ⇒ β = 2 精确命中；α 年化 = 0.001 × 252。"""
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    rb = pd.Series([0.01, 0.02, -0.01, 0.03, 0.005, -0.02], index=idx)
    rs = 0.001 + 2.0 * rb
    alpha, beta = alpha_beta(rs, rb, 252)
    assert beta == pytest.approx(2.0)
    assert alpha == pytest.approx(0.001 * 252)


def test_alpha_beta_degenerate() -> None:
    """基准方差为 0 / 样本不足 ⇒ (None, None)。"""
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    degenerate = alpha_beta(
        pd.Series([0.01, 0.02, 0.03, 0.04], index=idx),
        pd.Series([0.01, 0.01, 0.01, 0.01], index=idx),
    )
    assert degenerate == (None, None)
    assert alpha_beta([0.01], [0.02]) == (None, None)


def test_tracking_error_and_information_ratio() -> None:
    """跟踪误差 = 超额收益标准差 × √252；超额恒为 0（TE=0）时 IR 无意义 → None。"""
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    rb = pd.Series([0.01, -0.01, 0.02, -0.02], index=idx)
    # 策略与基准完全一致 ⇒ 超额恒为 0 ⇒ 跟踪误差 0、信息比率无意义
    assert tracking_error(rb, rb, 252) == pytest.approx(0.0)
    assert information_ratio(rb, rb, 252) is None

    excess = pd.Series([0.001, -0.002, 0.003, -0.001], index=idx)
    rs = rb + excess
    te = float(np.std(excess.to_numpy(), ddof=1)) * np.sqrt(252)
    assert tracking_error(rs, rb, 252) == pytest.approx(te)
    assert information_ratio(rs, rb, 252) == pytest.approx(
        float(excess.mean()) * 252 / te
    )


# --------------------------------------------------------------------------- #
# 交易类
# --------------------------------------------------------------------------- #
def _trades_frame() -> pd.DataFrame:
    """成交明细：2 笔买入（pnl 空，不计入统计）+ 4 笔卖出（2 盈 2 亏）。"""
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "symbol": "A", "side": "buy", "price": 10.0,
             "qty": 100, "amount": 1000.0, "cost": 5.0, "pnl": None},
            {"date": "2026-01-03", "symbol": "A", "side": "sell", "price": 10.5,
             "qty": 100, "amount": 1050.0, "cost": 5.5, "pnl": 44.5},
            {"date": "2026-01-04", "symbol": "B", "side": "buy", "price": 20.0,
             "qty": 100, "amount": 2000.0, "cost": 5.0, "pnl": None},
            {"date": "2026-01-05", "symbol": "B", "side": "sell", "price": 19.0,
             "qty": 100, "amount": 1900.0, "cost": 5.9, "pnl": -105.9},
            {"date": "2026-01-06", "symbol": "A", "side": "sell", "price": 10.0,
             "qty": 100, "amount": 1000.0, "cost": 5.5, "pnl": 20.0},
            {"date": "2026-01-07", "symbol": "C", "side": "sell", "price": 15.0,
             "qty": 100, "amount": 1500.0, "cost": 5.0, "pnl": -60.0},
        ]
    )


def test_closed_trades_win_rate_profit_loss() -> None:
    """已平仓筛选（买入剔除）→ 胜率 2/4；盈亏比 = 平均盈利 / |平均亏损|。"""
    trades = _trades_frame()
    closed = closed_trades(trades)
    assert len(closed) == 4
    assert trade_count(trades) == 4
    assert win_rate(trades) == pytest.approx(0.5)
    # 平均盈利 (44.5+20)/2=32.25；平均亏损 (105.9+60)/2=82.95
    assert profit_loss_ratio(trades) == pytest.approx(32.25 / 82.95)


def test_trade_metrics_empty_and_degenerate() -> None:
    """无成交 / 全胜 / 全负的边界：返回 None / 0，不抛异常。"""
    assert closed_trades(None).empty
    assert closed_trades(pd.DataFrame()).empty
    assert win_rate(None) is None
    assert profit_loss_ratio(None) is None
    assert trade_count(None) == 0
    only_win = _trades_frame().iloc[[1]]
    assert win_rate(only_win) == pytest.approx(1.0)
    assert profit_loss_ratio(only_win) is None   # 缺亏损侧


# --------------------------------------------------------------------------- #
# 集成
# --------------------------------------------------------------------------- #
def test_compute_metrics_full() -> None:
    """compute_metrics 全字段装配 + to_dict 键集合稳定。"""
    idx = pd.date_range("2026-01-01", periods=11, freq="D")
    nav = pd.Series(np.linspace(1.0, 1.2, 11), index=idx)
    bench = pd.Series(np.linspace(1.0, 1.1, 11), index=idx)
    metrics = compute_metrics(nav, bench, _trades_frame())

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_return == pytest.approx(0.2)
    assert metrics.benchmark_total_return == pytest.approx(0.1)
    assert metrics.excess_return == pytest.approx(0.1)
    assert metrics.max_drawdown == pytest.approx(0.0)   # 单调上升无回撤
    assert metrics.trade_count == 4
    assert metrics.win_rate == pytest.approx(0.5)
    assert set(metrics.to_dict()) == {
        "total_return", "annual_return", "annual_volatility", "sharpe",
        "max_drawdown", "calmar", "alpha", "beta", "tracking_error",
        "information_ratio", "benchmark_total_return", "excess_return",
        "win_rate", "profit_loss_ratio", "trade_count",
    }


def test_compute_metrics_degenerate_inputs() -> None:
    """空序列 / 全 NaN ⇒ 字段留 None（不抛异常），交易指标为 0 / None。"""
    metrics = compute_metrics([], [])
    assert metrics.total_return is None
    assert metrics.sharpe is None
    assert metrics.beta is None
    assert metrics.trade_count == 0

    metrics_nan = compute_metrics([np.nan, np.nan], [np.nan, np.nan])
    assert metrics_nan.total_return is None
    assert metrics_nan.max_drawdown is None