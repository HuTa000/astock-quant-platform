"""绩效归因模块测试（P7-S2）。

覆盖（全部手算对拍）：
    1. ``yearly_summary``：跨年收益链式口径（首年从首日起算）+ 年内最深回撤；
    2. ``yearly_summary``：无基准列 → 基准字段为 ``None``（不崩）；
    3. ``yearly_summary``：年内回撤按「历史最高点」而非「年内起点」计算；
    4. ``drawdown_episodes``：峰值 / 谷底 / 修复日、深度、下跌与修复交易日数；
    5. ``drawdown_episodes``：未修复区间（``recovered=False``）；
    6. ``drawdown_episodes``：净值单调向上 → 空表；``top_n`` 按深度截断；
    7. ``jensen_decomposition``：构造 β=1.5 / α 精确已知 → 分解字段手算对拍；
    8. ``jensen_decomposition``：缺基准 → 仅策略年化有值；
    9. 非法输入（``None`` / 空表 / 缺列）→ 空表 / 全 ``None``，**不抛异常**。
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.engine.backtest.attribution import (
    DECOMPOSITION_KEYS,
    DRAWDOWN_COLUMNS,
    YEARLY_COLUMNS,
    drawdown_episodes,
    jensen_decomposition,
    yearly_summary,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _equity(dates, nav, benchmark=None) -> pd.DataFrame:
    """构造最小净值表（``date/nav[/benchmark_nav]``，与引擎 equity 同契约列口径）。"""
    data = {"date": pd.to_datetime(list(dates)), "nav": list(nav)}
    if benchmark is not None:
        data["benchmark_nav"] = list(benchmark)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# 1~3. yearly_summary
# --------------------------------------------------------------------------- #
def test_yearly_summary_chain_and_drawdown() -> None:
    """跨年年收益 = 年末 / 上年末 − 1；年内最深回撤手算对拍。"""
    equity = _equity(
        ["2020-12-31", "2021-06-30", "2021-12-31", "2022-06-30", "2022-12-31"],
        [1.0, 1.1, 1.2, 1.0, 1.1],
        [1.0, 1.05, 1.1, 1.0, 1.05],
    )
    frame = yearly_summary(equity)
    assert list(frame.columns) == YEARLY_COLUMNS
    assert frame["year"].tolist() == [2020, 2021, 2022]

    y2020 = frame.iloc[0]
    assert y2020["days"] == 1
    assert y2020["strategy_return"] == pytest.approx(0.0)
    assert y2020["benchmark_return"] == pytest.approx(0.0)

    y2021 = frame.iloc[1]
    assert y2021["days"] == 2
    assert y2021["strategy_return"] == pytest.approx(1.2 / 1.0 - 1.0)
    assert y2021["benchmark_return"] == pytest.approx(1.1 / 1.0 - 1.0)
    assert y2021["excess_return"] == pytest.approx(0.1)
    assert y2021["strategy_max_dd"] == pytest.approx(0.0)
    assert y2021["benchmark_max_dd"] == pytest.approx(0.0)

    y2022 = frame.iloc[2]
    assert y2022["strategy_return"] == pytest.approx(1.1 / 1.2 - 1.0)
    assert y2022["benchmark_return"] == pytest.approx(1.05 / 1.1 - 1.0)
    assert y2022["excess_return"] == pytest.approx(1.1 / 1.2 - 1.05 / 1.1)
    # 年内最深回撤：2022-06-30 跌到 1.0（相对 1.2 的历史高点）。
    assert y2022["strategy_max_dd"] == pytest.approx(1.0 / 1.2 - 1.0)
    assert y2022["benchmark_max_dd"] == pytest.approx(1.0 / 1.1 - 1.0)


def test_yearly_summary_without_benchmark() -> None:
    """无 ``benchmark_nav`` 列：基准相关字段为 None，策略字段照常。"""
    frame = yearly_summary(_equity(["2021-06-30", "2021-12-31"], [1.0, 1.2]))
    row = frame.iloc[0]
    assert row["strategy_return"] == pytest.approx(0.2)
    assert row["benchmark_return"] is None
    assert row["excess_return"] is None
    assert row["benchmark_max_dd"] is None
    assert row["strategy_max_dd"] == pytest.approx(0.0)


def test_yearly_drawdown_uses_all_time_peak() -> None:
    """年内回撤按「历史最高点」计：年初已处于回撤中也要计入。"""
    equity = _equity(
        ["2020-01-31", "2020-12-31", "2021-01-04", "2021-12-31"],
        [1.0, 2.0, 1.5, 1.8],
    )
    frame = yearly_summary(equity)
    y2021 = frame[frame["year"] == 2021].iloc[0]
    # 2021 年内净值先 1.5 后 1.8（年内无新低），但相对 2020 年末峰值 2.0 曾跌 25%。
    assert y2021["strategy_max_dd"] == pytest.approx(1.5 / 2.0 - 1.0)
    assert y2021["strategy_return"] == pytest.approx(1.8 / 2.0 - 1.0)


# --------------------------------------------------------------------------- #
# 4~6. drawdown_episodes
# --------------------------------------------------------------------------- #
def test_drawdown_episodes_hand_calc() -> None:
    """两段回撤的峰值 / 谷底 / 修复日与交易日数手算对拍。"""
    dates = pd.date_range("2021-01-04", periods=9, freq="B")
    nav = [1.0, 1.2, 0.9, 0.6, 0.8, 1.0, 1.3, 1.1, 1.4]
    frame = drawdown_episodes(pd.Series(nav, index=dates))
    assert list(frame.columns) == DRAWDOWN_COLUMNS
    assert len(frame) == 2

    first = frame.iloc[0]
    assert first["peak_date"] == dates[1]
    assert first["trough_date"] == dates[3]
    assert first["recovery_date"] == dates[6]
    assert first["depth"] == pytest.approx(0.6 / 1.2 - 1.0)
    assert first["drawdown_days"] == 2   # pos3 − pos1
    assert first["recovery_days"] == 3   # pos6 − pos3
    assert bool(first["recovered"]) is True

    second = frame.iloc[1]
    assert second["peak_date"] == dates[6]
    assert second["trough_date"] == dates[7]
    assert second["recovery_date"] == dates[8]
    assert second["depth"] == pytest.approx(1.1 / 1.3 - 1.0)
    assert second["drawdown_days"] == 1
    assert second["recovery_days"] == 1


def test_drawdown_episodes_not_recovered() -> None:
    """数据结束时仍未收复前高：recovered=False、修复字段为 None。"""
    frame = drawdown_episodes([1.0, 1.2, 0.9, 0.95])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["depth"] == pytest.approx(0.9 / 1.2 - 1.0)
    assert row["drawdown_days"] == 1
    assert row["recovered"] is False or bool(row["recovered"]) is False
    assert row["recovery_date"] is None
    assert row["recovery_days"] is None


def test_drawdown_episodes_monotonic_and_top_n() -> None:
    """单调向上 → 空表；多段回撤按深度截断 top_n。"""
    assert drawdown_episodes([1.0, 1.1, 1.2]).empty

    nav = [1.0, 0.9, 1.0, 0.7, 1.0, 0.95, 1.0]
    frame = drawdown_episodes(nav, top_n=2)
    assert len(frame) == 2
    depths = frame["depth"].tolist()
    # 深度从深到浅：0.7/1.0−1 = −0.30 → 0.9/1.0−1 = −0.10（−0.05 被截掉）。
    assert depths[0] == pytest.approx(-0.3)
    assert depths[1] == pytest.approx(-0.1)


# --------------------------------------------------------------------------- #
# 7~8. jensen_decomposition
# --------------------------------------------------------------------------- #
def test_jensen_decomposition_exact_beta_alpha() -> None:
    """构造 r_p = 0.0004 + 1.5 × r_b（无噪声）→ β=1.5、α 年化=0.1008 精确。"""
    n_returns = 252
    dates = pd.date_range("2020-01-01", periods=n_returns + 1, freq="B")
    bench_nav = 1.0
    strategy_nav = 1.0
    bench_values = [1.0]
    strategy_values = [1.0]
    for i in range(n_returns):
        r_b = 0.002 if i % 2 == 0 else 0.0
        r_p = 0.0004 + 1.5 * r_b
        bench_nav *= 1.0 + r_b
        strategy_nav *= 1.0 + r_p
        bench_values.append(bench_nav)
        strategy_values.append(strategy_nav)

    equity = _equity(dates, strategy_values, bench_values)
    result = jensen_decomposition(equity)
    assert set(result) == set(DECOMPOSITION_KEYS)

    assert result["beta"] == pytest.approx(1.5, rel=1e-9, abs=1e-12)
    assert result["alpha_annual"] == pytest.approx(0.0004 * 252, rel=1e-9)
    # n_periods == periods_per_year → 几何年化 == 区间总收益（精确）。
    assert result["benchmark_annual"] == pytest.approx(bench_nav - 1.0)
    assert result["strategy_annual"] == pytest.approx(strategy_nav - 1.0)
    assert result["beta_component"] == pytest.approx(0.5 * result["benchmark_annual"])
    assert result["alpha_component"] == result["alpha_annual"]
    assert result["excess_annual"] == pytest.approx(
        result["strategy_annual"] - result["benchmark_annual"]
    )
    assert result["residual"] == pytest.approx(
        result["excess_annual"] - result["beta_component"] - result["alpha_component"]
    )
    # α = 10.08%/年 显著，残差应远小于 α 分量。
    assert abs(result["residual"]) < abs(result["alpha_component"])


def test_jensen_decomposition_without_benchmark() -> None:
    """缺基准：仅策略年化有值，其余为 None（用 253 个点使年化中性：n=252）。"""
    dates = pd.date_range("2020-01-01", periods=253, freq="B")
    nav = [1.2 ** (i / 252.0) for i in range(253)]  # 首=1.0、末=1.2 → 年化精确 0.2
    result = jensen_decomposition(_equity(dates, nav))
    assert result["strategy_annual"] == pytest.approx(0.2, rel=1e-9)
    assert result["benchmark_annual"] is None
    assert result["alpha_annual"] is None
    assert result["beta"] is None
    assert result["residual"] is None


# --------------------------------------------------------------------------- #
# 9. 非法输入
# --------------------------------------------------------------------------- #
def test_invalid_inputs_no_exception() -> None:
    """None / 空表 / 缺列：返回空表或全 None，绝不抛异常。"""
    assert list(yearly_summary(None).columns) == YEARLY_COLUMNS
    assert yearly_summary(pd.DataFrame({"foo": [1]})).empty
    assert list(drawdown_episodes(None).columns) == DRAWDOWN_COLUMNS
    assert drawdown_episodes([]).empty
    assert drawdown_episodes([1.0]).empty
    result = jensen_decomposition(None)
    assert set(result) == set(DECOMPOSITION_KEYS)
    assert all(value is None for value in result.values())