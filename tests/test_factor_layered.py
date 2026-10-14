"""P7-S3 分层回测单测（手算对拍）。

覆盖：
    1. 单调性：因子值与前瞻收益完全同序 ⇒ 各层收益逐档升高、单调性 = +1、
       多空为正；反向同序 ⇒ 单调性 = −1、多空为负；
    2. 分组手算：等数量分档每层股数、组均值、净值链乘、多空净值（两日轮动）；
    3. 闸门：有效标的不足 / 无区分度（因子值全同）⇒ 该日不建仓并记警告；
       全部日不建仓 ⇒ ok=False（不抛异常）；
    4. 参数非法：n_groups < 2、非 DataFrame 输入、空面板、无公共日期/标的
       ⇒ ok=False；
    5. 秩相关：完全正/负相关、常数、样本不足、含 NaN、非法输入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engine.factor.layered import (
    GROUP_STATS_COLUMNS,
    run_layered,
    spearman_rank_corr,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _frame(data: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    """构造面板：``{标的: [逐日值]}`` → ``dates × symbols``。"""
    return pd.DataFrame(
        {symbol: [float(v) for v in values] for symbol, values in data.items()},
        index=pd.to_datetime(dates),
    )


def _ranked(symbols: list[str], dates: list[str], scale: float) -> pd.DataFrame:
    """``S1..Sn`` 逐日值 = 序号 × scale（用于排序类构造）。"""
    return _frame(
        {symbol: [float(i) * scale] for i, symbol in enumerate(symbols, start=1)},
        dates,
    )


# --------------------------------------------------------------------------- #
# 1. 单调性
# --------------------------------------------------------------------------- #
def test_monotonic_ascending() -> None:
    """因子值与前瞻收益完全同序：Q1→Q5 逐档升高、单调性 = +1、多空为正。"""
    symbols = [f"S{i:02d}" for i in range(1, 26)]          # 25 只 → 每层 5 只
    dates = ["2024-01-02", "2024-01-03"]
    factor = _ranked(symbols, dates, 1.0)
    forward = _ranked(symbols, dates, 0.001)               # 收益 = 因子值 × 0.001

    result = run_layered(factor, forward, n_groups=5)

    assert result.ok
    assert result.n_groups == 5
    assert result.group_names == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert list(result.group_stats.columns) == GROUP_STATS_COLUMNS
    totals = result.group_stats["total_return"].to_numpy(dtype=float)
    assert np.all(np.diff(totals) > 0)                     # 逐档严格升高
    assert result.monotonicity == pytest.approx(1.0)
    assert result.days == 2
    # 每层 5 只：Q1 日收益 = (1+2+3+4+5)×0.001/5 = 0.003、Q5 = 0.023
    assert result.group_stats.iloc[0]["mean_daily_return"] == pytest.approx(0.003)
    assert result.group_stats.iloc[4]["mean_daily_return"] == pytest.approx(0.023)
    # 多空 = Q5 − Q1 = 0.02/日 → 两日链乘：1.02、1.02²
    assert float(result.long_short_nav.iloc[0]) == pytest.approx(1.02)
    assert float(result.long_short_nav.iloc[1]) == pytest.approx(1.02**2)
    assert result.long_short_annual is not None
    assert result.long_short_annual > 0


def test_monotonic_descending() -> None:
    """收益与因子值反向同序：Q1 最高、Q5 最低、单调性 = −1、多空为负。"""
    symbols = [f"S{i:02d}" for i in range(1, 21)]          # 20 只 → 每层 4 只
    dates = ["2024-01-02", "2024-01-03"]
    factor = _ranked(symbols, dates, 1.0)
    forward = _ranked(symbols, dates, -0.001)

    result = run_layered(factor, forward, n_groups=5)

    assert result.ok
    totals = result.group_stats["total_return"].to_numpy(dtype=float)
    assert np.all(np.diff(totals) < 0)                     # 逐档严格降低
    assert result.monotonicity == pytest.approx(-1.0)
    assert float(result.long_short_nav.iloc[-1]) < 1.0
    assert result.long_short_annual is not None
    assert result.long_short_annual < 0


# --------------------------------------------------------------------------- #
# 2. 两日轮动手算
# --------------------------------------------------------------------------- #
def test_two_day_rotation_hand_check() -> None:
    """两日轮动：Q1 净值 = 1.025×0.975、多空净值 = 1.16×0.84（逐点手算）。"""
    symbols = [f"S{i:02d}" for i in range(1, 21)]
    dates = ["2024-01-02", "2024-01-03"]
    factor = _ranked(symbols, dates, 1.0)
    # 第一天收益 = +序号×0.01、第二天 = −序号×0.01（组的日收益逐日反号）
    forward = pd.DataFrame(
        {
            symbol: [float(i) * 0.01, -float(i) * 0.01]
            for i, symbol in enumerate(symbols, start=1)
        },
        index=pd.to_datetime(dates),
    )

    result = run_layered(factor, forward, n_groups=5)

    assert result.ok
    assert result.days == 2
    # Q1 = 序号 1~4：日收益 ±(0.01+0.02+0.03+0.04)/4 = ±0.025
    assert float(result.group_nav.iloc[0]["Q1"]) == pytest.approx(1.025)
    assert float(result.group_nav.iloc[1]["Q1"]) == pytest.approx(1.025 * 0.975)
    # Q5 = 序号 17~20：日收益 ±0.185
    assert float(result.group_nav.iloc[0]["Q5"]) == pytest.approx(1.185)
    assert float(result.group_nav.iloc[1]["Q5"]) == pytest.approx(1.185 * 0.815)
    # 多空 = Q5 − Q1 = ±0.16
    assert float(result.long_short_nav.iloc[0]) == pytest.approx(1.16)
    assert float(result.long_short_nav.iloc[1]) == pytest.approx(1.16 * 0.84)
    # 两日反号 ⇒ 各层日均收益均为 0
    assert result.group_stats["mean_daily_return"].abs().max() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. 闸门与降级
# --------------------------------------------------------------------------- #
def test_insufficient_symbols_no_position() -> None:
    """有效标的不足 min_stocks ⇒ 全部日不建仓 ⇒ ok=False（不抛异常）。"""
    symbols = [f"S{i:02d}" for i in range(1, 9)]           # 8 只 < min_stocks=10
    factor = _ranked(symbols, ["2024-01-02"], 1.0)
    forward = _ranked(symbols, ["2024-01-02"], 0.001)

    result = run_layered(factor, forward, min_stocks=10)

    assert not result.ok
    assert "无任何有效建仓日" in result.error


def test_zero_dispersion_day_skipped() -> None:
    """某日因子值全同（无区分度）⇒ 该日不建仓并记警告，其余日照常建仓。"""
    symbols = [f"S{i:02d}" for i in range(1, 13)]
    dates = ["2024-01-02", "2024-01-03"]
    factor = _frame(
        {symbol: [float(7.0), float(i)] for i, symbol in enumerate(symbols, start=1)},
        dates,
    )
    forward = _ranked(symbols, dates, 0.001)

    result = run_layered(factor, forward, n_groups=5, min_stocks=6)

    assert result.ok
    assert result.days == 1                                 # 仅第二日建仓
    assert any("未建仓" in message for message in result.warnings)
    # 无持仓日从净值表中剔除：净值表只剩第二日，Q1 = 1 + 组均值
    assert len(result.group_nav) == 1
    assert float(result.group_nav.iloc[0]["Q1"]) == pytest.approx(1.0 + 0.001 * 1.5)


def test_all_days_ineligible_fails() -> None:
    """两日全等值 ⇒ 无任何有效建仓日 ⇒ ok=False。"""
    symbols = [f"S{i:02d}" for i in range(1, 13)]
    dates = ["2024-01-02", "2024-01-03"]
    factor = _frame({symbol: [1.0, 1.0] for symbol in symbols}, dates)
    forward = _ranked(symbols, dates, 0.001)

    result = run_layered(factor, forward, n_groups=5, min_stocks=6)

    assert not result.ok
    assert "无任何有效建仓日" in result.error


# --------------------------------------------------------------------------- #
# 4. 参数非法
# --------------------------------------------------------------------------- #
def test_invalid_arguments() -> None:
    """层数 < 2 / 非 DataFrame / 空面板 / 面板无交集 ⇒ ok=False，均不抛异常。"""
    symbols = [f"S{i:02d}" for i in range(1, 13)]
    factor = _ranked(symbols, ["2024-01-02"], 1.0)
    forward = _ranked(symbols, ["2024-01-02"], 0.001)

    too_few_groups = run_layered(factor, forward, n_groups=1)
    assert not too_few_groups.ok and "层数" in too_few_groups.error
    assert not run_layered("bad", forward).ok
    assert not run_layered(factor, forward_panel=None).ok
    assert not run_layered(pd.DataFrame(), pd.DataFrame()).ok

    other_symbols = [f"T{i:02d}" for i in range(1, 13)]
    disjoint = run_layered(factor, _ranked(other_symbols, ["2024-01-02"], 0.001))
    assert not disjoint.ok and "公共日期或标的" in disjoint.error


# --------------------------------------------------------------------------- #
# 5. 秩相关
# --------------------------------------------------------------------------- #
def test_spearman_rank_corr() -> None:
    """秩相关：完全正/负 = ±1；常数、样本 < 2、非法输入 ⇒ None；成对剔除 NaN。"""
    assert spearman_rank_corr([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_rank_corr([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    # 秩相关对非线性单调变换稳健（指数放大不影响）
    assert spearman_rank_corr([1, 2, 3], [1, 100, 10000]) == pytest.approx(1.0)
    assert spearman_rank_corr([1, 1, 1], [1, 2, 3]) is None
    assert spearman_rank_corr([1], [2]) is None
    assert spearman_rank_corr(["x", "y"], [1, 2]) is None
    # 成对剔除缺失：仅剩 (1,2)、(3,6) 两对 → 完全正相关
    assert spearman_rank_corr([1, np.nan, 3], [2, 5, 6]) == pytest.approx(1.0)