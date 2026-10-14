"""P7-S3 Rank IC 与统计检验单测（手算对拍）。

全部用例为纯函数测试（不依赖 Qt / 数据库）：
    1. Rank IC：完美正相关 / 反相关 / 小样本手算（Spearman 公式对拍）；
    2. 有效性闸门：横截面样本不足、因子值全同（无区分度）→ 该日 IC 置缺失；
    3. 对齐：两张面板列序不同 / 标的与日期不齐时取交集；
    4. Newey-West：滞后 0 阶与普通标准误一致（手算对拍）；
       自相关序列（AR(1)）下修正标准误必须大于普通标准误；
    5. 双侧 p 值：t 分布临界值对拍（0.05 / 0.10 / 大自由度趋于正态）；
    6. 汇总统计：均值 / ICIR / 胜率 / t 值 / 显著性标志（含样本不足降级）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.factor.ic import (
    newey_west_mean_se,
    rank_ic,
    summarize_ic,
    t_pvalue_two_sided,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _panel(values: np.ndarray, dates=None, symbols=None) -> pd.DataFrame:
    """把二维数组包成面板 DataFrame（默认 3 行 × 4 列）。"""
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    rows, columns = array.shape
    index = pd.RangeIndex(rows) if dates is None else pd.DatetimeIndex(dates)
    labels = [f"s{i}" for i in range(columns)] if symbols is None else list(symbols)
    return pd.DataFrame(array, index=index, columns=labels)


# --------------------------------------------------------------------------- #
# 1~3. Rank IC
# --------------------------------------------------------------------------- #
def test_rank_ic_perfect_and_reversed() -> None:
    """同一面板 IC=1（完美正相关）、取负后 IC=−1（反相关）。"""
    rng = np.random.default_rng(1)
    factor = _panel(rng.normal(size=(30, 20)))
    ic_same = rank_ic(factor, factor.copy(), min_stocks=5)
    assert len(ic_same) == 30
    assert np.allclose(ic_same.to_numpy(), 1.0)
    ic_reversed = rank_ic(factor, -factor, min_stocks=5)
    assert np.allclose(ic_reversed.to_numpy(), -1.0)


def test_rank_ic_small_hand_check() -> None:
    """手算对拍：因子 [1,2,3] vs 收益 [2,1,3] ⇒ Spearman ρ = 1 − 6×2/(3×8) = 0.5。"""
    factor = _panel([[1.0, 2.0, 3.0]])
    forward = _panel([[2.0, 1.0, 3.0]])
    ic = rank_ic(factor, forward, min_stocks=3)
    assert ic.iloc[0] == pytest.approx(0.5)


def test_rank_ic_gates_thin_and_constant_rows() -> None:
    """样本不足（< min_stocks）或因子值全同（无区分度）的交易日必须剔除。"""
    dates = pd.bdate_range("2024-01-01", periods=3)
    # 第 1 天：只有 2 个有效样本（< 5）；第 2 天：因子值全同；第 3 天：正常 5 个样本
    factor = _panel(
        [
            [1.0, np.nan, np.nan, np.nan, np.nan],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dates=dates,
    )
    forward = _panel(
        [
            [1.0, np.nan, np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dates=dates,
    )
    ic = rank_ic(factor, forward, min_stocks=5)
    assert len(ic) == 1
    assert ic.index[0] == dates[2]
    assert ic.iloc[0] == pytest.approx(-1.0)  # 完全反向


def test_rank_ic_aligns_on_intersection() -> None:
    """两面板列序不同 / 有额外标的与日期时，取交集且结果不受列序影响。"""
    dates = pd.bdate_range("2024-01-01", periods=4)
    factor = _panel(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 1.0, 2.0], [7.0, 8.0, 9.0]],
        dates=dates,
        symbols=["a", "b", "c"],
    )
    forward = _panel(
        [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0], [2.0, 1.0, 1.0], [9.0, 7.0, 8.0]],
        dates=dates,
        symbols=["c", "b", "a"],  # 列序颠倒
    )
    forward["extra"] = 1.0  # 额外标的（交集外）
    forward.loc[dates[0] - pd.Timedelta(days=1)] = [1.0, 1.0, 1.0, 1.0]
    ic = rank_ic(factor, forward, min_stocks=3)
    assert list(ic.index) == list(dates)
    assert ic.iloc[0] == pytest.approx(0.5)


def test_rank_ic_empty_inputs() -> None:
    """空表 / 非法类型 ⇒ 空 Series（不抛异常）。"""
    assert rank_ic(pd.DataFrame(), pd.DataFrame()).empty
    assert rank_ic(None, pd.DataFrame([[1.0]])) is not None  # type: ignore[arg-type]
    assert rank_ic(None, pd.DataFrame([[1.0]])).empty  # type: ignore[arg-type]
    empty = _panel([[1.0, 2.0]])
    assert rank_ic(empty.iloc[0:0], empty).empty


# --------------------------------------------------------------------------- #
# 4. Newey-West
# --------------------------------------------------------------------------- #
def test_newey_west_zero_lag_matches_naive_hand_check() -> None:
    """滞后 0 阶 = 普通标准误：values [1,2,3,4] ⇒ SE = √(1.25/4)。"""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    se = newey_west_mean_se(values, 0)
    assert se == pytest.approx(math.sqrt(1.25 / 4.0))
    naive = float(values.std(ddof=0)) / math.sqrt(len(values))
    assert se == pytest.approx(naive)


def test_newey_west_inflates_under_autocorrelation() -> None:
    """AR(1)（φ=0.9）序列：修正标准误必须明显大于普通标准误（自相关抬高方差）。"""
    rng = np.random.default_rng(123)
    count = 800
    series = np.zeros(count)
    for index in range(1, count):
        series[index] = 0.9 * series[index - 1] + rng.normal(0.0, 1.0)
    naive = newey_west_mean_se(series, 0)
    adjusted = newey_west_mean_se(series, 10)
    assert naive is not None and adjusted is not None
    assert adjusted > naive * 1.5


def test_newey_west_edge_cases() -> None:
    """样本不足 / 常数序列 / 非法输入 ⇒ None（不抛异常）。"""
    assert newey_west_mean_se(np.array([1.0]), 3) is None
    assert newey_west_mean_se(np.array([5.0, 5.0, 5.0]), 1) is None
    assert newey_west_mean_se(None, 1) is None  # type: ignore[arg-type]
    assert newey_west_mean_se(np.array([1.0, np.nan, 3.0, 4.0]), 1) is not None


# --------------------------------------------------------------------------- #
# 5. p 值
# --------------------------------------------------------------------------- #
def test_t_pvalue_known_critical_values() -> None:
    """t 分布临界值对拍：df=10 时 2.228→0.05、1.812→0.10、2.0→0.0734。"""
    assert t_pvalue_two_sided(2.228, 10) == pytest.approx(0.05, abs=1e-3)
    assert t_pvalue_two_sided(1.812, 10) == pytest.approx(0.10, abs=2e-3)
    assert t_pvalue_two_sided(2.0, 10) == pytest.approx(0.0734, abs=1e-3)
    assert t_pvalue_two_sided(0.0, 10) == pytest.approx(1.0)


def test_t_pvalue_large_df_approaches_normal() -> None:
    """大自由度下 t 分布趋于正态：1.959964 → p ≈ 0.05。"""
    assert t_pvalue_two_sided(1.959964, 10**7) == pytest.approx(0.05, abs=1e-3)


def test_t_pvalue_invalid_inputs() -> None:
    """非法输入（无法解析 / 自由度 ≤0 / 非有限）⇒ 1.0（视为不显著）。"""
    assert t_pvalue_two_sided("x", 10) == 1.0  # type: ignore[arg-type]
    assert t_pvalue_two_sided(3.0, 0) == 1.0
    assert t_pvalue_two_sided(float("nan"), 10) == 1.0
    assert t_pvalue_two_sided(float("inf"), 10) == 1.0


# --------------------------------------------------------------------------- #
# 6. 汇总统计
# --------------------------------------------------------------------------- #
def test_summarize_ic_hand_check() -> None:
    """均值 / ICIR / 胜率 / t 值手算对拍（ddof=1 口径）。"""
    values = np.array([0.10, 0.20, -0.10, 0.05, 0.15])
    ic = pd.Series(values)
    summary = summarize_ic(ic, horizon=10)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    assert summary.n_days == 5
    assert summary.ic_mean == pytest.approx(mean)
    assert summary.ic_std == pytest.approx(std)
    assert summary.ic_ir == pytest.approx(mean / std)
    assert summary.t_stat == pytest.approx(mean / (std / math.sqrt(5)))
    assert summary.ic_win_rate == pytest.approx(4.0 / 5.0)  # 与正均值同号 4/5
    assert summary.nw_lags == 9  # 默认 = 预测期 − 1
    assert summary.t_stat_nw is not None
    assert summary.p_value is not None
    assert summary.significant == (summary.p_value < 0.05)  # 显著 ⇔ 双侧 p < 0.05


def test_summarize_ic_insufficient_sample_degrades() -> None:
    """空序列 / 单点 ⇒ 除 n_days 外均为 None，显著标志 False（不抛异常）。"""
    empty = summarize_ic(pd.Series(dtype=float))
    assert empty.n_days == 0 and empty.ic_mean is None and not empty.significant
    single = summarize_ic(pd.Series([0.1]))
    assert single.n_days == 1 and single.t_stat_nw is None and not single.significant


def test_summarize_ic_strong_signal_detected() -> None:
    """强信号（均值 0.05、波动 0.02、500 天）⇒ 高度显著（p < 0.01）。"""
    rng = np.random.default_rng(9)
    ic = pd.Series(0.05 + rng.normal(0.0, 0.02, 500))
    summary = summarize_ic(ic, horizon=5)
    assert summary.significant
    assert summary.p_value is not None and summary.p_value < 0.01
    assert summary.t_stat_nw is not None and summary.t_stat_nw > 2.58
    assert summary.ic_win_rate is not None and summary.ic_win_rate > 0.9