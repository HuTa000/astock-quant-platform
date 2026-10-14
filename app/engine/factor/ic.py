"""Rank IC 计算与统计检验（P7-S3）。

定位：
    回答「这个因子的值排名靠前，未来收益是不是真的靠前」——
    每个交易日做一次**横截面秩相关**（factor 排名 vs 前瞻收益排名），
    得到 IC（Information Coefficient）时间序列，再用统计检验判断
    「预测能力是真本事还是噪声」。

为什么用 Rank IC（秩相关）而不是普通相关：
    * 收益分布有极端值（涨停、闪崩），普通皮尔逊相关会被个别极端样本带偏；
      换成**排名**（而非数值本身）计算相关，天然抗异常值；
    * 因子与收益的关系常是**非线性单调**（值越大收益越高即可），
      秩相关只关心「排序是否一致」，正好匹配。

汇总统计口径（通俗解释）：
    * IC 均值：平均每天「因子排名与未来收益排名」的一致程度，
      >0 = 因子值越大未来收益越高；A 股日频经验 |IC|>0.03 算有效、
      0.02~0.03 边缘、<0.02 基本是噪声；
    * IC 标准差 / ICIR（= 均值 / 标准差）：预测能力的**稳定性**，
      同样 0.03 的均值，波动越小越可信；ICIR > 0.5 良好；
    * IC 胜率：IC 与均值同号的交易日占比——「大多数日子方向都对」；
    * t 值：均值 / 标准误，衡量「均值不为 0」的统计显著性；
    * **Newey-West 修正**：5/10/20/60 日前瞻收益**相互重叠**（相邻交易日的
      未来窗口大量重合），IC 序列存在自相关，普通 t 检验会**高估**显著性；
      采用 Newey-West（Bartlett 核，滞后阶数默认 = 预测期 − 1）修正标准误后
      再算 t 值，才是可信的显著性；
    * p 值：双侧 Student-t（自由度 = 有效天数 − 1），用不完全 Beta 函数
      精确计算（**不依赖 scipy**）；|t|>1.96 ≈ 5% 显著、>2.58 ≈ 1% 高度显著。

对外函数**不抛业务异常**：样本不足 / 非法输入 → 返回空 Series 或含
``None`` 字段的汇总对象（与项目「引擎层容错、UI 层兜底」约定一致）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 摘要表契约列（分析编排产出成表时使用）。
IC_SUMMARY_COLUMNS: list[str] = [
    "n_days",
    "ic_mean",
    "ic_std",
    "ic_ir",
    "ic_win_rate",
    "t_stat",
    "t_stat_nw",
    "p_value",
    "nw_lags",
    "significant",
]


# --------------------------------------------------------------------------- #
# 基础：秩相关与 Newey-West
# --------------------------------------------------------------------------- #
def rank_ic(
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    min_stocks: int = 10,
) -> pd.Series:
    """逐日横截面 Rank IC 序列。

    每个交易日：取「因子值与前瞻收益均非 NaN」的标的，计算两组**排名**的
    皮尔逊相关（等价于 Spearman 秩相关，同分取平均秩）；当日有效标的数
    ``< min_stocks`` 或因子值全部相同（排名无差异）→ 该日 IC 记 NaN。

    Args:
        factor_panel: 因子面板（dates × symbols）。
        forward_panel: 前瞻收益面板（同形；口径见
            :meth:`~app.engine.factor.panel.FactorPanel.forward`）。
        min_stocks: 当日横截面最少有效标的数（低于则该日不参与统计，默认 10）。

    Returns:
        IC Series（索引 = 日期，仅含有效日）；非法输入返回空 Series。
    """
    if not isinstance(factor_panel, pd.DataFrame) or not isinstance(forward_panel, pd.DataFrame):
        return pd.Series(dtype=float)
    if factor_panel.empty or forward_panel.empty:
        return pd.Series(dtype=float)
    # 对齐到「日期取交集、标的取交集」，防止外部传入两张不同来源的表。
    common_dates = factor_panel.index.intersection(forward_panel.index)
    common_symbols = factor_panel.columns.intersection(forward_panel.columns)
    if len(common_dates) == 0 or len(common_symbols) == 0:
        return pd.Series(dtype=float)
    factor = factor_panel.reindex(index=common_dates, columns=common_symbols)
    forward = forward_panel.reindex(index=common_dates, columns=common_symbols)

    threshold = max(int(min_stocks), 2)  # 秩相关至少需要 2 个样本
    mask = factor.notna() & forward.notna()
    counts = mask.sum(axis=1)
    factor_rank = factor.where(mask).rank(axis=1)
    forward_rank = forward.where(mask).rank(axis=1)
    factor_centered = factor_rank.sub(factor_rank.mean(axis=1), axis=0)
    forward_centered = forward_rank.sub(forward_rank.mean(axis=1), axis=0)
    numerator = (factor_centered * forward_centered).sum(axis=1, skipna=True)
    factor_std = factor_rank.std(axis=1, ddof=0)
    forward_std = forward_rank.std(axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = numerator / (counts.astype(float) * factor_std * forward_std)
    ic = ic.replace([np.inf, -np.inf], np.nan)
    ic = ic.where(counts >= threshold)
    ic.name = "ic"
    return ic.dropna()


def newey_west_mean_se(values: np.ndarray, lags: int) -> Optional[float]:
    """Newey-West 修正后的**均值标准误**（Bartlett 核，无截距回归的 HAC）。

    公式：``S = γ₀ + 2 Σ_{l=1..L} (1 − l/(L+1)) γ_l``，``SE = √(S / n)``，
    其中 ``γ_l`` 为（去均值后）滞后 l 阶自协方差（除以 n）。
    ``L = 0`` 时退化为普通标准误 ``std/√n``（用于对拍测试）。

    Args:
        values: 时间序列（如 IC 序列）。
        lags: 滞后阶数上限 L（≥0）；对重叠收益取「预测期 − 1」。

    Returns:
        标准误；样本不足 2 个 / 方差非正 / 非法输入返回 ``None``。
    """
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    array = array[np.isfinite(array)]
    n = len(array)
    if n < 2:
        return None
    centered = array - array.mean()
    lag_count = max(int(lags), 0)
    gamma0 = float(centered @ centered) / n
    total = gamma0
    for lag in range(1, min(lag_count, n - 1) + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        gamma_lag = float(centered[lag:] @ centered[:-lag]) / n
        total += 2.0 * weight * gamma_lag
    if not np.isfinite(total) or total <= 0:
        return None
    return math.sqrt(total / n)


def t_pvalue_two_sided(t_stat: float, degrees_of_freedom: float) -> float:
    """双侧 Student-t 检验 p 值（**不依赖 scipy**，不完全 Beta 函数精确计算）。

    数学关系：``p = I_{df/(df+t²)}(df/2, 1/2)``（正则化不完全 Beta 函数）。
    边界：``t = 0`` → 1.0；``df <= 0`` 或非法输入 → 1.0（视为不显著）。

    Args:
        t_stat: t 统计量。
        degrees_of_freedom: 自由度（通常 = 样本数 − 1）。

    Returns:
        双侧 p 值 ∈ [0, 1]。
    """
    try:
        t_value = float(t_stat)
        df_value = float(degrees_of_freedom)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(t_value) or not np.isfinite(df_value) or df_value <= 0:
        return 1.0
    x = df_value / (df_value + t_value * t_value)
    return float(_regularized_incomplete_beta(df_value / 2.0, 0.5, x))


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 ``I_x(a, b)``（连分式展开，Numerical Recipes 口径）。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """不完全 Beta 的连分式（Lentz 法，收敛阈值 3e-12）。"""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for step in range(1, 201):
        twice = 2 * step
        numerator = step * (b - step) * x / ((qam + twice) * (a + twice))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        numerator = -(a + step) * (qab + step) * x / ((a + twice) * (qap + twice))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return result


# --------------------------------------------------------------------------- #
# 汇总统计
# --------------------------------------------------------------------------- #
@dataclass
class ICSummary:
    """IC 序列的汇总统计（样本不足时数值字段为 ``None``）。

    Attributes:
        n_days: 参与统计的有效交易日数。
        ic_mean: IC 均值（预测能力大小与方向）。
        ic_std: IC 标准差（预测能力波动）。
        ic_ir: ICIR = 均值 / 标准差（稳定性；越大越可信）。
        ic_win_rate: IC 与均值同号的交易日占比（方向稳定性）。
        t_stat: 普通 t 值（参考用；重叠收益下偏乐观）。
        t_stat_nw: Newey-West 修正 t 值（显著性判断以此为准）。
        p_value: 双侧 p 值（基于 NW 修正）。
        nw_lags: NW 使用的滞后阶数。
        significant: 是否显著（95% 水平：p < 0.05；t 缺失视为不显著）。
    """

    n_days: int = 0
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None
    ic_win_rate: Optional[float] = None
    t_stat: Optional[float] = None
    t_stat_nw: Optional[float] = None
    p_value: Optional[float] = None
    nw_lags: Optional[int] = None
    significant: bool = False


def summarize_ic(ic: pd.Series, horizon: int = 1, nw_lags: Optional[int] = None) -> ICSummary:
    """对 IC 序列做全套汇总统计（**不抛业务异常**）。

    Args:
        ic: :func:`rank_ic` 产出的 IC 序列。
        horizon: 预测期（交易日数）；仅用于确定 NW 默认滞后阶数。
        nw_lags: NW 滞后阶数；``None`` 时取 ``max(预测期 − 1, 1)``——重叠收益
            的自相关最长可持续「预测期 − 1」阶。

    Returns:
        :class:`ICSummary`；有效样本 < 2 时除 ``n_days`` 外均为 ``None``。
    """
    values = pd.to_numeric(pd.Series(ic), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    summary = ICSummary(n_days=int(len(values)))
    if len(values) < 2:
        return summary
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    summary.ic_mean = mean
    summary.ic_std = std
    if std > 0:
        summary.ic_ir = mean / std
    lag_count = int(nw_lags) if nw_lags is not None else max(int(horizon) - 1, 1)
    lag_count = max(lag_count, 0)
    summary.nw_lags = lag_count
    naive_se = std / math.sqrt(len(array))
    if naive_se > 0:
        summary.t_stat = mean / naive_se
    nw_se = newey_west_mean_se(array, lag_count)
    if nw_se is not None and nw_se > 0:
        nw_t = mean / nw_se
        summary.t_stat_nw = nw_t
        summary.p_value = t_pvalue_two_sided(nw_t, len(array) - 1)
        summary.significant = bool(summary.p_value < 0.05)
    if mean != 0:
        signs = np.sign(array)
        summary.ic_win_rate = float((signs == np.sign(mean)).mean())
    return summary