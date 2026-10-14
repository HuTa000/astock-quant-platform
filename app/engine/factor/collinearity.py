"""因子共线性诊断（相关矩阵 + VIF，P7-S3）。

定位：
    回答「这些因子是不是在说同一件事」——两个因子若高度相关，同时放进多因子
    模型等于**重复下注**：IC 各自看起来有效，合起来却不会带来增量信息，
    还会让回归系数不稳定。本模块产出：
    * **相关矩阵**：因子两两皮尔逊相关（买入前的原始因子值先做**每日横截面
      z-score 标准化**，消除量纲差异后再比较）；
    * **VIF（方差膨胀因子）**：因子 j 的方差中约有 ``1 − 1/VIF`` 的比例能被
      其他因子线性解释（VIF=1 完全独立、=5 约 80% 可被解释、≥10 严重共线）；
    * **高相关因子对清单**（|r| > 0.8）与警告。

计算口径与稳健处理：
    * 标准化：每个交易日横截面上 ``(x − 当日均值) / 当日标准差``——只做
      **横截面**标准化，不跨日，不引入未来信息；
    * 相关矩阵用**成对完整样本**（pandas ``corr`` 默认成对剔除缺失）；
    * VIF 用标准恒等式 ``VIF_j = [R⁻¹]_jj``（R 为相关矩阵；与逐个回归
      1/(1−R²) 在同样本下完全等价），并对角加 **1e-10 数值稳定项**——
      完全重复的因子会得到极大 VIF（≈5e9），如实反映「该组合近乎奇异」；
    * 两因子从未共同出现（相关为 NaN）时按 0 处理并记警告。

对外函数**不抛业务异常**：因子不足 2 个 / 无有效数据 → ``ok=False``。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 高相关因子对的告警阈值（|r| 超过即列出）。
HIGH_CORR_THRESHOLD: float = 0.8

# VIF 告警阈值：≥5 警戒、≥10 严重（业内通用经验值）。
VIF_WARNING_THRESHOLD: float = 5.0
VIF_SEVERE_THRESHOLD: float = 10.0

# 相关矩阵求逆时的对角稳定项（防完全共线导致奇异）。
_RIDGE: float = 1e-10


@dataclass
class CollinearityResult:
    """共线性诊断产出（**不抛异常**，问题封装为 ``ok=False`` + ``error``）。

    Attributes:
        ok: 是否完成诊断。
        error: 失败原因（成功时为空串）。
        corr: 因子两两相关矩阵（k × k，索引 = 因子名）。
        vif: 各因子 VIF（Series，索引 = 因子名；严重奇异处为 ``inf``）。
        high_corr_pairs: 高相关因子对 ``(因子a, 因子b, 相关系数)``，按 |r| 降序。
        n_obs: 参与诊断的（交易日 × 标的）观测数。
        warnings: 诊断警告（VIF 超限、因子对从未共同出现等）。
    """

    ok: bool
    error: str = ""
    corr: pd.DataFrame = field(default_factory=pd.DataFrame)
    vif: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    high_corr_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    n_obs: int = 0
    warnings: list[str] = field(default_factory=list)


def standardize_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """每日横截面 z-score 标准化（行内 ``(x − 均值) / 标准差``，``ddof=0``）。

    某日标准差为 0（因子值全同）/ 有效样本不足 2 个 → 该日全部记 NaN。

    Args:
        panel: 因子面板（dates × symbols）。

    Returns:
        同形标准化矩阵；非法输入返回空表。
    """
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return pd.DataFrame()
    mean = panel.mean(axis=1)
    std = panel.std(axis=1, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = panel.sub(mean, axis=0).div(std, axis=0)
    standardized = standardized.replace([np.inf, -np.inf], np.nan)
    return standardized


def compute_collinearity(panels: dict[str, pd.DataFrame]) -> CollinearityResult:
    """计算因子相关矩阵与 VIF（**不抛业务异常**）。

    Args:
        panels: ``{因子名: 因子面板}``（面板 = dates × symbols，至少 2 个因子）。

    Returns:
        :class:`CollinearityResult`。
    """
    if not isinstance(panels, dict) or len(panels) < 2:
        return CollinearityResult(ok=False, error="共线性诊断至少需要 2 个因子")
    warnings: list[str] = []
    stacked: dict[str, pd.Series] = {}
    for name, panel in panels.items():
        standardized = standardize_cross_section(panel)
        if standardized.empty:
            warnings.append(f"因子 {name} 无有效数据，已从共线性诊断中剔除")
            continue
        series = standardized.stack().dropna()
        if series.empty:
            warnings.append(f"因子 {name} 无有效观测，已从共线性诊断中剔除")
            continue
        stacked[str(name)] = pd.to_numeric(series, errors="coerce").astype(float)
    if len(stacked) < 2:
        return CollinearityResult(
            ok=False, error="有效因子不足 2 个，无法诊断共线性", warnings=warnings
        )

    frame = pd.DataFrame(stacked)
    n_obs = int(len(frame))
    corr = frame.corr()
    if corr.isna().any().any():
        never = [
            f"{a}×{b}"
            for index, a in enumerate(corr.index)
            for b in corr.columns[index + 1:]
            if not np.isfinite(corr.loc[a, b])
        ]
        if never:
            warnings.append(
                f"以下因子对从未共同出现，相关系数按 0 处理：{'、'.join(never[:5])}"
            )
        corr = corr.fillna(0.0)

    matrix = corr.to_numpy(dtype=float)
    try:
        inverse = np.linalg.inv(matrix + np.eye(len(corr)) * _RIDGE)
        diagonal = np.diag(inverse)
    except np.linalg.LinAlgError:  # pragma: no cover - 有稳定项时几乎不可能
        diagonal = np.full(len(corr), np.inf)
    values: list[float] = []
    for value in diagonal:
        if not np.isfinite(value):
            values.append(float("inf"))
        elif value <= 0:  # pragma: no cover - 理论非正，数值异常时按缺失处理
            values.append(float("nan"))
        else:
            values.append(float(value))
    vif = pd.Series(values, index=corr.index, name="vif")

    high_corr_pairs: list[tuple[str, str, float]] = []
    for index, first in enumerate(corr.index):
        for second in corr.columns[index + 1:]:
            value = float(corr.loc[first, second])
            if np.isfinite(value) and abs(value) > HIGH_CORR_THRESHOLD:
                high_corr_pairs.append((str(first), str(second), value))
    high_corr_pairs.sort(key=lambda item: abs(item[2]), reverse=True)

    severe = [str(name) for name, value in vif.items() if value >= VIF_SEVERE_THRESHOLD]
    warning_level = [
        str(name)
        for name, value in vif.items()
        if VIF_WARNING_THRESHOLD <= value < VIF_SEVERE_THRESHOLD
    ]
    if severe:
        warnings.append(f"严重共线（VIF≥10，建议剔除或合成）：{'、'.join(severe)}")
    if warning_level:
        warnings.append(f"共线警戒（VIF 5~10，谨慎同用）：{'、'.join(warning_level)}")
    for first, second, value in high_corr_pairs[:3]:
        warnings.append(f"高相关因子对：{first} × {second}（r={value:.2f}）")

    return CollinearityResult(
        ok=True,
        corr=corr,
        vif=vif,
        high_corr_pairs=high_corr_pairs,
        n_obs=n_obs,
        warnings=warnings,
    )