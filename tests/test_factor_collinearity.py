"""P7-S3 因子共线性诊断单测（手算对拍）。

覆盖：
    1. 横截面标准化：行内 z-score 手算、标准差为 0 → NaN、空输入 → 空表；
    2. 正交因子 ⇒ 相关系数 0、VIF = 1、无警告；
    3. 两因子恒等式：VIF = 1 / (1 − r²)（构造 r = 0.5 / 0.9 精确命中）；
    4. 三因子等相关（ρ = 0.5）⇒ VIF = 1.5（正交基底构造，精确值）；
    5. 完全重复因子 ⇒ 相关 1、VIF 极大、高相关对清单 + 严重共线警告；
    6. 从未共同出现 ⇒ 相关按 0 处理 + 警告；无效因子剔除后不足 2 个 ⇒ ok=False；
    7. 参数非法：入参非 dict / 少于 2 个因子 ⇒ ok=False（不抛异常）。

构造技巧：单日横截面上取 4 个两两正交、零均值、单位范数的基底
``v1=(1,1,−1,−1)/2``、``v2=(1,−1,1,−1)/2``、``v3=(1,−1,−1,1)/2``，
则 ``corr(vi, vj) = vi·vj``（行内 z-score 不改变皮尔逊相关），
相关系数可被**精确**构造（无需随机数，测试无抖动）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.factor.collinearity import (
    HIGH_CORR_THRESHOLD,
    VIF_SEVERE_THRESHOLD,
    VIF_WARNING_THRESHOLD,
    CollinearityResult,
    compute_collinearity,
    standardize_cross_section,
)

# 正交基底（4 只标的的单日横截面）：零均值、单位范数、两两点积为 0。
_BASIS_V1 = [0.5, 0.5, -0.5, -0.5]
_BASIS_V2 = [0.5, -0.5, 0.5, -0.5]
_BASIS_V3 = [0.5, -0.5, -0.5, 0.5]
_DATE = "2024-01-02"


def _panel(
    values: list[float], date: str = _DATE, prefix: str = "S"
) -> pd.DataFrame:
    """单日横截面面板：``{S1: v1, S2: v2, ...}``（dates × symbols）。"""
    return pd.DataFrame(
        {f"{prefix}{i + 1}": [float(v)] for i, v in enumerate(values)},
        index=pd.to_datetime([date]),
    )


def _combine(*terms: tuple[float, list[float]]) -> list[float]:
    """线性组合 ``Σ 系数 × 基底``（长度为 4 的向量）。"""
    total = [0.0] * len(_BASIS_V1)
    for coefficient, basis in terms:
        for index, value in enumerate(basis):
            total[index] += coefficient * value
    return total


# --------------------------------------------------------------------------- #
# 1. 横截面标准化
# --------------------------------------------------------------------------- #
def test_standardize_cross_section_hand_check() -> None:
    """行内 z-score：``[1,2,3,4]`` → 均值 2.5、ddof=0 标准差 √1.25 手算对拍。"""
    panel = _panel([1.0, 2.0, 3.0, 4.0])
    standardized = standardize_cross_section(panel)
    expected = [-1.5, -0.5, 0.5, 1.5]                       # 手算：(x − 2.5)
    actual = standardized.iloc[0].to_numpy(dtype=float)
    assert actual == pytest.approx(
        [value / math.sqrt(1.25) for value in expected]
    )
    # 行内标准化的均值 0、ddof=0 标准差 1
    assert float(standardized.iloc[0].mean()) == pytest.approx(0.0)
    assert float(standardized.iloc[0].std(ddof=0)) == pytest.approx(1.0)


def test_standardize_cross_section_edges() -> None:
    """标准差为 0（值全同）→ 该日全 NaN；空输入 / 非法输入 → 空表。"""
    constant = standardize_cross_section(_panel([5.0, 5.0, 5.0, 5.0]))
    assert bool(constant.isna().all().all())
    assert standardize_cross_section(pd.DataFrame()).empty
    assert standardize_cross_section("bad").empty


# --------------------------------------------------------------------------- #
# 2. 正交因子
# --------------------------------------------------------------------------- #
def test_orthogonal_factors_vif_one() -> None:
    """正交因子：相关系数 0、VIF = 1、无任何共线警告。"""
    panels = {
        "A": _panel(_combine((1.0, _BASIS_V1))),
        "B": _panel(_combine((1.0, _BASIS_V2))),
    }

    result = compute_collinearity(panels)

    assert result.ok
    assert result.n_obs == 4
    assert float(result.corr.loc["A", "B"]) == pytest.approx(0.0, abs=1e-12)
    assert float(result.vif["A"]) == pytest.approx(1.0)
    assert float(result.vif["B"]) == pytest.approx(1.0)
    assert result.high_corr_pairs == []
    assert result.warnings == []


# --------------------------------------------------------------------------- #
# 3. 两因子恒等式 VIF = 1/(1−r²)
# --------------------------------------------------------------------------- #
def test_two_factor_vif_identity() -> None:
    """两因子 VIF 恒等式：r = 0.5 → VIF = 4/3；r = 0.9 → VIF = 1/0.19 + 警戒。"""
    half = compute_collinearity(
        {
            "A": _panel(_combine((1.0, _BASIS_V1))),
            "B": _panel(
                _combine((0.5, _BASIS_V1), (math.sqrt(0.75), _BASIS_V2))
            ),
        }
    )
    assert half.ok
    assert float(half.corr.loc["A", "B"]) == pytest.approx(0.5)
    assert float(half.vif["A"]) == pytest.approx(1.0 / (1.0 - 0.25))
    assert float(half.vif["B"]) == pytest.approx(1.0 / (1.0 - 0.25))
    assert half.high_corr_pairs == []                       # 0.5 < 0.8 不告警

    # r = 0.9：VIF = 1/(1−0.81) = 5.263 落在 5~10 警戒区，且进高相关对清单
    strong = compute_collinearity(
        {
            "A": _panel(_combine((1.0, _BASIS_V1))),
            "B": _panel(
                _combine((0.9, _BASIS_V1), (math.sqrt(0.19), _BASIS_V2))
            ),
        }
    )
    assert float(strong.corr.loc["A", "B"]) == pytest.approx(0.9)
    assert float(strong.vif["A"]) == pytest.approx(1.0 / 0.19, rel=1e-6)
    assert len(strong.high_corr_pairs) == 1
    first, second, coefficient = strong.high_corr_pairs[0]
    assert (first, second) == ("A", "B")
    assert coefficient == pytest.approx(0.9)
    assert abs(coefficient) > HIGH_CORR_THRESHOLD
    assert VIF_WARNING_THRESHOLD <= float(strong.vif["A"]) < VIF_SEVERE_THRESHOLD
    assert any("共线警戒" in message for message in strong.warnings)


# --------------------------------------------------------------------------- #
# 4. 三因子等相关
# --------------------------------------------------------------------------- #
def test_three_factor_equicorrelation_vif() -> None:
    """三因子两两相关 0.5 ⇒ VIF = 1.5（等相关矩阵解析解精确命中）。"""
    third = _combine(
        (0.5, _BASIS_V1),
        (math.sqrt(3.0) / 6.0, _BASIS_V2),
        (math.sqrt(6.0) / 3.0, _BASIS_V3),
    )
    panels = {
        "A": _panel(_combine((1.0, _BASIS_V1))),
        "B": _panel(_combine((0.5, _BASIS_V1), (math.sqrt(0.75), _BASIS_V2))),
        "C": _panel(third),
    }

    result = compute_collinearity(panels)

    assert result.ok
    for left, right in [("A", "B"), ("A", "C"), ("B", "C")]:
        assert float(result.corr.loc[left, right]) == pytest.approx(0.5, abs=1e-9)
    # 等相关 ρ=0.5、k=3：VIF = 1 / (1 − 1/3) = 1.5
    for name in ("A", "B", "C"):
        assert float(result.vif[name]) == pytest.approx(1.5, rel=1e-6)
    assert result.high_corr_pairs == []


# --------------------------------------------------------------------------- #
# 5. 完全重复因子
# --------------------------------------------------------------------------- #
def test_duplicate_factor_severe_vif() -> None:
    """完全重复因子：相关 1、VIF 极大（≈5e9）、严重共线警告 + 高相关对清单。"""
    vector = _combine((1.0, _BASIS_V1))
    panels = {"A": _panel(vector), "dup": _panel(vector)}

    result = compute_collinearity(panels)

    assert result.ok
    assert float(result.corr.loc["A", "dup"]) == pytest.approx(1.0)
    assert float(result.vif["A"]) > 1e6                     # 近奇异：稳定项下 ≈5e9
    assert float(result.vif["dup"]) > 1e6
    assert len(result.high_corr_pairs) == 1
    assert result.high_corr_pairs[0][2] == pytest.approx(1.0)
    assert any("严重共线" in message for message in result.warnings)
    assert float(result.vif["A"]) >= VIF_SEVERE_THRESHOLD


# --------------------------------------------------------------------------- #
# 6. 从未共同出现 / 无效因子剔除
# --------------------------------------------------------------------------- #
def test_never_co_occurring_pair_warns() -> None:
    """两因子从不在同一天出现：相关按 0 处理并记警告（VIF = 1）。"""
    panels = {
        "A": _panel(_combine((1.0, _BASIS_V1)), date="2024-01-02"),
        "B": _panel(_combine((1.0, _BASIS_V2)), date="2024-01-03"),
    }

    result = compute_collinearity(panels)

    assert result.ok
    assert float(result.corr.loc["A", "B"]) == pytest.approx(0.0)
    assert float(result.vif["A"]) == pytest.approx(1.0)
    assert any("从未共同出现" in message for message in result.warnings)


def test_invalid_factor_panel_excluded() -> None:
    """单因子面板无效 ⇒ 剔除并记警告；不足 2 个有效因子 ⇒ ok=False。"""
    panels = {"A": _panel(_combine((1.0, _BASIS_V1))), "B": pd.DataFrame()}

    result = compute_collinearity(panels)

    assert not result.ok
    assert "有效因子不足 2 个" in result.error
    assert any("B 无有效数据" in message for message in result.warnings)

    # 全 NaN 面板（无有效观测）同样被剔除
    all_nan = pd.DataFrame(
        {"S1": [np.nan], "S2": [np.nan]}, index=pd.to_datetime([_DATE])
    )
    result_nan = compute_collinearity({"A": _panel([1.0, 2.0]), "B": all_nan})
    assert not result_nan.ok
    assert any("B 无有效观测" in message for message in result_nan.warnings)


# --------------------------------------------------------------------------- #
# 7. 参数非法
# --------------------------------------------------------------------------- #
def test_invalid_arguments() -> None:
    """入参非 dict / 少于 2 个因子 ⇒ ok=False（不抛异常）。"""
    assert not compute_collinearity({}).ok
    assert not compute_collinearity({"A": _panel([1.0, 2.0])}).ok
    assert not compute_collinearity("bad").ok
    assert not compute_collinearity(None).ok
    assert isinstance(compute_collinearity({}), CollinearityResult)