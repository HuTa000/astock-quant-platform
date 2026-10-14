"""P7-S3 因子计算库单测（手算对拍 + 因果性）。

全部用例为纯函数测试（不依赖 Qt / 数据库）：
    1. 动量 / 均线乖离 / 波动率 / 振幅 / 量比 / 成交额 / RSI / 位置 / 距高点
       逐因子手算对拍；
    2. 预热期置 NaN（样本不足不产生假值）；
    3. **因果性**：只改未来 bar，历史因子值必须逐点不变（防前视/未来函数）；
    4. 非法输入 / 未知因子名不抛异常（返回空 Series）；
    5. ±inf 清洗（除零结果统一按缺失处理）；
    6. 注册表元信息完整（12 个因子、名称唯一、中文名/类别/解释齐备）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.engine.factor.factors import (
    DEFAULT_FACTOR_NAMES,
    FACTOR_REGISTRY,
    compute_factor,
    compute_factors,
    factor_names,
    get_factor,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _bars(
    close,
    high=None,
    low=None,
    volume=None,
    amount=None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """按收盘价数组构造日线（high/low/volume/amount 未给则取默认值）。"""
    close_array = np.asarray(close, dtype=float)
    count = len(close_array)
    return pd.DataFrame(
        {
            "datetime": pd.bdate_range(start, periods=count),
            "open": close_array,
            "high": np.asarray(high, dtype=float) if high is not None else close_array + 0.5,
            "low": np.asarray(low, dtype=float) if low is not None else close_array - 0.5,
            "close": close_array,
            "volume": np.asarray(volume, dtype=float) if volume is not None else np.full(count, 100.0),
            "amount": np.asarray(amount, dtype=float) if amount is not None else np.full(count, 10_000.0),
        }
    )


# --------------------------------------------------------------------------- #
# 1. 逐因子手算对拍
# --------------------------------------------------------------------------- #
def test_ret_windows_hand_check() -> None:
    """涨幅因子：close=1..25，ret_5[5]=5.0、ret_20[20]=20.0，预热期为 NaN。"""
    bars = _bars(np.arange(1.0, 26.0))
    ret_5 = compute_factor("ret_5", bars)
    ret_20 = compute_factor("ret_20", bars)
    assert ret_5.iloc[5] == pytest.approx(6.0 / 1.0 - 1.0)
    assert ret_20.iloc[20] == pytest.approx(21.0 / 1.0 - 1.0)
    assert pd.isna(ret_5.iloc[4])  # 只有 5 个样本，凑不满 shift(5)
    ret_60 = compute_factor("ret_60", bars)
    assert ret_60.isna().all()  # 25 根 bar 凑不满 60 日回溯


def test_ma_dev_hand_check() -> None:
    """均线乖离：close=1..25，ma_dev_20[19] = 20 / mean(1..20) − 1。"""
    bars = _bars(np.arange(1.0, 26.0))
    dev = compute_factor("ma_dev_20", bars)
    assert dev.iloc[19] == pytest.approx(20.0 / 10.5 - 1.0)
    assert pd.isna(dev.iloc[18])


def test_vol_20_hand_check() -> None:
    """波动率：每日恒定 +1% 收益 ⇒ 20 日波动率为 0；样本不足为 NaN。"""
    close = 100.0 * np.power(1.01, np.arange(25))
    vol = compute_factor("vol_20", bars=_bars(close))
    assert vol.iloc[20] == pytest.approx(0.0, abs=1e-12)
    assert pd.isna(vol.iloc[19])


def test_amp_20_hand_check() -> None:
    """振幅：high−low=2、昨收=10 ⇒ 振幅恒为 0.2。"""
    bars = _bars(np.full(25, 10.0), high=np.full(25, 11.0), low=np.full(25, 9.0))
    amp = compute_factor("amp_20", bars)
    assert amp.iloc[20] == pytest.approx(0.2)
    assert pd.isna(amp.iloc[19])


def test_vr_5_20_hand_check() -> None:
    """量比：前 20 日量 100、后 5 日量 200 ⇒ MA5/MA20 = 200/125 = 1.6。"""
    volume = np.concatenate([np.full(20, 100.0), np.full(5, 200.0)])
    bars = _bars(np.full(25, 10.0), volume=volume)
    vr = compute_factor("vr_5_20", bars)
    assert vr.iloc[19] == pytest.approx(1.0)
    assert vr.iloc[24] == pytest.approx(200.0 / 125.0)
    assert vr.iloc[24] == pytest.approx(1.6)


def test_log_amt_hand_check() -> None:
    """成交额对数：恒 100 ⇒ ln(100)；非正成交额视为缺失（NaN）。"""
    bars = _bars(np.full(25, 10.0), amount=np.full(25, 100.0))
    log_amt = compute_factor("log_amt_20", bars)
    assert log_amt.iloc[19] == pytest.approx(math.log(100.0))
    negative = compute_factor("log_amt_20", _bars(np.full(25, 10.0), amount=np.full(25, -5.0)))
    assert negative.isna().all()


def test_rsi_hand_check() -> None:
    """RSI：单边上涨=100、完全走平=50、+1/−1 交替在 50 附近摆动。"""
    rising = compute_factor("rsi_14", _bars(10.0 + np.arange(20, dtype=float)))
    assert rising.iloc[19] == pytest.approx(100.0)
    assert pd.isna(rising.iloc[12])  # 预热期（不足 14 个收益样本）
    flat = compute_factor("rsi_14", _bars(np.full(20, 10.0)))
    assert flat.iloc[19] == pytest.approx(50.0)
    alternating = compute_factor("rsi_14", _bars(np.tile([10.0, 11.0], 10)))
    tail = alternating.dropna()
    assert tail.between(45.0, 55.0).all()
    assert tail.mean() == pytest.approx(50.0, abs=2.0)


def test_pos_20_and_dist_high_250_hand_check() -> None:
    """区间位置：最低 9、最高 13、收盘 12 ⇒ 0.75；距高点：close/最高−1。"""
    close = np.concatenate([np.full(24, 10.0), [12.0]])
    bars = _bars(close, high=close + 1.0, low=close - 1.0)
    pos = compute_factor("pos_20", bars)
    assert pos.iloc[24] == pytest.approx((12.0 - 9.0) / (13.0 - 9.0))
    assert pos.iloc[24] == pytest.approx(0.75)
    long_bars = _bars(np.full(260, 10.0), high=np.full(260, 11.0))
    dist = compute_factor("dist_high_250", long_bars)
    assert dist.iloc[249] == pytest.approx(10.0 / 11.0 - 1.0)
    assert pd.isna(dist.iloc[248])


# --------------------------------------------------------------------------- #
# 2~3. 因果性与清洗
# --------------------------------------------------------------------------- #
def test_all_factors_causal_under_future_change() -> None:
    """因果性：只改最后 30 根 bar，前 370 根的因子值必须逐点不变（防未来函数）。"""
    rng = np.random.default_rng(20261011)
    count = 400
    close = 10.0 + np.cumsum(rng.normal(0.0, 0.1, count))
    bars = _bars(
        close,
        high=close + np.abs(rng.normal(0.0, 0.2, count)),
        low=close - np.abs(rng.normal(0.0, 0.2, count)),
        volume=rng.uniform(1e6, 5e6, count),
        amount=rng.uniform(1e7, 5e7, count),
    )
    original = compute_factors(bars)

    changed = bars.copy()
    changed.loc[370:, "close"] = changed.loc[370:, "close"] + 10.0
    changed.loc[370:, "high"] = changed.loc[370:, "high"] + 10.0
    changed.loc[370:, "low"] = changed.loc[370:, "low"] + 10.0
    changed.loc[370:, "volume"] = changed.loc[370:, "volume"] * 3.0
    changed.loc[370:, "amount"] = changed.loc[370:, "amount"] * 3.0
    recomputed = compute_factors(changed)

    for name in DEFAULT_FACTOR_NAMES:
        before = original[name].iloc[:370].to_numpy(dtype=float)
        after = recomputed[name].iloc[:370].to_numpy(dtype=float)
        assert np.allclose(before, after, equal_nan=True), f"{name} 出现未来函数"


def test_inf_sanitized_to_nan() -> None:
    """除零产生的 ±inf 统一清洗为 NaN（振幅分母为 0 的场景）。"""
    close = np.concatenate([[0.0], np.full(24, 10.0)])
    amp = compute_factor("amp_20", _bars(close))
    assert not np.isinf(amp.to_numpy(dtype=float)).any()
    assert pd.isna(amp.iloc[19])  # 19 号窗口（0..19）含除零样本 ⇒ 整窗按缺失
    assert amp.iloc[24] == pytest.approx(0.1)  # 24 号窗口（5..24）已不含除零样本（1.0/10）

    zero_volume = compute_factor("vr_5_20", _bars(np.full(25, 10.0), volume=np.zeros(25)))
    assert zero_volume.isna().all()


# --------------------------------------------------------------------------- #
# 4~6. 容错与注册表
# --------------------------------------------------------------------------- #
def test_invalid_inputs_return_empty_without_raising() -> None:
    """未知因子名 / 空表 / 非法类型 ⇒ 空 Series 或空表（不抛异常）。"""
    bars = _bars(np.arange(1.0, 11.0))
    assert compute_factor("not_a_factor", bars).empty
    assert compute_factor("ret_5", pd.DataFrame()).empty
    assert compute_factor("ret_5", None).empty  # type: ignore[arg-type]
    frame = compute_factors(pd.DataFrame(), ["ret_5", "ret_20"])
    assert frame.empty and list(frame.columns) == ["ret_5", "ret_20"]


def test_registry_metadata_complete() -> None:
    """注册表：12 个因子、名称唯一、中文名/类别/解释齐备且可调用。"""
    names = factor_names()
    assert len(names) == 12
    assert len(set(names)) == 12
    assert names == list(FACTOR_REGISTRY.keys())
    assert DEFAULT_FACTOR_NAMES == names
    for name in names:
        definition = get_factor(name)
        assert definition is not None
        assert definition.label and definition.category and definition.description
        assert callable(definition.func)
    assert get_factor("不存在") is None


def test_compute_factors_column_subset() -> None:
    """批量计算：指定子集时列顺序与请求一致。"""
    bars = _bars(np.arange(1.0, 41.0))
    frame = compute_factors(bars, ["ma_dev_20", "ret_5"])
    assert list(frame.columns) == ["ma_dev_20", "ret_5"]
    assert len(frame) == 40