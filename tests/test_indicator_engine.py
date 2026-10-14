"""P3-S1 指标引擎测试：统一入口、pandas-ta 适配、内置指标、错误不抛。

覆盖设计验收标准：
    * ``calc("close.rolling(20).mean()", bars)`` 长度一致、前 19 NaN、第 20 起 = 20 日均值；
    * ``ta`` 命名空间暴露常用指标；
    * 内置 MA/EMA/MACD/RSI/KDJ/BOLL 可直接调用且与 pandas-ta 直算一致；
    * 非法表达式（语法错 / 未知名字 / 除零）返回错误 Result，不抛异常；
    * ``calc`` 返回 Series 时 index 与 bars 对齐；
    * ``calc`` 表达式内 ``ta`` / 内置指标（``MA`` 等）可正常调用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engine.indicator import (
    BOLL,
    EMA,
    KDJ,
    MA,
    MACD,
    RSI,
    IndicatorEngine,
    IndicatorResult,
    get_engine,
    ta,
)

N = 60


def make_bars(n: int = N, index: pd.Index | None = None) -> pd.DataFrame:
    """构造确定性 Bar 契约 DataFrame（close = 1..n，便于手算对照）。"""
    close = np.arange(1.0, n + 1.0)
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.5
    volume = np.arange(100.0, 100.0 + n)
    amount = close * volume
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )
    if index is not None:
        df.index = index
    return df


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return make_bars()


# --------------------------------------------------------------------------- #
# 验收 1：calc 与 rolling 均值
# --------------------------------------------------------------------------- #
def test_calc_rolling_mean_length_and_values(bars):
    engine = get_engine()
    result = engine.calc("close.rolling(20).mean()", bars)

    assert isinstance(result, IndicatorResult)
    assert result.ok is True, result.error
    series = result.series
    assert series is not None
    assert len(series) == len(bars)

    # 前 19 个为 NaN
    assert series.iloc[:19].isna().all()
    # 第 20 起为 20 日均值（手算：close=1..60，index19 = mean(1..20) = 10.5）
    assert series.iloc[19] == pytest.approx(10.5)
    assert series.iloc[20] == pytest.approx(11.5)
    assert series.iloc[-1] == pytest.approx(50.5)
    # 与 pandas 直算完全一致
    expected = bars["close"].rolling(20).mean()
    pd.testing.assert_series_equal(series, expected, check_names=False)


# --------------------------------------------------------------------------- #
# 验收 2：ta 命名空间暴露常用指标
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["sma", "ema", "macd", "rsi", "bbands", "kdj", "stoch"]
)
def test_ta_namespace_exposes_functions(name):
    assert hasattr(ta, name), f"ta 缺少指标：{name}"
    assert callable(getattr(ta, name))


def test_ta_functions_return_reasonable_results(bars):
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]

    sma = ta.sma(close, length=20)
    assert isinstance(sma, pd.Series) and len(sma) == N
    assert sma.iloc[:19].isna().all()

    ema = ta.ema(close, length=20)
    assert isinstance(ema, pd.Series) and len(ema) == N

    macd = ta.macd(close)
    assert isinstance(macd, pd.DataFrame)
    assert set(macd.columns) == {"MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"}
    assert len(macd) == N

    rsi = ta.rsi(close, length=14)
    assert isinstance(rsi, pd.Series) and len(rsi) == N

    bbands = ta.bbands(close, length=20, std=2.0)
    assert isinstance(bbands, pd.DataFrame) and len(bbands) == N
    assert "BBM_20_2.0" in bbands.columns

    kdj = ta.kdj(high=high, low=low, close=close)
    assert isinstance(kdj, pd.DataFrame) and len(kdj) == N
    assert {"K_9_3", "D_9_3", "J_9_3"}.issubset(kdj.columns)


# --------------------------------------------------------------------------- #
# 验收 3：内置指标可直接调用且值合理（对照 pandas-ta 直算）
# --------------------------------------------------------------------------- #
def test_builtin_ma_matches_ta(bars):
    pd.testing.assert_series_equal(
        MA(bars, 20), ta.sma(bars["close"], length=20), check_names=False
    )
    assert len(MA(bars, 20)) == N
    assert MA(bars, 20).iloc[:19].isna().all()


def test_builtin_ema_matches_ta(bars):
    pd.testing.assert_series_equal(
        EMA(bars, 20), ta.ema(bars["close"], length=20), check_names=False
    )


def test_builtin_macd_matches_ta(bars):
    expected = ta.macd(bars["close"])["MACD_12_26_9"]
    pd.testing.assert_series_equal(MACD(bars), expected, check_names=False)


def test_builtin_rsi_matches_ta(bars):
    pd.testing.assert_series_equal(
        RSI(bars, 14), ta.rsi(bars["close"], length=14), check_names=False
    )


def test_builtin_kdj_matches_ta(bars):
    expected = ta.kdj(
        high=bars["high"], low=bars["low"], close=bars["close"]
    )["K_9_3"]
    pd.testing.assert_series_equal(KDJ(bars), expected, check_names=False)


def test_builtin_boll_matches_ta(bars):
    expected = ta.bbands(bars["close"], length=20, std=2.0)["BBM_20_2.0"]
    pd.testing.assert_series_equal(BOLL(bars), expected, check_names=False)


def test_builtin_boll_other_columns(bars):
    upper = BOLL(bars, column="UPPER")
    lower = BOLL(bars, column="LOWER")
    mid = BOLL(bars, column="MID")
    assert len(upper) == len(bars)
    # 上轨 >= 中轨 >= 下轨（在有效值区间）
    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


# --------------------------------------------------------------------------- #
# 验收 4：非法表达式返回错误 Result，不抛异常
# --------------------------------------------------------------------------- #
def test_calc_syntax_error_returns_failure(bars):
    result = get_engine().calc("close.rolling(", bars)
    assert result.ok is False
    assert result.error
    assert result.series is None


def test_calc_unknown_name_returns_failure(bars):
    result = get_engine().calc("foo.rolling(5)", bars)
    assert result.ok is False
    assert result.error


def test_calc_divide_by_zero_returns_failure(bars):
    result = get_engine().calc("close/0", bars)
    assert result.ok is False
    assert result.error


def test_calc_bad_bars_returns_failure():
    result = get_engine().calc("close.rolling(5).mean()", None)
    assert result.ok is False
    assert result.error

    result = get_engine().calc("close.rolling(5).mean()", pd.DataFrame())
    assert result.ok is False
    assert result.error


def test_calc_bad_expr_returns_failure(bars):
    assert get_engine().calc("", bars).ok is False
    assert get_engine().calc(None, bars).ok is False


def test_calc_never_raises(bars):
    """引擎层不向 UI 抛异常：对一组病态输入逐一调用，均返回 Result。"""
    for expr in ["close.rolling(", "foo.rolling(5)", "close/0", "ta.nonexistent(close)"]:
        result = get_engine().calc(expr, bars)
        assert isinstance(result, IndicatorResult)


# --------------------------------------------------------------------------- #
# 验收 5：calc 返回 Series 时 index 与 bars 对齐
# --------------------------------------------------------------------------- #
def test_calc_series_index_aligns_with_bars():
    # 非默认 index，验证对齐逻辑对自定义 index 也成立。
    custom_index = pd.RangeIndex(100, 100 + N)
    frame = make_bars(index=custom_index)
    result = get_engine().calc("close.rolling(20).mean()", frame)
    assert result.ok is True
    assert result.series.index.equals(frame.index)
    assert len(result.series) == len(frame)


def test_calc_ta_macd_column_selection_aligns(bars):
    result = get_engine().calc("ta.macd(close)['MACD_12_26_9']", bars)
    assert result.ok is True
    assert result.series is not None
    assert len(result.series) == len(bars)
    assert result.series.index.equals(bars.index)


def test_get_engine_singleton():
    assert get_engine() is get_engine()


# --------------------------------------------------------------------------- #
# calc 表达式内 ta / 内置指标可正常调用（PRD C 项）
# --------------------------------------------------------------------------- #
def test_calc_ma_callable(bars):
    result = get_engine().calc("MA(bars, 20)", bars)
    assert result.ok is True, result.error
    assert result.series is not None
    assert len(result.series) == len(bars)
    assert result.series.iloc[:19].isna().all()


def test_calc_ta_callable(bars):
    result = get_engine().calc("ta.sma(close, 20)", bars)
    assert result.ok is True, result.error
    assert result.series is not None
    assert len(result.series) == len(bars)
    assert result.series.iloc[:19].isna().all()


# --------------------------------------------------------------------------- #
# 回归：短序列（行数 < 指标窗口）不崩溃，返回全 NaN（降级不抛）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [MA, EMA, MACD, RSI, KDJ, BOLL])
def test_builtin_short_sequence_returns_full_nan(fn):
    short = make_bars(n=5)
    # 5 根 bar 小于所有指标默认窗口，pandas-ta 会返回 None；
    # 内置指标应降级为与输入等长的全 NaN Series，而不是抛 NoneType 异常。
    result = fn(short)
    assert isinstance(result, pd.Series)
    assert len(result) == 5
    assert result.isna().all()


def test_calc_short_sequence_builtin_returns_nan_not_error():
    short = make_bars(n=5)
    # MA 已恢复注入 calc 命名空间；短序列时内置指标降级为全 NaN，不抛异常。
    result = get_engine().calc("MA(bars, 20)", short)
    assert result.ok is True
    assert result.series is not None
    assert len(result.series) == 5
    assert result.series.isna().all()
