"""``derive_bars`` / ``derive_minute_bars`` 纯函数单测（手算期望值，**不拿 resample 输出去对**）。

覆盖：
    1. 跨 3 周、含月初月末边界的日线 → 周线 / 月线**逐字段**手算断言；
    2. 单根日线 → 1 根，OHLCV / datetime 与输入相同；
    3. 空表 / ``None`` / 缺列 / 不可派生周期 → 空 Bar 契约表且**不抛**；
    4. 乱序输入 → 输出仍按周期时间升序；
    5. 结果列序 == :data:`app.data.store.db.BAR_COLUMNS`；
    6. 分钟派生（P6）：5m → 15/30/60m 逐字段手算、跨日不合并、午休不错桶、
       余根成桶、降级不抛、乱序输入升序。
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.resample import (
    RESAMPLABLE,
    can_derive,
    can_derive_minute,
    derive_bars,
    derive_minute_bars,
)
from app.data.store.db import BAR_COLUMNS


def _sample_daily() -> pd.DataFrame:
    """9 根日线：跨 3 个自然周（W-SUN），跨 2024-01/02 月初月末边界。

    日期（含星期）：01-25(四) 01-26(五) | 01-29(一) 01-30(二) 01-31(三) 02-01(四) 02-02(五)
    | 02-05(一) 02-06(二)。周分组：{01-25,01-26} / {01-29..02-02} / {02-05,02-06}；
    月分组：Jan{01-25..01-31} / Feb{02-01..02-06}。
    """
    rows = [
        # date,        open,  high,  low,   close, volume, amount
        ("2024-01-25", 10.0, 10.5, 9.8, 10.2, 100, 1000),
        ("2024-01-26", 10.2, 10.6, 10.0, 10.4, 200, 2000),
        ("2024-01-29", 10.4, 11.0, 10.3, 10.8, 300, 3000),
        ("2024-01-30", 10.8, 11.2, 10.6, 10.9, 400, 4000),
        ("2024-01-31", 10.9, 11.5, 10.7, 11.3, 500, 5000),
        ("2024-02-01", 11.3, 11.6, 11.0, 11.1, 600, 6000),
        ("2024-02-02", 11.1, 11.4, 10.9, 11.0, 700, 7000),
        ("2024-02-05", 11.0, 11.8, 10.95, 11.7, 800, 8000),
        ("2024-02-06", 11.7, 12.0, 11.5, 11.9, 900, 9000),
    ]
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp(r[0]) for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [float(r[5]) for r in rows],
            "amount": [float(r[6]) for r in rows],
        }
    )


def _expected_weekly() -> pd.DataFrame:
    """手算周线期望（open=首日开 / high=max / low=min / close=末日收 / vol,amt=求和 / dt=末日）。"""
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-26"), pd.Timestamp("2024-02-02"), pd.Timestamp("2024-02-06")],
            "open": [10.0, 10.4, 11.0],
            "high": [10.6, 11.6, 12.0],
            "low": [9.8, 10.3, 10.95],
            "close": [10.4, 11.0, 11.9],
            "volume": [300.0, 2500.0, 1700.0],
            "amount": [3000.0, 25000.0, 17000.0],
        }
    )


def _expected_monthly() -> pd.DataFrame:
    """手算月线期望。"""
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-06")],
            "open": [10.0, 11.3],
            "high": [11.5, 12.0],
            "low": [9.8, 10.9],
            "close": [11.3, 11.9],
            "volume": [1500.0, 3000.0],
            "amount": [15000.0, 30000.0],
        }
    )


# --------------------------------------------------------------------------- #
# 1. 手算期望：周线 / 月线逐字段
# --------------------------------------------------------------------------- #
def test_derive_weekly_matches_hand_computed() -> None:
    result = derive_bars(_sample_daily(), "w").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, _expected_weekly(), check_dtype=False)


def test_derive_monthly_matches_hand_computed() -> None:
    result = derive_bars(_sample_daily(), "m").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, _expected_monthly(), check_dtype=False)


# --------------------------------------------------------------------------- #
# 2. 单根日线
# --------------------------------------------------------------------------- #
def test_derive_single_daily_bar_passthrough() -> None:
    one = _sample_daily().iloc[[4]].reset_index(drop=True)  # 2024-01-31
    for period in ("w", "m"):
        result = derive_bars(one, period)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["datetime"] == pd.Timestamp("2024-01-31")
        assert row["open"] == 10.9
        assert row["high"] == 11.5
        assert row["low"] == 10.7
        assert row["close"] == 11.3
        assert row["volume"] == 500.0
        assert row["amount"] == 5000.0


# --------------------------------------------------------------------------- #
# 3. 降级：空 / None / 缺列 / 不可派生周期 → 空 Bar 契约，且不抛
# --------------------------------------------------------------------------- #
def test_derive_empty_and_none_no_raise() -> None:
    for bad in (None, pd.DataFrame()):
        result = derive_bars(bad, "w")
        assert result.empty
        assert list(result.columns) == BAR_COLUMNS


def test_derive_missing_columns_no_raise() -> None:
    bad = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-01")], "close": [1.0]})
    result = derive_bars(bad, "m")
    assert result.empty
    assert list(result.columns) == BAR_COLUMNS


def test_derive_not_resamplable_returns_empty() -> None:
    for period in ("d", "5m", "1m", ""):
        result = derive_bars(_sample_daily(), period)
        assert result.empty
        assert list(result.columns) == BAR_COLUMNS


def test_can_derive_only_week_month() -> None:
    assert RESAMPLABLE == ("w", "m")
    assert can_derive("w") is True
    assert can_derive("m") is True
    for period in ("d", "5m", "15m", "30m", "60m", "1m", ""):
        assert can_derive(period) is False


# --------------------------------------------------------------------------- #
# 4. 乱序输入 → 输出升序（且聚合以「时间先后」为准）
# --------------------------------------------------------------------------- #
def test_derive_handles_shuffled_input_and_sorts_ascending() -> None:
    shuffled = _sample_daily().sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert list(shuffled["datetime"]) != list(_sample_daily()["datetime"])  # 确实乱序

    weekly = derive_bars(shuffled, "w").reset_index(drop=True)
    # 与顺序输入结果一致（聚合以时间先后为准，不受行序影响）
    pd.testing.assert_frame_equal(weekly, _expected_weekly(), check_dtype=False)
    # 且严格升序
    dts = list(weekly["datetime"])
    assert dts == sorted(dts)


# --------------------------------------------------------------------------- #
# 5. 列序 == BAR_COLUMNS
# --------------------------------------------------------------------------- #
def test_derive_columns_match_bar_contract() -> None:
    for period in ("w", "m"):
        assert list(derive_bars(_sample_daily(), period).columns) == BAR_COLUMNS


# --------------------------------------------------------------------------- #
# 6. 分钟派生（5m → 15/30/60m）
# --------------------------------------------------------------------------- #
def _sample_5m(days: tuple[str, ...] = ("2026-01-05",), per_session: int = 24) -> pd.DataFrame:
    """构造 5 分钟 Bar（``datetime`` = bar **结束时刻**）：上午 09:35 起、下午 13:05 起各 ``per_session`` 根。

    第 i 根（跨日全局序号，0 起）：``open=10+0.1i``、``high=open+0.05``、``low=open-0.05``、
    ``close=open+0.02``、``volume=10+i``、``amount=1000+i``（手算期望值由此推出）。
    """
    rows: list[tuple] = []
    i = 0
    for day in days:
        stamps = [
            pd.Timestamp(f"{day} 09:35") + pd.Timedelta(minutes=5 * j)
            for j in range(per_session)
        ] + [
            pd.Timestamp(f"{day} 13:05") + pd.Timedelta(minutes=5 * j)
            for j in range(per_session)
        ]
        for dt in stamps:
            rows.append(
                (
                    dt,
                    10.0 + 0.10 * i,
                    10.05 + 0.10 * i,
                    9.95 + 0.10 * i,
                    10.02 + 0.10 * i,
                    10.0 + i,
                    1000.0 + i,
                )
            )
            i += 1
    return pd.DataFrame(
        {
            "datetime": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [float(r[5]) for r in rows],
            "amount": [float(r[6]) for r in rows],
        }
    )


def test_derive_minute_5m_to_15m_hand_computed() -> None:
    result = derive_minute_bars(_sample_5m(), "5m", "15m").reset_index(drop=True)
    assert len(result) == 16  # 每日 48 根 / 3 = 16 桶（上午 8 + 下午 8）

    first = result.iloc[0]  # 上午桶 0 = 09:35 / 09:40 / 09:45
    assert first["datetime"] == pd.Timestamp("2026-01-05 09:45")  # 取桶内末根（结束时刻）
    assert first["open"] == pytest.approx(10.00)
    assert first["high"] == pytest.approx(10.25)
    assert first["low"] == pytest.approx(9.95)
    assert first["close"] == pytest.approx(10.22)
    assert first["volume"] == pytest.approx(33.0)  # 10+11+12
    assert first["amount"] == pytest.approx(3003.0)  # 1000+1001+1002

    # 午休不错桶：上午末桶结束 11:30，下午首桶结束 13:15（不跨午休合并）
    assert result.iloc[7]["datetime"] == pd.Timestamp("2026-01-05 11:30")
    afternoon_first = result.iloc[8]  # 全局序号 24~26 = 13:05 / 13:10 / 13:15
    assert afternoon_first["datetime"] == pd.Timestamp("2026-01-05 13:15")
    assert afternoon_first["open"] == pytest.approx(12.40)
    assert afternoon_first["high"] == pytest.approx(12.65)
    assert afternoon_first["low"] == pytest.approx(12.35)
    assert afternoon_first["close"] == pytest.approx(12.62)
    assert afternoon_first["volume"] == pytest.approx(105.0)  # 34+35+36
    assert afternoon_first["amount"] == pytest.approx(3075.0)  # 1024+1025+1026


def test_derive_minute_5m_to_60m_hand_computed() -> None:
    result = derive_minute_bars(_sample_5m(), "5m", "60m").reset_index(drop=True)
    assert len(result) == 4  # 每日 48 根 / 12 = 4 桶

    first = result.iloc[0]  # 全局序号 0~11 → 结束 10:30
    assert first["datetime"] == pd.Timestamp("2026-01-05 10:30")
    assert first["open"] == pytest.approx(10.00)
    assert first["high"] == pytest.approx(11.15)  # open_11=11.10 + 0.05
    assert first["low"] == pytest.approx(9.95)
    assert first["close"] == pytest.approx(11.12)  # 11.10 + 0.02
    assert first["volume"] == pytest.approx(186.0)  # Σ(10+i), i=0..11
    assert first["amount"] == pytest.approx(12066.0)  # Σ(1000+i), i=0..11

    second = result.iloc[1]  # 全局序号 12~23 → 结束 11:30
    assert second["datetime"] == pd.Timestamp("2026-01-05 11:30")
    assert second["open"] == pytest.approx(11.20)
    assert second["high"] == pytest.approx(12.35)
    assert second["low"] == pytest.approx(11.15)
    assert second["close"] == pytest.approx(12.32)
    assert second["volume"] == pytest.approx(330.0)  # 12×10 + 210
    assert second["amount"] == pytest.approx(12210.0)  # 12×1000 + 210


@pytest.mark.parametrize("target,per_day", [("15m", 16), ("30m", 8), ("60m", 4)])
def test_derive_minute_bucket_counts(target: str, per_day: int) -> None:
    assert len(derive_minute_bars(_sample_5m(), "5m", target)) == per_day


def test_derive_minute_multiday_boundary_no_merge() -> None:
    result = derive_minute_bars(_sample_5m(days=("2026-01-05", "2026-01-06")), "5m", "15m")
    assert len(result) == 32
    days = [dt.normalize() for dt in result["datetime"]]
    assert days.count(pd.Timestamp("2026-01-05")) == 16
    assert days.count(pd.Timestamp("2026-01-06")) == 16
    # 第二日首桶从当日 09:45 起（未与第一日尾桶合并）
    assert result.iloc[16]["datetime"] == pd.Timestamp("2026-01-06 09:45")
    dts = list(result["datetime"])
    assert dts == sorted(dts)


def test_derive_minute_partial_last_bucket_is_emitted() -> None:
    # 每时段仅 5 根 → 15m 得 4 桶（每时段 3 + 余 2），余根也成桶且取桶内末根时间
    result = derive_minute_bars(_sample_5m(per_session=5), "5m", "15m").reset_index(drop=True)
    assert len(result) == 4
    remainder = result.iloc[1]  # 全局序号 3、4 = 09:50 / 09:55
    assert remainder["datetime"] == pd.Timestamp("2026-01-05 09:55")
    assert remainder["open"] == pytest.approx(10.30)  # i=3
    assert remainder["high"] == pytest.approx(10.45)  # i=4: 10.40+0.05
    assert remainder["low"] == pytest.approx(10.25)  # i=3: 10.30-0.05
    assert remainder["close"] == pytest.approx(10.42)  # i=4: 10.40+0.02
    assert remainder["volume"] == pytest.approx(27.0)  # (10+3)+(10+4)
    assert remainder["amount"] == pytest.approx(2007.0)  # 1003+1004


def test_can_derive_minute_pairs() -> None:
    for source, target in (
        ("5m", "15m"),
        ("5m", "30m"),
        ("5m", "60m"),
        ("1m", "15m"),
        ("15m", "60m"),
        ("30m", "60m"),
    ):
        assert can_derive_minute(source, target) is True
    for source, target in (
        ("5m", "5m"),
        ("15m", "5m"),
        ("60m", "15m"),
        ("d", "15m"),
        ("", "15m"),
        ("5m", "d"),
        ("5m", ""),
    ):
        assert can_derive_minute(source, target) is False


def test_derive_minute_invalid_inputs_no_raise() -> None:
    bad_frames = (
        None,
        pd.DataFrame(),
        pd.DataFrame({"datetime": [pd.Timestamp("2026-01-05 09:35")], "close": [1.0]}),
    )
    for bad in bad_frames:
        result = derive_minute_bars(bad, "5m", "15m")
        assert result.empty
        assert list(result.columns) == BAR_COLUMNS
    # 不可派生组合（大周期 → 小周期）→ 空表且不抛
    result = derive_minute_bars(_sample_5m(), "60m", "15m")
    assert result.empty
    assert list(result.columns) == BAR_COLUMNS


def test_derive_minute_shuffled_input_sorts_ascending() -> None:
    ordered = derive_minute_bars(_sample_5m(), "5m", "15m").reset_index(drop=True)
    shuffled = _sample_5m().sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert list(shuffled["datetime"]) != list(_sample_5m()["datetime"])  # 确实乱序

    result = derive_minute_bars(shuffled, "5m", "15m").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, ordered, check_dtype=False)


def test_derive_minute_columns_match_bar_contract() -> None:
    for target in ("15m", "30m", "60m"):
        assert list(derive_minute_bars(_sample_5m(), "5m", target).columns) == BAR_COLUMNS
