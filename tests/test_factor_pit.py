"""P7-S4 PIT 成分掩码与股票池口径单测（行为对拍）。

覆盖：
    1. 面板掩码未启用（``mask=None``）⇒ 取值与缓存行为与 P7-S3 完全一致；
    2. 掩码启用后：非成分日 / 非成分股的一切取值（因子与前瞻收益）置 NaN；
    3. 掩码与面板索引 / 列不完全对齐时，缺失格子按「非成分」处理；
    4. 编排 PIT 口径 E2E：研究池 = 快照并集；某交易日只用「当时生效名单」
       （严格 PIT——早于首份名单的交易日无样本）；
    5. 评估窗口过滤研究池；窗口起点落在某份名单区间中间时该名单成员不缺席；
    6. 降级路径：库内无快照 ⇒ ``ok=False``；显式池横截面不足 ⇒ ``ok=False``；
    7. 非法 ``universe`` 取值 ⇒ ``ValueError``；默认口径仍为 ``"current"``。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.models import IndexConstituent
from app.data.universe import membership_matrix
from app.data.store.db import Database
from app.engine.factor import FactorAnalysisSpec, FactorAnalyzer, build_panel


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _biz_days(count: int, start: str = "2024-01-01") -> list[str]:
    """生成 count 个「交易日」日期串（工作日口径）。"""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=count)]


def _linear_spec(days: list[str], start_price: float) -> dict:
    """线性上涨行情：每日 +1 元（open = close，便于手算）。"""
    return {
        day: (start_price + index, start_price + index + 0.5, start_price + index - 0.5,
              start_price + index)
        for index, day in enumerate(days)
    }


def _seed(db: Database, symbol: str, spec: dict) -> None:
    """按 ``{日期: (o, h, l, c)}`` 写入前复权日线。"""
    frame = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(day),
                "open": float(values[0]),
                "high": float(values[1]),
                "low": float(values[2]),
                "close": float(values[3]),
                "volume": 1_000_000.0,
                "amount": float(values[3]) * 1_000_000.0,
            }
            for day, values in spec.items()
        ]
    )
    frame["symbol"] = symbol
    frame["period"] = "d"
    frame["adjust"] = "qfq"
    db.upsert_bars(frame)


def _seed_constituents(
    db: Database, snapshots: dict[str, list[str]], index_code: str = "sh.000300"
) -> None:
    """按 ``{生效日: [标的, ...]}`` 写入成分股快照。"""
    rows = [
        IndexConstituent(
            index_code=index_code,
            symbol=symbol,
            name="",
            effective_date=pd.Timestamp(day),
        )
        for day, members in snapshots.items()
        for symbol in members
    ]
    db.upsert_index_constituents(rows)


# --------------------------------------------------------------------------- #
# 1~3. 面板掩码
# --------------------------------------------------------------------------- #
def test_panel_mask_none_matches_default_behavior(tmp_path) -> None:
    """未启用掩码：取值与「惰性缓存返回同一对象」的行为保持不变。"""
    db = _open(tmp_path / "nomask.db")
    days = _biz_days(10)
    _seed(db, "000001", _linear_spec(days, 10.0))

    panel, _ = build_panel(db, ["000001"])

    assert panel.mask is None
    factor = panel.factor("ret_5")
    assert panel.factor("ret_5") is factor                  # 缓存对象原样返回
    assert float(factor["000001"].loc[days[9]]) == pytest.approx(19.0 / 14.0 - 1.0)
    assert float(panel.forward(1)["000001"].loc[days[5]]) == pytest.approx(17.0 / 16.0 - 1.0)


def test_panel_mask_zeroes_non_constituent_cells(tmp_path) -> None:
    """掩码外的格子（非成分日 / 非成分股）因子与前瞻收益均为 NaN。"""
    db = _open(tmp_path / "mask.db")
    days = _biz_days(10)
    _seed(db, "000001", _linear_spec(days, 10.0))
    _seed(db, "000002", _linear_spec(days, 20.0))

    panel, _ = build_panel(db, ["000001", "000002"])

    keep = pd.DataFrame(False, index=panel.dates, columns=panel.symbols)
    keep.loc[pd.Timestamp(days[5]):pd.Timestamp(days[8]), "000001"] = True
    panel.mask = keep

    factor = panel.factor("ret_5")
    forward = panel.forward(1)
    # 成分区间内取值保留
    assert not pd.isna(factor.loc[days[5], "000001"])
    assert not pd.isna(forward.loc[days[5], "000001"])
    # 非成分日（成分区间之外）置 NaN
    assert pd.isna(factor.loc[days[4], "000001"])
    assert pd.isna(factor.loc[days[9], "000001"])
    assert pd.isna(forward.loc[days[9], "000001"])
    # 非成分股整列为 NaN
    assert factor["000002"].isna().all()
    assert forward["000002"].isna().all()


def test_panel_mask_misaligned_subset_treated_as_false(tmp_path) -> None:
    """掩码只覆盖部分日期 / 标的时，缺失格子按非成分处理（不抛异常）。"""
    db = _open(tmp_path / "partial.db")
    days = _biz_days(10)
    _seed(db, "000001", _linear_spec(days, 10.0))
    _seed(db, "000002", _linear_spec(days, 20.0))

    panel, _ = build_panel(db, ["000001", "000002"])

    tiny = pd.DataFrame(True, index=[pd.Timestamp(days[5])], columns=["000001"])
    panel.mask = tiny

    factor = panel.factor("ret_5")
    assert not pd.isna(factor.loc[days[5], "000001"])       # 掩码命中
    assert pd.isna(factor.loc[days[6], "000001"])           # 掩码缺行 ⇒ 非成分
    assert factor["000002"].isna().all()                    # 掩码缺列 ⇒ 非成分

    # 掩码结构完全不可对齐（非 DataFrame）⇒ 降级为不启用（不抛异常）
    panel.mask = object()
    factor = panel.factor("ret_5")
    assert not pd.isna(factor.loc[days[9], "000001"])


# --------------------------------------------------------------------------- #
# 4~5. 编排 PIT 口径
# --------------------------------------------------------------------------- #
def _pit_fixture(db: Database, days: int = 20) -> list[str]:
    """播种 4 只线性行情标的 + 2 份快照（A: 1~2 号 / B: 3~4 号）。"""
    calendar = _biz_days(days)
    for index, symbol in enumerate(["000001", "000002", "000003", "000004"]):
        _seed(db, symbol, _linear_spec(calendar, 10.0 + index * 10.0))
    _seed_constituents(
        db,
        {
            calendar[5]: ["000001", "000002"],              # 首份名单生效（第 6 个交易日）
            calendar[11]: ["000003", "000004"],             # 第二份名单生效
        },
    )
    return calendar


def test_analysis_pit_strict_schedule_and_union_pool(tmp_path) -> None:
    """PIT E2E：研究池 = 快照并集；严格按「当时生效名单」逐日取样本。"""
    db = _open(tmp_path / "pit.db")
    calendar = _pit_fixture(db)
    spec = FactorAnalysisSpec(
        factor_names=["ret_5"],
        horizons=[1],
        n_groups=2,
        min_stocks=2,
        universe="pit",
    )

    report = FactorAnalyzer(db=db).run(spec)

    assert report.ok, report.error
    assert report.universe_mode == "pit"
    # 研究池 = 快照并集（不是「当前成分」A 的 2 只）
    assert report.universe == ["000001", "000002", "000003", "000004"]
    # 严格 PIT：A 生效（第 6 日）之前无样本；末两日前瞻收益未走完 ⇒ 无 IC
    expected = [calendar[index] for index in range(5, 18)]
    assert [d.strftime("%Y-%m-%d") for d in report.ic_series[("ret_5", 1)].index] == expected
    assert report.ic_table["n_days"].tolist() == [13]
    # 名单内两只标的的因子排序与前瞻收益排序完全一致 ⇒ 每日 IC = +1
    assert float(report.ic_table["ic_mean"].iloc[0]) == pytest.approx(1.0)


def test_analysis_pit_window_filters_union(tmp_path) -> None:
    """评估窗口过滤研究池；窗口起点落在名单区间中间时该名单成员不缺席。"""
    db = _open(tmp_path / "pit_window.db")
    calendar = _pit_fixture(db)
    spec = FactorAnalysisSpec(
        factor_names=["ret_5"],
        horizons=[1],
        n_groups=2,
        min_stocks=2,
        start=calendar[12],                                 # 起点晚于 B 生效日
        universe="pit",
    )

    report = FactorAnalyzer(db=db).run(spec)

    assert report.ok, report.error
    # B 生效日早于窗口起点，但它是窗口起点「当时适用」的名单 ⇒ 成员必须纳入
    assert report.universe == ["000003", "000004"]
    assert report.ic_table["n_days"].tolist() == [6]        # calendar[12..17]


# --------------------------------------------------------------------------- #
# 6~7. 降级路径与参数校验
# --------------------------------------------------------------------------- #
def test_analysis_pit_without_snapshots_fails(tmp_path) -> None:
    """库内无成分股快照 ⇒ ok=False（不静默退回当前可见标的口径）。"""
    db = _open(tmp_path / "no_snap.db")
    calendar = _biz_days(10)
    _seed(db, "000001", _linear_spec(calendar, 10.0))

    report = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(factor_names=["ret_5"], universe="pit")
    )

    assert not report.ok
    assert "成分股快照" in report.error
    assert report.universe_mode == "pit"


def test_analysis_pit_explicit_symbols_cross_section_too_small(tmp_path) -> None:
    """显式池 + PIT：两只标的归属不同名单 ⇒ 每日横截面仅 1 只 ⇒ 无有效样本。"""
    db = _open(tmp_path / "pit_small.db")
    _pit_fixture(db)

    report = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(
            symbols=["000001", "000003"],
            factor_names=["ret_5"],
            horizons=[1],
            min_stocks=2,
            universe="pit",
        )
    )

    assert not report.ok
    assert "均无有效样本" in report.error
    assert report.universe == ["000001", "000003"]


def test_analysis_universe_invalid_value_raises(tmp_path) -> None:
    """非法 universe 取值 ⇒ ValueError（调用方输入错误）。"""
    db = _open(tmp_path / "bad_universe.db")
    with pytest.raises(ValueError):
        FactorAnalyzer(db=db).run(FactorAnalysisSpec(universe="live"))


def test_analysis_default_universe_is_current(tmp_path) -> None:
    """默认口径 = current：研究池为本地可见标的，掩码不启用（与 P7-S3 一致）。"""
    db = _open(tmp_path / "current.db")
    calendar = _biz_days(10)
    _seed(db, "000001", _linear_spec(calendar, 10.0))
    _seed(db, "000002", _linear_spec(calendar, 20.0))
    _seed_constituents(db, {calendar[0]: ["000001"]})       # 有快照也不应被使用

    report = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(factor_names=["ret_5"], horizons=[1], n_groups=2, min_stocks=2)
    )

    assert report.ok, report.error
    assert report.universe_mode == "current"
    assert report.universe == ["000001", "000002"]           # 未被快照裁剪
    assert report.ic_table["n_days"].tolist() == [3]         # calendar[5..7]


def test_membership_matrix_segments_by_snapshot(tmp_path) -> None:
    """掩码分段语义对拍：某日只认「生效日 ≤ 该日 的最近一份名单」。"""
    db = _open(tmp_path / "matrix.db")
    calendar = _pit_fixture(db)
    records = db.list_index_constituents("sh.000300")

    matrix = membership_matrix(calendar, ["000001", "000003"], records, "sh.000300")

    assert not bool(matrix.loc[calendar[4]].any())           # 首份名单生效前无成员
    assert bool(matrix.loc[calendar[5], "000001"])
    assert not bool(matrix.loc[calendar[5], "000003"])
    assert bool(matrix.loc[calendar[19], "000003"])
    assert not bool(matrix.loc[calendar[19], "000001"])      # 已被剔除
    assert matrix.to_numpy().dtype == np.bool_