"""PIT 指数成分股工具与存储 / 更新链路单测（P7-S4 / S1）。

覆盖三层：
    1. :mod:`app.data.universe` 纯逻辑（归一化、快照、PIT 取数、并集、成分掩码）；
    2. :class:`~app.data.store.db.Database` 的 ``index_constituent`` CRUD 往返；
    3. :meth:`~app.data.update_service.UpdateService.update_index_constituents`
       在**离线假数据源**下的落库与去重（不依赖网络）。

全部用例使用 ``tmp_path`` 临时库，绝不触碰真实库 ``data/app.db``。
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.data.models import IndexConstituent
from app.data.store.db import Database
from app.data.universe import (
    membership_matrix,
    normalize_symbol,
    pit_members,
    pit_snapshot,
    quarter_ends,
    snapshot_dates,
    to_date_str,
    universe_union,
)
from app.data.update_service import UpdateService

# --------------------------------------------------------------------------- #
# universe.py：基础归一化
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sh.600000", "600000"),
        ("sz.000001", "000001"),
        ("600000.SH", "600000"),
        ("600000", "600000"),
        (600000, "600000"),
        ("", ""),
        (None, ""),
        ("abc", ""),
    ],
)
def test_normalize_symbol(raw, expected) -> None:
    """带市场前缀 / 裸码 / 非法输入统一为裸 6 位码。"""
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-09-30", "2026-09-30"),
        ("2026/09/30", "2026-09-30"),
        ("20260930", "2026-09-30"),
        (datetime(2026, 9, 30, 15, 0), "2026-09-30"),
        (None, ""),
        ("", ""),
        (pd.NaT, ""),
        (float("nan"), ""),
    ],
)
def test_to_date_str(raw, expected) -> None:
    """多种日期表示统一为 ``YYYY-MM-DD``；空 / NaT / NaN 归一为空串。"""
    assert to_date_str(raw) == expected


def test_quarter_ends_covers_range() -> None:
    """季末采样应覆盖区间内的每个季末（含首尾），区间非法时为空。"""
    assert quarter_ends("2015-03-31", "2016-06-30") == [
        "2015-03-31",
        "2015-06-30",
        "2015-09-30",
        "2015-12-31",
        "2016-03-31",
        "2016-06-30",
    ]
    assert quarter_ends("2020-03-31", "2020-03-31") == ["2020-03-31"]
    assert quarter_ends("2020-06-30", "2020-03-31") == []


# --------------------------------------------------------------------------- #
# universe.py：快照 / PIT
# --------------------------------------------------------------------------- #
def _records() -> list[tuple[str, str, str, str]]:
    """两份快照：2020-01-01 → {A,B}；2020-07-01 → {B,C}。"""
    return [
        ("sh.000300", "sh.600000", "浦发银行", "2020-01-01"),
        ("sh.000300", "sh.600001", "邯郸钢铁", "2020-01-01"),
        ("sh.000300", "sh.600000", "浦发银行", "2020-07-01"),
        ("sh.000300", "sh.600002", "齐鲁石化", "2020-07-01"),
    ]


def test_snapshot_dates_sorted_unique() -> None:
    """快照生效日应升序去重。"""
    assert snapshot_dates(_records()) == ["2020-01-01", "2020-07-01"]


def test_pit_snapshot_strict_before_first() -> None:
    """早于首份名单的交易日不得借用未来名单（严格 PIT，返回空）。"""
    assert pit_snapshot(_records(), "2019-12-31") == ("", [])
    assert pit_members(_records(), "2019-12-31") == []


def test_pit_snapshot_picks_latest_not_after() -> None:
    """取生效日 ≤ as_of 的最近一份快照。"""
    assert pit_snapshot(_records(), "2020-01-01") == ("2020-01-01", ["600000", "600001"])
    assert pit_snapshot(_records(), "2020-06-30") == ("2020-01-01", ["600000", "600001"])
    assert pit_snapshot(_records(), "2020-07-01") == ("2020-07-01", ["600000", "600002"])
    assert pit_snapshot(_records(), "2026-10-11") == ("2020-07-01", ["600000", "600002"])


def test_pit_members_accepts_dataclass_and_dataframe() -> None:
    """三种入参写法（元组 / dataclass / DataFrame）等价。"""
    dataclass_rows = [
        IndexConstituent("sh.000300", "600000", "浦发银行", datetime(2020, 1, 1)),
        IndexConstituent("sh.000300", "600001", "邯郸钢铁", datetime(2020, 1, 1)),
        IndexConstituent("sh.000300", "600002", "齐鲁石化", datetime(2020, 7, 1)),
    ]
    frame = pd.DataFrame(
        [
            {"index_code": "sh.000300", "symbol": "sh.600000", "effective_date": "2020-01-01"},
            {"index_code": "sh.000300", "symbol": "sh.600001", "effective_date": "2020-01-01"},
            {"index_code": "sh.000300", "symbol": "sh.600002", "effective_date": "2020-07-01"},
        ]
    )
    assert pit_members(dataclass_rows, "2020-03-01") == ["600000", "600001"]
    assert pit_members(frame, "2020-03-01") == ["600000", "600001"]
    assert pit_members(_records(), "2020-03-01") == ["600000", "600001"]


def test_pit_snapshot_filters_by_index_code() -> None:
    """指定 index_code 时不混入其他指数的名单。"""
    rows = _records() + [("sh.000016", "sh.601398", "工商银行", "2020-01-01")]
    assert pit_members(rows, "2020-03-01", index_code="sh.000016") == ["601398"]
    assert pit_members(rows, "2020-03-01", index_code="sh.000300") == ["600000", "600001"]


def test_universe_union_includes_dropped_stocks() -> None:
    """并集必须包含「只在早期名单出现过」的标的（幸存者偏差的关键样本）。"""
    assert universe_union(_records()) == ["600000", "600001", "600002"]
    # 只统计 2020-07-01 之后的快照 → 早期标的被排除。
    assert universe_union(_records(), start="2020-06-01") == ["600000", "600002"]


def test_membership_matrix_stepwise() -> None:
    """成分掩码按快照分段赋值，早于首份名单的行保持 False。"""
    dates = ["2019-12-31", "2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31"]
    symbols = ["600000", "600001", "600002"]
    matrix = membership_matrix(dates, symbols, _records())
    assert list(matrix.index) == dates
    assert list(matrix.columns) == symbols
    assert matrix.loc["2019-12-31"].tolist() == [False, False, False]
    assert matrix.loc["2020-01-01"].tolist() == [True, True, False]
    assert matrix.loc["2020-06-30"].tolist() == [True, True, False]
    assert matrix.loc["2020-07-01"].tolist() == [True, False, True]
    assert matrix.loc["2020-12-31"].tolist() == [True, False, True]


def test_membership_matrix_degrades_on_empty_input() -> None:
    """无名单记录 / 无标的时返回全 False 矩阵，不抛异常。"""
    matrix = membership_matrix(["2020-01-01"], ["600000"], [])
    assert matrix.shape == (1, 1)
    assert not bool(matrix.to_numpy().any())
    assert membership_matrix(["2020-01-01"], [], _records()).empty


# --------------------------------------------------------------------------- #
# 存储层 CRUD
# --------------------------------------------------------------------------- #
@pytest.fixture()
def db(tmp_path) -> Database:
    """提供临时文件库（已建表），用后关闭。"""
    database = Database(tmp_path / "test.db")
    database.init_schema()
    yield database
    database.close()


def test_db_index_constituent_roundtrip(db: Database) -> None:
    """写入 → 读出：生效日按日期粒度保存，顺序为 (index_code, effective_date, symbol)。"""
    rows = [
        IndexConstituent("sh.000300", "600000", "浦发银行", datetime(2020, 1, 1)),
        IndexConstituent("sh.000300", "600001", "邯郸钢铁", datetime(2020, 1, 1)),
        IndexConstituent("sh.000300", "600000", "浦发银行", datetime(2020, 7, 1)),
    ]
    assert db.upsert_index_constituents(rows) == 3
    loaded = db.list_index_constituents("sh.000300")
    assert [(r.symbol, r.effective_date.strftime("%Y-%m-%d")) for r in loaded] == [
        ("600000", "2020-01-01"),
        ("600001", "2020-01-01"),
        ("600000", "2020-07-01"),
    ]
    assert loaded[0].name == "浦发银行"
    assert db.count_index_constituents() == 3
    assert db.count_index_constituents("sh.000300") == 3
    assert db.count_index_constituents("sh.000905") == 0
    assert db.list_index_codes() == ["sh.000300"]


def test_db_index_constituent_upsert_is_idempotent(db: Database) -> None:
    """同 (index_code, symbol, effective_date) 重复写入不新增行。"""
    rows = [IndexConstituent("sh.000300", "600000", "浦发银行", datetime(2020, 1, 1))]
    db.upsert_index_constituents(rows)
    db.upsert_index_constituents(rows)
    assert db.count_index_constituents() == 1


def test_db_index_constituent_normalizes_prefixed_symbol_and_dataframe(db: Database) -> None:
    """DataFrame 入参自动归一化市场前缀；非法行 / 未支持类型被正确处理。"""
    frame = pd.DataFrame(
        [
            {"index_code": "sh.000300", "symbol": "sh.600000", "name": "浦发银行",
             "effective_date": "2020-01-01"},
            {"index_code": "sh.000300", "symbol": "", "name": "非法", "effective_date": "2020-01-01"},
            {"index_code": "sh.000300", "symbol": "600001", "name": "缺日期"},
        ]
    )
    assert db.upsert_index_constituents(frame) == 1
    assert db.list_index_constituents("sh.000300")[0].symbol == "600000"
    with pytest.raises(TypeError):
        db.upsert_index_constituents("not-a-frame")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 更新服务（离线假数据源）
# --------------------------------------------------------------------------- #
class _FakeSource:
    """离线假数据源：按查询日返回「当天或之前最近一份」名单快照。"""

    name = "fake"

    def __init__(self, snapshots: dict[str, list[str]], empty: bool = False) -> None:
        self._snapshots = snapshots
        self._empty = empty
        self.calls: list[str] = []

    def get_index_constituents(self, index_code: str = "sh.000300", date_=None) -> pd.DataFrame:
        self.calls.append(str(date_))
        if self._empty:
            return pd.DataFrame()
        effective = ""
        for day in sorted(self._snapshots):
            if day <= str(date_):
                effective = day
        if not effective:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "index_code": index_code,
                "symbol": self._snapshots[effective],
                "name": "",
                "effective_date": effective,
            }
        )


def test_update_service_index_constituents_persists_and_dedupes(db: Database) -> None:
    """同一份名单被多个查询日命中时只落库一次（按 effective_date 去重）。"""
    source = _FakeSource({"2020-01-01": ["600000", "600001"], "2020-07-01": ["600000", "600002"]})
    service = UpdateService(db=db, primary_name="fake", fallback_name="", sources={"fake": source})
    report = service.update_index_constituents(
        dates=["2020-01-01", "2020-03-31", "2020-07-01", "2020-09-30"]
    )
    assert report.ok
    assert report.symbols_count == 4  # 查询了 4 次
    assert report.bars_written == 4  # 两份名单各 2 行，后两次查询被去重
    assert source.calls == ["2020-01-01", "2020-03-31", "2020-07-01", "2020-09-30"]
    loaded = db.list_index_constituents("sh.000300")
    assert [(r.symbol, r.effective_date.strftime("%Y-%m-%d")) for r in loaded] == [
        ("600000", "2020-01-01"),
        ("600001", "2020-01-01"),
        ("600000", "2020-07-01"),
        ("600002", "2020-07-01"),
    ]


def test_update_service_index_constituents_records_failure_offline(db: Database) -> None:
    """数据源无有效名单时如实记失败，且不抛异常。"""
    service = UpdateService(
        db=db, primary_name="fake", fallback_name="", sources={"fake": _FakeSource({}, empty=True)}
    )
    report = service.update_index_constituents(dates=["2020-01-01"])
    assert report.ok is False
    assert report.failed and "未取到成分股名单" in report.failed[0][1]
    assert db.count_index_constituents() == 0


def test_update_service_index_constituents_auto_quarter_dates(db: Database) -> None:
    """未显式给 dates 时按季末自动生成查询日（起止落在季末上）。"""
    source = _FakeSource({"2015-03-30": ["600000"]})
    service = UpdateService(db=db, primary_name="fake", fallback_name="", sources={"fake": source})
    report = service.update_index_constituents(start="2015-03-31", end="2015-06-30")
    assert report.symbols_count == 2
    assert source.calls == ["2015-03-31", "2015-06-30"]
