"""P3-S4 多条件组合选股引擎测试。

覆盖（对应《规格》逐条）：
    1. 技术面单条件（裸字段 ``close``）；
    2. 技术面指标表达式条件（``close.rolling(3).mean()``）；
    3. 基本面条件（``pe`` 数值比较 + 缺值不命中）；
    4. AND / OR 逻辑叠加；
    5. 基本面抓取失败容错（该股 failed、跳过、不中断整体）；
    6. 无日线数据标的（failed、跳过）；
    7. 空候选池（显式空 + 自选股空）；
    8. 空条件（AND 全命中 / OR 全不命中，真空真值语义）；
    9. logic / 条件结构非法边界（抛 ``ValueError``）；
    10. 自选股候选池 symbol 去重；周线派生路径；单例与序列化回归。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.datasource.base import DataSourceError
from app.data.store.db import Database
from app.engine.screener import (
    FUNDAMENTAL_FIELDS,
    Condition,
    ScreenResult,
    ScreenerEngine,
    get_screener,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_daily(db: Database, symbol: str, closes, period: str = "d") -> None:
    """写入 n 根日线（``adjust=""``，与读库默认一致）。"""
    close = np.asarray(closes, dtype=float)
    n = len(close)
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 100.0),
            "amount": close * 1000.0,
        }
    )
    frame["symbol"] = symbol
    frame["period"] = period
    frame["adjust"] = ""
    db.upsert_bars(frame)


def _fundamental(**kw) -> dict:
    """构造结构稳定的基本面 dict（缺省全 None / 空串）。"""
    base = {
        "symbol": "",
        "name": "",
        "industry": "",
        "list_date": "",
        "total_market_cap": None,
        "float_market_cap": None,
        "pe": None,
        "pb": None,
        "roe": None,
        "revenue_yoy": None,
        "report_period": "",
        "summary": "",
    }
    base.update(kw)
    return base


class FakeSource:
    """注入用基本面 fake 源（按 symbol 返回预设 dict，可指定抛错 symbol）。"""

    name = "fake"

    def __init__(self, fundamentals=None, fail_symbols=()) -> None:
        self._fundamentals = fundamentals or {}
        self._fail = set(fail_symbols)

    def get_fundamental(self, symbol: str) -> dict:
        if symbol in self._fail:
            raise DataSourceError(self.name, RuntimeError("boom"), "基本面抓取超时")
        return dict(self._fundamentals.get(symbol, {}))


# --------------------------------------------------------------------------- #
# 1. 技术面单条件
# --------------------------------------------------------------------------- #
def test_technical_single_condition(tmp_path) -> None:
    db = _open(tmp_path / "t1.db")
    _seed_daily(db, "000001", list(range(1, 21)))   # 最新 close=20
    _seed_daily(db, "000002", list(range(1, 11)))   # 最新 close=10
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run([Condition("close", ">", 15)], symbols=["000001", "000002"])

    assert result.total == 2
    assert result.hits == ["000001"]
    assert result.failed == []
    assert result.hit_count == 1
    assert result.failed_count == 0


# --------------------------------------------------------------------------- #
# 2. 技术面指标表达式条件
# --------------------------------------------------------------------------- #
def test_technical_indicator_expression(tmp_path) -> None:
    db = _open(tmp_path / "t2.db")
    _seed_daily(db, "600000", [1, 2, 3, 4, 5, 6])   # 近3均值=(4+5+6)/3=5
    _seed_daily(db, "600001", [1, 2, 3])            # 近3均值=2
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run(
        [Condition("close.rolling(3).mean()", ">", 4.5)],
        symbols=["600000", "600001"],
    )

    assert result.hits == ["600000"]
    assert result.failed == []


def test_illegal_technical_expression_is_false_not_failed(tmp_path) -> None:
    """非法技术面表达式 → 条件 False（不抛异常、不中断、不记 failed）。"""
    db = _open(tmp_path / "t2b.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run([Condition("no_such_name", ">", 1)], symbols=["000001"])

    assert result.hits == []
    assert result.failed == []
    assert result.total == 1


# --------------------------------------------------------------------------- #
# 3. 基本面条件
# --------------------------------------------------------------------------- #
def test_fundamental_condition(tmp_path) -> None:
    db = _open(tmp_path / "t3.db")
    _seed_daily(db, "000001", [10.0] * 5)
    _seed_daily(db, "000002", [10.0] * 5)
    _seed_daily(db, "000003", [10.0] * 5)   # 缺 pe
    source = FakeSource(
        fundamentals={
            "000001": _fundamental(pe=20.0),
            "000002": _fundamental(pe=40.0),
            "000003": _fundamental(pe=None),
        }
    )
    engine = ScreenerEngine(db=db, source=source)

    result = engine.run([Condition("pe", "<", 30)], symbols=["000001", "000002", "000003"])

    assert result.hits == ["000001"]
    assert result.failed == []
    assert result.total == 3


# --------------------------------------------------------------------------- #
# 4. AND / OR 逻辑叠加
# --------------------------------------------------------------------------- #
def test_and_or_logic(tmp_path) -> None:
    db = _open(tmp_path / "t4.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5, 6])   # close=6
    _seed_daily(db, "000002", [1, 2, 3, 4, 5, 6])   # close=6
    source = FakeSource(
        fundamentals={"000001": _fundamental(pe=10.0), "000002": _fundamental(pe=50.0)}
    )
    engine = ScreenerEngine(db=db, source=source)
    conds = [Condition("close", ">", 4), Condition("pe", "<", 30)]

    and_result = engine.run(conds, symbols=["000001", "000002"], logic="AND")
    or_result = engine.run(conds, symbols=["000001", "000002"], logic="OR")

    assert and_result.hits == ["000001"]
    assert or_result.hits == ["000001", "000002"]


# --------------------------------------------------------------------------- #
# 5. 基本面抓取失败容错
# --------------------------------------------------------------------------- #
def test_fundamental_failure_tolerance(tmp_path) -> None:
    db = _open(tmp_path / "t5.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])
    _seed_daily(db, "000002", [1, 2, 3, 4, 5])
    _seed_daily(db, "000003", [1, 2, 3, 4, 5])
    source = FakeSource(
        fundamentals={"000001": _fundamental(pe=10.0), "000003": _fundamental(pe=20.0)},
        fail_symbols={"000002"},
    )
    engine = ScreenerEngine(db=db, source=source)

    result = engine.run(
        [Condition("pe", "<", 30)], symbols=["000001", "000002", "000003"]
    )

    assert result.hits == ["000001", "000003"]
    assert result.total == 3
    assert [f["symbol"] for f in result.failed] == ["000002"]
    assert "基本面" in result.failed[0]["reason"]


# --------------------------------------------------------------------------- #
# 6. 无日线数据
# --------------------------------------------------------------------------- #
def test_no_daily_bars(tmp_path) -> None:
    db = _open(tmp_path / "t6.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])
    # "000002" 不写任何 K 线
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run([Condition("close", ">", 1)], symbols=["000001", "000002"])

    assert result.hits == ["000001"]
    assert [f["symbol"] for f in result.failed] == ["000002"]
    assert "无K线" in result.failed[0]["reason"]
    assert result.total == 2


# --------------------------------------------------------------------------- #
# 7. 空候选池
# --------------------------------------------------------------------------- #
def test_empty_pool(tmp_path) -> None:
    db = _open(tmp_path / "t7.db")
    engine = ScreenerEngine(db=db, source=FakeSource())

    explicit = engine.run([Condition("close", ">", 1)], symbols=[])
    watchlist = engine.run([Condition("close", ">", 1)])

    assert explicit.hits == [] and explicit.failed == [] and explicit.total == 0
    assert watchlist.hits == [] and watchlist.failed == [] and watchlist.total == 0


# --------------------------------------------------------------------------- #
# 8. 空条件（真空真值）
# --------------------------------------------------------------------------- #
def test_empty_conditions(tmp_path) -> None:
    db = _open(tmp_path / "t8.db")
    _seed_daily(db, "000001", [1, 2, 3])
    _seed_daily(db, "000002", [4, 5, 6])
    engine = ScreenerEngine(db=db, source=FakeSource())

    and_result = engine.run([], symbols=["000001", "000002"], logic="AND")
    or_result = engine.run([], symbols=["000001", "000002"], logic="OR")

    assert and_result.hits == ["000001", "000002"]
    assert or_result.hits == []
    assert or_result.total == 2


# --------------------------------------------------------------------------- #
# 9. 边界（非法 logic / 非法条件结构）
# --------------------------------------------------------------------------- #
def test_logic_and_condition_boundaries(tmp_path) -> None:
    db = _open(tmp_path / "t9.db")
    engine = ScreenerEngine(db=db, source=FakeSource())

    with pytest.raises(ValueError):
        engine.run([Condition("close", ">", 1)], symbols=[], logic="XOR")
    with pytest.raises(ValueError):
        engine.run([{"field": "", "op": ">", "value": 1}], symbols=[])
    with pytest.raises(ValueError):
        Condition("close", "~", 1)
    with pytest.raises(ValueError):
        Condition.from_dict({"field": "pe"})  # 缺 op/value


# --------------------------------------------------------------------------- #
# 10. 候选池去重 / 周线派生 / 序列化 / 单例
# --------------------------------------------------------------------------- #
def test_watchlist_pool_symbol_dedup(tmp_path) -> None:
    db = _open(tmp_path / "t10.db")
    g1 = db.create_group("核心", sort_order=0)
    g2 = db.create_group("观察", sort_order=1)
    db.add_watch_item(g1, "000001")
    db.add_watch_item(g1, "000002")
    db.add_watch_item(g2, "000001")   # 跨组重复 → 应去重
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])
    _seed_daily(db, "000002", [1, 2, 3, 4, 5])
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run([Condition("close", ">", 3)])

    assert result.total == 2
    assert sorted(result.hits) == ["000001", "000002"]


def test_week_period_derivation(tmp_path) -> None:
    db = _open(tmp_path / "t11.db")
    _seed_daily(db, "000001", list(range(1, 16)))   # 最新日 close=15
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run([Condition("close", ">", 14)], symbols=["000001"], period="w")

    assert result.hits == ["000001"]
    assert result.failed == []


def test_condition_roundtrip() -> None:
    cond = Condition("pe", "<", 30)
    data = cond.to_dict()
    assert data == {"field": "pe", "op": "<", "value": 30.0}

    restored = Condition.from_dict({"field": "pe", "op": "=", "value": "30"})
    assert restored.field == "pe"
    assert restored.op == "=="   # "=" 归一化
    assert restored.value == 30.0


def test_get_screener_singleton() -> None:
    a = get_screener()
    b = get_screener()
    assert isinstance(a, ScreenerEngine)
    assert a is b


def test_screen_result_and_fundamental_fields() -> None:
    result = ScreenResult(hits=["000001"], failed=[{"symbol": "000002", "reason": "x"}], total=2)
    assert result.hit_count == 1
    assert result.failed_count == 1
    assert "pe" in FUNDAMENTAL_FIELDS
    assert "close" not in FUNDAMENTAL_FIELDS


# --------------------------------------------------------------------------- #
# 11. P4-S3：ScreenResult.metrics 携带指标实际值
# --------------------------------------------------------------------------- #
def test_run_metrics_fundamental_and_technical(tmp_path) -> None:
    """命中股的 metrics 与 hits 一一对应，基本面 + 技术面条件实际值均正确。"""
    db = _open(tmp_path / "m1.db")
    _seed_daily(db, "000001", list(range(1, 21)))   # 最新 close=20.0
    _seed_daily(db, "000002", list(range(1, 11)))   # 最新 close=10.0
    source = FakeSource(
        fundamentals={
            "000001": _fundamental(pe=20.0),
            "000002": _fundamental(pe=40.0),
        }
    )
    engine = ScreenerEngine(db=db, source=source)
    conds = [Condition("close", ">", 15), Condition("pe", "<", 30)]

    result = engine.run(conds, symbols=["000001", "000002"], logic="AND")

    assert result.hits == ["000001"]
    assert result.metrics == [
        {
            "symbol": "000001",
            "conditions": [
                {"field": "close", "value": 20.0},
                {"field": "pe", "value": 20.0},
            ],
        },
    ]


def test_run_metrics_or_logic_records_all_condition_values(tmp_path) -> None:
    """OR 命中股即便某条件不满足，也记录该条件实际值（不因未命中丢值）。"""
    db = _open(tmp_path / "m2.db")
    _seed_daily(db, "000002", [1, 2, 3, 4, 5, 6])   # 最新 close=6.0
    source = FakeSource(fundamentals={"000002": _fundamental(pe=50.0)})
    engine = ScreenerEngine(db=db, source=source)
    conds = [Condition("close", ">", 4), Condition("pe", "<", 30)]

    result = engine.run(conds, symbols=["000002"], logic="OR")

    assert result.hits == ["000002"]
    assert result.metrics == [
        {
            "symbol": "000002",
            "conditions": [
                {"field": "close", "value": 6.0},
                {"field": "pe", "value": 50.0},
            ],
        },
    ]


def test_run_metrics_bad_technical_expression_value_none(tmp_path) -> None:
    """技术面条件求值异常 → 该条件 metrics 的 value 记 None（不中断、不崩）。"""
    db = _open(tmp_path / "m3.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])   # 最新 close=5.0
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run(
        [Condition("no_such_name", ">", 1), Condition("close", ">", 4)],
        symbols=["000001"],
        logic="OR",
    )

    assert result.hits == ["000001"]
    assert result.metrics[0]["conditions"][0] == {"field": "no_such_name", "value": None}
    assert result.metrics[0]["conditions"][1] == {"field": "close", "value": 5.0}
