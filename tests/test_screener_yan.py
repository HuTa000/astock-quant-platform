"""P3-S4 多条件组合选股引擎 —— QA 严过关 补充对抗性测试。

本文件只做**新增覆盖**，不改任何源码。针对工程师 15 用例未覆盖的边界做独立取证：

    A. 短序列（bar < 指标窗口）→ 全 NaN → 条件不命中且**不记 failed**；
    B. 指标结果前导 NaN / 尾部有效 → 取最新有效值（非 NaN）；
    C. 基本面字段缺值（None / 非数值字符串）→ 条件不命中、不崩溃；
    D. 基本面抓取成功但返回非 dict（None / 其它类型）→ 容错行为取证；
    E. ``value`` 为表达式（非数值）→ 构造期拒绝（与「value 必须数值」规格一致）；
    F. ``op`` 边界：``==`` 精确相等、``>=`` / ``<=`` 闭区间（等于阈值命中）、``!=``；
    G. ``logic`` 大小写归一化（``"and"`` / ``"or"``）；
    H. 显式 ``symbols`` 绕过自选股候选池（优先级：显式 > 自选股）；
    I. 多股「命中 + 无K线失败 + 有K线不匹配」混合时 hits / failed 不串号；
    J. OR 语义下技术面条件异常 + 其它条件命中 → 任一命中仍成立。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.datasource.base import DataSourceError
from app.data.store.db import Database
from app.engine.screener import Condition, ScreenerEngine


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_daily(db: Database, symbol: str, closes, period: str = "d") -> None:
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
    """按 symbol 返回预设 dict；可指定抛错 symbol；可返回非 dict（如 None / str）。"""

    name = "fake"

    def __init__(self, fundamentals=None, fail_symbols=(), non_dict=None) -> None:
        self._fundamentals = fundamentals or {}
        self._fail = set(fail_symbols)
        self._non_dict = non_dict or {}

    def get_fundamental(self, symbol: str):
        if symbol in self._fail:
            raise DataSourceError(self.name, RuntimeError("boom"), "基本面抓取超时")
        if symbol in self._non_dict:
            return self._non_dict[symbol]
        return dict(self._fundamentals.get(symbol, {}))


# --------------------------------------------------------------------------- #
# A. 短序列（bar < 指标窗口）→ 全 NaN → 不命中、不记 failed
# --------------------------------------------------------------------------- #
def test_short_series_ma_all_nan_is_false_not_failed(tmp_path) -> None:
    db = _open(tmp_path / "a.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])   # 5 根 < MA20 窗口 → 全 NaN
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run([Condition("MA(close,20)", ">", 0)], symbols=["000001"])

    assert result.hits == []
    assert result.failed == []          # 不因数据不足记 failed
    assert result.total == 1


# --------------------------------------------------------------------------- #
# B. 前导 NaN + 尾部有效 → 取最新有效值（非 NaN）
# --------------------------------------------------------------------------- #
def test_ma_leading_nan_returns_latest_valid(tmp_path) -> None:
    db = _open(tmp_path / "b.db")
    # 25 根：MA20 最新 = mean(6..25) = 15.5
    _seed_daily(db, "000001", list(range(1, 26)))
    engine = ScreenerEngine(db=db, source=FakeSource())

    hit = engine.run([Condition("MA(close,20)", ">", 15)], symbols=["000001"])
    miss = engine.run([Condition("MA(close,20)", ">", 15.5)], symbols=["000001"])

    assert hit.hits == ["000001"]
    assert miss.hits == []             # 15.5 不 > 15.5


# --------------------------------------------------------------------------- #
# C. 基本面字段缺值（None / 非数值字符串）→ 不命中、不崩溃
# --------------------------------------------------------------------------- #
def test_fundamental_missing_and_non_numeric_value(tmp_path) -> None:
    db = _open(tmp_path / "c.db")
    _seed_daily(db, "000001", [10.0] * 5)   # pe=None
    _seed_daily(db, "000002", [10.0] * 5)   # pe="N/A" 非数值
    _seed_daily(db, "000003", [10.0] * 5)   # pe=8.0 命中
    source = FakeSource(
        fundamentals={
            "000001": _fundamental(pe=None),
            "000002": _fundamental(pe="N/A"),
            "000003": _fundamental(pe=8.0),
        }
    )
    engine = ScreenerEngine(db=db, source=source)

    result = engine.run([Condition("pe", "<", 20)], symbols=["000001", "000002", "000003"])

    assert result.hits == ["000003"]
    assert result.failed == []
    assert result.total == 3


# --------------------------------------------------------------------------- #
# D. 基本面抓取成功但返回非 dict（None / str）→ 容错取证
# --------------------------------------------------------------------------- #
def test_fundamental_returns_none_is_tolerated(tmp_path) -> None:
    db = _open(tmp_path / "d1.db")
    _seed_daily(db, "000001", [10.0] * 5)
    engine = ScreenerEngine(db=db, source=FakeSource(non_dict={"000001": None}))

    result = engine.run([Condition("pe", "<", 20)], symbols=["000001"])

    # None → 不崩溃、不命中、不记 failed（按「字段缺值」处理）
    assert result.hits == []
    assert result.failed == []
    assert result.total == 1


def test_fundamental_returns_non_dict_str_is_tolerated(tmp_path) -> None:
    """数据源返回非 dict（违反协议）时引擎不向 UI 抛异常（run 层兜底为「未命中」）。"""
    db = _open(tmp_path / "d2.db")
    _seed_daily(db, "000001", [10.0] * 5)
    engine = ScreenerEngine(db=db, source=FakeSource(non_dict={"000001": "garbage"}))

    result = engine.run([Condition("pe", "<", 20)], symbols=["000001"])

    # 关键取证：不抛异常（run 兜底）；但未记入 failed（hits+failed < total 属预期）
    assert result.hits == []
    assert result.total == 1
    # 协议违规返回非 dict 时，_fundamental_value 抛 AttributeError，被 run 兜底为未命中；
    # 该股既不命中也不入 failed，属「扫描未匹配」语义，符合 ScreenResult.total 的 <= 约定。


# --------------------------------------------------------------------------- #
# E. value 为表达式（非数值）→ 构造期拒绝（与「value 必须数值」规格一致）
# --------------------------------------------------------------------------- #
def test_value_as_expression_rejected() -> None:
    with pytest.raises(ValueError):
        Condition("close", ">", "MA(close,20)")
    with pytest.raises(ValueError):
        Condition.from_dict({"field": "close", "op": ">", "value": "MA(close,20)"})


# --------------------------------------------------------------------------- #
# F. op 边界：== / >= / <= / !=（等于阈值命中闭区间）
# --------------------------------------------------------------------------- #
def test_op_boundaries(tmp_path) -> None:
    db = _open(tmp_path / "f.db")
    _seed_daily(db, "000001", [5.0, 6.0, 7.0])   # 最新 close = 7.0
    engine = ScreenerEngine(db=db, source=FakeSource())

    eq = engine.run([Condition("close", "==", 7.0)], symbols=["000001"])
    ge = engine.run([Condition("close", ">=", 7.0)], symbols=["000001"])
    le = engine.run([Condition("close", "<=", 7.0)], symbols=["000001"])
    ne = engine.run([Condition("close", "!=", 7.0)], symbols=["000001"])
    lt = engine.run([Condition("close", "<", 7.0)], symbols=["000001"])

    assert eq.hits == ["000001"]     # 精确相等命中
    assert ge.hits == ["000001"]     # 等于阈值命中（闭区间）
    assert le.hits == ["000001"]     # 等于阈值命中（闭区间）
    assert ne.hits == []
    assert lt.hits == []


# --------------------------------------------------------------------------- #
# G. logic 大小写归一化
# --------------------------------------------------------------------------- #
def test_logic_case_insensitive(tmp_path) -> None:
    db = _open(tmp_path / "g.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5, 6])   # close=6
    _seed_daily(db, "000002", [1, 2, 3, 4, 5, 6])   # close=6
    source = FakeSource(
        fundamentals={"000001": _fundamental(pe=10.0), "000002": _fundamental(pe=50.0)}
    )
    engine = ScreenerEngine(db=db, source=source)
    conds = [Condition("close", ">", 4), Condition("pe", "<", 30)]

    and_result = engine.run(conds, symbols=["000001", "000002"], logic="and")
    or_result = engine.run(conds, symbols=["000001", "000002"], logic="or")

    assert and_result.hits == ["000001"]
    assert or_result.hits == ["000001", "000002"]


# --------------------------------------------------------------------------- #
# H. 显式 symbols 绕过自选股候选池（显式 > 自选股）
# --------------------------------------------------------------------------- #
def test_explicit_symbols_bypass_watchlist(tmp_path) -> None:
    db = _open(tmp_path / "h.db")
    g = db.create_group("核心", sort_order=0)
    db.add_watch_item(g, "000001")
    db.add_watch_item(g, "000002")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])
    _seed_daily(db, "000002", [1, 2, 3, 4, 5])
    _seed_daily(db, "000003", [1, 2, 3, 4, 5])
    engine = ScreenerEngine(db=db, source=FakeSource())

    # 显式只传 000003 → 只扫它，自选股里的 000001/000002 不参与
    result = engine.run([Condition("close", ">", 0)], symbols=["000003"])

    assert result.hits == ["000003"]
    assert result.total == 1
    assert result.failed == []


# --------------------------------------------------------------------------- #
# I. 多股「命中 + 无K线失败 + 有K线不匹配」混合 → hits / failed 不串号
# --------------------------------------------------------------------------- #
def test_partial_hit_fail_no_match(tmp_path) -> None:
    db = _open(tmp_path / "i.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])   # close=5 → 命中
    _seed_daily(db, "000002", [1, 1, 1, 1, 1])   # close=1 → 不匹配
    # "000003" 无 K 线 → failed
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run([Condition("close", ">", 3)], symbols=["000001", "000002", "000003"])

    assert result.hits == ["000001"]
    assert [f["symbol"] for f in result.failed] == ["000003"]
    assert result.total == 3
    assert result.hit_count == 1
    assert result.failed_count == 1


# --------------------------------------------------------------------------- #
# J. OR 语义下：技术面条件异常 + 其它条件命中 → 任一命中仍成立
# --------------------------------------------------------------------------- #
def test_or_logic_with_one_bad_condition_still_hits(tmp_path) -> None:
    db = _open(tmp_path / "j.db")
    _seed_daily(db, "000001", [1, 2, 3, 4, 5])   # close=5
    engine = ScreenerEngine(db=db, source=FakeSource())

    result = engine.run(
        [Condition("no_such_name", ">", 1), Condition("close", ">", 4)],
        symbols=["000001"],
        logic="OR",
    )

    assert result.hits == ["000001"]
    assert result.failed == []
