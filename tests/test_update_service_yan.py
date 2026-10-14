"""R13 修复（update_daily 同时落未复权 + 前复权两套）—— QA 严过关 独立补测。

本文件只做**新增覆盖**，不改任何源码。针对工程师 ``test_update_service.py`` +
``test_friendly_error.py`` 已覆盖之外的真实缺陷场景，做独立取证：

    A. 单套已最新跳过：未复权 ``""`` 库内已含今日 → 跳过该套不请求；前复权 ``qfq``
       落后 → 仍从自身 ``max(dt)+1`` 起始（``eff_start > end_date`` 跳过语义 + 增量独立）。
    B. 反向失败：``""`` 失败、``qfq`` 成功 → failed 记 1 次，reason 点名「不复权」，
       qfq 仍落库（补足既有用例只测「qfq 失败」的盲区）。
    C. hfq 不落 + ``adjust`` 参数被忽略：传 ``adjust="hfq"`` 仍只落 ``""``/``"qfq"``，
       绝不请求/落 hfq。
    D. 双套互不覆盖（强取证）：同一 symbol 同一 dt，``""`` 与 ``"qfq"`` 落**不同**收盘价，
       读回互不串值（主键含 adjust 的落地正确性）。
    E. ``EVT_COMPLETED`` 的 ``bars_written`` 精确等于两套之和，``symbols_count``/``failed_count`` 正确。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.core.event_bus import get_bus
from app.data.datasource.base import BAR_COLUMNS
from app.data.store.db import Database
from app.data.update_service import (
    DEFAULT_START,
    EVT_COMPLETED,
    EVT_FAILED,
    UpdateService,
)


# --------------------------------------------------------------------------- #
# 辅助（自包含，避免与 test_update_service 耦合）
# --------------------------------------------------------------------------- #
def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _make_bars(start: str, end: str, close_value: float = 10.2) -> pd.DataFrame:
    """生成 [start, end] 内最多 3 根确定性 Bar，close 可指定以区分口径。"""
    base = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    rows = []
    for i in range(3):
        dt = base + pd.Timedelta(days=i)
        if dt > stop:
            break
        rows.append(
            {
                "datetime": dt,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": close_value,
                "volume": 100.0,
                "amount": 102000.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.DataFrame(rows)


def _seed(db: Database, symbol: str, start: str, end: str, adjust: str,
          close_value: float = 10.2) -> None:
    frame = _make_bars(start, end, close_value=close_value)
    frame["symbol"] = symbol
    frame["period"] = "d"
    frame["adjust"] = adjust
    db.upsert_bars(frame)


class _CallSource:
    """成功、按口径返回不同收盘价的假数据源（记录调用）。"""

    def __init__(self, raw_close: float = 9.99, qfq_close: float = 11.11) -> None:
        self.raw_close = raw_close
        self.qfq_close = qfq_close
        self.kline_calls: list[dict] = []

    def get_kline(self, symbol, period, start, end, adjust=""):
        self.kline_calls.append(
            dict(symbol=symbol, period=period, start=start, end=end, adjust=adjust)
        )
        close_value = self.qfq_close if adjust == "qfq" else self.raw_close
        return _make_bars(start, end, close_value=close_value)


class _SelectiveFailSource:
    """按复权口径选择性失败（一套失败不影响另一套）。"""

    def __init__(self, fail_adjusts=()) -> None:
        self.fail_adjusts = set(fail_adjusts)
        self.kline_calls: list[dict] = []

    def get_kline(self, symbol, period, start, end, adjust=""):
        self.kline_calls.append(
            dict(symbol=symbol, period=period, start=start, end=end, adjust=adjust)
        )
        if adjust in self.fail_adjusts:
            raise RuntimeError(f"模拟失败 adjust={adjust!r}")
        return _make_bars(start, end, close_value=10.2)


def _service(db, sources, primary="primary", fallback="fallback"):
    return UpdateService(
        db=db, primary_name=primary, fallback_name=fallback, sources=sources
    )


@pytest.fixture()
def bus():
    b = get_bus()
    b.clear()
    yield b
    b.clear()


# --------------------------------------------------------------------------- #
# A. 单套已最新跳过 + 增量独立
# --------------------------------------------------------------------------- #
def test_one_adjust_up_to_date_skipped_other_fetches_from_own_max() -> None:
    """``""`` 已含今日 → 跳过不请求；``qfq`` 落后 → 仍从自身 max(dt)+1 起始。"""
    db = _mem_db()
    today = date.today().strftime("%Y-%m-%d")
    _seed(db, "000001", today, today, adjust="")            # 未复权已最新
    _seed(db, "000001", "2015-01-01", "2015-01-02", adjust="qfq")  # 前复权落后

    primary = _CallSource()
    svc = _service(db, {"primary": primary, "fallback": _CallSource()})
    report = svc.update_daily(symbols=["000001"])

    starts = {c["adjust"]: c["start"] for c in primary.kline_calls}
    assert "" not in starts                    # 未复权已最新 → 根本没请求
    assert starts.get("qfq") == "2015-01-03"   # qfq 从自身 max+1 起始（非默认、非借用 ""）
    assert report.ok
    assert report.bars_written == 3            # 仅补落后的 qfq 一套（3 根）
    assert report.failed == []


# --------------------------------------------------------------------------- #
# B. 反向失败："" 失败、qfq 成功
# --------------------------------------------------------------------------- #
def test_raw_adjust_failure_qfq_still_persisted() -> None:
    """``""`` 失败仅跳过该套，``qfq`` 仍落库；failed 记 1 次、reason 点名「不复权」。"""
    db = _mem_db()
    svc = _service(
        db,
        {"primary": _SelectiveFailSource(fail_adjusts={""}),
         "fallback": _SelectiveFailSource(fail_adjusts={""})},
    )
    report = svc.update_daily(symbols=["000001"], end="2015-01-10")

    assert not report.ok
    assert [s for s, _ in report.failed] == ["000001"]
    assert "不复权" in report.failed[0][1]      # 点名失败口径为「不复权」
    assert "qfq" not in report.failed[0][1]    # qfq 成功，不点名
    assert report.bars_written == 3            # 仅 qfq 一套落库
    assert db.get_bars("000001", "d", adjust="").shape[0] == 0
    assert db.get_bars("000001", "d", adjust="qfq").shape[0] == 3


# --------------------------------------------------------------------------- #
# C. hfq 不落 + adjust 参数被忽略
# --------------------------------------------------------------------------- #
def test_hfq_never_persisted_and_adjust_param_ignored() -> None:
    """传 ``adjust="hfq"`` 也仍只落 ``""``/``"qfq"``，绝不请求/落 hfq。"""
    db = _mem_db()
    primary = _CallSource()
    svc = _service(db, {"primary": primary, "fallback": _CallSource()})
    report = svc.update_daily(symbols=["000001"], adjust="hfq", end="2015-01-10")

    requested = {c["adjust"] for c in primary.kline_calls}
    assert requested == {"", "qfq"}            # 请求口径严格两套
    assert "hfq" not in requested
    assert db.get_bars("000001", "d", adjust="hfq").shape[0] == 0  # 不落 hfq
    assert db.get_bars("000001", "d", adjust="").shape[0] == 3
    assert db.get_bars("000001", "d", adjust="qfq").shape[0] == 3
    assert report.ok


# --------------------------------------------------------------------------- #
# D. 双套互不覆盖（强取证）
# --------------------------------------------------------------------------- #
def test_two_adjusts_values_not_cross_contaminated() -> None:
    """同一 symbol 同一 dt，"" 与 qfq 落不同收盘价，读回互不串值。"""
    db = _mem_db()
    svc = _service(db, {"primary": _CallSource(raw_close=9.99, qfq_close=11.11),
                        "fallback": _CallSource()})
    svc.update_daily(symbols=["000001"], end="2015-01-10")

    raw = db.get_bars("000001", "d", adjust="")
    qfq = db.get_bars("000001", "d", adjust="qfq")
    assert raw.shape[0] == 3 and qfq.shape[0] == 3
    assert set(raw["close"]) == {9.99}
    assert set(qfq["close"]) == {11.11}
    # 同一交易日两套并存（主键含 adjust，未复权不被前复权覆盖）
    raw_days = set(pd.to_datetime(raw["datetime"]).dt.strftime("%Y-%m-%d"))
    qfq_days = set(pd.to_datetime(qfq["datetime"]).dt.strftime("%Y-%m-%d"))
    assert raw_days == qfq_days


# --------------------------------------------------------------------------- #
# E. EVT_COMPLETED 的 bars_written 精确等于两套之和
# --------------------------------------------------------------------------- #
def test_completed_event_bars_written_exact_sum(bus) -> None:
    """``EVT_COMPLETED`` payload 的 bars_written == 两套之和，计数正确。"""
    db = _mem_db()
    svc = _service(db, {"primary": _CallSource(), "fallback": _CallSource()})
    completed: list[dict] = []
    failed: list[dict] = []
    bus.subscribe(EVT_COMPLETED, lambda p: completed.append(p))
    bus.subscribe(EVT_FAILED, lambda p: failed.append(p))

    svc.update_daily(symbols=["000001"], end="2015-01-10")

    assert len(completed) == 1
    assert completed[0]["bars_written"] == 6   # 两套 × 3 根
    assert completed[0]["symbols_count"] == 1  # symbol 级，不随口径翻倍
    assert completed[0]["failed_count"] == 0
    assert failed == []                        # 两套都成功 → 无 FAILED 事件
