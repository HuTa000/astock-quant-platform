"""``UpdateService`` 单测（离线，fake 数据源 + 内存 SQLite）。

覆盖：增量起始日期、重试+兜底、失败不中断、事件顺序、meta 记录、报告计数、
空自选回退演示清单、分钟更新。
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.core.event_bus import get_bus
from app.data.datasource.base import BAR_COLUMNS
from app.data.store.db import Database
from app.data.update_service import (
    DEFAULT_START,
    DEMO_SYMBOLS,
    EVT_COMPLETED,
    EVT_FAILED,
    EVT_PROGRESS,
    EVT_STARTED,
    META_LAST_UPDATE,
    UpdateReport,
    UpdateService,
)


# --------------------------------------------------------------------------- #
# 工具：内存库 / 假数据源 / 事件收集
# --------------------------------------------------------------------------- #
def _mem_db() -> Database:
    """新建并初始化一个内存库。"""
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _make_bars(start: str, end: str, adjust: str = "") -> pd.DataFrame:
    """生成 [start, end] 内最多 3 根确定性 Bar。"""
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
                "close": 10.2,
                "volume": 100.0,   # 手
                "amount": 102000.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.DataFrame(rows)


class FakeSource:
    """离线假数据源（满足 DataSource 协议）。"""

    def __init__(self, name: str, fail_times: int = 0) -> None:
        self.name = name
        self.fail_times = fail_times
        self.kline_calls: list[dict] = []
        self.minute_calls: list[dict] = []

    def get_kline(self, symbol, period, start, end, adjust=""):
        self.kline_calls.append(
            dict(symbol=symbol, period=period, start=start, end=end, adjust=adjust)
        )
        if len(self.kline_calls) <= self.fail_times:
            raise RuntimeError(f"{self.name} 模拟失败#{len(self.kline_calls)}")
        return _make_bars(start, end, adjust)

    def get_minute(self, symbol, date_):
        self.minute_calls.append(dict(symbol=symbol, date=date_))
        if len(self.minute_calls) <= self.fail_times:
            raise RuntimeError(f"{self.name} 分钟模拟失败")
        base = pd.Timestamp(date_)
        return pd.DataFrame(
            [
                {"datetime": base + pd.Timedelta(minutes=1), "open": 10.0, "high": 10.1,
                 "low": 9.9, "close": 10.05, "volume": 50.0, "amount": 50000.0},
                {"datetime": base + pd.Timedelta(minutes=2), "open": 10.05, "high": 10.2,
                 "low": 10.0, "close": 10.15, "volume": 60.0, "amount": 60000.0},
            ]
        )

    def list_stocks(self):
        return pd.DataFrame(columns=["code", "name", "market"])

    def get_spot(self):
        return pd.DataFrame(columns=["code", "name", "price", "pct_change"])

    def get_adj_factor(self, symbol, start, end):
        return pd.DataFrame(columns=["datetime", "factor"])

    def get_fundamental(self, symbol):
        return {}


class SelectiveFailSource:
    """按复权口径选择性失败的假数据源（用于验证「一套失败不影响另一套」）。"""

    def __init__(self, fail_adjusts=()) -> None:
        self.fail_adjusts = set(fail_adjusts)
        self.kline_calls: list[dict] = []

    def get_kline(self, symbol, period, start, end, adjust=""):
        self.kline_calls.append(
            dict(symbol=symbol, period=period, start=start, end=end, adjust=adjust)
        )
        if adjust in self.fail_adjusts:
            raise RuntimeError(f"模拟失败 adjust={adjust!r}")
        return _make_bars(start, end, adjust)


@pytest.fixture()
def bus():
    """隔离事件总线：用例前清空订阅。"""
    b = get_bus()
    b.clear()
    yield b
    b.clear()


class EventCollector:
    """记录事件序列（用于断言顺序）。"""

    def __init__(self, bus, events) -> None:
        self.records: list[tuple[str, dict]] = []
        for ev in events:
            bus.subscribe(ev, lambda payload, ev=ev: self.records.append((ev, payload)))

    def names(self) -> list[str]:
        return [ev for ev, _ in self.records]


def _service(db, sources, primary="primary", fallback="fallback"):
    return UpdateService(db=db, primary_name=primary, fallback_name=fallback, sources=sources)


# --------------------------------------------------------------------------- #
# 增量起始
# --------------------------------------------------------------------------- #
def test_resolve_start_default_when_empty() -> None:
    """无历史数据 → 默认起始。"""
    db = _mem_db()
    src = FakeSource("primary")
    svc = _service(db, {"primary": src, "fallback": FakeSource("fallback")})
    assert svc._resolve_start("000001", "qfq", None) == DEFAULT_START


def test_resolve_start_next_day_after_existing() -> None:
    """已有 max dt → 次日。"""
    db = _mem_db()
    src = FakeSource("primary")
    svc = _service(db, {"primary": src, "fallback": FakeSource("fallback")})
    svc.update_daily(symbols=["000001"], adjust="qfq")
    # 第一批写了 2015-01-01..03（默认起始）
    first = db.get_bars("000001", "d", adjust="qfq")
    max_dt = first["datetime"].max()
    assert svc._resolve_start("000001", "qfq", None) == (
        pd.Timestamp(max_dt) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")


def test_resolve_start_explicit_override() -> None:
    """显式 start 优先。"""
    db = _mem_db()
    svc = _service(db, {"primary": FakeSource("primary"), "fallback": FakeSource("fallback")})
    assert svc._resolve_start("000001", "qfq", "2018-06-01") == "2018-06-01"


# --------------------------------------------------------------------------- #
# 重试 + 兜底
# --------------------------------------------------------------------------- #
def test_retry_on_primary_then_success() -> None:
    """主源首次失败、重试成功（不触发兜底）。"""
    db = _mem_db()
    primary = FakeSource("primary", fail_times=1)
    fallback = FakeSource("fallback")
    svc = _service(db, {"primary": primary, "fallback": fallback})
    report = svc.update_daily(symbols=["000001"], adjust="qfq")
    assert report.ok
    # fail_times 为全局计数：仅首个调用（未复权第 1 次）失败，重试成功后再跑 qfq。
    # → 未复权 2 次（失败 + 重试成功）+ qfq 1 次（直接成功）= 3 次
    assert len(primary.kline_calls) == 3
    assert fallback.kline_calls == []        # 未用到兜底
    assert db.count_bars("000001", "d") > 0


def test_fallback_after_primary_exhausted() -> None:
    """主源连续失败（重试后仍失败）→ 自动切兜底成功。"""
    db = _mem_db()
    primary = FakeSource("primary", fail_times=999)
    fallback = FakeSource("fallback")
    svc = _service(db, {"primary": primary, "fallback": fallback})
    report = svc.update_daily(symbols=["000001"], adjust="qfq")
    assert report.ok
    # 落两套口径：主源每套「1 + 重试 1」共 2 次 → 2 套 × 2 = 4 次；兜底每套 1 次 → 2 次
    assert len(primary.kline_calls) == 4
    assert len(fallback.kline_calls) == 2
    assert db.count_bars("000001", "d") > 0


def test_all_fail_does_not_interrupt_and_counts_failures() -> None:
    """主/兜底全失败：计入 failed，不中断其它 symbol，不抛异常。"""
    db = _mem_db()
    svc = _service(
        db,
        {"primary": FakeSource("primary", fail_times=999),
         "fallback": FakeSource("fallback", fail_times=999)},
    )
    report = svc.update_daily(symbols=["000001", "600000"], adjust="qfq")
    assert not report.ok
    assert report.bars_written == 0
    assert [s for s, _ in report.failed] == ["000001", "600000"]
    # 两套口径都失败时，symbol 级仅记一次，reason 应点名两套口径
    assert "不复权" in report.failed[0][1]
    assert "qfq" in report.failed[0][1]
    assert db.count_bars() == 0


# --------------------------------------------------------------------------- #
# 事件
# --------------------------------------------------------------------------- #
def test_event_sequence_success(bus) -> None:
    """成功路径事件顺序：STARTED → PROGRESS → COMPLETED。"""
    db = _mem_db()
    svc = _service(db, {"primary": FakeSource("primary"), "fallback": FakeSource("fallback")})
    collector = EventCollector(bus, [EVT_STARTED, EVT_PROGRESS, EVT_FAILED, EVT_COMPLETED])
    svc.update_daily(symbols=["000001", "600000"], adjust="qfq")
    names = collector.names()
    assert names[0] == EVT_STARTED
    assert names[-1] == EVT_COMPLETED
    assert names.count(EVT_PROGRESS) == 2
    assert EVT_FAILED not in names
    # 完成事件 payload 计数
    completed = [p for ev, p in collector.records if ev == EVT_COMPLETED][0]
    assert completed["symbols_count"] == 2
    assert completed["bars_written"] > 0
    assert completed["failed_count"] == 0


def test_event_failed_emitted(bus) -> None:
    """失败路径应发出 FAILED 事件。"""
    db = _mem_db()
    svc = _service(
        db,
        {"primary": FakeSource("primary", fail_times=999),
         "fallback": FakeSource("fallback", fail_times=999)},
    )
    collector = EventCollector(bus, [EVT_FAILED, EVT_COMPLETED])
    svc.update_daily(symbols=["000001"], adjust="qfq")
    assert EVT_FAILED in collector.names()
    failed_payload = [p for ev, p in collector.records if ev == EVT_FAILED][0]
    assert failed_payload["symbol"] == "000001"
    assert "error" in failed_payload


# --------------------------------------------------------------------------- #
# meta / 报告
# --------------------------------------------------------------------------- #
def test_meta_written_and_report_counts() -> None:
    """结束后写入 meta，且 UpdateReport 计数正确。"""
    db = _mem_db()
    svc = _service(db, {"primary": FakeSource("primary"), "fallback": FakeSource("fallback")})
    assert svc.last_update_time() is None
    report = svc.update_daily(symbols=["000001"], adjust="qfq")
    assert isinstance(report, UpdateReport)
    assert report.symbols_count == 1
    assert report.bars_written == 6          # 两套口径 × 2015-01-01/02/03
    assert report.ok
    assert svc.last_update_time() is not None
    assert db.get_meta(META_LAST_UPDATE) is not None


# --------------------------------------------------------------------------- #
# 符号来源
# --------------------------------------------------------------------------- #
def test_empty_watchlist_falls_back_to_demo_list() -> None:
    """自选股为空 → 使用内置演示清单。"""
    db = _mem_db()
    primary = FakeSource("primary")
    svc = _service(db, {"primary": primary, "fallback": FakeSource("fallback")})
    report = svc.update_daily(symbols=None, adjust="qfq")
    assert report.symbols_count == len(DEMO_SYMBOLS)
    used = {c["symbol"] for c in primary.kline_calls}
    assert used == set(DEMO_SYMBOLS)


def test_watchlist_symbols_used_and_deduped() -> None:
    """非空自选股 → 用之，且去重。"""
    db = _mem_db()
    gid = db.create_group("默认")
    db.add_watch_item(gid, "000001")
    db.add_watch_item(gid, "600000")
    db.add_watch_item(gid, "000001")  # 重复（同组 OR IGNORE，不会重复；另加跨组重复）
    gid2 = db.create_group("其它")
    db.add_watch_item(gid2, "000001")
    primary = FakeSource("primary")
    svc = _service(db, {"primary": primary, "fallback": FakeSource("fallback")})
    svc.update_daily(symbols=None, adjust="qfq")
    used = [c["symbol"] for c in primary.kline_calls]
    # 去重后 000001 仍只处理一次，但落两套口径 → 每 symbol 出现 2 次
    assert used.count("000001") == 2
    assert set(used) == {"000001", "600000"}


# --------------------------------------------------------------------------- #
# 分钟更新
# --------------------------------------------------------------------------- #
def test_update_minute_persists_1m() -> None:
    """update_minute 走分钟源并写入 period='1m'。"""
    db = _mem_db()
    minute = FakeSource("pytdx")
    svc = UpdateService(db=db, sources={"pytdx": minute})
    report = svc.update_minute("000001", "2026-09-19")
    assert report.ok
    assert report.bars_written == 2
    assert db.count_bars("000001", "1m") == 2
    assert minute.minute_calls[0]["symbol"] == "000001"


def test_update_minute_failure_reported() -> None:
    """分钟源失败 → 计入 failed，不抛异常。"""
    db = _mem_db()
    svc = UpdateService(db=db, sources={"pytdx": FakeSource("pytdx", fail_times=999)})
    report = svc.update_minute("000001", "2026-09-19")
    assert not report.ok
    assert report.failed[0][0] == "000001"


# --------------------------------------------------------------------------- #
# P6 优化：分钟线批量更新（baostock 分钟接口实测可用，供模拟训练按周期回放）
# --------------------------------------------------------------------------- #
class _PeriodFailSource(FakeSource):
    """指定周期固定失败的假数据源（验证「单周期失败不中断其余周期」）。"""

    def __init__(self, name: str, fail_period: str) -> None:
        super().__init__(name)
        self._fail_period = fail_period

    def get_kline(self, symbol, period, start, end, adjust=""):
        self.kline_calls.append(
            dict(symbol=symbol, period=period, start=start, end=end, adjust=adjust)
        )
        if period == self._fail_period:
            raise RuntimeError(f"{self.name} 模拟 {period} 失败")
        return _make_bars(start, end, adjust)


def test_update_minutes_persists_all_periods() -> None:
    """update_minutes 默认一次拉全 5/15/30/60m，各周期独立落库（未复权）。"""
    db = _mem_db()
    src = FakeSource("baostock")
    svc = UpdateService(db=db, sources={"baostock": src, "akshare": FakeSource("akshare")})
    report = svc.update_minutes(symbols=["000001"], start="2026-09-25", end="2026-09-30")

    assert report.ok
    assert [c["period"] for c in src.kline_calls] == ["5m", "15m", "30m", "60m"]
    assert all(c["adjust"] == "" for c in src.kline_calls)
    total = sum(db.count_bars("000001", p) for p in ("5m", "15m", "30m", "60m"))
    assert report.bars_written == total
    for period in ("5m", "15m", "30m", "60m"):
        assert db.count_bars("000001", period) >= 1


def test_update_minutes_ignores_non_minute_periods() -> None:
    """periods 里的非分钟周期（如 'd'）被忽略，不发起请求。"""
    db = _mem_db()
    src = FakeSource("baostock")
    svc = UpdateService(db=db, sources={"baostock": src})
    report = svc.update_minutes(
        symbols=["000001"], periods=("5m", "d"), start="2026-09-25", end="2026-09-30"
    )

    assert report.ok
    assert [c["period"] for c in src.kline_calls] == ["5m"]
    assert db.count_bars("000001", "d") == 0


def test_update_minutes_default_start_recent() -> None:
    """未显式起始日 → 默认结束日前 10 自然日（请求量小且回放够用）。"""
    assert UpdateService._minute_default_start("2026-09-30") == "2026-09-20"


def test_update_minutes_failure_reported() -> None:
    """某周期（主源重试 + 兜底均失败）→ 该标的计入 failed 一次，其余周期照常落库。"""
    db = _mem_db()
    src = _PeriodFailSource("baostock", fail_period="5m")
    svc = UpdateService(db=db, sources={"baostock": src})  # 仅主源，无兜底
    report = svc.update_minutes(symbols=["000001"], start="2026-09-25", end="2026-09-30")

    assert not report.ok
    assert report.failed[0][0] == "000001"
    assert "5m" in report.failed[0][1]
    # 5m 失败后仍继续拉了 15/30/60m（未中断整体）
    assert db.count_bars("000001", "15m") >= 1
    assert db.count_bars("000001", "60m") >= 1


# --------------------------------------------------------------------------- #
# R13：日线同时落「未复权 + 前复权」两套
# --------------------------------------------------------------------------- #
def test_two_adjusts_both_persisted() -> None:
    """一次 update_daily 同时落两套口径，行数一致、互不覆盖。"""
    db = _mem_db()
    svc = _service(db, {"primary": FakeSource("primary"), "fallback": FakeSource("fallback")})
    report = svc.update_daily(symbols=["000001"], end="2015-01-10")

    raw = db.get_bars("000001", "d", adjust="")
    qfq = db.get_bars("000001", "d", adjust="qfq")

    assert report.ok
    assert report.bars_written == 6           # 两套 × 3 根
    assert raw.shape[0] == 3
    assert qfq.shape[0] == 3
    assert raw.shape[0] == qfq.shape[0]


def test_incremental_two_adjusts_independent_start() -> None:
    """两套口径各自从各自库内 max(dt)+1 增量起始（互不借用）。"""
    db = _mem_db()
    # 预置 qfq 到 2015-01-02；未复权留空
    qfq = _make_bars("2015-01-01", "2015-01-02")
    qfq["symbol"] = "000001"
    qfq["period"] = "d"
    qfq["adjust"] = "qfq"
    db.upsert_bars(qfq)

    primary = FakeSource("primary")
    svc = _service(db, {"primary": primary, "fallback": FakeSource("fallback")})
    svc.update_daily(symbols=["000001"], end="2015-01-10")

    starts = {c["adjust"]: c["start"] for c in primary.kline_calls}
    assert starts[""] == DEFAULT_START           # 未复权无历史 → 默认起始
    assert starts["qfq"] == "2015-01-03"         # qfq max=2015-01-02 → 次日


def test_one_adjust_failure_other_persisted() -> None:
    """某套口径失败仅跳过该套、另一套仍落库，failed 正确点名失败的 adjust。"""
    db = _mem_db()
    svc = _service(
        db,
        {"primary": SelectiveFailSource(fail_adjusts={"qfq"}),
         "fallback": SelectiveFailSource(fail_adjusts={"qfq"})},
    )
    report = svc.update_daily(symbols=["000001"], end="2015-01-10")

    assert not report.ok
    assert [s for s, _ in report.failed] == ["000001"]
    assert "qfq" in report.failed[0][1]          # reason 点名失败的复权口径
    assert report.bars_written == 3              # 仅未复权一套落库
    assert db.get_bars("000001", "d", adjust="").shape[0] == 3
    assert db.get_bars("000001", "d", adjust="qfq").shape[0] == 0
