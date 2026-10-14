# -*- coding: utf-8 -*-
"""网络类报错「讲人话」单测 + 主源配置默认值守卫（离线）。

覆盖：
    * ``friendly_error`` 五类关键字映射（入参为「异常对象」与「字符串」两种形态）；
    * ``DataSourceError`` 前缀穿透（``str()`` 形如 ``"[akshare] HTTPSConnectionPool(...)"``）；
    * 未命中映射 → 截断到 80 字符 + ``…``；空/None → ``未知错误``；
    * ``update_daily`` 全失败时 ``report.failed`` 与 ``EVT_FAILED`` payload 均为**人话**，
      且 payload **不新增键**（保持 §1.3 契约）；
    * ``UpdateService`` 未显式传源名时，默认取配置的 ``baostock``/``akshare``。
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.core.event_bus import get_bus
from app.data.datasource.base import BAR_COLUMNS, DataSourceError
from app.data.store.db import Database
from app.data.update_service import (
    EVT_FAILED,
    UpdateService,
    _root_cause,
    friendly_error,
)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _mem_db() -> Database:
    """新建并初始化一个内存库。"""
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


class _AlwaysFailSource:
    """恒失败的假数据源（模拟东财/代理类网络异常）。"""

    def __init__(self, name: str, message: str) -> None:
        self.name = name
        self.message = message
        self.calls = 0

    def get_kline(self, symbol, period, start, end, adjust=""):
        self.calls += 1
        raise ConnectionError(self.message)

    def get_minute(self, symbol, date_):
        self.calls += 1
        raise ConnectionError(self.message)

    def list_stocks(self):
        return pd.DataFrame(columns=["code", "name", "market"])

    def get_spot(self):
        return pd.DataFrame(columns=["code", "name", "price", "pct_change"])

    def get_adj_factor(self, symbol, start, end):
        return pd.DataFrame(columns=["datetime", "factor"])

    def get_fundamental(self, symbol):
        return {}


class _OkSource:
    """恒成功的假数据源（提供空 Bar 契约表）。"""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_kline(self, symbol, period, start, end, adjust=""):
        return pd.DataFrame(columns=BAR_COLUMNS)


@pytest.fixture()
def bus():
    """隔离事件总线。"""
    b = get_bus()
    b.clear()
    yield b
    b.clear()


# --------------------------------------------------------------------------- #
# friendly_error：五类映射
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPSConnectionPool(host='x', port=443): (Caused by ProxyError('Unable to connect to proxy'))",
         "网络代理不可用"),
        ("requests.exceptions.ProxyError: boom", "网络代理不可用"),
        ("http.client.RemoteDisconnected: Remote end closed connection without response",
         "连接被中断（服务端或中间设备重置）"),
        ("ConnectionError: ('Connection aborted.', RemoteDisconnected(...))",
         "连接被中断（服务端或中间设备重置）"),
        ("socket.gaierror: [Errno 11001] getaddrinfo failed", "域名解析失败"),
        ("Name or service not known", "域名解析失败"),
        ("HTTPSConnectionPool: Read timed out. (read timeout=15)", "连接超时"),
        ("TimeoutError: request Timeout", "连接超时"),
        ("Max retries exceeded with url: /api", "连接失败（重试耗尽）"),
    ],
)
def test_friendly_error_maps_network_keywords(raw: str, expected: str) -> None:
    """字符串入参：五类网络关键字应映射为对应人话。"""
    assert friendly_error(raw) == expected


def test_friendly_error_accepts_exception_object() -> None:
    """入参为异常对象时应等价于其 ``str()``。"""
    assert friendly_error(ConnectionError("ConnectTimeout: timed out")) == "连接超时"


def test_friendly_error_penetrates_datasource_error_prefix() -> None:
    """``DataSourceError`` 的 ``str()`` 带 ``[源名]`` 前缀，匹配必须能穿透。"""
    err = DataSourceError("akshare", ConnectionError("getaddrinfo failed"))
    assert "[akshare]" in str(err)  # 前提：前缀确实存在
    assert friendly_error(err) == "域名解析失败"


def test_friendly_error_truncates_unknown_message() -> None:
    """未命中映射 → 截断到 80 字符并以 ``…`` 结尾。"""
    result = friendly_error("Z" * 200)
    assert len(result) == 81
    assert result.endswith("…")
    assert result.startswith("Z" * 80)


def test_friendly_error_keeps_short_unknown_message() -> None:
    """短且未命中映射 → 原样返回（不截断、不加省略号）。"""
    assert friendly_error("某些未知问题") == "某些未知问题"


def test_friendly_error_collapses_whitespace_and_handles_empty() -> None:
    """多行/空白应压平；空串与 None → ``未知错误``。"""
    assert friendly_error("line1\n  line2\ttab") == "line1 line2 tab"
    assert friendly_error("") == "未知错误"
    assert friendly_error(None) == "未知错误"
    assert friendly_error("   \n  ") == "未知错误"


# --------------------------------------------------------------------------- #
# _root_cause：解包到最内层原始异常
# --------------------------------------------------------------------------- #
def test_root_cause_returns_self_for_plain_exception() -> None:
    """无 ``original`` 链的普通异常应原样返回。"""
    plain = RuntimeError("x")
    assert _root_cause(plain) is plain
    assert _root_cause(None) is None


def test_root_cause_unwraps_multiple_levels() -> None:
    """多层 ``DataSourceError`` 嵌套应一路解到最内层。"""
    inner = ConnectionError("boom")
    nested = DataSourceError("outer", DataSourceError("inner", inner))
    assert _root_cause(nested) is inner


def test_friendly_error_unwraps_aggregated_datasource_error() -> None:
    """聚合异常的 message 已不含关键词，必须先解包再映射 —— 否则会把人话喂成内部文案。

    这是本项目真实踩到的坑：``_fetch_with_fallback`` 抛出的 ``DataSourceError``
    message 形如「所有数据源拉取 000001 失败：网络代理不可用」，关键词已丢失。
    """
    inner = ConnectionError("Unable to connect to proxy")
    err = DataSourceError("baostock", inner, message="所有数据源拉取 000001 失败")
    assert "proxy" not in str(err).lower()  # 前提：聚合串确实不含关键词
    assert friendly_error(err) == "网络代理不可用"
    assert _root_cause(err) is inner


# --------------------------------------------------------------------------- #
# 集成：update_daily 全失败 → 人话（且 payload 契约不变）
# --------------------------------------------------------------------------- #
def test_update_daily_all_fail_reports_friendly_error(bus) -> None:
    """所有数据源均失败时：report.failed 与 EVT_FAILED 的 error 都是人话，不是 urllib3 长栈。"""
    db = _mem_db()
    records: list[dict] = []
    bus.subscribe(EVT_FAILED, lambda payload: records.append(payload))

    primary = _AlwaysFailSource(
        "baostock",
        "HTTPSConnectionPool(host='push2his.eastmoney.com', port=443): "
        "Max retries exceeded (Caused by ProxyError('Unable to connect to proxy'))",
    )
    fallback = _AlwaysFailSource("akshare", "getaddrinfo failed")
    svc = UpdateService(
        db=db, primary_name="baostock", fallback_name="akshare",
        sources={"baostock": primary, "akshare": fallback},
    )

    report = svc.update_daily(["000001"], adjust="qfq", start="2026-09-01", end="2026-09-18")

    assert report.symbols_count == 1
    assert report.bars_written == 0
    assert len(report.failed) == 1
    symbol, message = report.failed[0]
    assert symbol == "000001"
    # 两套口径各「主源重试 1 次 + 兜底源 1 次」→ 主源 2×2=4 次、兜底 1×2=2 次
    assert primary.calls == 4 and fallback.calls == 2
    # 聚合消息优先反映**主源**原因（baostock 的 ProxyError → 网络代理不可用），
    # 不再被最后一个兜底源（akshare 的 getaddrinfo → 域名解析失败）覆盖；
    # R13 起 reason 带复权口径前缀（两套都失败 → 点名两套）。
    assert message == "不复权:网络代理不可用；qfq:网络代理不可用"
    assert "HTTPSConnectionPool" not in message

    assert len(records) == 1
    assert records[0]["symbol"] == "000001"
    assert records[0]["error"] == "不复权:网络代理不可用；qfq:网络代理不可用"
    # §1.3 payload 契约：不新增键
    assert set(records[0].keys()) == {"symbol", "error"}


def test_update_daily_proxy_error_maps_to_proxy_message(bus) -> None:
    """两源都报代理不可用 → 人话为「网络代理不可用」。"""
    db = _mem_db()
    svc = UpdateService(
        db=db, primary_name="baostock", fallback_name="akshare",
        sources={
            "baostock": _AlwaysFailSource("baostock", "ProxyError('Unable to connect to proxy')"),
            "akshare": _AlwaysFailSource("akshare", "Unable to connect to proxy"),
        },
    )
    report = svc.update_daily(["000001"], adjust="qfq", start="2026-09-01", end="2026-09-18")
    assert report.failed[0][1] == "不复权:网络代理不可用；qfq:网络代理不可用"


def test_update_daily_success_has_no_failed_entries(bus) -> None:
    """对照：成功路径不应产生 failed 记录（防止误报）。"""
    db = _mem_db()
    svc = UpdateService(
        db=db, primary_name="baostock", fallback_name="akshare",
        sources={"baostock": _OkSource("baostock"), "akshare": _OkSource("akshare")},
    )
    report = svc.update_daily(["000001"], adjust="qfq", start="2026-09-01", end="2026-09-18")
    assert report.failed == []
    assert report.ok is True


# --------------------------------------------------------------------------- #
# 增量跳过 + 主源报错优先（2026-09-21 真机日志暴露的两个缺陷）
# --------------------------------------------------------------------------- #
class _CallTrackingOkSource:
    """成功但记录调用次数的假数据源。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.kline_calls: list[dict] = []

    def get_kline(self, symbol, period, start, end, adjust=""):
        self.kline_calls.append(
            dict(symbol=symbol, period=period, start=start, end=end, adjust=adjust)
        )
        return pd.DataFrame(columns=BAR_COLUMNS)


def test_update_daily_skips_when_already_up_to_date(bus) -> None:
    """库内已含今日数据（两套口径都含）时，update_daily 应跳过：不请求、不报错、不写。"""
    db = _mem_db()
    today = date.today().strftime("%Y-%m-%d")
    for adjust in ("", "qfq"):
        frame = pd.DataFrame(
            [{
                "datetime": pd.Timestamp(today),
                "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2,
                "volume": 100.0, "amount": 102000.0,
            }]
        )
        frame["symbol"] = "000001"
        frame["period"] = "d"
        frame["adjust"] = adjust
        db.upsert_bars(frame)

    primary = _CallTrackingOkSource("baostock")
    svc = UpdateService(
        db=db, primary_name="baostock", fallback_name="akshare",
        sources={"baostock": primary, "akshare": _CallTrackingOkSource("akshare")},
    )
    report = svc.update_daily(["000001"], adjust="qfq")
    assert report.ok
    assert report.bars_written == 0        # 两套都已最新，无新增
    assert report.failed == []             # 不报错
    assert primary.kline_calls == []       # 根本没去请求主源


def test_update_daily_surface_primary_reason_not_fallback(bus) -> None:
    """主源报「起止日期无效」、兜底报网络错时，人话应取主源而非被兜底掩盖。

    复现用户真机日志：baostock 因 start>end 报「起始日期大于终止日期」，
    akshare 因网络重置报 Connection aborted；旧实现只会显示兜底的
    「连接被中断」，把主源真实原因埋掉。
    """
    db = _mem_db()
    records: list[dict] = []
    bus.subscribe(EVT_FAILED, lambda payload: records.append(payload))
    primary = _AlwaysFailSource("baostock", "起始日期大于终止日期，请修改。")
    fallback = _AlwaysFailSource("akshare", "Remote end closed connection")
    svc = UpdateService(
        db=db, primary_name="baostock", fallback_name="akshare",
        sources={"baostock": primary, "akshare": fallback},
    )
    report = svc.update_daily(["000001"], adjust="qfq", start="2026-09-01", end="2026-09-18")
    symbol, message = report.failed[0]
    assert symbol == "000001"
    # 人话仍取主源原因（不被兜底掩盖），R13 起 reason 带复权口径前缀（两套都失败 → 点名两套）。
    assert message == "不复权:起始日期大于终止日期，请修改。；qfq:起始日期大于终止日期，请修改。"
    assert records[0]["error"] == "不复权:起始日期大于终止日期，请修改。；qfq:起始日期大于终止日期，请修改。"
    # payload 契约不变
    assert set(records[0].keys()) == {"symbol", "error"}


# --------------------------------------------------------------------------- #
# 配置默认值守卫（换源后防回退）
# --------------------------------------------------------------------------- #
def test_update_service_reads_primary_and_fallback_from_config() -> None:
    """未显式传源名时应取配置：主源 baostock、兜底 akshare。"""
    db = _mem_db()
    svc = UpdateService(db=db)
    assert svc._primary_name == "baostock"
    assert svc._fallback_name == "akshare"
    assert svc._minute_name == "pytdx"
