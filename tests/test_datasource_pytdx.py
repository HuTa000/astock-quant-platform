"""``PytdxSource`` 单测。

全部用例通过 monkeypatch 注入 **离线假 pytdx API / reader**，不依赖真实网络，
确定性验证：分页累积（不丢根/升序/去重，跨 800 边界）、每日 \u2264240 根与按日过滤、
连接 failover、异常包装、成交量口径、本地 ``.lc1`` 分支与不支持能力返空。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.data.datasource import pytdx_src
from app.data.datasource.base import BAR_COLUMNS, DataSource, DataSourceError
from app.data.datasource.pytdx_src import PytdxSource


# --------------------------------------------------------------------------- #
# 数据集构造：N 个交易日的 1 分钟线（每日 240 根，09:31-11:30 + 13:01-15:00）
# --------------------------------------------------------------------------- #
def _trading_minutes() -> list[str]:
    """一天 240 个交易分钟（HH:MM）。"""
    out: list[str] = []
    for start_h, start_m, count in ((9, 31, 120), (13, 1, 120)):
        t = datetime(2000, 1, 1, start_h, start_m)
        for _ in range(count):
            out.append(t.strftime("%H:%M"))
            t += timedelta(minutes=1)
    return out


def _build_dataset(days: int, base: str = "2026-08-25") -> list[dict]:
    """构造按时间升序排列的分钟线数据集（模拟 TDX：越晚越靠后）。"""
    minutes = _trading_minutes()
    base_dt = datetime.strptime(base, "%Y-%m-%d")
    data: list[dict] = []
    for d in range(days):
        day = base_dt + timedelta(days=d)
        for hm in minutes:
            data.append(
                {
                    "datetime": f"{day:%Y-%m-%d} {hm}",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.2,
                    "vol": 100.0,      # 单位：手（网络口径，无需换算）
                    "amount": 102000.0,
                }
            )
    return data


class FakeTdxApi:
    """离线假 pytdx API：模拟 connect / get_security_bars 的分页语义。

    类属性 ``dataset``、``fail_ips`` 供用例配置；``instances`` 记录实例以便断言 failover。
    """

    dataset: list[dict] = []
    fail_ips: set[str] = set()
    instances: list["FakeTdxApi"] = []
    raise_on_bars: bool = False

    def __init__(self) -> None:
        self.connected = False
        self.connected_ip: str | None = None
        self.calls: list[dict] = []
        FakeTdxApi.instances.append(self)

    def connect(self, ip, port, time_out=None) -> bool:
        self.connected_ip = ip
        if ip in FakeTdxApi.fail_ips:
            self.connected = False
            return False
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def get_security_bars(self, category, market, code, start, count):
        self.calls.append(dict(category=category, market=market, code=code, start=start, count=count))
        if FakeTdxApi.raise_on_bars:
            raise RuntimeError("模拟取数错误")
        data = FakeTdxApi.dataset
        n = len(data)
        if start >= n:
            return []
        end = n - start
        begin = max(0, end - count)
        # 页内按时间升序返回（TDX 语义）
        return [dict(b) for b in data[begin:end]]


@pytest.fixture()
def fake_api(monkeypatch):
    """注入离线假 API 并重置类状态。"""
    FakeTdxApi.dataset = []
    FakeTdxApi.fail_ips = set()
    FakeTdxApi.instances = []
    FakeTdxApi.raise_on_bars = False
    monkeypatch.setattr(pytdx_src, "TdxHq_API", FakeTdxApi)
    return FakeTdxApi


def _source(hosts=None) -> PytdxSource:
    """构造连接假 API 的数据源（默认单台假服务器）。"""
    return PytdxSource(hosts=hosts or [("fake", "127.0.0.1", 7709)], timeout=1.0)


# --------------------------------------------------------------------------- #
# 分页累积
# --------------------------------------------------------------------------- #
def test_get_kline_1m_pagination_no_loss_sorted_dedup(fake_api) -> None:
    """跨 800 边界分页：不丢根、不重复、时间升序。"""
    fake_api.dataset = _build_dataset(days=5)  # 1200 根，> 800，触发两页
    assert len(fake_api.dataset) == 1200
    src = _source()
    df = src.get_kline("000001", "1m", "2026-08-01", "2026-09-30")

    assert list(df.columns) == BAR_COLUMNS
    assert len(df) == 1200                       # 不丢根
    assert df["datetime"].is_monotonic_increasing  # 升序
    assert df["datetime"].is_unique              # 不重复
    # 确认确实发生了翻页（第二次请求 start=800）
    starts = [c["start"] for c in fake_api.instances[-1].calls]
    assert 800 in starts


def test_get_kline_1m_boundary_rows_present_once(fake_api) -> None:
    """页边界处（第 800/801 根）的记录恰好出现一次。"""
    fake_api.dataset = _build_dataset(days=5)
    src = _source()
    df = src.get_kline("000001", "1m", "2026-08-01", "2026-09-30")
    # 数据集按升序；第 800、801 根（1-based）应各出现一次
    boundary_ts = pd.to_datetime(fake_api.dataset[799]["datetime"])
    assert (df["datetime"] == boundary_ts).sum() == 1


def test_get_kline_non_1m_returns_empty_and_no_connect(fake_api) -> None:
    """非 1 分钟周期返回空契约，且不触发连接。"""
    fake_api.dataset = _build_dataset(days=1)
    src = _source()
    df = src.get_kline("000001", "d", "2026-08-25", "2026-08-25")
    assert list(df.columns) == BAR_COLUMNS
    assert df.empty
    assert fake_api.instances == []  # 未连接


def test_get_kline_category_and_market(fake_api) -> None:
    """应使用 1 分钟类别常量，并按市场传递 market。"""
    fake_api.dataset = _build_dataset(days=1)
    src = _source()
    src.get_kline("600000", "1m", "2026-08-25", "2026-08-25")  # 沪市
    call = fake_api.instances[-1].calls[0]
    assert call["category"] == pytdx_src._KLINE_1MIN
    assert call["market"] == pytdx_src._MARKET_SH
    assert call["code"] == "600000"


# --------------------------------------------------------------------------- #
# get_minute
# --------------------------------------------------------------------------- #
def test_get_minute_day_filter_and_cap(fake_api) -> None:
    """get_minute 仅返回当日、\u2264240 根、升序。"""
    fake_api.dataset = _build_dataset(days=3, base="2026-08-25")
    src = _source()
    target = "2026-08-26"
    df = src.get_minute("000001", target)
    assert len(df) == 240
    assert df["datetime"].dt.strftime("%Y-%m-%d").unique().tolist() == [target]
    assert df["datetime"].is_monotonic_increasing


def test_get_minute_volume_unit_is_lots(fake_api) -> None:
    """网络 vol 单位已是「手」，不得再换算。"""
    fake_api.dataset = _build_dataset(days=1)
    src = _source()
    df = src.get_minute("000001", "2026-08-25")
    assert float(df.loc[0, "volume"]) == 100.0


# --------------------------------------------------------------------------- #
# failover / 异常
# --------------------------------------------------------------------------- #
def test_failover_to_second_host(fake_api) -> None:
    """第一台失败应自动切第二台。"""
    fake_api.dataset = _build_dataset(days=1)
    fake_api.fail_ips = {"1.1.1.1"}
    src = PytdxSource(
        hosts=[("bad", "1.1.1.1", 7709), ("good", "2.2.2.2", 7709)], timeout=1.0
    )
    df = src.get_minute("000001", "2026-08-25")
    assert len(df) == 240
    # 第一台被尝试且失败，第二台成功
    assert fake_api.instances[0].connected_ip == "1.1.1.1"
    assert fake_api.instances[0].connected is False
    assert any(inst.connected and inst.connected_ip == "2.2.2.2" for inst in fake_api.instances)


def test_all_hosts_fail_raises(fake_api) -> None:
    """所有服务器均失败应抛 DataSourceError。"""
    fake_api.dataset = _build_dataset(days=1)
    fake_api.fail_ips = {"1.1.1.1", "2.2.2.2"}
    src = PytdxSource(hosts=[("a", "1.1.1.1", 7709), ("b", "2.2.2.2", 7709)], timeout=1.0)
    with pytest.raises(DataSourceError) as excinfo:
        src.get_minute("000001", "2026-08-25")
    assert excinfo.value.source == "pytdx"


def test_request_exception_wrapped(fake_api) -> None:
    """取数异常应包装为 DataSourceError 且保留 original。"""
    fake_api.dataset = _build_dataset(days=1)
    fake_api.raise_on_bars = True
    src = _source()
    with pytest.raises(DataSourceError) as excinfo:
        src.get_minute("000001", "2026-08-25")
    assert excinfo.value.source == "pytdx"
    assert isinstance(excinfo.value.original, Exception)


def test_lazy_connect_and_reuse(fake_api) -> None:
    """懒连接：构造不连接；多次取数复用同一连接。"""
    fake_api.dataset = _build_dataset(days=2, base="2026-08-25")
    src = _source()
    assert fake_api.instances == []
    src.get_minute("000001", "2026-08-25")
    src.get_minute("000001", "2026-08-26")
    assert len(fake_api.instances) == 1  # 只连接一次


def test_close_disconnects(fake_api) -> None:
    """close() 应断开且幂等。"""
    fake_api.dataset = _build_dataset(days=1)
    src = _source()
    src.get_minute("000001", "2026-08-25")
    src.close()
    assert fake_api.instances[-1].connected is False
    src.close()  # 幂等


# --------------------------------------------------------------------------- #
# 本地 .lc1 兜底
# --------------------------------------------------------------------------- #
class _FakeMinBarReader:
    """假 .lc1 reader：返回含「股」口径 volume 的 DataFrame。"""

    frame: pd.DataFrame | None = None

    def get_df(self, path):  # noqa: ANN001
        return _FakeMinBarReader.frame.copy()


def test_local_minute_without_config_returns_empty(monkeypatch) -> None:
    """未配置 tdx_dir 时返回空表。"""
    monkeypatch.setattr(pytdx_src.PytdxSource, "_configured_tdx_dir", staticmethod(lambda: None))
    df = _source().read_local_minute("000001", "2026-09-01")
    assert list(df.columns) == BAR_COLUMNS
    assert df.empty


def test_local_minute_missing_file_returns_empty(tmp_path) -> None:
    """文件不存在时返回空表。"""
    df = _source().read_local_minute("000001", "2026-09-01", tdx_dir=str(tmp_path))
    assert list(df.columns) == BAR_COLUMNS
    assert df.empty


def test_local_minute_reads_and_converts_volume(monkeypatch, tmp_path) -> None:
    """.lc1 读取：按日过滤，且 volume 由「股」÷100 转为「手」。"""
    lc1 = tmp_path / "vipdoc" / "sz" / "minline" / "sz000001.lc1"
    lc1.parent.mkdir(parents=True)
    lc1.write_bytes(b"")  # 占位，实际由假 reader 提供数据

    _FakeMinBarReader.frame = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "amount": [101000.0, 102000.0, 103000.0],
            "volume": [100000.0, 200000.0, 300000.0],  # 单位：股
        },
        index=pd.to_datetime(
            ["2026-09-01 09:31", "2026-09-01 09:32", "2026-09-02 09:31"]
        ),
    )
    monkeypatch.setattr(pytdx_src, "TdxMinBarReader", _FakeMinBarReader)

    df = _source().read_local_minute("000001", "2026-09-01", tdx_dir=str(tmp_path))
    assert list(df.columns) == BAR_COLUMNS
    assert len(df) == 2  # 仅当日
    assert float(df.loc[0, "volume"]) == 1000.0  # 100000 股 -> 1000 手


# --------------------------------------------------------------------------- #
# 非本源职责能力
# --------------------------------------------------------------------------- #
def test_unsupported_methods_return_empty() -> None:
    """list_stocks / get_spot / get_adj_factor 返回空表；get_fundamental 返回空骨架 dict。"""
    src = _source()
    assert src.list_stocks().empty
    assert src.get_spot().empty
    assert src.get_adj_factor("000001", "2026-01-01", "2026-12-31").empty
    info = src.get_fundamental("000001")
    assert set(info.keys()) == set(pytdx_src._FUNDAMENTAL_KEYS)
    assert info["symbol"] == "000001"
    assert info["roe"] is None


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def test_get_source_pytdx_registered() -> None:
    """注册表应能实例化 pytdx 数据源，且满足 DataSource 协议。"""
    from app.data.datasource import SOURCES, get_source

    assert "pytdx" in SOURCES
    src = get_source("pytdx")
    assert isinstance(src, PytdxSource)
    assert isinstance(src, DataSource)
    assert src.name == "pytdx"
