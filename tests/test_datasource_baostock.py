"""``BaoStockSource`` 单测。

全部用例通过 monkeypatch 注入 **离线假 baostock**，不依赖真实网络，
用于确定性验证：代码前缀映射、周期/复权映射、**成交量股→手换算**、
时间解析、Bar 契约归一化、异常包装与注册表可用性。
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.data.datasource import baostock_src
from app.data.datasource.baostock_src import (
    BaoStockSource,
    _parse_bs_time,
    _to_bs_code,
)
from app.data.datasource.base import BAR_COLUMNS, DataSource, DataSourceError


class FakeResult:
    """模拟 baostock ``ResultData``：``fields`` + ``next()``/``get_row_data()``。"""

    def __init__(
        self,
        fields: list[str],
        rows: list[list[str]],
        error_code: str = "0",
        error_msg: str = "",
    ) -> None:
        self.fields = fields
        self._rows = rows
        self._idx = -1
        self.error_code = error_code
        self.error_msg = error_msg

    def next(self) -> bool:  # noqa: A003 - 对齐 baostock API 命名
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self) -> list[str]:
        return self._rows[self._idx]


# 日线两行：volume 单位为「股」（100000 股 = 1000 手）。
_DAILY_ROWS: list[dict[str, str]] = [
    {"date": "2026-09-01", "code": "sz.000001", "open": "10.0", "high": "10.6",
     "low": "9.9", "close": "10.5", "volume": "100000", "amount": "1050000.0"},
    {"date": "2026-09-02", "code": "sz.000001", "open": "11.0", "high": "11.6",
     "low": "10.9", "close": "11.5", "volume": "200000", "amount": "2300000.0"},
]

# 分钟两行：time 为 ``YYYYMMDDHHMMSSsss``（毫秒部分补 000）。
_MINUTE_ROWS: list[dict[str, str]] = [
    {"date": "2026-09-01", "time": "20260901093100000", "code": "sz.000001",
     "open": "10.0", "high": "10.15", "low": "9.95", "close": "10.1",
     "volume": "10000", "amount": "101000.0"},
    {"date": "2026-09-01", "time": "20260901093200000", "code": "sz.000001",
     "open": "10.1", "high": "10.25", "low": "10.05", "close": "10.2",
     "volume": "20000", "amount": "204000.0"},
]


class FakeBs:
    """离线假 baostock：返回与真实 fields/rows 结构一致的确定性数据。"""

    __version__ = "00.9.30"

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}
        self.login_count = 0
        self.logout_count = 0
        self.logged_in = False

    # -- 连接 -- #
    def login(self) -> FakeResult:
        self.login_count += 1
        self.logged_in = True
        return FakeResult([], [], "0", "")

    def logout(self) -> FakeResult:
        self.logout_count += 1
        self.logged_in = False
        return FakeResult([], [], "0", "")

    # -- K 线 -- #
    def query_history_k_data_plus(
        self, code, fields, start_date="", end_date="", frequency="d", adjustflag="3"
    ) -> FakeResult:
        self.calls["hist"] = dict(
            code=code, fields=fields, start_date=start_date,
            end_date=end_date, frequency=frequency, adjustflag=adjustflag,
        )
        field_list = [f for f in fields.split(",") if f]
        source_rows = _MINUTE_ROWS if frequency in ("5", "15", "30", "60") else _DAILY_ROWS
        rows = [[row.get(f, "") for f in field_list] for row in source_rows]
        return FakeResult(field_list, rows)

    # -- 列表 -- #
    def query_all_stock(self, day="") -> FakeResult:
        self.calls["all_stock"] = dict(day=day)
        return FakeResult(
            ["code", "tradeStatus", "code_name"],
            [
                ["sh.600000", "1", "浦发银行"],
                ["sz.000001", "1", "平安银行"],
                ["bj.830799", "1", "艾融软件"],
            ],
        )

    # -- 基本面 -- #
    def query_stock_basic(self, code="", code_name="") -> FakeResult:
        self.calls["basic"] = dict(code=code)
        return FakeResult(
            ["code", "code_name", "ipoDate", "outDate", "type", "status"],
            [["sz.000001", "平安银行", "1991-04-03", "", "1", "1"]],
        )

    def query_stock_industry(self, code="", date="") -> FakeResult:
        self.calls["industry"] = dict(code=code)
        return FakeResult(
            ["updateDate", "code", "code_name", "industry", "industryClassification"],
            [["2026-01-01", "sz.000001", "平安银行", "银行", "证监会"]],
        )

    def query_dupont_data(self, code="", year=0, quarter=0) -> FakeResult:
        return FakeResult(
            ["code", "pubDate", "statDate", "dupontROE"],
            [["sz.000001", "2026-08-01", "2026-06-30", "0.046746"]],
        )

    def query_growth_data(self, code="", year=0, quarter=0) -> FakeResult:
        return FakeResult(
            ["code", "pubDate", "statDate", "YOYNI"],
            [["sz.000001", "2026-08-01", "2026-06-30", "0.033213"]],
        )

    def query_adjust_factor(self, code="", start_date="", end_date="") -> FakeResult:
        self.calls["adj"] = dict(code=code, start_date=start_date, end_date=end_date)
        return FakeResult(
            ["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor"],
            [
                ["sz.000001", "2026-06-12", "1.0", "1.0"],
                ["sz.000001", "2025-10-15", "1.0329067641682", "1.0"],
            ],
        )

    # -- 指数成分股（P7-S4） -- #
    def query_hs300_stocks(self, date="") -> FakeResult:
        """返回首列 = 名单**生效日**（与真实接口一致，非查询日）。"""
        self.calls["hs300"] = dict(date=date)
        return FakeResult(
            ["date", "code", "code_name"],
            [
                ["2015-03-30", "sh.600000", "浦发银行"],
                ["2015-03-30", "sh.600005", "武钢股份"],
                ["2015-03-30", "sz.000001", "平安银行"],
            ],
        )

    def query_zz500_stocks(self, date="") -> FakeResult:
        self.calls["zz500"] = dict(date=date)
        return FakeResult(
            ["date", "code", "code_name"],
            [["2020-06-15", "sh.600100", "同方股份"]],
        )


class FailingBs(FakeBs):
    """让 K 线接口抛错，用于验证异常包装。"""

    def query_history_k_data_plus(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("模拟网络错误")


class LoginFailingBs(FakeBs):
    """登录失败。"""

    def login(self) -> FakeResult:
        return FakeResult([], [], "1", "登录失败：用户名或密码错误")


@pytest.fixture()
def fake_bs(monkeypatch) -> FakeBs:
    """注入离线假 baostock。"""
    fake = FakeBs()
    monkeypatch.setattr(baostock_src, "bs", fake)
    return fake


@pytest.fixture()
def source(fake_bs: FakeBs) -> BaoStockSource:
    """基于假 baostock 的 BaoStockSource。"""
    return BaoStockSource()


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("000001", "sz.000001"), ("600000", "sh.600000"), ("300750", "sz.300750"),
     ("688981", "sh.688981"), ("830799", "bj.830799")],
)
def test_to_bs_code_prefix_mapping(symbol: str, expected: str) -> None:
    """代码应映射为带市场前缀的 baostock 代码。"""
    assert _to_bs_code(symbol) == expected


def test_parse_bs_time_variants() -> None:
    """分钟时间字段应正确解析（含毫秒后缀与常见格式）。"""
    assert _parse_bs_time("20260901093100000") == datetime(2026, 9, 1, 9, 31, 0)
    assert _parse_bs_time("2026-09-01 09:31:00") == datetime(2026, 9, 1, 9, 31, 0)
    assert pd.isna(_parse_bs_time(""))


# --------------------------------------------------------------------------- #
# get_kline
# --------------------------------------------------------------------------- #
def test_get_kline_daily_contract_and_params(source: BaoStockSource, fake_bs: FakeBs) -> None:
    """日线：代码/周期/复权映射正确 + 返回 Bar 契约。"""
    df = source.get_kline("000001", "d", "2026-09-01", "2026-09-02", "qfq")
    assert list(df.columns) == BAR_COLUMNS
    assert df["datetime"].is_monotonic_increasing
    assert float(df.loc[0, "close"]) == 10.5

    call = fake_bs.calls["hist"]
    assert call["code"] == "sz.000001"
    assert call["frequency"] == "d"
    assert call["adjustflag"] == "2"  # qfq -> 2
    assert call["start_date"] == "2026-09-01"
    assert call["end_date"] == "2026-09-02"


def test_get_kline_volume_shares_to_lots_conversion(source: BaoStockSource) -> None:
    """⚠️ 关键：BaoStock 成交量单位为「股」，应除以 100 转为「手」以对齐 AkShare。"""
    df = source.get_kline("000001", "d", "2026-09-01", "2026-09-02")
    # 100000 股 -> 1000 手；200000 股 -> 2000 手
    assert float(df.loc[0, "volume"]) == 1000.0
    assert float(df.loc[1, "volume"]) == 2000.0


@pytest.mark.parametrize(
    ("period", "bs_freq"),
    [("d", "d"), ("w", "w"), ("m", "m"), ("5m", "5"), ("15m", "15"),
     ("30m", "30"), ("60m", "60")],
)
def test_get_kline_frequency_mapping(
    source: BaoStockSource, fake_bs: FakeBs, period: str, bs_freq: str
) -> None:
    """周期编码应映射到 baostock frequency。"""
    source.get_kline("000001", period, "2026-09-01", "2026-09-02")
    assert fake_bs.calls["hist"]["frequency"] == bs_freq


@pytest.mark.parametrize(
    ("adjust", "adjustflag"),
    [("", "3"), ("qfq", "2"), ("hfq", "1")],
)
def test_get_kline_adjust_mapping(
    source: BaoStockSource, fake_bs: FakeBs, adjust: str, adjustflag: str
) -> None:
    """复权口径应映射到 baostock adjustflag（''->3 / qfq->2 / hfq->1）。"""
    source.get_kline("000001", "d", "2026-09-01", "2026-09-02", adjust)
    assert fake_bs.calls["hist"]["adjustflag"] == adjustflag


# --------------------------------------------------------------------------- #
# get_kline：指数（P7-S1 回测基准 sh.000300）
# --------------------------------------------------------------------------- #
def test_get_kline_index_uses_index_fields(
    source: BaoStockSource, fake_bs: FakeBs
) -> None:
    """指数应使用不含 adjustflag 的专用字段集，返回 Bar 契约。"""
    df = source.get_kline("sh.000300", "d", "2026-09-01", "2026-09-02")

    call = fake_bs.calls["hist"]
    assert call["code"] == "sh.000300"
    assert call["adjustflag"] == "3"          # 指数不复权
    assert call["fields"] == baostock_src._INDEX_FIELDS
    assert "adjustflag" not in call["fields"]
    assert list(df.columns) == BAR_COLUMNS
    assert float(df.loc[0, "close"]) == 10.5


def test_get_kline_index_error_no_minimal_fallback(monkeypatch) -> None:
    """指数字段集失败时应直接抛错，**不**回退最小字段集（后者对指数同样不被接受）。"""

    class IndexFailingBs(FakeBs):
        def __init__(self) -> None:
            super().__init__()
            self.hist_calls = 0

        def query_history_k_data_plus(
            self, code, fields, start_date="", end_date="", frequency="d", adjustflag="3"
        ) -> FakeResult:
            self.hist_calls += 1
            self.calls["hist"] = dict(code=code, fields=fields)
            return FakeResult([], [], "1", "字段不存在")

    fake = IndexFailingBs()
    monkeypatch.setattr(baostock_src, "bs", fake)
    src = BaoStockSource()

    with pytest.raises(DataSourceError):
        src.get_kline("sh.000300", "d", "2026-09-01", "2026-09-02")
    assert fake.hist_calls == 1  # 只请求一次（无最小字段回退）


def test_get_kline_minute_uses_time_field(source: BaoStockSource) -> None:
    """分钟线应以 time 字段构造 datetime。"""
    df = source.get_kline("000001", "5m", "2026-09-01", "2026-09-01")
    assert list(df.columns) == BAR_COLUMNS
    assert df.loc[0, "datetime"] == pd.Timestamp("2026-09-01 09:31:00")
    # 10000 股 -> 100 手
    assert float(df.loc[0, "volume"]) == 100.0


def test_get_kline_unsupported_period_raises(source: BaoStockSource) -> None:
    """不支持的周期应包装为 DataSourceError。"""
    with pytest.raises(DataSourceError):
        source.get_kline("000001", "3m", "2026-09-01", "2026-09-02")


def test_get_kline_unsupported_adjust_raises(source: BaoStockSource) -> None:
    """不支持的复权口径应包装为 DataSourceError。"""
    with pytest.raises(DataSourceError):
        source.get_kline("000001", "d", "2026-09-01", "2026-09-02", "bfq")


def test_get_kline_wraps_exception(monkeypatch) -> None:
    """底层异常应包装成 DataSourceError 且保留 original。"""
    monkeypatch.setattr(baostock_src, "bs", FailingBs())
    src = BaoStockSource()
    with pytest.raises(DataSourceError) as excinfo:
        src.get_kline("000001", "d", "2026-09-01", "2026-09-02")
    assert excinfo.value.source == "baostock"
    assert isinstance(excinfo.value.original, Exception)


def test_login_failure_raises(monkeypatch) -> None:
    """登录失败（error_code != 0）应包装为 DataSourceError。"""
    monkeypatch.setattr(baostock_src, "bs", LoginFailingBs())
    src = BaoStockSource()
    with pytest.raises(DataSourceError):
        src.get_kline("000001", "d", "2026-09-01", "2026-09-02")


def test_login_lazy_and_reused(source: BaoStockSource, fake_bs: FakeBs) -> None:
    """懒登录：构造时未登录，多次调用只登录一次。"""
    assert fake_bs.login_count == 0
    source.get_kline("000001", "d", "2026-09-01", "2026-09-02")
    source.list_stocks()
    assert fake_bs.login_count == 1


def test_close_logs_out(source: BaoStockSource, fake_bs: FakeBs) -> None:
    """close() 应登出且幂等。"""
    source.get_kline("000001", "d", "2026-09-01", "2026-09-02")
    source.close()
    assert fake_bs.logout_count == 1
    source.close()  # 幂等，不重复登出
    assert fake_bs.logout_count == 1


def test_bs_unavailable_raises(monkeypatch) -> None:
    """baostock 未导入时调用应抛出 DataSourceError。"""
    monkeypatch.setattr(baostock_src, "bs", None)
    src = BaoStockSource()
    with pytest.raises(DataSourceError):
        src.get_kline("000001", "d", "2026-09-01", "2026-09-02")


# --------------------------------------------------------------------------- #
# get_index_constituents（P7-S4：历史成分股快照 / 幸存者偏差治理）
# --------------------------------------------------------------------------- #
_CONSTITUENT_COLUMNS = ["index_code", "symbol", "name", "effective_date"]


def test_get_index_constituents_hs300_contract(
    source: BaoStockSource, fake_bs: FakeBs
) -> None:
    """沪深300：返回列契约 + **裸码归一化** + effective_date 取名单生效日。"""
    df = source.get_index_constituents("sh.000300", "2015-03-31")

    assert list(df.columns) == _CONSTITUENT_COLUMNS
    assert len(df) == 3
    # sh.600000 -> 600000（与本地库 kline.symbol 同口径）
    assert list(df["symbol"]) == ["600000", "600005", "000001"]
    assert set(df["index_code"]) == {"sh.000300"}
    # 首列是**名单生效日**（2015-03-30），而非查询日（2015-03-31）
    assert set(df["effective_date"]) == {"2015-03-30"}
    assert df.loc[0, "name"] == "浦发银行"


@pytest.mark.parametrize(
    ("date_", "expected"),
    [
        ("2015-03-31", "2015-03-31"),
        ("2015/03/31", "2015-03-31"),                # 斜杠归一化
        ("20150331", "2015-03-31"),                  # 8 位纯数字归一化
        (datetime(2015, 3, 31), "2015-03-31"),       # datetime 归一化
    ],
)
def test_get_index_constituents_date_passthrough(
    source: BaoStockSource, fake_bs: FakeBs, date_, expected: str
) -> None:
    """查询日期应归一化为 ``YYYY-MM-DD`` 后透传给 baostock。"""
    source.get_index_constituents("sh.000300", date_)
    assert fake_bs.calls["hs300"]["date"] == expected


def test_get_index_constituents_none_date_uses_latest(
    source: BaoStockSource, fake_bs: FakeBs
) -> None:
    """``date_=None`` 应**不带参数**调用（取最新一份名单）。"""
    source.get_index_constituents("sh.000300", None)
    assert fake_bs.calls["hs300"]["date"] == ""  # 假接口默认值 = 未传参


def test_get_index_constituents_zz500_mapping(
    source: BaoStockSource, fake_bs: FakeBs
) -> None:
    """中证500 应路由到 query_zz500_stocks 且 index_code 列如实回填。"""
    df = source.get_index_constituents("sh.000905", "2020-06-30")

    assert "zz500" in fake_bs.calls
    assert fake_bs.calls["zz500"]["date"] == "2020-06-30"
    assert list(df["symbol"]) == ["600100"]
    assert set(df["index_code"]) == {"sh.000905"}


def test_get_index_constituents_unsupported_index_returns_empty(
    source: BaoStockSource, fake_bs: FakeBs
) -> None:
    """不支持的指数：返回**带列名**空表、不抛异常、且不触发登录/查询。"""
    df = source.get_index_constituents("sh.000001")

    assert df.empty
    assert list(df.columns) == _CONSTITUENT_COLUMNS
    assert "hs300" not in fake_bs.calls
    assert fake_bs.login_count == 0


def test_get_index_constituents_error_returns_empty(
    monkeypatch, fake_bs: FakeBs
) -> None:
    """接口返回错误码：降级为空表，不抛异常。"""

    class ErrorBs(FakeBs):
        def query_hs300_stocks(self, date="") -> FakeResult:
            self.calls["hs300"] = dict(date=date)
            return FakeResult([], [], "1", "网络超时")

    fake = ErrorBs()
    monkeypatch.setattr(baostock_src, "bs", fake)
    df = BaoStockSource().get_index_constituents("sh.000300", "2015-03-31")

    assert df.empty
    assert list(df.columns) == _CONSTITUENT_COLUMNS


def test_get_index_constituents_exception_returns_empty(monkeypatch) -> None:
    """底层抛异常：降级为空表（供上层「跳过该快照」），不中断调用方。"""

    class RaisingBs(FakeBs):
        def query_hs300_stocks(self, date="") -> FakeResult:
            raise RuntimeError("模拟网络错误")

    monkeypatch.setattr(baostock_src, "bs", RaisingBs())
    df = BaoStockSource().get_index_constituents("sh.000300", "2015-03-31")

    assert df.empty
    assert list(df.columns) == _CONSTITUENT_COLUMNS


def test_get_index_constituents_empty_rows_returns_empty(
    monkeypatch,
) -> None:
    """接口成功但无数据行：返回带列名空表。"""

    class EmptyBs(FakeBs):
        def query_hs300_stocks(self, date="") -> FakeResult:
            return FakeResult(["date", "code", "code_name"], [])

    monkeypatch.setattr(baostock_src, "bs", EmptyBs())
    df = BaoStockSource().get_index_constituents("sh.000300", "2015-03-31")

    assert df.empty
    assert list(df.columns) == _CONSTITUENT_COLUMNS


def test_get_index_constituents_filters_unparsable_code(monkeypatch) -> None:
    """无法提取 6 位码的脏数据行应被剔除，不污染外键口径。"""

    class DirtyBs(FakeBs):
        def query_hs300_stocks(self, date="") -> FakeResult:
            return FakeResult(
                ["date", "code", "code_name"],
                [
                    ["2015-03-30", "sh.600000", "浦发银行"],
                    ["2015-03-30", "sh.ABCDEF", "脏数据"],
                ],
            )

    monkeypatch.setattr(baostock_src, "bs", DirtyBs())
    df = BaoStockSource().get_index_constituents("sh.000300", "2015-03-31")

    assert list(df["symbol"]) == ["600000"]


# --------------------------------------------------------------------------- #
# list_stocks
# --------------------------------------------------------------------------- #
def test_list_stocks_market_derivation(source: BaoStockSource) -> None:
    """应剥离市场前缀、补 6 位并推导 market 列。"""
    df = source.list_stocks()
    assert list(df.columns) == ["code", "name", "market"]
    mapping = dict(zip(df["code"], df["market"]))
    assert mapping["000001"] == "sz"
    assert mapping["600000"] == "sh"
    assert mapping["830799"] == "bj"
    assert "平安银行" in list(df["name"])


# --------------------------------------------------------------------------- #
# get_minute / get_spot（不支持）
# --------------------------------------------------------------------------- #
def test_get_minute_returns_empty_contract(source: BaoStockSource) -> None:
    """1 分钟应返回 Bar 契约空表。"""
    df = source.get_minute("000001", "2026-09-01")
    assert list(df.columns) == BAR_COLUMNS
    assert df.empty


def test_get_spot_returns_empty(source: BaoStockSource) -> None:
    """实时 spot 应返回空表。"""
    df = source.get_spot()
    assert df.empty


# --------------------------------------------------------------------------- #
# get_adj_factor
# --------------------------------------------------------------------------- #
def test_get_adj_factor_contract(source: BaoStockSource, fake_bs: FakeBs) -> None:
    """复权因子：列名 [datetime, factor]、升序、接口参数正确。"""
    df = source.get_adj_factor("000001", "2025-01-01", "2026-12-31")
    assert list(df.columns) == ["datetime", "factor"]
    assert len(df) == 2
    assert df["datetime"].is_monotonic_increasing
    assert df.loc[0, "datetime"] == pd.Timestamp("2025-10-15")
    assert fake_bs.calls["adj"]["code"] == "sz.000001"
    assert fake_bs.calls["adj"]["start_date"] == "2025-01-01"


# --------------------------------------------------------------------------- #
# get_fundamental
# --------------------------------------------------------------------------- #
def test_get_fundamental_structure(source: BaoStockSource) -> None:
    """基本面 dict 键完整，且能从 basic/industry/dupont/growth 拼出字段。"""
    info = source.get_fundamental("000001")
    expected_keys = {
        "symbol", "name", "industry", "list_date", "total_market_cap",
        "float_market_cap", "pe", "pb", "roe", "revenue_yoy", "report_period", "summary",
    }
    assert set(info.keys()) == expected_keys
    assert info["symbol"] == "000001"
    assert info["name"] == "平安银行"
    assert info["industry"] == "银行"
    assert info["list_date"] == "1991-04-03"
    assert info["report_period"] == "2026-06-30"
    assert info["summary"]


def test_get_fundamental_ratio_to_percent(source: BaoStockSource) -> None:
    """⚠️ 关键：BaoStock 基本面为小数比率，须 ×100 转为百分比数值（对齐 AkShare）。"""
    info = source.get_fundamental("000001")
    # dupontROE 0.046746 -> 4.6746（百分比数值）
    assert info["roe"] == pytest.approx(4.6746)
    # YOYNI 0.033213 -> 3.3213（百分比数值）
    assert info["revenue_yoy"] == pytest.approx(3.3213)
    # summary 文案按百分比口径
    assert "ROE：4.6746%" in info["summary"]
    assert "净利同比：3.3213%" in info["summary"]


def test_get_fundamental_degrade_no_raise(monkeypatch) -> None:
    """子接口失败时应静默降级、不抛异常、结构完整。"""

    class PartialFailingBs(FakeBs):
        def query_stock_basic(self, code="", code_name=""):  # noqa: ANN001
            raise RuntimeError("basic 挂了")

        def query_dupont_data(self, code="", year=0, quarter=0):  # noqa: ANN001
            raise RuntimeError("dupont 挂了")

    monkeypatch.setattr(baostock_src, "bs", PartialFailingBs())
    info = BaoStockSource().get_fundamental("000001")
    assert info["name"] is None  # basic 失败 -> name 缺失
    assert info["roe"] is None  # dupont 失败 -> roe 缺失
    assert info["industry"] == "银行"  # industry 仍可用


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def test_get_source_baostock_registered() -> None:
    """注册表应能实例化 baostock 数据源，且满足 DataSource 协议。"""
    from app.data.datasource import SOURCES, get_source

    assert "baostock" in SOURCES
    src = get_source("baostock")
    assert isinstance(src, BaoStockSource)
    assert isinstance(src, DataSource)
    assert src.name == "baostock"
