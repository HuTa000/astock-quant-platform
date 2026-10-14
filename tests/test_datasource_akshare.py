"""``AkShareSource`` 单测。

全部用例通过 monkeypatch 注入 **离线假 akshare**，不依赖真实网络，
用于确定性验证列映射、周期映射、契约归一化与异常包装。
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.datasource import akshare_src
from app.data.datasource.akshare_src import AkShareSource
from app.data.datasource.base import BAR_COLUMNS, DataSource, DataSourceError


class FakeAk:
    """离线假 akshare：返回与真实列名一致的确定性数据。"""

    def __init__(self) -> None:
        self.calls: dict[str, dict] = {}

    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust="", timeout=None):
        self.calls["hist"] = dict(
            symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust=adjust
        )
        return pd.DataFrame(
            {
                "日期": ["2026-09-01", "2026-09-02"],
                "股票代码": ["000001", "000001"],
                "开盘": [10.0, 11.0],
                "收盘": [10.5, 11.5],
                "最高": [10.6, 11.6],
                "最低": [9.9, 10.9],
                "成交量": [1000, 2000],
                "成交额": [10500.0, 23000.0],
                "振幅": [1.0, 1.0],
                "涨跌幅": [0.5, 0.5],
                "涨跌额": [0.05, 0.05],
                "换手率": [0.1, 0.1],
            }
        )

    def stock_zh_a_hist_min_em(self, symbol, start_date, end_date, period="5", adjust=""):
        self.calls["hist_min"] = dict(
            symbol=symbol, start_date=start_date, end_date=end_date, period=period, adjust=adjust
        )
        return pd.DataFrame(
            {
                "时间": ["2026-09-01 09:31:00", "2026-09-01 09:32:00"],
                "开盘": [10.0, 10.1],
                "收盘": [10.1, 10.2],
                "最高": [10.15, 10.25],
                "最低": [9.95, 10.05],
                "成交量": [100, 200],
                "成交额": [1010.0, 2040.0],
                "均价": [10.05, 10.15],
            }
        )

    def stock_info_a_code_name(self):
        return pd.DataFrame(
            {"code": ["000001", "600000", "830799"], "name": ["平安银行", "浦发银行", "艾融软件"]}
        )

    def stock_zh_a_spot_em(self):
        return pd.DataFrame(
            {
                "代码": ["000001", "600000"],
                "名称": ["平安银行", "浦发银行"],
                "最新价": [10.0, 20.0],
                "涨跌幅": [1.2, -0.5],
                "市盈率-动态": [5.0, 6.0],
                "市净率": [0.6, 0.7],
                "成交量": [100, 200],
            }
        )

    def stock_individual_info_em(self, symbol, timeout=None):
        return pd.DataFrame(
            {
                "item": ["股票简称", "行业", "上市时间", "总市值", "流通市值"],
                "value": ["平安银行", "银行", "1991-04-03", 100.0, 90.0],
            }
        )

    def stock_zh_a_daily(self, symbol, start_date=None, end_date=None, adjust=""):
        self.calls["daily"] = dict(symbol=symbol, adjust=adjust)
        return pd.DataFrame(
            {"date": ["2026-06-12", "2025-10-15"], "qfq_factor": [1.0, 1.0329067641682]}
        )

    def stock_financial_analysis_indicator(self, symbol, start_year=None):
        return pd.DataFrame(
            {"日期": ["2025-12-31"], "净资产收益率(%)": [10.5], "主营业务收入增长率(%)": [8.2]}
        )


class FailingAk(FakeAk):
    """让日线接口抛错，用于验证异常包装。"""

    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust="", timeout=None):
        raise RuntimeError("模拟网络错误")


@pytest.fixture()
def fake_ak(monkeypatch):
    """注入离线假 akshare。"""
    fake = FakeAk()
    monkeypatch.setattr(akshare_src, "ak", fake)
    return fake


@pytest.fixture()
def source(fake_ak) -> AkShareSource:
    """基于假 akshare 的 AkShareSource。"""
    return AkShareSource()


# --------------------------------------------------------------------------- #
# get_kline
# --------------------------------------------------------------------------- #
def test_get_kline_daily_contract_and_params(source: AkShareSource, fake_ak: FakeAk) -> None:
    """日线：参数映射正确 + 返回 Bar 契约。"""
    df = source.get_kline("000001", "d", "20260101", "20260920", "qfq")
    assert list(df.columns) == BAR_COLUMNS
    assert df["datetime"].is_monotonic_increasing
    assert float(df.loc[0, "close"]) == 10.5

    call = fake_ak.calls["hist"]
    assert call["period"] == "daily"
    assert call["start_date"] == "20260101"
    assert call["end_date"] == "20260920"
    assert call["adjust"] == "qfq"


@pytest.mark.parametrize(
    ("period", "ak_period"),
    [("d", "daily"), ("w", "weekly"), ("m", "monthly")],
)
def test_get_kline_period_mapping(source: AkShareSource, fake_ak: FakeAk, period: str, ak_period: str) -> None:
    """d/w/m 应映射到 daily/weekly/monthly。"""
    source.get_kline("000001", period, "20260101", "20260920")
    assert fake_ak.calls["hist"]["period"] == ak_period


@pytest.mark.parametrize(
    ("period", "ak_period"),
    [("5m", "5"), ("15m", "15"), ("30m", "30"), ("60m", "60")],
)
def test_get_kline_minute_mapping(source: AkShareSource, fake_ak: FakeAk, period: str, ak_period: str) -> None:
    """5m/15m/30m/60m 应映射到 '5'/'15'/'30'/'60'，且时间为 datetime 文本。"""
    df = source.get_kline("000001", period, "2026-09-01", "2026-09-01")
    assert list(df.columns) == BAR_COLUMNS
    call = fake_ak.calls["hist_min"]
    assert call["period"] == ak_period
    assert call["start_date"] == "2026-09-01 09:00:00"
    assert call["end_date"] == "2026-09-01 15:30:00"


def test_get_kline_unsupported_period_raises(source: AkShareSource) -> None:
    """不支持的周期应包装为 DataSourceError。"""
    with pytest.raises(DataSourceError):
        source.get_kline("000001", "3m", "20260101", "20260920")


def test_get_kline_wraps_exception(monkeypatch) -> None:
    """底层异常应包装成 DataSourceError 且保留 original。"""
    monkeypatch.setattr(akshare_src, "ak", FailingAk())
    src = AkShareSource()
    with pytest.raises(DataSourceError) as excinfo:
        src.get_kline("000001", "d", "20260101", "20260920")
    assert excinfo.value.source == "akshare"
    assert isinstance(excinfo.value.original, Exception)


# --------------------------------------------------------------------------- #
# list_stocks
# --------------------------------------------------------------------------- #
def test_list_stocks_market_derivation(source: AkShareSource) -> None:
    """应补出 market 列并正确推导市场。"""
    df = source.list_stocks()
    assert list(df.columns) == ["code", "name", "market"]
    mapping = dict(zip(df["code"], df["market"]))
    assert mapping["000001"] == "sz"
    assert mapping["600000"] == "sh"
    assert mapping["830799"] == "bj"


# --------------------------------------------------------------------------- #
# get_minute
# --------------------------------------------------------------------------- #
def test_get_minute_returns_empty_contract(source: AkShareSource) -> None:
    """1 分钟应返回 Bar 契约空表。"""
    df = source.get_minute("000001", "2026-09-01")
    assert list(df.columns) == BAR_COLUMNS
    assert df.empty


# --------------------------------------------------------------------------- #
# get_spot
# --------------------------------------------------------------------------- #
def test_get_spot_front_columns(source: AkShareSource) -> None:
    """spot 前四列应为 [code, name, price, pct_change]，其余列保留。"""
    df = source.get_spot()
    assert list(df.columns[:4]) == ["code", "name", "price", "pct_change"]
    assert float(df.loc[0, "price"]) == 10.0
    assert float(df.loc[0, "pct_change"]) == 1.2
    assert "市盈率-动态" in df.columns


# --------------------------------------------------------------------------- #
# get_adj_factor
# --------------------------------------------------------------------------- #
def test_get_adj_factor_contract_and_filter(source: AkShareSource, fake_ak: FakeAk) -> None:
    """复权因子：列名 [datetime, factor]、升序、区间过滤、接口参数正确。"""
    df = source.get_adj_factor("000001", "2025-01-01", "2026-12-31")
    assert list(df.columns) == ["datetime", "factor"]
    assert len(df) == 2
    assert df["datetime"].is_monotonic_increasing
    assert fake_ak.calls["daily"]["symbol"] == "sz000001"
    assert fake_ak.calls["daily"]["adjust"] == "qfq-factor"

    sub = source.get_adj_factor("000001", "2026-01-01", "2026-12-31")
    assert len(sub) == 1
    assert sub.loc[0, "datetime"] == pd.Timestamp("2026-06-12")


# --------------------------------------------------------------------------- #
# get_fundamental
# --------------------------------------------------------------------------- #
def test_get_fundamental_structure(source: AkShareSource) -> None:
    """基本面 dict 键完整，且能从 info/spot/financial 拼出字段。"""
    info = source.get_fundamental("000001")
    expected_keys = {
        "symbol", "name", "industry", "list_date", "total_market_cap",
        "float_market_cap", "pe", "pb", "roe", "revenue_yoy", "report_period", "summary",
    }
    assert set(info.keys()) == expected_keys
    assert info["symbol"] == "000001"
    assert info["name"] == "平安银行"
    assert info["industry"] == "银行"
    assert info["total_market_cap"] == 100.0
    assert info["float_market_cap"] == 90.0
    assert info["pe"] == 5.0
    assert info["pb"] == 0.6
    assert info["roe"] == 10.5
    assert info["revenue_yoy"] == 8.2
    assert info["summary"]


def test_get_fundamental_missing_subfields_no_raise(monkeypatch) -> None:
    """估值 spot 子调用失败时应静默降级、不抛异常、结构完整。"""

    class SpotFailingAk(FakeAk):
        def stock_zh_a_spot_em(self):
            raise RuntimeError("spot 挂了")

    monkeypatch.setattr(akshare_src, "ak", SpotFailingAk())
    info = AkShareSource().get_fundamental("000001")
    assert info["name"] == "平安银行"
    assert info["pe"] is None  # spot 失败 -> 缺字段为 None
    assert info["roe"] == 10.5  # 财务接口仍可用


def test_get_fundamental_latest_period_not_stale(monkeypatch) -> None:
    """最新报告期「主营业务收入增长率」为占位符时须返回 None，不得回退历史旧值 -100.0。"""

    class StaleFinancialAk(FakeAk):
        def stock_financial_analysis_indicator(self, symbol, start_year=None):
            return pd.DataFrame(
                {
                    "日期": ["2010-12-31", "2026-06-30"],
                    "净资产收益率(%)": [8.0, 4.69],
                    "主营业务收入增长率(%)": [-100.0, "-"],  # 历史 -100.0，最新期占位符
                }
            )

    monkeypatch.setattr(akshare_src, "ak", StaleFinancialAk())
    info = AkShareSource().get_fundamental("000001")
    assert info["roe"] == 4.69          # 最新期有效
    assert info["revenue_yoy"] is None  # 最新期占位符 -> None（不回退 -100.0）
    assert info["report_period"] == "2026-06-30"


@pytest.mark.parametrize("placeholder", ["", "-", "--", None])
def test_parse_num_placeholders_return_none(placeholder) -> None:
    """空值/占位符必须解析为 None，不得伪造数值。"""
    from app.data.datasource.akshare_src import _parse_num

    assert _parse_num(placeholder) is None


def test_parse_num_valid_and_invalid() -> None:
    """有效数值正常解析；非数值文本返回 None。"""
    from app.data.datasource.akshare_src import _parse_num

    assert _parse_num("8.2") == pytest.approx(8.2)
    assert _parse_num(-100.0) == pytest.approx(-100.0)
    assert _parse_num("abc") is None
    assert pd.isna(_parse_num(float("nan")))


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def test_get_source_akshare_registered() -> None:
    """注册表应能实例化 akshare 数据源，且满足 DataSource 协议。"""
    from app.data.datasource import get_source

    src = get_source("akshare")
    assert isinstance(src, AkShareSource)
    assert isinstance(src, DataSource)
    assert src.name == "akshare"
