"""``DataSource`` 接口与通用工具的单测（不使用网络）。"""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.datasource.base import (
    BAR_COLUMNS,
    DataSource,
    DataSourceError,
    SOURCES,
    derive_market,
    get_source,
    is_index_symbol,
    normalize_bars,
    register_source,
)


# --------------------------------------------------------------------------- #
# normalize_bars
# --------------------------------------------------------------------------- #
def test_normalize_bars_english_columns() -> None:
    """英文列名应归一化为 Bar 契约且类型正确。"""
    raw = pd.DataFrame(
        {
            "datetime": ["2026-09-02", "2026-09-01"],
            "open": ["1", "2"],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [100, 200],
            "amount": [105, 410],
        }
    )
    out = normalize_bars(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert str(out.dtypes["datetime"]).startswith("datetime64")
    assert out["datetime"].is_monotonic_increasing
    assert float(out.loc[0, "open"]) == 2.0  # 2026-09-01 在前
    assert float(out.loc[1, "open"]) == 1.0


def test_normalize_bars_chinese_columns() -> None:
    """akshare 中文列名应正确映射。"""
    raw = pd.DataFrame(
        {
            "日期": ["2026-09-01", "2026-09-02"],
            "开盘": [10.0, 11.0],
            "收盘": [10.5, 11.5],
            "最高": [10.6, 11.6],
            "最低": [9.9, 10.9],
            "成交量": [1000, 2000],
            "成交额": [10500, 23000],
        }
    )
    out = normalize_bars(raw)
    assert list(out.columns) == BAR_COLUMNS
    assert float(out.loc[0, "close"]) == 10.5
    assert float(out.loc[1, "volume"]) == 2000.0


def test_normalize_bars_sorts_and_dedups() -> None:
    """乱序输入应升序排列，重复时间戳应去重（保留最后一条）。"""
    raw = pd.DataFrame(
        {
            "日期": ["2026-09-03", "2026-09-01", "2026-09-02", "2026-09-02"],
            "开盘": [3, 1, 2, 20],
            "收盘": [3, 1, 2, 20],
            "最高": [3, 1, 2, 20],
            "最低": [3, 1, 2, 20],
            "成交量": [3, 1, 2, 20],
            "成交额": [3, 1, 2, 20],
        }
    )
    out = normalize_bars(raw)
    assert len(out) == 3
    assert out["datetime"].is_monotonic_increasing
    # 2026-09-02 的两条中保留最后一条（open=20）
    row = out[out["datetime"] == pd.Timestamp("2026-09-02")]
    assert float(row["open"].iloc[0]) == 20.0


def test_normalize_bars_missing_columns_raises() -> None:
    """缺必需列应抛 ValueError。"""
    with pytest.raises(ValueError):
        normalize_bars(pd.DataFrame({"日期": ["2026-09-01"], "开盘": [1.0]}))


def test_normalize_bars_requires_dataframe() -> None:
    """非 DataFrame 入参应抛 TypeError。"""
    with pytest.raises(TypeError):
        normalize_bars([1, 2, 3])  # type: ignore[arg-type]


def test_normalize_bars_empty_returns_contract() -> None:
    """空输入返回契约列空表。"""
    out = normalize_bars(pd.DataFrame())
    assert list(out.columns) == BAR_COLUMNS
    assert out.empty


# --------------------------------------------------------------------------- #
# derive_market
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("600000", "sh"),
        ("688981", "sh"),
        ("900901", "sh"),
        ("000001", "sz"),
        ("300750", "sz"),
        ("200011", "sz"),
        ("830799", "bj"),
        ("430047", "bj"),
        ("sh600000", "sh"),
        ("600000.SH", "sh"),
        ("SZ000001", "sz"),
    ],
)
def test_derive_market(code: str, expected: str) -> None:
    """市场推导规则（60/68/9→sh，00/30/2→sz，8/4/43→bj）。"""
    assert derive_market(code) == expected


# --------------------------------------------------------------------------- #
# is_index_symbol
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("sh.000300", True),      # 沪深300（P7-S1 默认基准）
        ("SH.000300", True),      # 大小写不敏感
        ("sz.399001", True),      # 深证成指
        ("sh.950090", True),      # 中证细分指数
        ("sh.600000", False),     # 普通沪市股票
        ("000300", False),        # 裸代码不判指数（避免与深市股票混淆）
        ("000001", False),
        ("sh.00030", False),      # 位数不足
        ("sh.000300x", False),    # 多余字符
        ("", False),
    ],
)
def test_is_index_symbol(code: str, expected: bool) -> None:
    """指数识别需同时满足「显式市场前缀 + 6 位数字 + 常见指数号段」。"""
    assert is_index_symbol(code) is expected


# --------------------------------------------------------------------------- #
# DataSourceError
# --------------------------------------------------------------------------- #
def test_datasource_error_attributes() -> None:
    """异常应保留 source / original / message。"""
    original = ValueError("boom")
    err = DataSourceError("akshare", original)
    assert err.source == "akshare"
    assert err.original is original
    assert "akshare" in str(err)
    assert "boom" in str(err)

    err2 = DataSourceError("akshare", original, message="自定义")
    assert err2.message == "自定义"


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
class _DummySource:
    """用于注册表测试的最小实现。"""

    name = "dummy"

    def list_stocks(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["code", "name", "market"])

    def get_kline(self, symbol, period, start, end, adjust=""):
        return pd.DataFrame(columns=BAR_COLUMNS)

    def get_minute(self, symbol, date):
        return pd.DataFrame(columns=BAR_COLUMNS)

    def get_fundamental(self, symbol):
        return {}

    def get_spot(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_adj_factor(self, symbol, start, end):
        return pd.DataFrame(columns=["datetime", "factor"])


def test_register_and_get_source() -> None:
    """注册后可按名获取；未知名称抛 KeyError 并给出可用源。"""
    @register_source("dummy_test")
    def _factory() -> _DummySource:
        return _DummySource()

    source = get_source("dummy_test")
    assert isinstance(source, _DummySource)
    assert "dummy_test" in SOURCES

    with pytest.raises(KeyError) as excinfo:
        get_source("不存在的数据源")
    assert "可用数据源" in str(excinfo.value)

    # 清理注册表，避免影响其它用例
    SOURCES.pop("dummy_test", None)


def test_datasource_protocol_runtime_checkable() -> None:
    """结构化协议：实现了全部方法的对象应被 isinstance 认可。"""
    assert isinstance(_DummySource(), DataSource)
    assert not isinstance(object(), DataSource)
