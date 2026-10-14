"""回测基准净值构造（P7-S1）。

定位：
    回答「策略跑赢大盘了吗」必须有一个公平对照物。本模块提供两种口径：

    * ``INDEX``：本地库内已有基准指数日线（默认 ``sh.000300`` 沪深300）→
      直接用其收盘价归一化为净值（指数无需复权，读 ``adjust=""``）；
    * ``EQUAL_WEIGHT_PROXY``：库内**没有指数数据**时的降级方案——用「回测股票池
      等权日收益」复合出「全市场代理」。

⚠️ 代理口径的已知偏差（结论中必须标注）：
    1. **等权 ≠ 市值加权**：沪深300 是市值加权（大金融/白酒权重高），等权代理把
       小盘股的影响放大，两者涨跌幅可能差很多；
    2. **幸存者偏差**：代理只含本地库现有股票，不含已退市股票（阶段 4 量化报告）；
    3. **停牌近似**：停牌日按前收照延，贡献 0% 收益（真实指数会把该股权重再分配）。

约定：所有对外函数**不抛异常**——读库失败 / 无数据 → 返回空 Series 或代理口径，
由引擎记录警告并如实标注 ``benchmark_kind``。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.data.store.db import Database

logger = logging.getLogger(__name__)

# 基准口径常量（回测结果 ``benchmark_kind`` 字段取值）。
BENCHMARK_KIND_INDEX: str = "INDEX"
BENCHMARK_KIND_PROXY: str = "EQUAL_WEIGHT_PROXY"


def index_nav(
    db: Database,
    symbol: str,
    calendar: pd.DatetimeIndex,
) -> Optional[pd.Series]:
    """读取库内指数日线并归一化为净值（首日 = 1.0）；无数据 / 读库失败返回 ``None``。

    Args:
        db: 存储实例（只读）。
        symbol: 指数代码（如 ``"sh.000300"``）。
        calendar: 回测交易日历（净值对齐到该索引，缺失日按前收照延）。

    Returns:
        与 ``calendar`` 对齐的净值 Series（指数数据起点之前为 NaN）；不可用时 ``None``。
    """
    if not symbol:
        return None
    try:
        bars = db.get_bars(str(symbol), "d", adjust="")
    except Exception as exc:  # noqa: BLE001 - 读库失败降级为「无基准」
        logger.warning("读取基准指数 %s 失败：%s", symbol, exc)
        return None
    if bars is None or bars.empty:
        return None
    closes = pd.Series(
        pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(bars["datetime"], errors="coerce")),
    )
    closes = closes[closes.index.notna()]
    closes = closes.replace([np.inf, -np.inf], np.nan).dropna()
    if closes.empty:
        return None
    closes = closes[~closes.index.duplicated(keep="last")].sort_index()
    aligned = closes.reindex(calendar, method="ffill")
    if aligned.notna().sum() < 2:
        return None
    base = float(aligned.dropna().iloc[0])
    if base <= 0:
        return None
    return (aligned / base).astype(float)


def equal_weight_nav(
    close_map: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
) -> Optional[pd.Series]:
    """用股票池等权日收益复合成「全市场代理」净值（首日 = 1.0）。

    Args:
        close_map: ``{symbol: 收盘价 Series}``（索引为交易日历，已前收照延）。
        calendar: 回测交易日历。

    Returns:
        与 ``calendar`` 对齐的代理净值；池为空返回 ``None``。

    Notes:
        每个交易日取「当日有收益的标的」的算术平均收益（停牌/未上市标的自动跳过），
        再逐日复利——等价于每日再平衡的等权组合。
    """
    usable = {sym: series for sym, series in (close_map or {}).items() if series is not None}
    if not usable:
        return None
    frame = pd.DataFrame(usable).reindex(calendar)
    if frame.empty:
        return None
    returns = frame.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    mean_return = returns.mean(axis=1, skipna=True).fillna(0.0)
    nav = (1.0 + mean_return).cumprod()
    nav.name = "benchmark_nav"
    return nav.astype(float)


def load_benchmark(
    db: Database,
    symbol: str,
    calendar: pd.DatetimeIndex,
    close_map: dict[str, pd.Series],
) -> tuple[pd.Series, str]:
    """加载基准净值：优先真实指数，缺失时降级为股票池等权代理。

    Args:
        db: 存储实例。
        symbol: 基准指数代码（如 ``"sh.000300"``）。
        calendar: 回测交易日历。
        close_map: 股票池收盘价矩阵（构造代理用）。

    Returns:
        ``(benchmark_nav, kind)``；``kind`` 取 :data:`BENCHMARK_KIND_INDEX` 或
        :data:`BENCHMARK_KIND_PROXY`。代理也不可用时返回全 NaN 序列 + 代理口径
        （引擎据此记警告，指标层会忽略 NaN）。
    """
    nav = index_nav(db, symbol, calendar)
    if nav is not None:
        return nav, BENCHMARK_KIND_INDEX
    proxy = equal_weight_nav(close_map, calendar)
    if proxy is None:
        logger.warning("基准 %s 无数据且股票池为空，代理基准不可用", symbol)
        return pd.Series(np.nan, index=pd.DatetimeIndex(calendar)), BENCHMARK_KIND_PROXY
    return proxy, BENCHMARK_KIND_PROXY


__all__ = [
    "BENCHMARK_KIND_INDEX",
    "BENCHMARK_KIND_PROXY",
    "index_nav",
    "equal_weight_nav",
    "load_benchmark",
]