"""牛 / 熊市场环境拆分（P7-S1）。

定位：
    「一个策略整体赚钱」不等于「一直好用」。本模块把回测区间按**基准自身**的
    牛熊状态切开，分别统计策略/基准/超额收益，回答：**策略在牛市里是不是纯靠
    大盘（β）？在熊市里抗不抗跌（防回撤能力）？**

口径：基准净值相对其 ``window`` 日均线——在上方记「牛市」、下方记「熊市」
    （``window`` 默认 200 个交易日 ≈ 年线；样本不足时用 ``min_periods=1`` 优雅降级）。

⚠️ 重要提醒：牛熊标签是**事后归因标签**（用到了「当天及之前」的均线，本可实时
    判定，但做的是「事后分类统计」），**不是交易信号**——用它回测出的择时收益会
    有争议（临界点判断、未来函数的边界），本模块仅用于**把区间切开做归因**。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 环境标签常量（英文键，供程序判断）。
REGIME_BULL: str = "bull"
REGIME_BEAR: str = "bear"

# 展示用中文标签。
REGIME_LABELS: dict[str, str] = {REGIME_BULL: "牛市", REGIME_BEAR: "熊市"}

# 结果表契约列。
REGIME_COLUMNS: list[str] = [
    "regime",
    "days",
    "share",
    "strategy_return",
    "benchmark_return",
    "excess_return",
]

# 默认均线窗口（约年线）。
DEFAULT_REGIME_WINDOW: int = 200


def classify_regime(
    benchmark_nav: Any,
    window: int = DEFAULT_REGIME_WINDOW,
) -> pd.Series:
    """按「基准净值 vs 其 N 日均线」逐日打牛/熊标签。

    Args:
        benchmark_nav: 基准净值/价格序列（索引为交易日）。
        window: 均线窗口（默认 200）。

    Returns:
        与输入等索引的标签 Series（值为 :data:`REGIME_BULL` / :data:`REGIME_BEAR`）；
        样本不足 / 全 NaN 时返回空 Series（**不抛异常**）。
    """
    if benchmark_nav is None:
        return pd.Series(dtype=object)
    series = pd.to_numeric(pd.Series(benchmark_nav), errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return pd.Series(dtype=object)
    window_size = max(1, int(window))
    moving_average = series.rolling(window_size, min_periods=1).mean()
    labels = pd.Series(
        np.where(series >= moving_average, REGIME_BULL, REGIME_BEAR),
        index=series.index,
        dtype=object,
    )
    return labels


def regime_summary(
    nav: Any,
    benchmark_nav: Any,
    regime: Any,
) -> pd.DataFrame:
    """按牛/熊分组统计策略、基准与超额收益（复利口径）。

    Args:
        nav: 策略净值序列。
        benchmark_nav: 基准净值序列（同索引口径）。
        regime: :func:`classify_regime` 输出的标签序列。

    Returns:
        契约列为 :data:`REGIME_COLUMNS` 的 DataFrame（中文环境名）：
        各组交易日数、占比、区间收益、基准收益、超额收益；无有效数据时返回空表。
    """
    if regime is None:
        return pd.DataFrame(columns=REGIME_COLUMNS)
    frame = pd.DataFrame(
        {
            "nav": pd.to_numeric(pd.Series(nav), errors="coerce"),
            "bench": pd.to_numeric(pd.Series(benchmark_nav), errors="coerce"),
            "regime": pd.Series(regime),
        }
    ).dropna(subset=["nav", "regime"])
    if frame.empty:
        return pd.DataFrame(columns=REGIME_COLUMNS)
    frame["ret"] = frame["nav"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["bench_ret"] = frame["bench"].pct_change().replace([np.inf, -np.inf], np.nan)

    total_days = int(len(frame))
    rows: list[dict] = []
    for label in (REGIME_BULL, REGIME_BEAR):
        group = frame[frame["regime"] == label]
        if group.empty:
            continue
        strategy_return = float((1.0 + group["ret"]).prod() - 1.0)
        bench_valid = group["bench_ret"].dropna()
        benchmark_return: Optional[float] = (
            float((1.0 + bench_valid).prod() - 1.0) if not bench_valid.empty else None
        )
        rows.append(
            {
                "regime": REGIME_LABELS.get(label, label),
                "days": int(len(group)),
                "share": float(len(group)) / total_days if total_days else None,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": (
                    strategy_return - benchmark_return
                    if benchmark_return is not None
                    else None
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=REGIME_COLUMNS)
    return pd.DataFrame(rows, columns=REGIME_COLUMNS)


__all__ = [
    "REGIME_BULL",
    "REGIME_BEAR",
    "REGIME_LABELS",
    "REGIME_COLUMNS",
    "DEFAULT_REGIME_WINDOW",
    "classify_regime",
    "regime_summary",
]