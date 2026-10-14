"""回测绩效指标（P7-S1）。

定位：
    纯函数指标库——输入净值序列 / 成交明细，输出标准绩效指标，**不依赖
    Qt、数据库与指标引擎**，便于手算对拍单测。

设计要点：
    * 所有对外函数**不抛异常**：输入非法 / 样本不足 → 返回 ``None``
      （与项目「引擎层容错、UI 层兜底」约定一致）；
    * 年化口径统一用 ``periods_per_year``（默认 252，业内通用；A 股实际年均
      约 242 个交易日，用 252 会**略微高估**年化收益与波动，仅作横向对比口径）；
    * α/β 用最小二乘（``np.cov``/``np.var`` 手算，不引入 scipy/statsmodels）；
    * 胜率 / 盈亏比按**已卖出成交的实现盈亏**统计（未平仓浮盈不计入）。

指标口径速查（通俗解释）：
    * 总收益：期末净值 / 期初净值 − 1，区间内一共赚了百分之多少；
    * 年化收益：把区间收益按几何折算成「平均每年」，便于跨区间比较；
    * 年化波动率：日收益标准差 × √252，衡量净值上下颠簸的剧烈程度；
    * 夏普比率：每承担 1 单位波动换来的超额收益（>1 良好、>2 优秀、<0 跑输无风险利率）；
    * 最大回撤：净值从历史最高点最深的一次跌幅（≤0），决定「最难受时刻」的亏损；
    * Calmar：年化收益 / |最大回撤|，>0.5 尚可、>1 较好；
    * α（年化）：剔除大盘（β）影响后，选股带来的超额收益；
    * β：策略相对大盘的波动敏感度（1=同涨同跌、>1 放大、<1 防守）；
    * 跟踪误差：超额收益（策略−基准）的波动率，越小越「稳」；
    * 信息比率：超额收益 / 跟踪误差，衡量「在基准之上多赚」的性价比；
    * 胜率：赚钱的已平仓交易占比；盈亏比：平均盈利 / 平均亏损。

典型用法::

    from app.engine.backtest.metrics import compute_metrics

    metrics = compute_metrics(equity["nav"], equity["benchmark_nav"], trades)
    print(metrics.sharpe, metrics.max_drawdown, metrics.alpha)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 年化换算用的「每年交易日数」（业内通用 252；A 股实际年均约 242，属高估口径）。
TRADING_DAYS_PER_YEAR: int = 252

# 成交明细契约列（引擎产出、指标消费）。
TRADE_COLUMNS: list[str] = [
    "date",
    "symbol",
    "side",
    "price",
    "qty",
    "amount",
    "cost",
    "pnl",
]


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _as_float_series(value: Any) -> pd.Series:
    """把序列/列表/数组统一为 float Series（非法元素转 NaN，索引保留）。"""
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").astype(float)
    digits = pd.to_numeric(pd.Series(list(value)), errors="coerce")
    return digits.astype(float)


def _clean(series: pd.Series) -> pd.Series:
    """去 NaN / ±inf，返回纯数值序列。"""
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _align_pair(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """按索引对齐两条序列并去 NaN（用于 α/β 与跟踪误差）。"""
    frame = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    return frame["a"], frame["b"]


# --------------------------------------------------------------------------- #
# 收益 / 风险
# --------------------------------------------------------------------------- #
def nav_returns(nav: Any) -> pd.Series:
    """净值序列 → 周期收益率序列（``pct_change`` 后去 NaN/inf）。

    Args:
        nav: 净值序列（任意可转 float 的序列）。

    Returns:
        收益率 Series；样本不足 2 个点时返回空 Series（**不抛异常**）。
    """
    series = _clean(_as_float_series(nav))
    if len(series) < 2:
        return pd.Series(dtype=float)
    returns = series.pct_change()
    return _clean(returns)


def total_return(nav: Any) -> Optional[float]:
    """区间总收益：``期末净值 / 期初净值 − 1``；样本不足 / 净值非正返回 ``None``。"""
    series = _clean(_as_float_series(nav))
    if len(series) < 2:
        return None
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    if first <= 0 or last <= 0:
        return None
    return last / first - 1.0


def annual_return(nav: Any, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> Optional[float]:
    """年化收益（几何）：``(1 + 总收益) ** (每年周期数 / 持有周期数) − 1``。

    Notes:
        持有周期数取 ``len(nav) − 1``（净值点数减一）。样本不足 1 个收益周期返回 ``None``。
    """
    series = _clean(_as_float_series(nav))
    n_periods = len(series) - 1
    if n_periods < 1:
        return None
    total = total_return(series)
    if total is None:
        return None
    growth = 1.0 + total
    if growth <= 0:
        return None
    try:
        return float(growth ** (float(periods_per_year) / n_periods) - 1.0)
    except (OverflowError, ValueError):  # pragma: no cover - 极端数值兜底
        return None


def annual_volatility(returns: Any, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> Optional[float]:
    """年化波动率：日收益样本标准差（ddof=1）× √每年周期数；样本不足 2 个返回 ``None``。"""
    series = _clean(_as_float_series(returns))
    if len(series) < 2:
        return None
    sd = float(series.std(ddof=1))
    if not np.isfinite(sd):
        return None
    return sd * float(np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Any,
    rf: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """夏普比率：``(日收益均值 − 日无风险利率) / 日收益标准差 × √每年周期数``。

    Args:
        returns: 周期（日）收益率序列。
        rf: **年化**无风险利率（默认 0；日频换算 ``rf / periods_per_year``）。
        periods_per_year: 每年周期数。

    Returns:
        夏普比率；样本不足 / 标准差为 0（收益恒定，比率无意义）返回 ``None``。
    """
    series = _clean(_as_float_series(returns))
    if len(series) < 2:
        return None
    sd = float(series.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    excess_mean = float(series.mean()) - float(rf) / float(periods_per_year)
    return float(excess_mean / sd * np.sqrt(periods_per_year))


def drawdown_series(nav: Any) -> pd.Series:
    """回撤序列：``净值 / 历史最高净值 − 1``（≤0，逐点）。

    Returns:
        与输入等索引的回撤 Series；样本不足 2 个点返回空 Series。
    """
    series = _clean(_as_float_series(nav))
    if len(series) < 2:
        return pd.Series(dtype=float)
    peak = series.cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = series / peak - 1.0
    return drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def max_drawdown(nav: Any) -> Optional[float]:
    """最大回撤（**≤0**，如 ``-0.25`` 表示最深亏 25%）；样本不足返回 ``None``。"""
    drawdown = drawdown_series(nav)
    if drawdown.empty:
        return None
    return float(drawdown.min())


def calmar_ratio(nav: Any, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> Optional[float]:
    """Calmar 比率：``年化收益 / |最大回撤|``；任一分量缺失或回撤为 0 返回 ``None``。"""
    annual = annual_return(nav, periods_per_year)
    mdd = max_drawdown(nav)
    if annual is None or mdd is None or mdd == 0:
        return None
    return float(annual / abs(mdd))


# --------------------------------------------------------------------------- #
# 相对基准：α / β / 跟踪误差 / 信息比率
# --------------------------------------------------------------------------- #
def alpha_beta(
    returns: Any,
    benchmark_returns: Any,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> tuple[Optional[float], Optional[float]]:
    """相对基准的 α（**年化**）与 β（最小二乘 OLS 手算）。

    模型：``策略日收益 = α_d + β × 基准日收益 + 残差``；
    α 年化采用**简单年化**（``α_d × 每年周期数``，业内通行口径）。

    Args:
        returns: 策略周期（日）收益率。
        benchmark_returns: 基准周期（日）收益率（按索引对齐）。
        periods_per_year: 每年周期数。
        rf: **年化**无风险利率（仅用于 α 的超额扣减）。

    Returns:
        ``(alpha_annual, beta)``；样本不足 / 基准方差为 0 时返回 ``(None, None)``。
    """
    pair_a, pair_b = _align_pair(_as_float_series(returns), _as_float_series(benchmark_returns))
    if len(pair_a) < 2:
        return None, None
    values_b = pair_b.to_numpy(dtype=float)
    var_b = float(np.var(values_b, ddof=1))
    if not np.isfinite(var_b) or var_b <= 0:
        return None, None
    cov = float(np.cov(pair_a.to_numpy(dtype=float), values_b, ddof=1)[0, 1])
    if not np.isfinite(cov):
        return None, None
    beta = cov / var_b
    rf_daily = float(rf) / float(periods_per_year)
    alpha_daily = float(pair_a.mean()) - rf_daily - beta * (float(pair_b.mean()) - rf_daily)
    return float(alpha_daily * periods_per_year), float(beta)


def tracking_error(
    returns: Any,
    benchmark_returns: Any,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """跟踪误差：超额收益（策略−基准）的样本标准差 × √每年周期数。"""
    pair_a, pair_b = _align_pair(_as_float_series(returns), _as_float_series(benchmark_returns))
    if len(pair_a) < 2:
        return None
    sd = float((pair_a - pair_b).std(ddof=1))
    if not np.isfinite(sd):
        return None
    return sd * float(np.sqrt(periods_per_year))


def information_ratio(
    returns: Any,
    benchmark_returns: Any,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """信息比率：``年化超额收益 / 跟踪误差``；跟踪误差为 0 或缺失时返回 ``None``。"""
    te = tracking_error(returns, benchmark_returns, periods_per_year)
    if te is None or te <= 0:
        return None
    pair_a, pair_b = _align_pair(_as_float_series(returns), _as_float_series(benchmark_returns))
    if len(pair_a) < 2:
        return None
    excess_mean = float((pair_a - pair_b).mean())
    return float(excess_mean * float(periods_per_year) / te)


# --------------------------------------------------------------------------- #
# 交易层面：胜率 / 盈亏比
# --------------------------------------------------------------------------- #
def closed_trades(trades: Optional[pd.DataFrame]) -> pd.DataFrame:
    """筛出**已平仓**（``side == 'sell'`` 且 ``pnl`` 非空）的成交记录。

    Returns:
        已平仓成交子表；无成交 / 结构不符时返回空表（保持 :data:`TRADE_COLUMNS` 列）。
    """
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    if "side" not in trades.columns:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    closed = trades[trades["side"] == "sell"]
    if "pnl" in closed.columns:
        closed = closed[closed["pnl"].notna()]
    return closed


def trade_count(trades: Optional[pd.DataFrame]) -> int:
    """已平仓交易笔数（买入建仓不计入）。"""
    return int(len(closed_trades(trades)))


def win_rate(trades: Optional[pd.DataFrame]) -> Optional[float]:
    """胜率：已平仓交易中 ``pnl > 0`` 的占比；无已平仓交易返回 ``None``。

    Notes:
        恰好打平（``pnl == 0``）不计为盈利。
    """
    closed = closed_trades(trades)
    if closed.empty:
        return None
    pnls = _clean(pd.to_numeric(closed["pnl"], errors="coerce"))
    if pnls.empty:
        return None
    return float((pnls > 0).mean())


def profit_loss_ratio(trades: Optional[pd.DataFrame]) -> Optional[float]:
    """盈亏比：``平均盈利 / |平均亏损|``；缺任一侧（全胜 / 全负）返回 ``None``。"""
    closed = closed_trades(trades)
    if closed.empty:
        return None
    pnls = _clean(pd.to_numeric(closed["pnl"], errors="coerce"))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    if wins.empty or losses.empty:
        return None
    return float(wins.mean() / abs(float(losses.mean())))


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
@dataclass
class PerformanceMetrics:
    """一次回测的绩效指标汇总（字段缺失为 ``None``，含义见模块 docstring）。"""

    total_return: Optional[float] = None          # 区间总收益
    annual_return: Optional[float] = None         # 年化收益（几何）
    annual_volatility: Optional[float] = None     # 年化波动率
    sharpe: Optional[float] = None                # 夏普比率
    max_drawdown: Optional[float] = None          # 最大回撤（≤0）
    calmar: Optional[float] = None                # 年化收益 / |最大回撤|
    alpha: Optional[float] = None                 # 年化 α（选股超额）
    beta: Optional[float] = None                  # 大盘敏感度
    tracking_error: Optional[float] = None        # 跟踪误差
    information_ratio: Optional[float] = None     # 信息比率
    benchmark_total_return: Optional[float] = None  # 基准区间总收益
    excess_return: Optional[float] = None         # 总收益 − 基准总收益
    win_rate: Optional[float] = None              # 已平仓胜率
    profit_loss_ratio: Optional[float] = None     # 盈亏比
    trade_count: int = 0                          # 已平仓交易笔数

    def to_dict(self) -> dict[str, Any]:
        """转为普通 dict（便于 UI 表格 / 报告渲染）。"""
        return asdict(self)


def compute_metrics(
    nav: Any,
    benchmark_nav: Any,
    trades: Optional[pd.DataFrame] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> PerformanceMetrics:
    """汇总一份完整绩效指标（**不抛异常**，缺失项留 ``None``）。

    Args:
        nav: 策略净值序列（建议日频、索引为交易日）。
        benchmark_nav: 基准净值序列（与 ``nav`` 同索引口径）。
        trades: 成交明细（契约列见 :data:`TRADE_COLUMNS`）；``None`` 视为无交易。
        periods_per_year: 年化换算周期数（默认 :data:`TRADING_DAYS_PER_YEAR`）。
        rf: 年化无风险利率（默认 0）。

    Returns:
        :class:`PerformanceMetrics`。
    """
    nav_series = _clean(_as_float_series(nav)) if nav is not None else pd.Series(dtype=float)
    bench_series = (
        _clean(_as_float_series(benchmark_nav))
        if benchmark_nav is not None
        else pd.Series(dtype=float)
    )
    returns = nav_returns(nav_series)
    benchmark_returns = nav_returns(bench_series)
    alpha, beta = alpha_beta(returns, benchmark_returns, periods_per_year, rf)
    total = total_return(nav_series)
    benchmark_total = total_return(bench_series)
    excess = None
    if total is not None and benchmark_total is not None:
        excess = total - benchmark_total
    return PerformanceMetrics(
        total_return=total,
        annual_return=annual_return(nav_series, periods_per_year),
        annual_volatility=annual_volatility(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, rf, periods_per_year),
        max_drawdown=max_drawdown(nav_series),
        calmar=calmar_ratio(nav_series, periods_per_year),
        alpha=alpha,
        beta=beta,
        tracking_error=tracking_error(returns, benchmark_returns, periods_per_year),
        information_ratio=information_ratio(returns, benchmark_returns, periods_per_year),
        benchmark_total_return=benchmark_total,
        excess_return=excess,
        win_rate=win_rate(trades),
        profit_loss_ratio=profit_loss_ratio(trades),
        trade_count=trade_count(trades),
    )


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "TRADE_COLUMNS",
    "PerformanceMetrics",
    "nav_returns",
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "drawdown_series",
    "max_drawdown",
    "calmar_ratio",
    "alpha_beta",
    "tracking_error",
    "information_ratio",
    "closed_trades",
    "trade_count",
    "win_rate",
    "profit_loss_ratio",
    "compute_metrics",
]