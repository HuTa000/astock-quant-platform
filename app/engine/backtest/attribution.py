"""绩效归因（P7-S2）。

定位：
    回测引擎（P7-S1）回答了「赚了多少、最坏亏多少」，本模块回答**「钱从哪来」**：
    * :func:`jensen_decomposition` —— 把区间收益拆成「跟着大盘涨的 β 部分」与
      「选股赚的 α 部分」（Jensen 分解），回答「是本事还是运气」；
    * :func:`yearly_summary` —— 逐年拆解策略 / 基准 / 超额收益与年内最深回撤，
      防止「一段大牛市拉高整体平均」掩盖路径问题；
    * :func:`drawdown_episodes` —— 把最大回撤从「一个数字」升级为
      「峰值日 → 谷底日 → 修复日 + 下跌 / 修复交易日数」，对应「能不能拿住」。

口径（通俗解释）：
    * α（年化）：剔除大盘影响后，选股规则本身贡献的年化超额收益；
    * β：大盘每涨 1%，策略平均涨 β%（<1 防守、>1 进攻）；
    * β 分量 = ``(β − 1) × 基准年化``：「敏感度偏离大盘」带来的差异
      （β < 1 而大盘上涨时该项为负——防守是有代价的）；
    * α 分量 = 年化 α 本身；
    * 残差 = ``年化超额 − β 分量 − α 分量``：日频回归（算术）与复利之间的
      近似误差 + 噪声，正常应远小于 α 分量。

⚠️ 口径说明：
    * 三块贡献是**算术年化近似**（日频 OLS 的线性分解，复利下有微小残差），
      用于「结构判断」（α 还是 β 主导），不是精确会计恒等式；
    * 全部函数**不抛异常**：输入非法 / 样本不足 → 返回空表或 ``None`` 字段
      （与项目「引擎层容错、UI 层兜底」约定一致）。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.engine.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    alpha_beta,
    annual_return,
    drawdown_series,
    nav_returns,
)

logger = logging.getLogger(__name__)

# 逐年表现表契约列。
YEARLY_COLUMNS: list[str] = [
    "year",
    "days",
    "strategy_return",
    "benchmark_return",
    "excess_return",
    "strategy_max_dd",
    "benchmark_max_dd",
]

# 回撤区间表契约列。
DRAWDOWN_COLUMNS: list[str] = [
    "peak_date",
    "trough_date",
    "recovery_date",
    "depth",
    "drawdown_days",
    "recovery_days",
    "recovered",
]

# 归因分解结果的键（顺序固定，便于 UI 逐行展示）。
DECOMPOSITION_KEYS: tuple[str, ...] = (
    "strategy_annual",
    "benchmark_annual",
    "excess_annual",
    "alpha_annual",
    "beta",
    "beta_component",
    "alpha_component",
    "residual",
)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _equity_frame(equity: Any) -> pd.DataFrame:
    """把回测净值表标准化为「DatetimeIndex + nav[/benchmark_nav]」的 DataFrame。

    Args:
        equity: :class:`~app.engine.backtest.engine.BacktestResult` 的 ``equity``
            （契约列 ``date/nav/benchmark_nav/drawdown``）；非法输入返回空表。

    Returns:
        按日期升序、去 NaT 的 DataFrame；无 ``nav`` 列 / 非法输入返回空表。
    """
    if not isinstance(equity, pd.DataFrame) or "nav" not in equity.columns:
        return pd.DataFrame()
    if "date" in equity.columns:
        stamps = pd.to_datetime(equity["date"], errors="coerce")
    else:
        stamps = pd.to_datetime(equity.index, errors="coerce")
    frame = pd.DataFrame(index=pd.DatetimeIndex(stamps))
    frame["nav"] = pd.to_numeric(equity["nav"], errors="coerce").to_numpy(dtype=float)
    if "benchmark_nav" in equity.columns:
        frame["benchmark_nav"] = pd.to_numeric(
            equity["benchmark_nav"], errors="coerce"
        ).to_numpy(dtype=float)
    frame = frame[~frame.index.isna()]
    return frame.sort_index()


def _nav_series(nav: Any) -> pd.Series:
    """把净值输入统一为 float Series（保留原索引，去 NaN / ±inf）。"""
    if isinstance(nav, pd.Series):
        series = pd.to_numeric(nav, errors="coerce").astype(float)
    else:
        raw = pd.Series(dtype=float) if nav is None else pd.to_numeric(pd.Series(list(nav)), errors="coerce")
        series = raw.astype(float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _opt_min(series: pd.Series) -> Optional[float]:
    """序列最小值；全 NaN / 空序列返回 ``None``。"""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min())


# --------------------------------------------------------------------------- #
# 逐年表现
# --------------------------------------------------------------------------- #
def yearly_summary(equity: Any) -> pd.DataFrame:
    """逐年收益与年内最深回撤（策略 vs 基准）。

    收益口径：**年收益 = 该年年末净值 / 上一年年末净值 − 1**（首年从回测首日起算），
    即「把逐年收益链乘起来 ≈ 区间总收益」——便于看出「哪几年在贡献、哪几年在拖后腿」。

    回撤口径：``strategy_max_dd`` / ``benchmark_max_dd`` 为该年内净值相对
    **历史最高点**（含此前年份）的最深回撤（≤0）。

    Args:
        equity: 回测净值表（``date/nav/benchmark_nav``）。

    Returns:
        契约列 :data:`YEARLY_COLUMNS` 的 DataFrame（按年份升序）；
        无有效数据 / 输入非法返回空表（**不抛异常**）。
    """
    frame = _equity_frame(equity)
    if frame.empty or "nav" not in frame.columns:
        return pd.DataFrame(columns=YEARLY_COLUMNS)
    frame = frame.dropna(subset=["nav"])
    if frame.empty:
        return pd.DataFrame(columns=YEARLY_COLUMNS)

    frame = frame.copy()
    frame["dd"] = drawdown_series(frame["nav"])
    has_benchmark = "benchmark_nav" in frame.columns
    if has_benchmark:
        frame["bench_dd"] = drawdown_series(frame["benchmark_nav"])

    rows: list[dict] = []
    strategy_base: Optional[float] = None   # 上一年末策略净值
    benchmark_base: Optional[float] = None  # 上一年末基准净值
    for year, group in frame.groupby(frame.index.year, sort=True):
        start_nav = float(group["nav"].iloc[0])
        end_nav = float(group["nav"].iloc[-1])
        base = strategy_base if strategy_base is not None else start_nav
        strategy_return = float(end_nav / base - 1.0) if base > 0 else None

        benchmark_return: Optional[float] = None
        if has_benchmark:
            bench_group = group["benchmark_nav"].dropna()
            if not bench_group.empty:
                bench_end = float(bench_group.iloc[-1])
                bench_base = (
                    benchmark_base
                    if benchmark_base is not None
                    else float(bench_group.iloc[0])
                )
                if bench_base > 0:
                    benchmark_return = float(bench_end / bench_base - 1.0)
                benchmark_base = bench_end

        rows.append(
            {
                "year": int(year),
                "days": int(len(group)),
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": (
                    strategy_return - benchmark_return
                    if strategy_return is not None and benchmark_return is not None
                    else None
                ),
                "strategy_max_dd": _opt_min(group["dd"]),
                "benchmark_max_dd": _opt_min(group["bench_dd"]) if has_benchmark else None,
            }
        )
        strategy_base = end_nav
    return pd.DataFrame(rows, columns=YEARLY_COLUMNS)


# --------------------------------------------------------------------------- #
# 回撤区间明细
# --------------------------------------------------------------------------- #
def drawdown_episodes(nav: Any, top_n: int = 5) -> pd.DataFrame:
    """回撤区间明细（按深度从深到浅取前 ``top_n`` 段）。

    每段回撤回答三件事：**从哪里开始跌（峰值日）→ 跌到哪天见底（谷底日，最深点）
    → 哪天爬回前高（修复日）**；``drawdown_days`` / ``recovery_days`` 为
    **交易日数**（下跌历时 / 从谷底爬回前高的历时）；未修复时 ``recovery_date``
    与 ``recovery_days`` 为 ``None``、``recovered`` 为 ``False``。

    Args:
        nav: 净值序列（Series / 列表 / 数组；类型非法按空处理）。
        top_n: 最多返回的回撤段数（按 ``depth`` 升序，即最深在前）。

    Returns:
        契约列 :data:`DRAWDOWN_COLUMNS` 的 DataFrame；净值单调向上（无回撤）
        / 样本不足返回空表（**不抛异常**）。
    """
    series = _nav_series(nav)
    if len(series) < 2:
        return pd.DataFrame(columns=DRAWDOWN_COLUMNS)

    values = series.to_numpy(dtype=float)
    running_peak = np.maximum.accumulate(values)
    positions = series.index

    def _row(peak_pos: int, trough_pos: int, recovery_pos: Optional[int]) -> dict:
        """构造一段回撤记录（交易日数按位置差计）。"""
        depth = (
            float(values[trough_pos] / values[peak_pos] - 1.0)
            if values[peak_pos] > 0
            else None
        )
        return {
            "peak_date": positions[peak_pos],
            "trough_date": positions[trough_pos],
            "recovery_date": positions[recovery_pos] if recovery_pos is not None else None,
            "depth": depth,
            "drawdown_days": int(trough_pos - peak_pos),
            "recovery_days": (
                int(recovery_pos - trough_pos) if recovery_pos is not None else None
            ),
            "recovered": bool(recovery_pos is not None),
        }

    episodes: list[dict] = []
    peak_pos = 0                        # 当前历史最高点位置
    start_pos: Optional[int] = None     # 本段回撤起点（进入回撤的第一根）
    trough_pos = 0                      # 本段最深点位置
    for i in range(len(values)):
        if values[i] >= running_peak[i]:
            # 创新高（或与历史最高持平）：若此前在回撤中，则本段「已修复」收尾。
            if start_pos is not None:
                episodes.append(_row(peak_pos, trough_pos, i))
                start_pos = None
            peak_pos = i
            continue
        if start_pos is None:
            start_pos = i
            trough_pos = i
        elif values[i] < values[trough_pos]:
            trough_pos = i
    if start_pos is not None:  # 数据结束仍未修复
        episodes.append(_row(peak_pos, trough_pos, None))

    if not episodes:
        return pd.DataFrame(columns=DRAWDOWN_COLUMNS)
    frame = pd.DataFrame(episodes, columns=DRAWDOWN_COLUMNS)
    frame = frame.sort_values("depth", ascending=True, na_position="last")
    return frame.head(max(1, int(top_n))).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 收益来源分解（Jensen）
# --------------------------------------------------------------------------- #
def jensen_decomposition(
    equity: Any,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> dict[str, Optional[float]]:
    """Jensen 收益来源分解：把区间收益拆成「大盘（β）+ 选股（α）+ 残差」。

    模型：``策略日收益 = α_d + β × 基准日收益 + 残差``（日频 OLS）；
    α 年化采用简单年化（``α_d × 每年周期数``，业内通行口径）：

        * ``strategy_annual`` / ``benchmark_annual``：区间年化收益（几何）；
        * ``excess_annual``：年化超额 = 策略年化 − 基准年化（算术差）；
        * ``alpha_annual`` / ``beta``：回归截距年化 / 斜率；
        * ``beta_component = (β − 1) × 基准年化``；
        * ``alpha_component = alpha_annual``；
        * ``residual = excess_annual − beta_component − alpha_component``。

    Returns:
        键为 :data:`DECOMPOSITION_KEYS` 的 dict；缺项为 ``None``（**不抛异常**）。
    """
    result: dict[str, Optional[float]] = {key: None for key in DECOMPOSITION_KEYS}
    frame = _equity_frame(equity)
    if frame.empty:
        return result

    nav = frame["nav"].dropna()
    if nav.empty:
        return result
    result["strategy_annual"] = annual_return(nav, periods_per_year)

    if "benchmark_nav" not in frame.columns:
        return result
    benchmark = frame["benchmark_nav"].dropna()
    if benchmark.empty:
        return result
    result["benchmark_annual"] = annual_return(benchmark, periods_per_year)

    alpha_annual, beta = alpha_beta(
        nav_returns(nav),
        nav_returns(benchmark),
        periods_per_year=periods_per_year,
        rf=rf,
    )
    result["alpha_annual"] = alpha_annual
    result["beta"] = beta

    strategy_annual = result["strategy_annual"]
    benchmark_annual = result["benchmark_annual"]
    excess_annual: Optional[float] = None
    if strategy_annual is not None and benchmark_annual is not None:
        excess_annual = strategy_annual - benchmark_annual
    result["excess_annual"] = excess_annual

    beta_component: Optional[float] = None
    if beta is not None and benchmark_annual is not None:
        beta_component = (beta - 1.0) * benchmark_annual
    result["beta_component"] = beta_component
    result["alpha_component"] = alpha_annual

    if (
        excess_annual is not None
        and beta_component is not None
        and alpha_annual is not None
    ):
        result["residual"] = excess_annual - beta_component - alpha_annual
    return result


__all__ = [
    "YEARLY_COLUMNS",
    "DRAWDOWN_COLUMNS",
    "DECOMPOSITION_KEYS",
    "yearly_summary",
    "drawdown_episodes",
    "jensen_decomposition",
]