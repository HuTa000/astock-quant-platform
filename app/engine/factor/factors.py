"""价量因子计算库（P7-S3）。

定位：
    多因子有效性分析的**最底层**——把单只股票的一段日线（OHLCV）换算成
    12 个「因子」时间序列，供 :mod:`app.engine.factor.ic`（Rank IC）、
    :mod:`app.engine.factor.layered`（分层回测）与
    :mod:`app.engine.factor.collinearity`（共线性）消费。
    **零 Qt 依赖、不依赖数据库**，纯函数，便于手算对拍单测。

设计要点：
    * **全部因子都是因果型**（只引用「当前及历史」bar）：``shift(正数)`` /
      ``rolling`` / ``ewm`` 均满足；**绝不出现** ``shift(-n)``、``center=True``
      等未来函数——这是因子分析防前视偏差的第一道闸门；
    * **预热期一律置 NaN**（``rolling`` 默认 ``min_periods=窗口``），交由上层
      按「缺失不参与横截面」处理，**不静默填 0**；
    * 因子**保持原始方向**（不预先取负号）：IC 的正负号本身就是结论
      （如 5 日涨幅 IC 为负 = 短期反转效应），方向由数据说话；
    * 对外 API **不抛业务异常**：未知因子名 / 非法输入 → 返回空 ``Series``
      并记警告日志（与项目「引擎层容错、UI 层兜底」约定一致）。

因子速查（12 个，通俗解释）：
    * ret_5 / ret_20 / ret_60：近 5/20/60 个交易日涨幅（动量）；
      正 IC = 强者恒强（动量效应），负 IC = 涨多了会跌（反转效应）；
    * ma_dev_20 / ma_dev_60：收盘价相对 20/60 日均线的乖离率
      （``close / MA − 1``）；回答「站上均线」这个信号有多强；
    * vol_20：20 日收益率标准差（波动率）——低波动异象的检验对象；
    * amp_20：20 日平均振幅 ``(high − low) / 昨收``——日内波动剧烈程度；
    * vr_5_20：量比 = 5 日均量 / 20 日均量，>1 为近期放量；
    * log_amt_20：20 日平均成交额取对数——规模 / 流动性代理
      （对数化是因为成交额跨数量级，取对数后分布更对称）；
    * rsi_14：14 日相对强弱指标（0~100），>70 常称超买、<30 常称超卖；
    * pos_20：收盘价在近 20 日高低区间中的位置（0=区间最低、1=最高）；
    * dist_high_250：收盘价距近 250 日（约一年）最高价的百分比
      （``close / 一年最高价 − 1``，≤0）；越接近 0 越靠近一年高点。

典型用法::

    from app.engine.factor.factors import compute_factors

    frame = compute_factors(bars, ["ret_20", "vol_20"])  # 列 = 因子，行 = bar
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 因子窗口常量（集中定义便于口径查阅与测试引用）。
MOMENTUM_WINDOWS: tuple[int, ...] = (5, 20, 60)
MA_WINDOWS: tuple[int, ...] = (20, 60)
VOL_WINDOW: int = 20
AMP_WINDOW: int = 20
VOLUME_RATIO_SHORT: int = 5
VOLUME_RATIO_LONG: int = 20
AMOUNT_WINDOW: int = 20
RSI_WINDOW: int = 14
POSITION_WINDOW: int = 20
HIGH_WINDOW: int = 250


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _column(bars: pd.DataFrame, column: str) -> pd.Series:
    """取 bars 中某一列并转成 float ``Series``（非法值转 NaN，索引重置）。

    索引重置为 ``RangeIndex``（与 bars 行序一一对应）——上层
    :mod:`app.engine.factor.panel` 负责再对齐到日期。

    Args:
        bars: 单只标的的日线（契约列 ``datetime/open/high/low/close/volume/amount``）。
        column: 列名。

    Returns:
        float Series；列缺失 / 非 DataFrame 输入返回空 Series。
    """
    if not isinstance(bars, pd.DataFrame) or column not in bars.columns:
        return pd.Series(dtype=float)
    return pd.Series(pd.to_numeric(bars[column], errors="coerce").to_numpy(dtype=float))


def _sanitize(series: pd.Series) -> pd.Series:
    """把 ±inf 归并成 NaN（如除零产生的 inf），保证下游统一按缺失处理。"""
    return series.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# 12 个因子计算函数（签名统一：bars -> pd.Series）
# --------------------------------------------------------------------------- #
def factor_ret_5(bars: pd.DataFrame) -> pd.Series:
    """近 5 日涨幅：``close / close.shift(5) − 1``。"""
    close = _column(bars, "close")
    return close / close.shift(5) - 1.0


def factor_ret_20(bars: pd.DataFrame) -> pd.Series:
    """近 20 日涨幅：``close / close.shift(20) − 1``。"""
    close = _column(bars, "close")
    return close / close.shift(20) - 1.0


def factor_ret_60(bars: pd.DataFrame) -> pd.Series:
    """近 60 日涨幅：``close / close.shift(60) − 1``。"""
    close = _column(bars, "close")
    return close / close.shift(60) - 1.0


def factor_ma_dev_20(bars: pd.DataFrame) -> pd.Series:
    """20 日均线乖离：``close / MA20 − 1``；>0 即「站上 20 日线」。"""
    close = _column(bars, "close")
    return close / close.rolling(MA_WINDOWS[0]).mean() - 1.0


def factor_ma_dev_60(bars: pd.DataFrame) -> pd.Series:
    """60 日均线乖离：``close / MA60 − 1``。"""
    close = _column(bars, "close")
    return close / close.rolling(MA_WINDOWS[1]).mean() - 1.0


def factor_vol_20(bars: pd.DataFrame) -> pd.Series:
    """20 日波动率：日收益率（``pct_change``）的样本标准差（``ddof=1``）。"""
    close = _column(bars, "close")
    returns = close.pct_change()
    return returns.rolling(VOL_WINDOW).std(ddof=1)


def factor_amp_20(bars: pd.DataFrame) -> pd.Series:
    """20 日平均振幅：``(high − low) / 昨收`` 的 20 日均值。"""
    high = _column(bars, "high")
    low = _column(bars, "low")
    prev_close = _column(bars, "close").shift(1)
    return ((high - low) / prev_close).rolling(AMP_WINDOW).mean()


def factor_vr_5_20(bars: pd.DataFrame) -> pd.Series:
    """量比：``5 日均量 / 20 日均量``（>1 表示近期放量）。"""
    volume = _column(bars, "volume")
    return volume.rolling(VOLUME_RATIO_SHORT).mean() / volume.rolling(
        VOLUME_RATIO_LONG
    ).mean()


def factor_log_amt_20(bars: pd.DataFrame) -> pd.Series:
    """20 日均成交额取自然对数（规模 / 流动性代理；成交额为负/零视为缺失）。"""
    amount = _column(bars, "amount")
    average = amount.where(amount > 0).rolling(AMOUNT_WINDOW).mean()
    return pd.Series(np.log(average.to_numpy(dtype=float)), index=average.index)


def _wilder_smoothed(values: pd.Series, window: int) -> pd.Series:
    """经典 Wilder 平滑：首值 = 前 ``window`` 个样本的简单平均（SMA 种子），
    其后递归 ``avg_t = ((window − 1) · avg_{t−1} + x_t) / window``。

    实现技巧：``ewm(alpha=1/window, adjust=False)`` 本身即该递归（但首值取
    第一个样本，预热期会带偏），故先算 ewm，再在种子位置之后按
    ``(1−α)^(t−k)`` 修正两条序列的记忆偏差，避免逐点 Python 循环。
    样本不足 ``window`` 个 → 全 NaN；仅支持前置连续缺失（因子序列口径）。
    """
    series = values.reset_index(drop=True)
    alpha = 1.0 / float(window)
    ewm = series.ewm(alpha=alpha, adjust=False).mean()
    valid_positions = np.flatnonzero(series.notna().to_numpy())
    if valid_positions.size < window:
        return pd.Series(np.nan, index=values.index, dtype=float)
    first_position = int(valid_positions[0])
    seed_position = first_position + window - 1
    seed = float(series.iloc[first_position : seed_position + 1].mean())
    base = float(ewm.iloc[seed_position])
    positions = np.arange(len(series))
    decay = np.where(
        positions >= seed_position,
        (1.0 - alpha) ** np.maximum(positions - seed_position, 0),
        np.nan,
    )
    corrected = ewm.to_numpy(dtype=float) + (seed - base) * decay
    return pd.Series(corrected, index=values.index, dtype=float)


def factor_rsi_14(bars: pd.DataFrame) -> pd.Series:
    """14 日 RSI（Wilder 平滑）：``100 − 100 / (1 + 平均涨幅 / 平均跌幅)``。

    首值 = 前 14 个涨跌幅的简单平均（经典 Wilder 口径），其后按
    ``((14 − 1) · 均值 + 当日) / 14`` 递归平滑。
    边界口径：窗口内**无下跌**（平均跌幅为 0）→ RSI = 100；完全无波动
    （涨跌幅均为 0）→ RSI = 50；预热期（不足 14 个收益样本）→ NaN。
    """
    close = _column(bars, "close")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    average_gain = _wilder_smoothed(gain, RSI_WINDOW)
    average_loss = _wilder_smoothed(loss, RSI_WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = average_gain / average_loss
    rsi = 100.0 - 100.0 / (1.0 + ratio)
    # 平均跌幅为 0 时比值发散：有涨幅记 100、完全无波动记 50；
    # 预热期（平均跌幅为 NaN）不满足 !=0 的替换条件，保持 NaN。
    flat = pd.Series(
        np.where(average_gain.to_numpy(dtype=float) > 0, 100.0, 50.0),
        index=rsi.index,
        dtype=float,
    )
    return rsi.where(average_loss != 0, flat)


def factor_pos_20(bars: pd.DataFrame) -> pd.Series:
    """20 日区间位置：``(close − 20 日最低) / (20 日最高 − 20 日最低)``，∈ [0, 1]。

    区间上下界同为一天（高低价相等）时分母为 0 → 置 NaN（无「位置」可言）。
    """
    close = _column(bars, "close")
    lowest = _column(bars, "low").rolling(POSITION_WINDOW).min()
    highest = _column(bars, "high").rolling(POSITION_WINDOW).max()
    return (close - lowest) / (highest - lowest)


def factor_dist_high_250(bars: pd.DataFrame) -> pd.Series:
    """距一年高点：``close / max(high, 250 日) − 1``（≤0，越接近 0 越靠近高点）。"""
    close = _column(bars, "close")
    return close / _column(bars, "high").rolling(HIGH_WINDOW).max() - 1.0


# --------------------------------------------------------------------------- #
# 因子注册表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FactorDef:
    """一个因子的元信息 + 计算函数（注册表条目，UI 展示也直接读它）。

    Attributes:
        name: 英文标识（如 ``"ret_20"``，面板列名 / 结果表主键）。
        label: 中文名（如 ``"近20日涨幅"``，UI 展示）。
        category: 类别（动量 / 趋势 / 波动 / 量能 / 位置 / 超买超卖）。
        description: 通俗解释（UI 提示与文档共用）。
        func: 计算函数，签名 ``(bars: DataFrame) -> Series``，因果型。
    """

    name: str
    label: str
    category: str
    description: str
    func: Callable[[pd.DataFrame], pd.Series]


# 注册表（**保持插入顺序** = 默认分析顺序 = UI 勾选顺序）。
FACTOR_REGISTRY: dict[str, FactorDef] = {
    item.name: item
    for item in (
        FactorDef(
            "ret_5", "近5日涨幅", "动量",
            "近 5 个交易日涨幅（close/close.shift(5)−1）；正 IC 为动量、负 IC 为短期反转",
            factor_ret_5,
        ),
        FactorDef(
            "ret_20", "近20日涨幅", "动量",
            "近 20 个交易日涨幅；中期动量效应检验",
            factor_ret_20,
        ),
        FactorDef(
            "ret_60", "近60日涨幅", "动量",
            "近 60 个交易日涨幅；长期动量效应检验",
            factor_ret_60,
        ),
        FactorDef(
            "ma_dev_20", "20日均线乖离", "趋势",
            "close/MA20−1；>0 即「站上 20 日线」，回答该信号有多强",
            factor_ma_dev_20,
        ),
        FactorDef(
            "ma_dev_60", "60日均线乖离", "趋势",
            "close/MA60−1；中期趋势偏离程度",
            factor_ma_dev_60,
        ),
        FactorDef(
            "vol_20", "20日波动率", "波动",
            "日收益率 20 日样本标准差；低波动异象检验对象",
            factor_vol_20,
        ),
        FactorDef(
            "amp_20", "20日平均振幅", "波动",
            "(high−low)/昨收 的 20 日均值；日内波动剧烈程度",
            factor_amp_20,
        ),
        FactorDef(
            "vr_5_20", "量比(5/20)", "量能",
            "5 日均量 / 20 日均量；>1 表示近期放量",
            factor_vr_5_20,
        ),
        FactorDef(
            "log_amt_20", "20日均额对数", "量能",
            "20 日平均成交额取对数；规模/流动性代理，越大越是大盘股",
            factor_log_amt_20,
        ),
        FactorDef(
            "rsi_14", "14日RSI", "超买超卖",
            "14 日相对强弱（0~100）；>70 超买、<30 超卖",
            factor_rsi_14,
        ),
        FactorDef(
            "pos_20", "20日区间位置", "位置",
            "收盘价在近 20 日高低区间的位置（0=最低、1=最高）",
            factor_pos_20,
        ),
        FactorDef(
            "dist_high_250", "距一年高点", "位置",
            "close/近 250 日最高价−1（≤0）；越接近 0 越靠近一年高点",
            factor_dist_high_250,
        ),
    )
}

# 默认参与分析的因子（全量，按注册顺序）。
DEFAULT_FACTOR_NAMES: list[str] = list(FACTOR_REGISTRY.keys())


# --------------------------------------------------------------------------- #
# 对外 API
# --------------------------------------------------------------------------- #
def factor_names() -> list[str]:
    """全部注册因子名（按注册顺序）。"""
    return list(FACTOR_REGISTRY.keys())


def get_factor(name: str) -> Optional[FactorDef]:
    """按名取因子定义；未知返回 ``None``（不抛异常）。"""
    return FACTOR_REGISTRY.get(str(name))


def compute_factor(name: str, bars: pd.DataFrame) -> pd.Series:
    """计算单个因子（**不抛业务异常**）。

    Args:
        name: 因子名（:data:`FACTOR_REGISTRY` 键）。
        bars: 单只标的日线（按时间升序，契约列同数据库）。

    Returns:
        与 ``bars`` 行序对齐的因子值 Series（预热期 / 非法值为 NaN）；
        未知因子名 / 空输入返回空 Series。
    """
    definition = get_factor(name)
    if definition is None:
        logger.warning("未知因子名：%s", name)
        return pd.Series(dtype=float)
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return pd.Series(dtype=float)
    try:
        return _sanitize(definition.func(bars).astype(float))
    except Exception:  # noqa: BLE001 - 计算异常按缺失处理，不向调用方抛
        logger.exception("因子 %s 计算异常（按缺失处理）", name)
        return pd.Series(dtype=float)


def compute_factors(
    bars: pd.DataFrame,
    names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """批量计算多个因子（列 = 因子名，行 = bar 序号）。

    Args:
        bars: 单只标的日线。
        names: 因子名列表；``None`` 取 :data:`DEFAULT_FACTOR_NAMES`（全量）。

    Returns:
        因子矩阵 DataFrame；输入非法 / 全部因子计算失败时返回空表（**不抛异常**）。
    """
    selected = list(names) if names else list(DEFAULT_FACTOR_NAMES)
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return pd.DataFrame(columns=selected)
    data: dict[str, pd.Series] = {}
    for name in selected:
        series = compute_factor(name, bars)
        if not series.empty:
            data[name] = series
    return pd.DataFrame(data)