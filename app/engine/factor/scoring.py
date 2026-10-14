"""多因子打分合成与 Top-N 选股（P7-S5）。

定位：
    P7-S3 回答「哪些因子单独有效」，P7-S4 解决「研究池怎么选取才无幸存者
    偏差」。本模块把这两步落地成**可直接用于选股的组合打分**：把选定的几个
    因子按「ICIR 动态加权」合成为一个综合分，每个交易日按得分取 Top-N，
    供 :mod:`app.engine.backtest` 验证「照这个分买，钱能不能赚」。

    * :func:`cross_section_zscore`  —— 每日横截面 z-score（+ 截断），把不同
      量纲的因子拉到同一把尺子上；
    * :func:`dynamic_icir_weights`  —— **扩展窗口** ICIR 权重：只用「截至
      当日已经完全可实现」的历史 IC 估计（含前瞻收益的兑现滞后），
      绝不用全样本 IC——否则等于拿未来数据给自己打分（in-sample 乐观）；
    * :func:`combine_scores`        —— 加权合成出打分面板；
    * :func:`top_picks`             —— 每个交易日按得分取 Top-N 名单；
    * :class:`FactorScorer`         —— 编排入口（读库、PIT 掩码、权重、打分）。

为什么用 ICIR 加权而不是等权：
    等权隐含「每个因子一样可信」，实际不同因子的预测能力与稳定性差异很大。
    ICIR = IC 均值 / IC 标准差——不只问「平均对多少」，还问「对得稳不稳」，
    更稳的因子获得更大话语权。IC 为负（如低成交额异象）时 ICIR 为负 →
    权重自动取负号 → 高得分落在「因子值低」的一侧，方向无需人工指定。

前视偏差的三道闸门（与项目统一口径一致）：
    1. 因子值只用 ≤ 当日数据（见 :mod:`app.engine.factor.factors` 的因果约束）；
    2. 信号在 T 日收盘产生、T+1 开盘成交（回测引擎口径）；
    3. **权重估计也只用历史**：基于 T+1 开盘前瞻收益的 IC，在「卖出日
       （T+1+前瞻期）收盘」之后才完全可知，故某日 IC 最早从
       ``该日 + 前瞻期 + 1`` 个交易日之后的权重日期开始参与估计
       （见 :func:`dynamic_icir_weights` 的「兑现滞后」）。

数据缺失口径：**缺失不填 0**——参与合成（权重非零）的因子必须全部有值，
任一缺失该股当日得分记 NaN（不参与 Top-N）；权重为 0（预热 / 无效）的因子
既不参与合成、也不要求有值。

对外 API **不抛业务异常**：非法输入降级为空结果并在 ``warnings`` 说明；
编排层把数据问题封装为 ``ok=False``（与项目「引擎层容错、UI 层兜底」一致）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from app.data.store.db import Database, get_db
from app.data.universe import (
    DEFAULT_INDEX_CODE,
    membership_matrix,
    pit_members,
    universe_union,
)
from app.engine.factor.collinearity import standardize_cross_section
from app.engine.factor.factors import DEFAULT_FACTOR_NAMES, FactorDef, get_factor
from app.engine.factor.ic import rank_ic
from app.engine.factor.panel import build_panel

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 默认口径常量（集中定义，UI 与测试引用同一份）
# --------------------------------------------------------------------------- #
#: 默认参与打分的**核心三因子**（P7-S3/S4 双口径检验后确认）：
#: 低成交额（规模/流动性代理，IC 负）、短期反转（ret_5，IC 负）、
#: 距一年高点远（dist_high_250，IC 正）。
DEFAULT_SCORING_FACTORS: tuple[str, ...] = ("log_amt_20", "ret_5", "dist_high_250")

#: 权重估计所用前瞻期（交易日）：默认 20 日 ≈ 一个自然月。
DEFAULT_IC_HORIZON: int = 20

#: 扩展窗口最少 IC 样本数：不足则该因子权重暂记 0（预热期不建仓）。
DEFAULT_MIN_IC_DAYS: int = 120

#: 横截面 z-score 截断边界（±3σ）：防止个别极端值主导当日得分。
DEFAULT_Z_CLIP: float = 3.0

#: 「最新权重」表契约列（顺序固定，便于 UI 逐列展示）。
WEIGHT_TABLE_COLUMNS: list[str] = [
    "factor", "label", "n_ic_days", "ic_mean", "ic_std", "ic_ir", "weight",
]


# --------------------------------------------------------------------------- #
# 基础：横截面标准化
# --------------------------------------------------------------------------- #
def cross_section_zscore(
    panel: pd.DataFrame,
    clip: float = DEFAULT_Z_CLIP,
) -> pd.DataFrame:
    """每日横截面 z-score 标准化 + 截断（**不抛业务异常**）。

    复用共线性模块的 :func:`~app.engine.factor.collinearity.standardize_cross_section`
    （口径：行内 ``(x − 均值) / 标准差``，``ddof=0``；某日标准差为 0 / 有效
    样本不足 2 个 → 该日全 NaN），再对 z 值做 ``±clip`` 截断。

    Args:
        panel: 因子面板（dates × symbols）。
        clip: 截断边界（>0 生效；``None`` / 非法值 = 不截断）。

    Returns:
        同形标准化矩阵；非法输入返回空表。
    """
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        return pd.DataFrame()
    standardized = standardize_cross_section(panel)
    if not isinstance(standardized, pd.DataFrame) or standardized.empty:
        return pd.DataFrame()
    bound: Optional[float] = None
    try:
        bound = float(clip) if clip is not None else None
    except (TypeError, ValueError):
        bound = None
    if bound is not None and bound > 0:
        standardized = standardized.clip(lower=-bound, upper=bound)
    return standardized


# --------------------------------------------------------------------------- #
# 扩展窗口 ICIR（权重估计的核心）
# --------------------------------------------------------------------------- #
def _expanding_icir(
    ic: pd.Series,
    calendar: pd.DatetimeIndex,
    horizon: int,
    min_ic_days: int,
) -> pd.DataFrame:
    """单因子的**扩展窗口** ICIR 统计（含「兑现滞后」，只用历史信息）。

    口径（严格无未来函数）：
        * IC 在评估日 d 由 ``open[d+1+horizon] / open[d+1] − 1`` 的前瞻收益
          算出——卖出发生在 d+1+horizon 开盘，该 IC 从**该日收盘**起才完全
          可知；权重日期 T 最早从 ``pos(d) + horizon + 1`` 起纳入这份 IC；
        * 每个日历位置的统计量 = 「可用位置 ≤ 当前位置」的全部 IC 的
          扩展均值 / 样本标准差（``ddof=1``）；有效样本数 < ``min_ic_days``
          → 统计量为 NaN（预热期，权重按 0 处理）；
        * 标准差为 0（IC 恒定）→ ICIR 记 NaN（无「稳定性」可言）。

    Args:
        ic: 该因子的逐日 IC 序列（索引 = 评估日，来自 :func:`~app.engine.factor.ic.rank_ic`）。
        calendar: 全局交易日历（升序去重）。
        horizon: 前瞻期（交易日，≥1）。
        min_ic_days: 最少有效 IC 样本数（≥2）。

    Returns:
        index = calendar 的 DataFrame，列 ``ic_mean / ic_std / ic_ir / n_ic``；
        非法输入返回空表。
    """
    columns = ["ic_mean", "ic_std", "ic_ir", "n_ic"]
    if not isinstance(calendar, pd.DatetimeIndex) or len(calendar) == 0:
        return pd.DataFrame(columns=columns)
    size = len(calendar)
    step = max(int(horizon), 1)
    min_days = max(int(min_ic_days), 2)

    # 1) 把 IC 值搬到「信息可用位置」：pos(d) + step + 1。
    blank = pd.DataFrame(
        {"ic": np.full(size, np.nan), "used": np.zeros(size, dtype=bool)},
        index=calendar,
    )
    series = pd.to_numeric(pd.Series(ic, dtype="float64"), errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if not series.empty:
        stamps = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce")).normalize()
        positions = calendar.get_indexer(stamps)
        valid = positions >= 0
        available = positions[valid] + step + 1
        within = available <= size - 1  # 迟到信息对任何权重日期都无效，直接丢弃
        if within.any():
            blank.iloc[available[within], 0] = series.to_numpy(dtype=float)[valid][within]
            blank.iloc[available[within], 1] = True

    # 2) 扩展均值 / 样本标准差 / 有效样本数（pandas 自动跳过 NaN）。
    expanding = blank["ic"].expanding(min_periods=min_days)
    mean = expanding.mean()
    std = expanding.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic_ir = mean / std
    ic_ir = ic_ir.replace([np.inf, -np.inf], np.nan)
    ic_ir = ic_ir.where(std > 0)  # IC 恒定（无波动）→ 无稳定性可言
    result = pd.DataFrame(
        {
            "ic_mean": mean,
            "ic_std": std,
            "ic_ir": ic_ir,
            "n_ic": blank["ic"].expanding().count(),
        }
    )
    return result


def dynamic_icir_weights(
    ic_series: dict[str, pd.Series],
    calendar: pd.DatetimeIndex,
    horizon: int = DEFAULT_IC_HORIZON,
    min_ic_days: int = DEFAULT_MIN_IC_DAYS,
) -> pd.DataFrame:
    """逐日 ICIR 权重（扩展窗口 + 归一化，**不抛业务异常**）。

    每个日历位置 T：``w_f(T) = ICIR_f(T) / Σ_g |ICIR_g(T)|``——保留正负号
    （负 IC 因子自动反向），归一化让「权重之和的绝对值 = 1」，得分量纲稳定、
    因子增减不改变其他因子的相对话语权。全部因子权重为 0（预热 / 无效）时
    该日权重全 0（下游不建仓）。

    Args:
        ic_series: ``{因子名: IC 序列}``（完整历史，窗口截取由调用方负责）。
        calendar: 全局交易日历（升序去重）。
        horizon: 前瞻期（交易日），决定「兑现滞后」时长。
        min_ic_days: 最少有效 IC 样本数。

    Returns:
        index = calendar、columns = 因子名的权重矩阵；非法输入返回空表。
    """
    if not isinstance(ic_series, dict) or not ic_series:
        return pd.DataFrame()
    if not isinstance(calendar, pd.DatetimeIndex) or len(calendar) == 0:
        return pd.DataFrame()
    raw: dict[str, pd.Series] = {}
    for name, ic in ic_series.items():
        stats = _expanding_icir(ic, calendar, horizon=horizon, min_ic_days=min_ic_days)
        if stats.empty:
            raw[str(name)] = pd.Series(0.0, index=calendar)
            continue
        raw[str(name)] = stats["ic_ir"].fillna(0.0)
    weights = pd.DataFrame(raw, index=calendar)
    total = weights.abs().sum(axis=1)
    normalized = weights.div(total.replace(0.0, np.nan), axis=0)
    return normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# --------------------------------------------------------------------------- #
# 合成打分与 Top-N
# --------------------------------------------------------------------------- #
def combine_scores(
    zscore_panels: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """加权合成打分面板：``score[T, s] = Σ_f w_f(T) · z_f(T, s)``（**不抛异常**）。

    缺失口径（缺失不填 0）：
        * 某因子当日权重为 0（预热 / 无效）→ 不参与合成、也不要求有值；
        * 某因子当日权重非零但该股因子值为 NaN → 该股当日得分记 NaN
          （宁可放弃一只，也不拿「部分因子」的分数冒充完整打分）；
        * 当日全部因子权重为 0 → 当日全部得分 NaN（不建仓）。

    面板对齐：以第一个非空因子面板为基准形状，其余因子按日期 / 标的 reindex
    补齐（缺失即 NaN）。

    Args:
        zscore_panels: ``{因子名: 标准化面板}``。
        weights: :func:`dynamic_icir_weights` 产出的权重矩阵。

    Returns:
        打分面板（dates × symbols）；非法输入返回空表。
    """
    if not isinstance(zscore_panels, dict) or not zscore_panels:
        return pd.DataFrame()
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return pd.DataFrame()
    base: Optional[pd.DataFrame] = None
    for frame in zscore_panels.values():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            base = frame
            break
    if base is None:
        return pd.DataFrame()
    index = base.index
    columns = base.columns
    if len(columns) == 0:
        return pd.DataFrame(index=index)

    total = pd.DataFrame(0.0, index=index, columns=columns)
    complete = pd.DataFrame(True, index=index, columns=columns)
    any_weight = pd.Series(False, index=index)
    for name, frame in zscore_panels.items():
        if name not in weights.columns:
            continue
        weight = (
            pd.to_numeric(weights[name], errors="coerce").reindex(index).fillna(0.0)
        )
        nonzero = weight != 0.0
        any_weight = any_weight | nonzero
        needed = np.repeat(nonzero.to_numpy()[:, None], len(columns), axis=1)
        needed_frame = pd.DataFrame(needed, index=index, columns=columns)
        aligned = frame.reindex(index=index, columns=columns)
        complete = complete & ~(needed_frame & aligned.isna())
        contribution = aligned.where(needed_frame, 0.0).fillna(0.0)
        weight_frame = pd.DataFrame(
            np.repeat(weight.to_numpy()[:, None], len(columns), axis=1),
            index=index,
            columns=columns,
        )
        total = total + contribution * weight_frame
    active = pd.DataFrame(
        np.repeat(any_weight.to_numpy()[:, None], len(columns), axis=1),
        index=index,
        columns=columns,
    )
    return total.where(complete & active)


def top_picks(score: pd.DataFrame, n: int) -> dict[pd.Timestamp, list[str]]:
    """每个交易日按得分取 Top-N 名单（**不抛业务异常**）。

    单日口径：剔除 NaN 得分 → 按「得分降序 + 代码升序（同分确定性次序）」排序
    → 取前 ``n`` 只；当日无任何有效得分则不产出该日记录（= 不建仓）。

    Args:
        score: :func:`combine_scores` 产出的打分面板。
        n: 每日选股数量（≥1）；非法值返回空 dict。

    Returns:
        ``{交易日: [标的代码, ...]}``（仅含非空名单的日子）。
    """
    picks: dict[pd.Timestamp, list[str]] = {}
    if not isinstance(score, pd.DataFrame) or score.empty:
        return picks
    try:
        top = int(n)
    except (TypeError, ValueError):
        logger.warning("Top-N 数量非法：%r，返回空名单", n)
        return picks
    if top < 1:
        return picks
    for date, row in score.iterrows():
        valid = pd.to_numeric(row, errors="coerce").dropna()
        if valid.empty:
            continue
        frame = valid.rename("score").reset_index()
        frame.columns = ["symbol", "score"]
        frame = frame.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort")
        picks[pd.Timestamp(date)] = list(frame["symbol"].head(top))
    return picks


# --------------------------------------------------------------------------- #
# 输入 / 输出容器
# --------------------------------------------------------------------------- #
@dataclass
class ScoringSpec:
    """一次多因子打分的输入参数。

    Attributes:
        symbols: 标的清单；``None`` = 本地库现有 qfq 日线标的（排除指数）；
            ``universe="pit"`` 且为 ``None`` 时 = 成分股快照并集。
        factor_names: 参与合成的因子名；``None`` = :data:`DEFAULT_SCORING_FACTORS`
            （核心三因子）。
        ic_horizon: 权重估计所用前瞻期（交易日，≥1，默认 20）。
        min_ic_days: 扩展窗口最少 IC 样本数（默认 120；预热期内权重为 0）。
        z_clip: 横截面 z 值截断边界（默认 ±3σ）。
        min_stocks: 横截面最少有效标的数（低于则该日 IC / 标准化无效，默认 10）。
        start: 打分输出窗口起始日（含）；``None`` = 不设下限。注意：权重仍
            用**全历史**估计（预热需要窗口外数据），窗口只截取输出。
        end: 打分输出窗口结束日（含）；``None`` = 不设上限。
        universe: 股票池口径——``"current"``（默认）= 当前可见标的；
            ``"pit"`` = 严格 PIT 成分股（消除幸存者偏差）。
        index_code: PIT 口径使用的指数代码（默认 :data:`DEFAULT_INDEX_CODE`）。
    """

    symbols: Optional[Sequence[str]] = None
    factor_names: Optional[Sequence[str]] = None
    ic_horizon: int = DEFAULT_IC_HORIZON
    min_ic_days: int = DEFAULT_MIN_IC_DAYS
    z_clip: float = DEFAULT_Z_CLIP
    min_stocks: int = 10
    start: Any = None
    end: Any = None
    universe: str = "current"
    index_code: str = DEFAULT_INDEX_CODE


@dataclass
class ScoreReport:
    """一次多因子打分的完整产出（**不抛异常**，问题封装为 ``ok=False`` + ``error``）。

    Attributes:
        ok: 是否成功完成打分。
        error: 失败原因（成功时为空串）。
        score: 打分面板（窗口内 dates × symbols，权重非零且因子齐全处才有值）。
        weights: 各因子权重时序 ``{因子名: Series}``（窗口内，UI 画图用）。
        latest_weights: 窗口末日的权重诊断表，契约列 :data:`WEIGHT_TABLE_COLUMNS`
            （因子 / 有效 IC 样本数 / IC 均值 · 标准差 · ICIR / 权重）。
        factors: 实际参与打分的因子定义（按注册顺序）。
        universe: 实际参与打分的标的清单。
        dates: 打分输出窗口内的交易日。
        scored_days: 至少有一只标的有有效得分的交易日数。
        ic_horizon / min_ic_days / z_clip: 本次采用的口径（回显）。
        universe_mode: 股票池口径（``"current"`` / ``"pit"``）。
        warnings: 过程警告（剔除标的、权重预热、无 IC 样本等）。
    """

    ok: bool
    error: str = ""
    score: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: dict[str, pd.Series] = field(default_factory=dict)
    latest_weights: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=WEIGHT_TABLE_COLUMNS)
    )
    factors: list[FactorDef] = field(default_factory=list)
    universe: list[str] = field(default_factory=list)
    dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    scored_days: int = 0
    ic_horizon: int = DEFAULT_IC_HORIZON
    min_ic_days: int = DEFAULT_MIN_IC_DAYS
    z_clip: float = DEFAULT_Z_CLIP
    universe_mode: str = "current"
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 编排入口
# --------------------------------------------------------------------------- #
class FactorScorer:
    """多因子打分编排器（只读本地库，不依赖 Qt）。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`~app.data.store.db.get_db`。
        """
        self._db: Database = db if db is not None else get_db()

    # ------------------------------------------------------------------ #
    def run(self, spec: ScoringSpec) -> ScoreReport:
        """执行打分（**不抛业务异常**，数据问题封装为 ``ok=False``）。

        Args:
            spec: 打分参数（:class:`ScoringSpec`）。

        Returns:
            :class:`ScoreReport`。

        Raises:
            TypeError: ``spec`` 类型错误（调用方编程错误）。
            ValueError: 起止日期无法解析 / ``universe`` 取值非法（调用方输入错误）。
        """
        if not isinstance(spec, ScoringSpec):
            raise TypeError(f"spec 必须是 ScoringSpec，实际：{type(spec).__name__}")
        mode = str(spec.universe or "current").strip().lower()
        if mode not in ("current", "pit"):
            raise ValueError(f"universe 只能是 'current' 或 'pit'，实际：{spec.universe!r}")

        warnings: list[str] = []
        # 1) 参数规范化：因子、前瞻期、预热门槛、窗口。
        selected = self._resolve_factors(spec.factor_names, warnings)
        if not selected:
            return ScoreReport(
                ok=False,
                error="没有可用的因子（因子名全部无效）",
                universe_mode=mode,
                warnings=warnings,
            )
        horizon = _positive_int(spec.ic_horizon, DEFAULT_IC_HORIZON)
        min_ic_days = max(_positive_int(spec.min_ic_days, DEFAULT_MIN_IC_DAYS), 2)
        start, end = self._resolve_window(spec.start, spec.end)

        # 1b) PIT 口径：研究池 = 成分股快照并集（与 P7-S4 分析口径一致）。
        records: list[Any] = []
        symbols = spec.symbols
        if mode == "pit":
            records = self._load_constituents(spec.index_code, warnings)
            if not records:
                return ScoreReport(
                    ok=False,
                    error=f"库内没有 {spec.index_code} 成分股快照，无法启用 PIT 股票池"
                    "（请先更新成分股名单）",
                    factors=selected,
                    universe_mode=mode,
                    warnings=warnings,
                )
            if symbols is None:
                union = set(universe_union(records, spec.index_code, start=start, end=end))
                if start is not None:
                    # 窗口起点所在区间生效的那份名单也要纳入，否则该区间成员会缺席。
                    union.update(pit_members(records, start, spec.index_code))
                symbols = sorted(union)
                if not symbols:
                    return ScoreReport(
                        ok=False,
                        error="评估窗口内没有任何成分股快照，PIT 股票池为空",
                        factors=selected,
                        universe_mode=mode,
                        warnings=warnings,
                    )

        # 2) 组装面板（全历史加载：权重预热与前瞻收益需要窗口外数据）。
        panel, panel_warnings = build_panel(self._db, symbols)
        warnings.extend(panel_warnings)
        if panel is None or panel.empty:
            return ScoreReport(
                ok=False,
                error=panel_warnings[0] if panel_warnings else "无可用数据",
                factors=selected,
                universe_mode=mode,
                warnings=warnings,
            )

        # 2b) PIT 口径：套上成分掩码（非成分日 / 非成分股的因子值与前瞻收益置 NaN）。
        if mode == "pit":
            try:
                panel.mask = membership_matrix(
                    panel.dates, panel.symbols, records, spec.index_code
                )
                if not bool(panel.mask.to_numpy().any()):
                    warnings.append("PIT 掩码全为 False：所选标的与成分股名单无交集")
            except Exception:  # noqa: BLE001 - 掩码异常时降级为当前可见标的
                logger.exception("构造 PIT 成分掩码失败（降级为当前可见标的口径）")
                warnings.append("PIT 成分掩码构造失败，本次按「当前可见标的」口径计算")
                panel.mask = None

        # 3) 逐因子：IC 序列（全历史，权重估计用）+ 横截面 z 值。
        ic_by_factor: dict[str, pd.Series] = {}
        for definition in selected:
            ic = rank_ic(
                panel.factor(definition.name),
                panel.forward(horizon),
                min_stocks=spec.min_stocks,
            )
            ic_by_factor[definition.name] = ic
            if ic.empty:
                warnings.append(
                    f"[{definition.label}] 无有效 IC 样本，权重恒为 0（不参与合成）"
                )
        z_panels = {
            definition.name: cross_section_zscore(
                panel.factor(definition.name), clip=spec.z_clip
            )
            for definition in selected
        }

        # 4) 扩展窗口 ICIR 权重 → 合成打分（全历史计算，再截取输出窗口）。
        weights_full = dynamic_icir_weights(
            ic_by_factor, panel.dates, horizon=horizon, min_ic_days=min_ic_days
        )
        if weights_full.empty:
            return ScoreReport(
                ok=False,
                error="权重估计失败（日历或因子面板为空）",
                factors=selected,
                universe=panel.symbols,
                universe_mode=mode,
                warnings=warnings,
            )
        score_full = combine_scores(z_panels, weights_full)

        # 5) 截取输出窗口。
        window_dates = panel.dates
        if start is not None:
            window_dates = window_dates[window_dates >= start]
        if end is not None:
            window_dates = window_dates[window_dates <= end]
        if len(window_dates) == 0:
            return ScoreReport(
                ok=False,
                error="评估窗口内没有任何交易日",
                factors=selected,
                universe=panel.symbols,
                universe_mode=mode,
                warnings=warnings,
            )
        score = score_full.reindex(index=window_dates)
        weights = {
            name: weights_full[name].reindex(window_dates) for name in weights_full.columns
        }
        scored_days = int(score.notna().any(axis=1).sum())
        if scored_days == 0:
            return ScoreReport(
                ok=False,
                error=(
                    f"评估窗口内打分全部无效（扩展窗口 IC 预热不足："
                    f"最少需 {min_ic_days} 个 IC 样本 + {horizon} 个兑现滞后交易日，"
                    "请扩大窗口或补充历史数据）"
                ),
                factors=selected,
                universe=panel.symbols,
                dates=window_dates,
                ic_horizon=horizon,
                min_ic_days=min_ic_days,
                z_clip=spec.z_clip,
                universe_mode=mode,
                warnings=warnings,
            )
        warmup = int(len(window_dates) - scored_days)
        if warmup > 0:
            warnings.append(
                f"权重预热：窗口内前 {warmup} 个交易日无有效打分（IC 扩展窗口样本"
                f"不足 {min_ic_days} 个或兑现滞后未走完），不建仓"
            )

        latest_weights = self._latest_weight_table(
            selected, ic_by_factor, panel.dates, window_dates[-1], horizon, min_ic_days
        )

        return ScoreReport(
            ok=True,
            score=score,
            weights=weights,
            latest_weights=latest_weights,
            factors=selected,
            universe=panel.symbols,
            dates=window_dates,
            scored_days=scored_days,
            ic_horizon=horizon,
            min_ic_days=min_ic_days,
            z_clip=spec.z_clip,
            universe_mode=mode,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # 内部：诊断表与参数解析（全部不抛业务异常）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _latest_weight_table(
        selected: list[FactorDef],
        ic_by_factor: dict[str, pd.Series],
        calendar: pd.DatetimeIndex,
        last_date: pd.Timestamp,
        horizon: int,
        min_ic_days: int,
    ) -> pd.DataFrame:
        """窗口末日的权重诊断表（契约列 :data:`WEIGHT_TABLE_COLUMNS`）。"""
        rows: list[dict] = []
        stats_map: dict[str, pd.DataFrame] = {}
        for definition in selected:
            stats_map[definition.name] = _expanding_icir(
                ic_by_factor.get(definition.name, pd.Series(dtype=float)),
                calendar,
                horizon=horizon,
                min_ic_days=min_ic_days,
            )
        raw_ir = {
            name: (frame["ic_ir"] if not frame.empty else pd.Series(dtype=float))
            for name, frame in stats_map.items()
        }
        raw_frame = pd.DataFrame(raw_ir, index=calendar)
        total = raw_frame.abs().sum(axis=1)
        normalized = raw_frame.div(total.replace(0.0, np.nan), axis=0)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for definition in selected:
            stats = stats_map[definition.name]
            if stats.empty or last_date not in stats.index:
                rows.append(
                    {
                        "factor": definition.name,
                        "label": definition.label,
                        "n_ic_days": 0,
                        "ic_mean": None,
                        "ic_std": None,
                        "ic_ir": None,
                        "weight": 0.0,
                    }
                )
                continue
            row = stats.loc[last_date]
            rows.append(
                {
                    "factor": definition.name,
                    "label": definition.label,
                    "n_ic_days": int(row["n_ic"]),
                    "ic_mean": _optional_float(row["ic_mean"]),
                    "ic_std": _optional_float(row["ic_std"]),
                    "ic_ir": _optional_float(row["ic_ir"]),
                    "weight": float(normalized.loc[last_date, definition.name]),
                }
            )
        return pd.DataFrame(rows, columns=WEIGHT_TABLE_COLUMNS)

    def _load_constituents(self, index_code: str, warnings: list[str]) -> list[Any]:
        """读取库内成分股快照；失败时记警告并返回空列表（不抛异常）。"""
        try:
            return list(self._db.list_index_constituents(index_code))
        except Exception:  # noqa: BLE001 - 库异常按「无快照」处理
            logger.exception("读取成分股快照失败")
            warnings.append("读取成分股快照失败，请检查数据库连接")
            return []

    @staticmethod
    def _resolve_factors(
        names: Optional[Sequence[str]], warnings: list[str]
    ) -> list[FactorDef]:
        """解析因子名列表：过滤未知名（记警告），去重，按注册顺序排列。"""
        requested = list(names) if names else list(DEFAULT_SCORING_FACTORS)
        resolved: list[FactorDef] = []
        seen: set[str] = set()
        unknown: list[str] = []
        for name in requested:
            definition = get_factor(name)
            if definition is None:
                unknown.append(str(name))
                continue
            if definition.name in seen:
                continue
            seen.add(definition.name)
            resolved.append(definition)
        if unknown:
            warnings.append(f"无效因子名已忽略：{'、'.join(unknown)}")
        order = list(DEFAULT_FACTOR_NAMES)
        resolved.sort(
            key=lambda item: order.index(item.name) if item.name in order else len(order)
        )
        return resolved

    @staticmethod
    def _resolve_window(
        start: Any, end: Any
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """解析输出窗口；无法解析的日期抛 ``ValueError``（调用方输入错误）。"""
        start_ts: Optional[pd.Timestamp] = None
        end_ts: Optional[pd.Timestamp] = None
        if start is not None:
            parsed = pd.to_datetime(start, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(f"起始日期无法解析：{start!r}")
            start_ts = pd.Timestamp(parsed).normalize()
        if end is not None:
            parsed = pd.to_datetime(end, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(f"结束日期无法解析：{end!r}")
            end_ts = pd.Timestamp(parsed).normalize()
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError(f"起始日期晚于结束日期：{start_ts.date()} > {end_ts.date()}")
        return start_ts, end_ts


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _positive_int(value: Any, fallback: int) -> int:
    """把输入转成 ≥1 的整数；非法 / 非正取 ``fallback``。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(fallback)
    return parsed if parsed >= 1 else int(fallback)


def _optional_float(value: Any) -> Optional[float]:
    """把统计值转成 float；NaN / 非法返回 ``None``（表中显示空白）。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None