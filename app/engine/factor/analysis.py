"""多因子有效性分析编排（P7-S3）。

定位：
    把阶段 1/2 的「策略」拆成「因子」，一次性回答三个业务问题：
    ① **每个因子单独是否有效、能预测多远**——对每个因子 × 每个预测期
       （默认 5/10/20/60 日）计算 Rank IC 与显著性（:mod:`ic`）；
    ② **按因子分档买是不是一档比一档好**——分层回测与单调性（:mod:`layered`）；
    ③ **因子之间是否信息重复**——相关矩阵与 VIF（:mod:`collinearity`）。

    本模块是引擎包的**唯一编排入口**（UI 与脚本都调它），自身不抛业务异常：
    数据不足 / 参数非法 → ``FactorReport(ok=False, error=...)``。

评估窗口口径：
    * ``start`` / ``end`` 限定的是**评估窗口**（IC 与分层统计取窗口内的交易日），
      但面板仍加载全部历史——因子预热（如 60 日均线、250 日高点）与 T+1 开盘
      前瞻收益需要窗口外的前后数据，截断加载会造成假缺失与假收益；
    * 前瞻收益一律「T 日收盘算因子 → T+1 开盘买入 → T+1+N 开盘卖出」
      （详见 :meth:`~app.engine.factor.panel.FactorPanel.forward`）。

股票池口径（P7-S4，幸存者偏差处理）：
    * ``universe="current"``（默认，与 P7-S3 完全一致）：研究池 = 本地库现有 qfq
      标的——退市 / 被剔除成分股缺席，收益被系统性高估；
    * ``universe="pit"``：研究池 = 库内成分股快照并集，并给面板套上
      ``dates × symbols`` 的 **PIT 布尔掩码**——某交易日只保留「该日之前最近一份
      生效名单」里的成分股，非成分日的因子值与前瞻收益一律 NaN。早于首份名单的
      交易日严格无样本（不用未来名单，避免未来函数）。

结果表契约：
    * :data:`IC_TABLE_COLUMNS`：IC 汇总长表（因子 × 预测期逐行）；
    * :func:`ic_decay_table`：把长表旋转成「因子 × 预测期」的 IC 均值衰减表。

典型用法::

    from app.engine.factor import FactorAnalyzer, FactorAnalysisSpec

    spec = FactorAnalysisSpec(horizons=(5, 20), start="2020-01-01", end="2025-12-31")
    report = FactorAnalyzer().run(spec)
    print(report.ic_table)          # IC 汇总
    print(report.collinearity.vif)  # VIF
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
from app.engine.factor.collinearity import CollinearityResult, compute_collinearity
from app.engine.factor.factors import (
    DEFAULT_FACTOR_NAMES,
    FactorDef,
    get_factor,
)
from app.engine.factor.ic import IC_SUMMARY_COLUMNS, rank_ic, summarize_ic
from app.engine.factor.layered import LayeredResult, run_layered
from app.engine.factor.panel import FactorPanel, build_panel

logger = logging.getLogger(__name__)

# 默认预测期（交易日）：短（1 周）/ 中短（2 周）/ 中（1 月）/ 长（1 季）。
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20, 60)

# IC 汇总长表契约列（顺序固定，便于 UI 逐列展示）。
IC_TABLE_COLUMNS: list[str] = ["factor", "label", "category", "horizon"] + IC_SUMMARY_COLUMNS


# --------------------------------------------------------------------------- #
# 输入 / 输出容器
# --------------------------------------------------------------------------- #
@dataclass
class FactorAnalysisSpec:
    """一次多因子分析的输入参数。

    Attributes:
        symbols: 标的清单；``None`` = 本地库现有 qfq 日线标的（排除指数）；
            ``universe="pit"`` 且 ``symbols`` 为 ``None`` 时 = 成分股快照并集
            （评估窗口内出现过即可）。
        factor_names: 因子名列表；``None`` = 全部注册因子（12 个）。
        horizons: 预测期（交易日，≥1）；默认 :data:`DEFAULT_HORIZONS`。
        start: 评估窗口起始日（含）；``None`` = 不设下限。
        end: 评估窗口结束日（含）；``None`` = 不设上限。
        n_groups: 分层回测层数（≥2，默认 5）。
        min_stocks: 横截面最少有效标的数（低于则该日不参与，默认 10）。
        universe: 股票池口径——``"current"``（默认）= 当前可见标的（与 P7-S3 一致，
            存在幸存者偏差）；``"pit"`` = 严格 PIT 成分股（消除幸存者偏差）。
        index_code: PIT 口径使用的指数代码（默认 :data:`DEFAULT_INDEX_CODE`）。
    """

    symbols: Optional[Sequence[str]] = None
    factor_names: Optional[list[str]] = None
    horizons: Sequence[int] = DEFAULT_HORIZONS
    start: Any = None
    end: Any = None
    n_groups: int = 5
    min_stocks: int = 10
    universe: str = "current"
    index_code: str = DEFAULT_INDEX_CODE


@dataclass
class FactorReport:
    """一次多因子分析的完整产出（**不抛异常**，问题封装为 ``ok=False`` + ``error``）。

    Attributes:
        ok: 是否成功完成分析。
        error: 失败原因（成功时为空串）。
        ic_table: IC 汇总长表，契约列 :data:`IC_TABLE_COLUMNS`。
        ic_series: IC 时序 ``{(因子名, 预测期): Series}``（UI 画图用）。
        layered: 分层结果 ``{因子名: LayeredResult}``（每日轮动口径）。
        collinearity: 共线性诊断（相关矩阵 + VIF）。
        factors: 实际参与分析的因子定义（按注册顺序）。
        universe: 实际参与分析的标的清单。
        evaluation_days: 评估窗口内的交易日数。
        n_groups: 分层层数。
        universe_mode: 本次采用的股票池口径（``"current"`` / ``"pit"``）。
        warnings: 过程警告（剔除标的、无效因子名、样本不足等）。
    """

    ok: bool
    error: str = ""
    ic_table: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=IC_TABLE_COLUMNS)
    )
    ic_series: dict[tuple[str, int], pd.Series] = field(default_factory=dict)
    layered: dict[str, LayeredResult] = field(default_factory=dict)
    collinearity: Optional[CollinearityResult] = None
    factors: list[FactorDef] = field(default_factory=list)
    universe: list[str] = field(default_factory=list)
    evaluation_days: int = 0
    n_groups: int = 5
    universe_mode: str = "current"
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 结果表辅助
# --------------------------------------------------------------------------- #
def ic_decay_table(ic_table: pd.DataFrame) -> pd.DataFrame:
    """把 IC 汇总长表旋转成「因子 × 预测期」的 IC 均值衰减表。

    Args:
        ic_table: :data:`IC_TABLE_COLUMNS` 契约长表。

    Returns:
        行 = 因子中文名（含类别），列 = 预测期（如 ``"IC@5日"``）的宽表；
        输入非法 / 空表返回空表（**不抛异常**）。
    """
    if not isinstance(ic_table, pd.DataFrame) or ic_table.empty:
        return pd.DataFrame()
    required = {"factor", "label", "horizon", "ic_mean"}
    if not required.issubset(ic_table.columns):
        return pd.DataFrame()
    data = ic_table.copy()
    data["展示名"] = data["label"] + "（" + data["factor"] + "）"
    pivot = data.pivot_table(
        index="展示名", columns="horizon", values="ic_mean", aggfunc="mean"
    )
    pivot = pivot.sort_index()
    pivot.columns = [f"IC@{int(column)}日" for column in pivot.columns]
    return pivot


# --------------------------------------------------------------------------- #
# 编排入口
# --------------------------------------------------------------------------- #
class FactorAnalyzer:
    """多因子有效性分析编排器（只读本地库，不依赖 Qt）。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`~app.data.store.db.get_db`。
        """
        self._db: Database = db if db is not None else get_db()

    # ------------------------------------------------------------------ #
    def run(self, spec: FactorAnalysisSpec) -> FactorReport:
        """执行完整分析（**不抛业务异常**，数据问题封装为 ``ok=False``）。

        Args:
            spec: 分析参数（:class:`FactorAnalysisSpec`）。

        Returns:
            :class:`FactorReport`。

        Raises:
            TypeError: ``spec`` 类型错误（调用方编程错误）。
            ValueError: 起止日期无法解析 / ``universe`` 取值非法（调用方输入错误）。
        """
        if not isinstance(spec, FactorAnalysisSpec):
            raise TypeError(f"spec 必须是 FactorAnalysisSpec，实际：{type(spec).__name__}")
        mode = str(spec.universe or "current").strip().lower()
        if mode not in ("current", "pit"):
            raise ValueError(f"universe 只能是 'current' 或 'pit'，实际：{spec.universe!r}")

        warnings: list[str] = []
        # 1) 参数规范化：因子名、预测期、评估窗口。
        selected = self._resolve_factors(spec.factor_names, warnings)
        if not selected:
            return FactorReport(
                ok=False, error="没有可分析的因子（因子名全部无效）", universe_mode=mode
            )
        horizons = self._resolve_horizons(spec.horizons)
        if not horizons:
            return FactorReport(
                ok=False,
                error="预测期列表为空或全部非法（应为 ≥1 的整数）",
                universe_mode=mode,
            )
        start, end = self._resolve_window(spec.start, spec.end)

        # 1b) PIT 口径：先取成分股快照 → 研究池 = 快照并集（补数据清单同口径）。
        records: list[Any] = []
        symbols = spec.symbols
        if mode == "pit":
            records = self._load_constituents(spec.index_code, warnings)
            if not records:
                return FactorReport(
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
                    return FactorReport(
                        ok=False,
                        error="评估窗口内没有任何成分股快照，PIT 股票池为空",
                        factors=selected,
                        universe_mode=mode,
                        warnings=warnings,
                    )

        # 2) 组装面板（全历史加载，窗口只限定评估范围）。
        panel, panel_warnings = build_panel(self._db, symbols)
        warnings.extend(panel_warnings)
        if panel is None or panel.empty:
            return FactorReport(
                ok=False,
                error=panel_warnings[0] if panel_warnings else "无可用数据",
                factors=selected,
                universe_mode=mode,
                warnings=warnings,
            )

        # 2b) PIT 口径：给面板套上成分掩码（非成分日 / 非成分股 → NaN）。
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

        evaluation_dates = panel.dates
        if start is not None:
            evaluation_dates = evaluation_dates[evaluation_dates >= start]
        if end is not None:
            evaluation_dates = evaluation_dates[evaluation_dates <= end]
        if len(evaluation_dates) == 0:
            return FactorReport(
                ok=False,
                error="评估窗口内没有任何交易日",
                factors=selected,
                universe=panel.symbols,
                universe_mode=mode,
                warnings=warnings,
            )

        # 3) IC：因子 × 预测期。
        rows: list[dict] = []
        ic_series: dict[tuple[str, int], pd.Series] = {}
        for definition in selected:
            factor_all = panel.factor(definition.name).reindex(evaluation_dates)
            for horizon in horizons:
                forward = panel.forward(horizon).reindex(evaluation_dates)
                ic = rank_ic(factor_all, forward, min_stocks=spec.min_stocks)
                summary = summarize_ic(ic, horizon=horizon)
                rows.append(
                    {
                        "factor": definition.name,
                        "label": definition.label,
                        "category": definition.category,
                        "horizon": horizon,
                        "n_days": summary.n_days,
                        "ic_mean": summary.ic_mean,
                        "ic_std": summary.ic_std,
                        "ic_ir": summary.ic_ir,
                        "ic_win_rate": summary.ic_win_rate,
                        "t_stat": summary.t_stat,
                        "t_stat_nw": summary.t_stat_nw,
                        "p_value": summary.p_value,
                        "nw_lags": summary.nw_lags,
                        "significant": summary.significant,
                    }
                )
                if not ic.empty:
                    ic_series[(definition.name, horizon)] = ic
        ic_table = pd.DataFrame(rows, columns=IC_TABLE_COLUMNS)

        # 4) 分层回测（每日轮动口径，与 IC 的预测期无关）。
        forward_one_day = panel.forward(1).reindex(evaluation_dates)
        layered: dict[str, LayeredResult] = {}
        for definition in selected:
            factor_window = panel.factor(definition.name).reindex(evaluation_dates)
            result = run_layered(
                factor_window,
                forward_one_day,
                n_groups=spec.n_groups,
                min_stocks=spec.min_stocks,
            )
            layered[definition.name] = result
            for message in result.warnings[:1]:
                warnings.append(f"[{definition.label}] {message}")

        # 5) 共线性（窗口内、先横截面标准化再汇总）。
        panels = {
            definition.name: panel.factor(definition.name).reindex(evaluation_dates)
            for definition in selected
        }
        collinearity = compute_collinearity(panels)
        if collinearity.warnings:
            warnings.extend(collinearity.warnings)

        if not ic_series and all(not item.ok for item in layered.values()):
            return FactorReport(
                ok=False,
                error="所有因子在评估窗口内均无有效样本（请扩大窗口或补充数据）",
                factors=selected,
                universe=panel.symbols,
                evaluation_days=int(len(evaluation_dates)),
                universe_mode=mode,
                warnings=warnings,
            )

        return FactorReport(
            ok=True,
            ic_table=ic_table,
            ic_series=ic_series,
            layered=layered,
            collinearity=collinearity if collinearity.ok else None,
            factors=selected,
            universe=panel.symbols,
            evaluation_days=int(len(evaluation_dates)),
            n_groups=spec.n_groups,
            universe_mode=mode,
            warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # 参数解析（全部不抛业务异常，非法部分降级并记警告）
    # ------------------------------------------------------------------ #
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
        names: Optional[list[str]], warnings: list[str]
    ) -> list[FactorDef]:
        """解析因子名列表：过滤未知名（记警告），保持注册顺序，去重。"""
        requested = list(names) if names else list(DEFAULT_FACTOR_NAMES)
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
        resolved.sort(key=lambda item: list(DEFAULT_FACTOR_NAMES).index(item.name)
                      if item.name in DEFAULT_FACTOR_NAMES else len(DEFAULT_FACTOR_NAMES))
        return resolved

    @staticmethod
    def _resolve_horizons(horizons: Sequence[int]) -> list[int]:
        """解析预测期列表：仅保留 ≥1 的整数，去重后升序。"""
        resolved: list[int] = []
        try:
            items = list(horizons)
        except TypeError:
            return resolved
        for item in items:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value >= 1 and value not in resolved:
                resolved.append(value)
        resolved.sort()
        return resolved

    @staticmethod
    def _resolve_window(start: Any, end: Any) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """解析评估窗口；无法解析的日期抛 ``ValueError``（调用方输入错误）。

        Returns:
            ``(start_ts, end_ts)``，``None`` 表示不设边界。
        """
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