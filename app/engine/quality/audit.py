"""数据质量审计编排（P7-S4）。

定位：
    「在做任何研究结论之前，先证明数据本身可信」——本模块把六项检查
    （:mod:`app.engine.quality.checks`）串成一次完整审计：从本地库装载日线，
    跑覆盖度 / OHLC / 日历 / 异常波动 / 复权自洽 / 幸存者偏差，
    产出一份可展示、可讲业务结论的 :class:`QualityAuditReport`。

设计约定（与引擎层其它模块一致）：
    * **零 Qt**（《开发手册》§3 分层红线）；
    * **不抛业务异常**：无数据 / 单项失败 → ``ok=False`` + ``error`` / 单项 ``None``，
      其余检查照常完成（部分结果仍可用）；
    * 结果表列名一律英文（UI 层负责中文表头），与
      :mod:`app.engine.factor.analysis` 的表契约风格一致。

典型用法::

    from app.engine.quality import QualityAuditor, QualityAuditSpec

    report = QualityAuditor().run(QualityAuditSpec(start="2015-01-01"))
    print(report.survivorship.summary)   # 幸存者偏差规模
    print(report.outliers.table.head())  # 异常波动样本
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from app.data.datasource.base import is_index_symbol
from app.data.store.db import Database, get_db
from app.data.universe import DEFAULT_INDEX_CODE
from app.engine.quality.checks import (
    DEFAULT_OUTLIER_THRESHOLD,
    DEFAULT_SUSPENSION_GAP,
    AdjustResult,
    CalendarResult,
    CoverageResult,
    OhlcResult,
    OutlierResult,
    SurvivorshipResult,
    check_adjust,
    check_calendar,
    check_coverage,
    check_ohlc,
    check_outliers,
    check_survivorship,
)

logger = logging.getLogger(__name__)

#: 复权自洽对账的默认抽样数（全量对账没有必要，抽样即可发现口径混用）。
DEFAULT_SAMPLE_SIZE: int = 50

#: 市场日历候选指数（按优先级）：沪深300 → 上证综指 → 中证500。
CALENDAR_INDEX_CANDIDATES: tuple[str, ...] = ("sh.000300", "sh.000001", "sh.000905")

#: 六项检查的固定清单（报告 items 字段与 UI 展示顺序）。
AUDIT_ITEMS: tuple[str, ...] = (
    "coverage", "ohlc", "calendar", "outliers", "adjust", "survivorship",
)

#: 中文名（UI / 日志展示用）。
AUDIT_ITEM_LABELS: dict[str, str] = {
    "coverage": "覆盖度 / 停牌率",
    "ohlc": "OHLC 逻辑校验",
    "calendar": "日历一致性",
    "outliers": "异常波动",
    "adjust": "复权自洽对账",
    "survivorship": "幸存者偏差量化",
}


# --------------------------------------------------------------------------- #
# 输入 / 输出容器
# --------------------------------------------------------------------------- #
@dataclass
class QualityAuditSpec:
    """一次质量审计的输入参数。

    Attributes:
        symbols: 审计标的；``None`` = 本地库全部 qfq 日线标的（自动排除指数）。
        index_code: 用于幸存者偏差量化的指数（默认沪深300）。
        start: 评估窗口起始（含）；``None`` 不限（覆盖度按各自上市区间统计）。
        end: 评估窗口结束（含）；``None`` 不限。
        sample_size: 复权自洽对账的抽样标的数（≥1，默认 50）。
        outlier_threshold: 异常波动阈值（默认 ±21%）。
        suspension_gap: 复牌跳空判定间隔（交易日，默认 5）。
    """

    symbols: Optional[Sequence[str]] = None
    index_code: str = DEFAULT_INDEX_CODE
    start: Any = None
    end: Any = None
    sample_size: int = DEFAULT_SAMPLE_SIZE
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD
    suspension_gap: int = DEFAULT_SUSPENSION_GAP


@dataclass
class QualityAuditReport:
    """一次质量审计的完整产出（部分失败时 ``ok=False``，其余结果照常可用）。

    Attributes:
        ok: 六项检查是否全部完成（单项内部发现的问题不影响 ``ok``——
            「发现偏差」本身就是审计的目的）。
        error: 失败说明（全部完成时为空串）。
        coverage / ohlc / calendar / outliers / adjust / survivorship:
            六项检查结果；某项失败为 ``None``。
        universe: 实际参与审计的标的清单。
        calendar_source: 市场日历来源描述。
        warnings: 聚合警告（装载 + 各项检查）。
    """

    ok: bool = False
    error: str = ""
    coverage: Optional[CoverageResult] = None
    ohlc: Optional[OhlcResult] = None
    calendar: Optional[CalendarResult] = None
    outliers: Optional[OutlierResult] = None
    adjust: Optional[AdjustResult] = None
    survivorship: Optional[SurvivorshipResult] = None
    universe: list[str] = field(default_factory=list)
    calendar_source: str = ""
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 编排入口
# --------------------------------------------------------------------------- #
class QualityAuditor:
    """数据质量审计编排器（只读本地库，不依赖 Qt）。"""

    def __init__(self, db: Optional[Database] = None) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`~app.data.store.db.get_db`。
        """
        self._db: Database = db if db is not None else get_db()

    # ------------------------------------------------------------------ #
    def run(self, spec: QualityAuditSpec) -> QualityAuditReport:
        """执行六项审计（**不抛业务异常**，数据问题封装为 ``ok=False``）。

        Args:
            spec: 审计参数（:class:`QualityAuditSpec`）。

        Returns:
            :class:`QualityAuditReport`。

        Raises:
            TypeError: ``spec`` 类型错误（调用方编程错误）。
        """
        if not isinstance(spec, QualityAuditSpec):
            raise TypeError(f"spec 必须是 QualityAuditSpec，实际：{type(spec).__name__}")

        report = QualityAuditReport()
        # 1) 标的池：默认本地全部 qfq 日线（排除指数）。
        try:
            pool, pool_warnings = self._resolve_universe(spec.symbols)
        except Exception:  # noqa: BLE001 - 库异常按空池处理
            logger.exception("读取审计标的清单失败")
            report.error = "读取本地标的清单失败，请检查数据库连接"
            return report
        report.warnings.extend(pool_warnings)
        if not pool:
            report.error = "本地库中没有可审计的前复权日线标的（请先更新数据）"
            return report
        report.universe = pool

        # 2) 装载数据：全池 qfq；抽样标的额外装载未复权（复权对账与对账用）。
        qfq_bars, load_warnings = self._load_bars(pool, "qfq")
        report.warnings.extend(load_warnings)
        if not qfq_bars:
            report.error = "本地库中没有可读取的前复权日线数据"
            return report
        usable = [symbol for symbol in pool if symbol in qfq_bars]

        # 3) 市场日历：优先指数日线（真实交易日历），缺失时回退到全池日期并集。
        calendar, calendar_source = self._load_calendar(qfq_bars)
        report.calendar_source = calendar_source

        # 4) 依次执行六项检查（单项失败不影响其它项）。
        report.coverage = self._safe(
            "coverage", report.warnings,
            lambda: check_coverage(
                {s: qfq_bars[s] for s in usable}, calendar, spec.start, spec.end
            ),
        )
        report.calendar = self._safe(
            "calendar", report.warnings,
            lambda: check_calendar(
                {s: qfq_bars[s] for s in usable}, calendar, calendar_source
            ),
        )
        report.outliers = self._safe(
            "outliers", report.warnings,
            lambda: check_outliers(
                {s: qfq_bars[s] for s in usable}, calendar,
                threshold=spec.outlier_threshold, suspension_gap=spec.suspension_gap,
            ),
        )

        sample = self._sample_symbols(usable, spec.sample_size)
        raw_bars, raw_warnings = self._load_bars(sample, "")
        report.warnings.extend(raw_warnings)
        report.ohlc = self._safe(
            "ohlc", report.warnings,
            lambda: self._run_ohlc(qfq_bars, usable, raw_bars, sample),
        )
        report.adjust = self._safe(
            "adjust", report.warnings,
            lambda: check_adjust(raw_bars, qfq_bars, sample_symbols=[
                symbol for symbol in sample if symbol in raw_bars
            ]),
        )
        report.survivorship = self._safe(
            "survivorship", report.warnings,
            lambda: self._run_survivorship(spec),
        )

        # 5) 汇总。
        missing_items = [
            name for name in AUDIT_ITEMS if getattr(report, name) is None
        ]
        if missing_items:
            report.ok = False
            labels = "、".join(AUDIT_ITEM_LABELS[name] for name in missing_items)
            report.error = f"以下审计项未能完成：{labels}"
        else:
            report.ok = True
        for name in AUDIT_ITEMS:
            item = getattr(report, name)
            for message in getattr(item, "warnings", []) or []:
                report.warnings.append(f"[{AUDIT_ITEM_LABELS[name]}] {message}")
        return report

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe(name: str, warnings: list[str], fn) -> Any:
        """执行单项检查；异常降级为 ``None`` + 警告（不中断其它项）。"""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 单项失败不影响整体
            logger.exception("质量审计项 %s 失败", name)
            warnings.append(f"{AUDIT_ITEM_LABELS.get(name, name)}审计失败：{exc}")
            return None

    def _resolve_universe(
        self, symbols: Optional[Sequence[str]]
    ) -> tuple[list[str], list[str]]:
        """解析审计标的池（默认本地 qfq 标的，排除指数）。"""
        warnings: list[str] = []
        if symbols is None:
            pool = [
                symbol
                for symbol in self._db.list_kline_symbols("d", "qfq")
                if not is_index_symbol(symbol)
            ]
        else:
            pool = []
            seen: set[str] = set()
            for item in symbols:
                key = str(item or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    pool.append(key)
        if not pool:
            warnings.append("标的池为空")
        return pool, warnings

    def _load_bars(
        self, symbols: Sequence[str], adjust: str
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """批量装载日线（单标的失败只跳过并记警告）。"""
        bars: dict[str, pd.DataFrame] = {}
        warnings: list[str] = []
        for symbol in symbols:
            try:
                frame = self._db.get_bars(symbol, "d", adjust=adjust)
            except Exception:  # noqa: BLE001 - 单标的读失败不影响整体
                logger.exception("读取 %s（%s）日线失败", symbol, adjust or "未复权")
                warnings.append(f"{symbol} 日线读取失败（复权口径：{adjust or '未复权'}）")
                continue
            if frame is None or frame.empty:
                continue
            bars[symbol] = frame
        return bars, warnings

    def _load_calendar(
        self, qfq_bars: dict[str, pd.DataFrame]
    ) -> tuple[pd.DatetimeIndex, str]:
        """装载市场交易日历：优先指数日线，缺失时回退全池日期并集。"""
        for candidate in CALENDAR_INDEX_CANDIDATES:
            try:
                frame = self._db.get_bars(candidate, "d", adjust="")
            except Exception:  # noqa: BLE001 - 指数读取失败仅降级
                logger.warning("读取指数日历 %s 失败", candidate, exc_info=True)
                continue
            if frame is None or frame.empty:
                continue
            stamps = pd.DatetimeIndex(
                pd.to_datetime(frame["datetime"], errors="coerce")
            ).dropna()
            if len(stamps):
                return stamps.sort_values().unique(), candidate
        # 回退：全池日期并集（精度稍差——停牌日可能被误当成交易日，
        # 但市场级数据缺失时仍能保证日历校验可用）。
        union: set[pd.Timestamp] = set()
        for frame in qfq_bars.values():
            stamps = pd.DatetimeIndex(
                pd.to_datetime(frame["datetime"], errors="coerce")
            ).dropna()
            union.update(pd.Timestamp(value).normalize() for value in stamps)
        return pd.DatetimeIndex(sorted(union)), "本地全池日期并集（未找到指数日历）"

    @staticmethod
    def _sample_symbols(pool: Sequence[str], sample_size: int) -> list[str]:
        """确定性抽样：排序后按固定步长等距抽取（保证可复现、覆盖全池）。"""
        ordered = sorted(pool)
        try:
            size = max(1, int(sample_size))
        except (TypeError, ValueError):
            size = DEFAULT_SAMPLE_SIZE
        if len(ordered) <= size:
            return ordered
        step = len(ordered) / size
        picked: list[str] = []
        for position in range(size):
            index = int(position * step)
            picked.append(ordered[min(index, len(ordered) - 1)])
        # 去重（步长取整可能造成重复）后保序。
        out: list[str] = []
        for symbol in picked:
            if symbol not in out:
                out.append(symbol)
        return out

    def _run_ohlc(
        self,
        qfq_bars: dict[str, pd.DataFrame],
        usable: Sequence[str],
        raw_bars: dict[str, pd.DataFrame],
        sample: Sequence[str],
    ) -> OhlcResult:
        """OHLC 校验：全池跑前复权口径 + 抽样跑未复权口径，合并为一份结果。"""
        qfq_result = check_ohlc({s: qfq_bars[s] for s in usable}, adjust="qfq")
        raw_result = check_ohlc(
            {s: raw_bars[s] for s in sample if s in raw_bars}, adjust=""
        )
        table = pd.concat([qfq_result.table, raw_result.table], ignore_index=True)
        samples = pd.concat(
            [qfq_result.samples, raw_result.samples], ignore_index=True
        )
        merged = OhlcResult(
            ok=qfq_result.ok and raw_result.ok,
            table=table,
            samples=samples,
            summary={
                "qfq": qfq_result.summary,
                "raw": raw_result.summary,
                "total_errors": int(
                    qfq_result.summary.get("total_errors", 0)
                    + raw_result.summary.get("total_errors", 0)
                ),
            },
            warnings=list(qfq_result.warnings) + list(raw_result.warnings),
        )
        return merged

    def _run_survivorship(self, spec: QualityAuditSpec) -> SurvivorshipResult:
        """幸存者偏差量化：名单取库内快照，本地可用 = 全部 qfq 标的。"""
        records = self._db.list_index_constituents(spec.index_code or None)
        local = [
            symbol
            for symbol in self._db.list_kline_symbols("d", "qfq")
            if not is_index_symbol(symbol)
        ]
        return check_survivorship(
            records,
            local,
            index_code=spec.index_code or None,
            start=spec.start,
            end=spec.end,
        )