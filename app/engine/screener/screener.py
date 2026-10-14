"""多条件组合选股引擎（P3-S4）。

职责：
    * 在**自选股范围**内，用「技术面 + 基本面」结构化条件做多条件逻辑叠加（AND / OR）
      选股，返回符合全部（AND）或任一（OR）条件的标的集；
    * 引擎接口预留全市场扩展（``symbols`` 显式传入即可绕过自选股候选池）；
    * **不向 UI 抛异常**：数据缺失、基本面抓取失败、技术面求值失败等一律内部消化，
      结果统一封装为 :class:`ScreenResult`。

条件形态（结构化条件）：
    * 每个条件为 :class:`Condition`（``field + op + value``）；
    * ``field`` 为基本面字段名（``pe/pb/roe/revenue_yoy/total_market_cap/float_market_cap``）
      之一时，走基本面数值比较；
    * 其它 ``field`` 一律按**技术面字段 / 指标表达式**处理，最终取序列最后一个有效值
      与 ``value`` 比较。例如 ``"close"`` 表示收盘价最新值，
      ``"close.rolling(20).mean()"`` 表示 20 日均线最新值，``"ta.rsi(close, 14)"``
      表示 RSI14 最新值；
    * ``op`` 支持 ``> / >= / < / <= / == / != / =``（``=`` 归一化为 ``==``）；
    * ``value`` 为数值阈值。

容错语义：
    * **基本面抓取失败**（``get_fundamental(symbol)`` 抛异常）→ 该股记入 ``failed``
      （``reason`` 含原因），跳过，不中断整体；
    * **无日线数据** → 该股记入 ``failed``（``reason`` 含「无K线数据」），跳过；
    * **技术面求值异常** → 该条件返回 ``False``，**不抛异常、不中断、不记 failed**；
    * 基本面字段缺值（``None`` / 非数值）→ 该条件返回 ``False``。

边界语义：
    * 空条件：AND 语义下 `all([]) == True`（全部命中），OR 语义下 `any([]) == False`
      （全部不命中），遵循集合论「真空真值」惯例；
    * 非法 ``logic`` / 非法条件结构 → 抛 :class:`ValueError`（属调用方输入错误，UI 需前置校验）。

Bar 契约：``bars`` 为 DataFrame，列 ``[datetime, open, high, low, close, volume, amount]``，
``datetime`` 升序；``volume`` 单位「手」。周/月线由本地日线经 :func:`derive_bars` 派生。
"""
from __future__ import annotations

import logging
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from app.core.config import get_config
from app.data.datasource.base import DataSource, DataSourceError, get_source
from app.data.resample import RESAMPLABLE, derive_bars
from app.data.store.db import BAR_COLUMNS, Database, get_db
from app.engine.indicator.engine import IndicatorEngine, get_engine

logger = logging.getLogger(__name__)

# 可参与数值比较的基本面字段（其余 fundamental 键为文本/摘要，不参与比较）。
FUNDAMENTAL_FIELDS: frozenset[str] = frozenset(
    {
        "pe",
        "pb",
        "roe",
        "revenue_yoy",
        "total_market_cap",
        "float_market_cap",
    }
)

# 操作符 → 比较函数映射（``=`` 在 ``_normalize_op`` 归一化为 ``==``）。
_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _normalize_op(op: Any) -> str:
    """归一化操作符（``=`` → ``==``），非法操作符抛 :class:`ValueError`。"""
    text = str(op).strip()
    if text == "=":
        text = "=="
    if text not in _OPERATORS:
        raise ValueError(f"不支持的操作符：{op!r}（支持 {sorted(_OPERATORS)}）")
    return text


@dataclass
class Condition:
    """选股条件（``field + op + value``）。

    Attributes:
        field: 字段名或技术面指标表达式。基本面字段名见 :data:`FUNDAMENTAL_FIELDS`，
            其余按技术面字段 / 指标表达式处理。
        op: 比较操作符，取值 ``> / >= / < / <= / == / != / =``。
        value: 数值阈值（构造时强制转 ``float``）。
    """

    field: str
    op: str
    value: float

    def __post_init__(self) -> None:
        """构造后严格校验：空字段 / 非法操作符 / 非数值阈值均抛 :class:`ValueError`。"""
        self.field = str(self.field).strip()
        if not self.field:
            raise ValueError("条件字段不能为空")
        self.op = _normalize_op(self.op)
        try:
            self.value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"条件阈值必须是数值：{self.value!r}") from exc

    @classmethod
    def from_dict(cls, data: dict) -> "Condition":
        """从 dict 构造（要求 ``field/op/value`` 三键齐全）。"""
        if not isinstance(data, dict):
            raise TypeError(f"条件必须是 dict，实际：{type(data).__name__}")
        missing = [k for k in ("field", "op", "value") if k not in data]
        if missing:
            raise ValueError(f"条件缺少字段：{missing}")
        return cls(field=data["field"], op=data["op"], value=data["value"])

    def to_dict(self) -> dict:
        """序列化为 dict（``field/op/value``）。"""
        return {"field": self.field, "op": self.op, "value": self.value}


@dataclass
class ScreenResult:
    """一次选股结果。

    Attributes:
        hits: 命中标的列表（按候选池顺序）。
        failed: 被跳过的标的清单，每项为 ``{"symbol": str, "reason": str}``。
        total: 候选池规模（扫描的标的总数，``hits + failed <= total``）。
        metrics: 命中标的的各条件指标实际值（与 ``hits`` **一一对应**，可选）。
            每项为 ``{"symbol": str, "conditions": [{"field": str, "value": float|None}, ...]}``，
            ``conditions`` 与本次选股条件顺序一致，``value`` 为该股该条件算出的实际值；
            求值异常 / 缺值时 ``value`` 为 ``None``。不传时默认空列表（**向后兼容**，
            不影响既有 ``hits/failed/total`` 语义）。
    """

    hits: list[str] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    total: int = 0
    metrics: list[dict] = field(default_factory=list)

    @property
    def hit_count(self) -> int:
        """命中数量。"""
        return len(self.hits)

    @property
    def failed_count(self) -> int:
        """被跳过数量。"""
        return len(self.failed)


class ScreenerEngine:
    """多条件组合选股引擎。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        source: Optional[DataSource] = None,
        engine: Optional[IndicatorEngine] = None,
    ) -> None:
        """初始化（依赖注入，默认全局单例 / 配置主数据源）。

        Args:
            db: 存储实例；默认 :func:`app.data.store.db.get_db`。
            source: 基本面数据源；默认按配置 ``data.source.primary`` 惰性解析
                :func:`get_source`。测试可注入 fake 源。
            engine: 指标计算引擎；默认 :func:`get_engine`。
        """
        cfg = get_config()
        self._db: Database = db if db is not None else get_db()
        self._source: Optional[DataSource] = source
        self._engine: IndicatorEngine = engine if engine is not None else get_engine()
        self._source_name: str = str(cfg.get("data.source.primary", "baostock"))
        self._resolved_source: Optional[DataSource] = None

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def run(
        self,
        conditions: Any,
        symbols: Optional[Sequence[str]] = None,
        period: str = "d",
        logic: str = "AND",
    ) -> ScreenResult:
        """执行选股，返回 :class:`ScreenResult`（**不抛异常**，数据错误走容错）。

        Args:
            conditions: 条件列表；支持单个 :class:`Condition` / 单个 dict / 由
                ``Condition`` 或 dict 组成的 list/tuple（dict 经 ``Condition.from_dict``
                转换）。
            symbols: 显式候选池；``None`` 时取 ``db.list_watch_items()`` 的 symbol 去重。
            period: 周期（``d``/``w``/``m``；``w``/``m`` 由日线本地派生）。
            logic: 逻辑叠加方式，``"AND"``（默认，全部命中）或 ``"OR"``（任一命中）。

        Returns:
            :class:`ScreenResult`：``hits`` 命中标的，``failed`` 被跳过标的，``total`` 候选总数。

        Raises:
            ValueError: 条件结构非法或 ``logic`` 非 ``AND/OR``（输入错误，需调用方前置校验）。
        """
        conds = self._normalize_conditions(conditions)
        logic = self._normalize_logic(logic)
        pool = self._resolve_pool(symbols)
        total = len(pool)
        needs_fundamental = any(self._is_fundamental(c) for c in conds)

        hits: list[str] = []
        metrics: list[dict] = []
        failed: list[dict] = []
        for symbol in pool:
            bars = self._load_bars(symbol, period)
            if bars is None or bars.empty:
                failed.append({"symbol": symbol, "reason": "无K线数据，无法选股"})
                continue

            fundamental: Optional[dict] = None
            if needs_fundamental:
                try:
                    fundamental = self._fetch_fundamental(symbol)
                except Exception as exc:  # noqa: BLE001 - 单股基本面失败不中断
                    failed.append({"symbol": symbol, "reason": f"基本面获取失败：{exc}"})
                    continue

            values: list[dict] = []
            try:
                passed, values = self._eval_all_with_values(conds, bars, fundamental, logic)
            except Exception as exc:  # noqa: BLE001 - 单股求值失败不中断
                logger.warning("选股求值失败：%s -> %s", symbol, exc)
                passed = False
                values = [{"field": c.field, "value": None} for c in conds]

            if passed:
                hits.append(symbol)
                metrics.append({"symbol": symbol, "conditions": values})

        return ScreenResult(hits=hits, failed=failed, total=total, metrics=metrics)

    # ------------------------------------------------------------------ #
    # 入参归一化
    # ------------------------------------------------------------------ #
    def _normalize_conditions(self, conditions: Any) -> list[Condition]:
        """把多种入参形态统一为 ``list[Condition]``（非法结构抛 :class:`ValueError`）。"""
        if conditions is None:
            return []
        if isinstance(conditions, Condition):
            return [conditions]
        if isinstance(conditions, dict):
            return [Condition.from_dict(conditions)]
        if isinstance(conditions, (list, tuple)):
            result: list[Condition] = []
            for cond in conditions:
                if isinstance(cond, Condition):
                    result.append(cond)
                elif isinstance(cond, dict):
                    result.append(Condition.from_dict(cond))
                else:
                    raise ValueError(f"无法识别的条件类型：{type(cond).__name__}")
            return result
        raise ValueError(f"无法识别的条件类型：{type(conditions).__name__}")

    @staticmethod
    def _normalize_logic(logic: Any) -> str:
        """归一化逻辑叠加方式（大小写不敏感，默认 ``AND``），非法抛 :class:`ValueError`。"""
        text = str(logic).strip().upper() if logic is not None else "AND"
        if text not in ("AND", "OR"):
            raise ValueError(f"logic 必须是 'AND' 或 'OR'，实际：{logic!r}")
        return text

    def _resolve_pool(self, symbols: Optional[Sequence[str]]) -> list[str]:
        """确定候选池（显式 > 自选股 symbol 去重；无自选股返回空，**不落演示清单**）。"""
        result: list[str] = []
        seen: set[str] = set()
        if symbols is not None:
            for symbol in symbols:
                key = str(symbol).strip()
                if key and key not in seen:
                    seen.add(key)
                    result.append(key)
            return result
        try:
            items = self._db.list_watch_items()
        except Exception as exc:  # noqa: BLE001 - 读自选股失败降级为空池
            logger.warning("读取自选股失败：%s", exc)
            return []
        for item in items:
            key = str(getattr(item, "symbol", "")).strip()
            if key and key not in seen:
                seen.add(key)
                result.append(key)
        return result

    # ------------------------------------------------------------------ #
    # 数据取用
    # ------------------------------------------------------------------ #
    def _load_bars(self, symbol: str, period: str) -> pd.DataFrame:
        """读取本地 K 线；周/月线由日线派生；读库失败降级为空表（**不抛**）。

        复权口径沿用指标编辑器的读库约定（``adjust=""``，未复权），保证与技术面
        表达式预览口径一致。
        """
        period = str(period).strip() or "d"
        try:
            if period == "d":
                return self._db.get_bars(symbol, "d")
            if period in RESAMPLABLE:
                daily = self._db.get_bars(symbol, "d")
                return derive_bars(daily, period)
            return self._db.get_bars(symbol, period)
        except Exception as exc:  # noqa: BLE001 - 读库失败降级
            logger.warning("读取 %s(%s) K线失败：%s", symbol, period, exc)
            return pd.DataFrame(columns=BAR_COLUMNS)

    def _fetch_fundamental(self, symbol: str) -> dict:
        """抓取个股基本面摘要；数据源不可用 / 抓取失败均向上抛（由 ``run`` 容错）。"""
        source = self._source
        if source is None:
            source = self._resolve_source()
        if source is None:
            raise DataSourceError(
                self._source_name,
                RuntimeError("数据源不可用"),
                "选股基本面数据源不可用",
            )
        return source.get_fundamental(symbol)

    def _resolve_source(self) -> Optional[DataSource]:
        """惰性解析配置主数据源（失败返回 ``None``，不阻断构造）。"""
        if self._resolved_source is not None:
            return self._resolved_source
        try:
            self._resolved_source = get_source(self._source_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("选股数据源 %r 不可用：%s", self._source_name, exc)
            self._resolved_source = None
        return self._resolved_source

    # ------------------------------------------------------------------ #
    # 条件求值
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_fundamental(cond: Condition) -> bool:
        """判断条件是否基本面条件（``field`` ∈ :data:`FUNDAMENTAL_FIELDS`）。"""
        return cond.field in FUNDAMENTAL_FIELDS

    def _eval_all(
        self,
        conds: list[Condition],
        bars: pd.DataFrame,
        fundamental: Optional[dict],
        logic: str,
    ) -> bool:
        """按逻辑叠加方式汇总各条件真值（AND 全真 / OR 任一真）。

        保留原有签名与语义；仅汇总真值，不返回各条件实际值（供既有调用方兼容）。
        """
        return self._eval_all_with_values(conds, bars, fundamental, logic)[0]

    def _eval_all_with_values(
        self,
        conds: list[Condition],
        bars: pd.DataFrame,
        fundamental: Optional[dict],
        logic: str,
    ) -> tuple[bool, list[dict]]:
        """按逻辑叠加方式汇总各条件真值，并返回各条件的实际值。

        Returns:
            ``(passed, values)``：``passed`` 为整体命中与否；``values`` 为与 ``conds``
            一一对应的 ``[{"field": str, "value": float|None}, ...]`` 列表
            （求值异常 / 缺值的 ``value`` 记 ``None``）。
        """
        evaluated = [self._eval_condition_value(c, bars, fundamental) for c in conds]
        values = [
            {"field": c.field, "value": value}
            for c, (_, value) in zip(conds, evaluated)
        ]
        if logic == "AND":
            passed = all(ok for ok, _ in evaluated)
        else:
            passed = any(ok for ok, _ in evaluated)
        return passed, values

    def _eval_condition(
        self, cond: Condition, bars: pd.DataFrame, fundamental: Optional[dict]
    ) -> bool:
        """求值单个条件（**不抛异常**，任何求值异常 / 缺值 → ``False``）。

        保留原有签名与语义；内部委托 :meth:`_eval_condition_value` 并只返回命中与否。
        """
        return self._eval_condition_value(cond, bars, fundamental)[0]

    def _eval_condition_value(
        self, cond: Condition, bars: pd.DataFrame, fundamental: Optional[dict]
    ) -> tuple[bool, Optional[float]]:
        """求值单个条件并返回实际值（**不抛异常**，任何求值异常 / 缺值 → ``(False, None)``）。

        Returns:
            ``(passed, value)``：``passed`` 为是否命中；``value`` 为该条件算出的实际值
            （基本面字段取实际值，技术面字段取序列末位有效值）。求值异常 / 缺值 / 非有限值
            时 ``value`` 为 ``None``。
        """
        if self._is_fundamental(cond):
            actual = self._fundamental_value(cond.field, fundamental)
        else:
            try:
                actual = self._eval_technical(cond.field, bars)
            except Exception as exc:  # noqa: BLE001 - 技术面求值异常 → 该条件 False
                logger.debug("技术面条件求值失败：%r -> %s", cond.field, exc)
                return False, None

        if actual is None or not math.isfinite(actual):
            return False, None
        return _OPERATORS[cond.op](actual, cond.value), actual

    @staticmethod
    def _fundamental_value(field: str, fundamental: Optional[dict]) -> Optional[float]:
        """取基本面数值；缺值 / 非数值 / 非有限值返回 ``None``。"""
        if not fundamental:
            return None
        raw = fundamental.get(field)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _eval_technical(self, expr: str, bars: pd.DataFrame) -> Optional[float]:
        """计算技术面字段 / 指标表达式，返回序列最后一个有效数值。

        Raises:
            ValueError: 指标求值失败（由 ``_eval_condition`` 捕获 → ``False``）。
        """
        result = self._engine.calc(expr, bars)
        if not result.ok:
            raise ValueError(result.error or "指标求值失败")
        return self._last_numeric(result.data)

    @staticmethod
    def _last_numeric(value: Any) -> Optional[float]:
        """从指标结果（Series / DataFrame / 数组 / 标量）取最后一个有效数值。"""
        if value is None:
            return None
        if isinstance(value, pd.Series):
            arr = pd.to_numeric(value, errors="coerce").dropna()
            return float(arr.iloc[-1]) if not arr.empty else None
        if isinstance(value, pd.DataFrame):
            for col in value.columns:
                arr = pd.to_numeric(value[col], errors="coerce").dropna()
                if not arr.empty:
                    return float(arr.iloc[-1])
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = pd.to_numeric(pd.Series(np.asarray(value)), errors="coerce").dropna()
            return float(arr.iloc[-1]) if not arr.empty else None
        if isinstance(value, (int, float, bool, np.number, np.bool_)):
            num = float(value)
            return num if math.isfinite(num) else None
        return None


# --------------------------------------------------------------------------- #
# 模块级单例
# --------------------------------------------------------------------------- #
_screener_singleton: Optional[ScreenerEngine] = None


def get_screener() -> ScreenerEngine:
    """获取全局选股引擎单例。"""
    global _screener_singleton
    if _screener_singleton is None:
        _screener_singleton = ScreenerEngine()
    return _screener_singleton


__all__ = [
    "Condition",
    "ScreenResult",
    "ScreenerEngine",
    "get_screener",
    "FUNDAMENTAL_FIELDS",
]
