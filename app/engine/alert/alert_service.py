"""技术指标信号预警服务（P4-S1）。

职责：
    * 为个股维护「技术指标信号预警规则」（预置信号 + 自定义表达式），规则持久化到
      ``alert_rule`` 表（重启保留）；
    * **事件驱动**检测：在「切到该股」/「数据更新完成」时，由 UI / 数据层调用
      :meth:`on_symbol_selected` / :meth:`on_data_updated`，本服务对启用规则逐一判定，
      触发则经事件总线发布 ``ALERT_TRIGGERED``（无常驻轮询线程）；
    * 规则类型（``kind``）：
        - ``ma_cross``     均线交叉（金叉 / 死叉，由 ``params.direction`` 指定，默认金叉）；
        - ``macd_cross``   MACD 金叉；
        - ``volume_surge`` 成交量放大；
        - ``expr``         自定义表达式（走指标引擎沙箱）；
    * **绝不向 UI 抛异常**：无 K 线 / 指标求值异常 / 参数非法 → 该规则跳过不触发、
      不中断其它规则。

规则参数（``params_json``，按 ``kind`` 区分）：
    * ``ma_cross``     -> ``{"fast": 5, "slow": 20, "direction": "cross_up"}``
      ``direction`` 可选，取值 ``"cross_up"``（金叉，默认，保持向后兼容）/
      ``"cross_down"``（死叉）。
      金叉：前一根 fast 均线 ≤ slow 均线 且 当前根 fast > slow（上穿）；
      死叉：前一根 fast 均线 ≥ slow 均线 且 当前根 fast < slow（下穿）。
    * ``macd_cross``   -> ``{"fast": 12, "slow": 26, "signal": 9}``
      DIF 上穿 DEA。
    * ``volume_surge`` -> ``{"multiple": 2.0, "window": 5}``
      当前 volume > 前 window 日均量 × multiple。
    * ``expr``         -> ``{"expr": "close > MA(close, 20)"}``
      用 ``eng.calc`` 求值，取末位值，布尔真值视为触发。

关键契约：
    * 事件名常量 :data:`EVT_ALERT_TRIGGERED` == ``"ALERT_TRIGGERED"``；
    * 事件 payload：``{"symbol": str, "kind": str, "rule_id": int, "params": dict,
      "value": float, "time": str}``（``time`` 为 ISO 文本）；
    * ``ma_cross/macd_cross`` 需比较「前一根 vs 当前根」，序列至少 2 根；数据不足视为不触发。

Bar 契约：``bars`` 为 DataFrame，列 ``[datetime, open, high, low, close, volume, amount]``，
``datetime`` 升序；``volume`` 单位「手」。周/月线由本地日线经 :func:`derive_bars` 派生。
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from app.core.event_bus import EventBus, get_bus
from app.data.resample import RESAMPLABLE, derive_bars
from app.data.store.db import BAR_COLUMNS, Database, get_db
from app.engine.indicator.builtins import MACD, MA
from app.engine.indicator.engine import IndicatorEngine, get_engine

logger = logging.getLogger(__name__)

# 事件名（事件驱动检测触发后经事件总线发布）。
EVT_ALERT_TRIGGERED: str = "ALERT_TRIGGERED"

# 规则类型。
KIND_MA_CROSS: str = "ma_cross"
KIND_MACD_CROSS: str = "macd_cross"
KIND_VOLUME_SURGE: str = "volume_surge"
KIND_EXPR: str = "expr"

SUPPORTED_KINDS: frozenset[str] = frozenset(
    {KIND_MA_CROSS, KIND_MACD_CROSS, KIND_VOLUME_SURGE, KIND_EXPR}
)

# ma_cross 交叉方向（``cross_up`` 为默认，保持历史规则向后兼容）。
DIRECTION_CROSS_UP: str = "cross_up"
DIRECTION_CROSS_DOWN: str = "cross_down"
CROSS_DIRECTIONS: frozenset[str] = frozenset({DIRECTION_CROSS_UP, DIRECTION_CROSS_DOWN})


# --------------------------------------------------------------------------- #
# 数值工具（不抛异常，一律收敛为 None / False）
# --------------------------------------------------------------------------- #
def _as_int(value: Any) -> Optional[int]:
    """尽力转 int；失败 / ``None`` / 非有限值返回 ``None``。"""
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _as_float(value: Any) -> Optional[float]:
    """尽力转 float；失败 / ``None`` / NaN / ±inf 返回 ``None``。"""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_number(value: Any) -> bool:
    """判断是否为可用的有限数值（NaN / inf / None 均为 False）。"""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class AlertService:
    """技术指标信号预警服务（事件驱动 + 规则持久化）。"""

    def __init__(
        self,
        db: Optional[Database] = None,
        engine: Optional[IndicatorEngine] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        """初始化（依赖注入，默认全局单例）。

        Args:
            db: 存储实例；默认 :func:`app.data.store.db.get_db`。
            engine: 指标计算引擎（仅 ``expr`` 规则使用）；默认 :func:`get_engine`。
            bus: 事件总线；默认 :func:`app.core.event_bus.get_bus`。
        """
        self._db: Database = db if db is not None else get_db()
        self._engine: IndicatorEngine = engine if engine is not None else get_engine()
        self._bus: EventBus = bus if bus is not None else get_bus()

    # ------------------------------------------------------------------ #
    # 规则管理
    # ------------------------------------------------------------------ #
    def add_rule(
        self,
        symbol: str,
        kind: str,
        params: dict[str, Any],
        enabled: bool = True,
        rule_id: Optional[int] = None,
    ) -> int:
        """新增（或更新）一条预警规则，返回规则 id。

        Args:
            symbol: 标的代码。
            kind: 规则类型（:data:`SUPPORTED_KINDS`）。
            params: 参数 dict（按 ``kind`` 区分，见模块 docstring）。
            enabled: 是否启用（默认 ``True``）。
            rule_id: 更新目标 id；``None`` 表示新增（透传 ``db.save_alert_rule``）。

        Returns:
            落库后的规则 id。

        Raises:
            ValueError: ``symbol`` 为空 / ``kind`` 不受支持 / ``params`` 非法
                （调用方输入错误，UI 需前置校验；与选股引擎对非法条件抛
                ``ValueError`` 的约定一致）。
        """
        symbol = str(symbol).strip()
        kind = str(kind).strip()
        if not symbol:
            raise ValueError("预警标的代码不能为空")
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"不支持的预警类型：{kind!r}（支持 {sorted(SUPPORTED_KINDS)}）")
        params = self._validate_params(kind, params)
        params_json = json.dumps(params, ensure_ascii=False)
        return int(
            self._db.save_alert_rule(
                symbol, kind, params_json, enabled=enabled, rule_id=rule_id
            )
        )

    @staticmethod
    def _validate_params(kind: str, params: Any) -> dict[str, Any]:
        """校验并归一化参数；非法参数抛 :class:`ValueError`。"""
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"预警参数必须是 dict，实际：{type(params).__name__}")

        if kind == KIND_MA_CROSS:
            fast = _as_int(params.get("fast"))
            slow = _as_int(params.get("slow"))
            if fast is None or slow is None or fast <= 0 or slow <= 0:
                raise ValueError("ma_cross 需要正整数 fast / slow")
            direction = str(params.get("direction", DIRECTION_CROSS_UP)).strip()
            if direction not in CROSS_DIRECTIONS:
                raise ValueError(
                    f"ma_cross 的 direction 非法：{direction!r}"
                    f"（支持 {sorted(CROSS_DIRECTIONS)}）"
                )
            normalized = {"fast": fast, "slow": slow}
            # 仅在调用方显式传入 direction 时落库，保持历史规则 params_json 不变。
            if "direction" in params:
                normalized["direction"] = direction
            return normalized

        if kind == KIND_MACD_CROSS:
            fast = _as_int(params.get("fast", 12))
            slow = _as_int(params.get("slow", 26))
            signal = _as_int(params.get("signal", 9))
            if (
                fast is None
                or slow is None
                or signal is None
                or min(fast, slow, signal) <= 0
            ):
                raise ValueError("macd_cross 需要正整数 fast / slow / signal")
            return {"fast": fast, "slow": slow, "signal": signal}

        if kind == KIND_VOLUME_SURGE:
            multiple = _as_float(params.get("multiple"))
            window = _as_int(params.get("window"))
            if multiple is None or window is None or multiple <= 0 or window <= 0:
                raise ValueError("volume_surge 需要 multiple > 0 且 window > 0")
            return {"multiple": multiple, "window": window}

        # kind == KIND_EXPR
        expr = str(params.get("expr", "")).strip()
        if not expr:
            raise ValueError("expr 规则需要非空表达式")
        return {"expr": expr}

    # ------------------------------------------------------------------ #
    # 事件驱动钩子
    # ------------------------------------------------------------------ #
    def on_symbol_selected(self, symbol: str, period: str = "d") -> list[dict]:
        """「切到该股」钩子：检测该 symbol 的全部启用规则（**不抛异常**）。"""
        return self.check_symbol(symbol, period)

    def on_data_updated(
        self, symbols: Optional[Sequence[str]], period: str = "d"
    ) -> list[dict]:
        """「数据更新完成」钩子：检测一批 symbol 的全部启用规则（**不抛异常**）。"""
        triggered: list[dict] = []
        if symbols is None:
            return triggered
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols:
            try:
                triggered.extend(self.check_symbol(str(symbol), period))
            except Exception as exc:  # noqa: BLE001 - 单股失败不中断整体
                logger.warning("预警检测失败：%s -> %s", symbol, exc)
        return triggered

    # ------------------------------------------------------------------ #
    # 检测核心
    # ------------------------------------------------------------------ #
    def check_symbol(self, symbol: str, period: str = "d") -> list[dict]:
        """检测某 symbol 的全部**启用**规则，返回触发的规则列表（**不抛异常**）。

        Args:
            symbol: 标的代码。
            period: 周期（``d``/``w``/``m``；``w``/``m`` 由日线本地派生）。

        Returns:
            触发的规则列表，每项为
            ``{"rule_id": int, "kind": str, "symbol": str, "value": float, "message": str}``。
            无 K 线 / 规则读取失败 / 指标求值异常时返回空列表（跳过不触发）。
        """
        triggered: list[dict] = []
        symbol = str(symbol).strip()
        if not symbol:
            return triggered

        try:
            rules = self._db.list_alert_rules(symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 %s 预警规则失败：%s", symbol, exc)
            return triggered

        enabled_rules = [r for r in rules if r.enabled]
        if not enabled_rules:
            return triggered

        try:
            bars = self._load_bars(symbol, period)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 %s(%s) K线失败：%s", symbol, period, exc)
            return triggered
        if bars is None or bars.empty:
            return triggered

        for rule in enabled_rules:
            try:
                ok, value = self._evaluate(rule, bars)
            except Exception as exc:  # noqa: BLE001 - 单规则求值异常 → 跳过
                logger.warning("预警规则 %d(%s) 求值失败：%s", rule.id, rule.kind, exc)
                continue
            if not ok:
                continue

            params = self._parse_params(rule)
            payload = {
                "symbol": symbol,
                "kind": rule.kind,
                "rule_id": int(rule.id),
                "params": params,
                "value": float(value),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                self._bus.publish(EVT_ALERT_TRIGGERED, payload)
            except Exception as exc:  # noqa: BLE001 - 发布失败不中断
                logger.warning("发布 ALERT_TRIGGERED 失败：%s", exc)

            triggered.append(
                {
                    "rule_id": int(rule.id),
                    "kind": rule.kind,
                    "symbol": symbol,
                    "value": float(value),
                    "message": self._message(rule.kind, rule.params_json, value),
                }
            )
        return triggered

    # ------------------------------------------------------------------ #
    # 数据取用
    # ------------------------------------------------------------------ #
    def _load_bars(self, symbol: str, period: str) -> pd.DataFrame:
        """读取本地 K 线；周/月线由日线派生；读库失败降级为空表（**不抛**）。

        复权口径沿用指标编辑器 / 选股引擎的读库约定（``adjust=""``，未复权）。
        """
        period = str(period).strip() or "d"
        try:
            if period == "d":
                return self._db.get_bars(symbol, "d")
            if period in RESAMPLABLE:
                daily = self._db.get_bars(symbol, "d")
                return derive_bars(daily, period)
            return self._db.get_bars(symbol, period)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取 %s(%s) K线失败：%s", symbol, period, exc)
            return pd.DataFrame(columns=BAR_COLUMNS)

    # ------------------------------------------------------------------ #
    # 规则求值
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_params(rule: Any) -> dict[str, Any]:
        """安全解析 ``rule.params_json`` 为 dict（非法 JSON / 非 dict 降级为空 dict）。"""
        try:
            raw = json.loads(rule.params_json) if rule.params_json else {}
        except (TypeError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _evaluate(self, rule: Any, bars: pd.DataFrame) -> tuple[bool, float]:
        """按 ``kind`` 分发求值，返回 ``(是否触发, 触发值)``（异常由调用方容错）。"""
        params = self._parse_params(rule)

        if rule.kind == KIND_MA_CROSS:
            return self._eval_ma_cross(params, bars)
        if rule.kind == KIND_MACD_CROSS:
            return self._eval_macd_cross(params, bars)
        if rule.kind == KIND_VOLUME_SURGE:
            return self._eval_volume_surge(params, bars)
        if rule.kind == KIND_EXPR:
            return self._eval_expr(params, bars)
        return False, 0.0

    @staticmethod
    def _eval_ma_cross(params: dict, bars: pd.DataFrame) -> tuple[bool, float]:
        """均线交叉：按 ``direction`` 判定金叉（默认）/ 死叉。

        金叉：前一根 fast ≤ slow 且当前根 fast > slow；
        死叉：前一根 fast ≥ slow 且当前根 fast < slow。

        返回 ``(触发, 当前 fast 均线值)``；数据不足 / 参数非法返回 ``(False, 0.0)``。
        """
        fast = _as_int(params.get("fast"))
        slow = _as_int(params.get("slow"))
        direction = str(params.get("direction", DIRECTION_CROSS_UP)).strip()
        if fast is None or slow is None or fast <= 0 or slow <= 0:
            return False, 0.0
        if direction not in CROSS_DIRECTIONS:
            return False, 0.0
        if "close" not in bars.columns or len(bars) < 2:
            return False, 0.0
        fast_ma = pd.to_numeric(bars["close"], errors="coerce").rolling(fast).mean()
        slow_ma = pd.to_numeric(bars["close"], errors="coerce").rolling(slow).mean()
        prev_fast, prev_slow = fast_ma.iloc[-2], slow_ma.iloc[-2]
        curr_fast, curr_slow = fast_ma.iloc[-1], slow_ma.iloc[-1]
        if not all(_valid_number(v) for v in (prev_fast, prev_slow, curr_fast, curr_slow)):
            return False, 0.0
        prev_fast_f, prev_slow_f = float(prev_fast), float(prev_slow)
        curr_fast_f, curr_slow_f = float(curr_fast), float(curr_slow)
        if direction == DIRECTION_CROSS_DOWN:
            if prev_fast_f >= prev_slow_f and curr_fast_f < curr_slow_f:
                return True, curr_fast_f
            return False, curr_fast_f
        if prev_fast_f <= prev_slow_f and curr_fast_f > curr_slow_f:
            return True, curr_fast_f
        return False, curr_fast_f

    @staticmethod
    def _eval_macd_cross(params: dict, bars: pd.DataFrame) -> tuple[bool, float]:
        """MACD 金叉：前一根 DIF ≤ DEA 且当前根 DIF > DEA。

        返回 ``(触发, 当前 DIF 值)``；数据不足 / 参数非法返回 ``(False, 0.0)``。
        """
        fast = _as_int(params.get("fast", 12))
        slow = _as_int(params.get("slow", 26))
        signal = _as_int(params.get("signal", 9))
        if (
            fast is None
            or slow is None
            or signal is None
            or min(fast, slow, signal) <= 0
        ):
            return False, 0.0
        if "close" not in bars.columns or len(bars) < 2:
            return False, 0.0
        try:
            dif = MACD(bars, fast=fast, slow=slow, signal=signal, column="MACD")
            dea = MACD(bars, fast=fast, slow=slow, signal=signal, column="MACDs")
        except Exception:  # noqa: BLE001 - pandas-ta 计算异常 → 不触发
            return False, 0.0
        prev_dif, prev_dea = dif.iloc[-2], dea.iloc[-2]
        curr_dif, curr_dea = dif.iloc[-1], dea.iloc[-1]
        if not all(_valid_number(v) for v in (prev_dif, prev_dea, curr_dif, curr_dea)):
            return False, 0.0
        if float(prev_dif) <= float(prev_dea) and float(curr_dif) > float(curr_dea):
            return True, float(curr_dif)
        return False, float(curr_dif)

    @staticmethod
    def _eval_volume_surge(params: dict, bars: pd.DataFrame) -> tuple[bool, float]:
        """成交量放大：当前 volume > 前 window 日均量 × multiple。

        返回 ``(触发, 当前 volume)``；数据不足 / 参数非法返回 ``(False, 0.0)``。
        """
        multiple = _as_float(params.get("multiple"))
        window = _as_int(params.get("window"))
        if multiple is None or window is None or multiple <= 0 or window <= 0:
            return False, 0.0
        if "volume" not in bars.columns or len(bars) < window + 1:
            return False, 0.0
        volume = pd.to_numeric(bars["volume"], errors="coerce")
        prev_mean = volume.iloc[-window - 1 : -1].mean()
        curr = volume.iloc[-1]
        if not _valid_number(prev_mean) or not _valid_number(curr):
            return False, 0.0
        if float(curr) > float(prev_mean) * multiple:
            return True, float(curr)
        return False, float(curr)

    def _eval_expr(self, params: dict, bars: pd.DataFrame) -> tuple[bool, float]:
        """自定义表达式：``eng.calc`` 求值，取末位值，布尔真值视为触发。

        返回 ``(触发, 末位数值)``；求值失败 / 无有效末位值返回 ``(False, 0.0)``。
        """
        expr = str(params.get("expr", "")).strip()
        if not expr:
            return False, 0.0
        try:
            result = self._engine.calc(expr, bars)
        except Exception:  # noqa: BLE001 - 引擎兜底不抛，双保险
            return False, 0.0
        if not result.ok:
            return False, 0.0
        value = self._last_scalar(result.data)
        if value is None or not math.isfinite(value):
            return False, 0.0
        return bool(value), float(value)

    @staticmethod
    def _last_scalar(value: Any) -> Optional[float]:
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
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if bool(value) else 0.0
        if isinstance(value, (int, float, np.number)):
            num = float(value)
            return num if math.isfinite(num) else None
        return None

    @staticmethod
    def _message(kind: str, params_json: str, value: float) -> str:
        """生成触发结果的人类可读文案。"""
        try:
            params = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError):
            params = {}
        if kind == KIND_MA_CROSS:
            direction = str(params.get("direction", DIRECTION_CROSS_UP)).strip()
            if direction == DIRECTION_CROSS_DOWN:
                return f"均线死叉：MA{params.get('fast', '')} 下穿 MA{params.get('slow', '')}"
            return f"均线金叉：MA{params.get('fast', '')} 上穿 MA{params.get('slow', '')}"
        if kind == KIND_MACD_CROSS:
            return "MACD 金叉（DIF 上穿 DEA）"
        if kind == KIND_VOLUME_SURGE:
            return f"成交量放大：当前量 {value:g}"
        if kind == KIND_EXPR:
            return f"表达式触发：{params.get('expr', '')}"
        return str(kind)


# --------------------------------------------------------------------------- #
# 模块级单例
# --------------------------------------------------------------------------- #
_alert_service_singleton: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """获取全局预警服务单例。"""
    global _alert_service_singleton
    if _alert_service_singleton is None:
        _alert_service_singleton = AlertService()
    return _alert_service_singleton


__all__ = [
    "AlertService",
    "get_alert_service",
    "EVT_ALERT_TRIGGERED",
    "KIND_MA_CROSS",
    "KIND_MACD_CROSS",
    "KIND_VOLUME_SURGE",
    "KIND_EXPR",
    "SUPPORTED_KINDS",
    "DIRECTION_CROSS_UP",
    "DIRECTION_CROSS_DOWN",
    "CROSS_DIRECTIONS",
]
