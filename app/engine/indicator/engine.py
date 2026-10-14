"""指标计算引擎（统一入口 ``IndicatorEngine.calc(expr, bars, ctx=None)``）。

职责：
    * 把用户表达式（如 ``"close.rolling(20).mean()"`` 或
      ``"ta.macd(close)['MACD_12_26_9']"``）在受限命名空间中求值；
    * 求值结果统一对齐到 ``bars.index``（保证「长度一致的 Series」）；
    * **绝不向 UI 抛异常**：一切错误（语法错、安全违规、未知名字、类型错、除零…）都封装为
      :class:`IndicatorResult`，UI 据 ``ok`` / ``error_type`` 决定绘制或 Toast。

安全边界（P3-S2 起，AST 白名单沙箱已就位）：
    * ``calc`` 不再直接 ``eval`` 用户字符串，而是：
      ``ast.parse(expr, mode='eval')`` → :func:`app.engine.indicator.security.validate_tree`
      白名单校验（禁危险节点 / ``__`` 属性 / 未知名）→ ``compile`` → 受限命名空间 ``eval``；
    * 命名空间注入 Bar 契约纯数据列、计算库 ``pd``/``np``（**已收窄为纯计算函数白名单，
      不含 I/O 与反序列化**）/``ta``（**已收窄为指标函数白名单代理，不含 ``core`` /
      ``_indicator_loader`` 等内部模块属性**）、上下文 ``symbol``/``period``/``n``、
      内置指标（``MA/EMA/MACD/RSI/KDJ/BOLL``）与 ``bars``，``__builtins__`` 置空；
      ``bars`` 仍注入真实 DataFrame 用于 ``MA(bars, 20)`` 等内置指标，但其
      ``.eval/.query/to_*/apply/pipe`` 等危险方法已在 AST 属性黑名单层被拦截，
      用户表达式无法触达字符串重入与写盘通道；
    * 属性名以 ``__`` 开头/结尾、显式 dunder 黑名单、以及普通危险方法名黑名单
      （``eval``/``query``/``to_*``/``pipe``/``apply`` 等）在 AST 层即被拒绝，杜绝
      ``close.__class__.__init__.__globals__['__builtins__']['eval']`` 类链逃逸与
      ``bars.eval("...")`` / ``bars.to_csv(...)`` 类字符串重入 / 写盘；
      ``Import``/``Lambda``/推导式等节点亦被白名单拒绝；
    * 失败统一带 ``error_type``（``Syntax``/``Security``/``Name``/``Type``/``Runtime``）。

Bar 契约：``bars`` 为 DataFrame，列 ``[datetime, open, high, low, close, volume, amount]``，
``datetime`` 升序；``volume`` 单位「手」。
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.data.store.db import BAR_COLUMNS
from app.engine.indicator import builtins as _builtins
from app.engine.indicator.safe_libs import build_safe_np, build_safe_pd, build_safe_ta
from app.engine.indicator.security import IndicatorSecurityError, validate_tree
from app.engine.indicator.talib_adapter import TalibAdapter

logger = logging.getLogger(__name__)

# calc 命名空间里可引用的内置指标名（把内置指标也注入，方便写 MA(bars, 20) 等）。
_BUILTIN_NAMES: tuple[str, ...] = ("MA", "EMA", "MACD", "RSI", "KDJ", "BOLL")

# 错误分类（design §2.4）。
ERROR_SYNTAX: str = "Syntax"
ERROR_SECURITY: str = "Security"
ERROR_NAME: str = "Name"
ERROR_TYPE: str = "Type"
ERROR_RUNTIME: str = "Runtime"


@dataclass
class IndicatorResult:
    """指标计算结果（成功 / 失败统一载体，引擎层不向 UI 抛异常）。

    Attributes:
        ok: 是否计算成功。
        data: 成功时的结果。通常为**与 bars 等长**的 :class:`pd.Series`；
            多列指标表达式若直接返回 DataFrame 则保留 DataFrame（已对齐 index）；
            表达式若为标量（如 ``ta.rsi(close, 14).iloc[-1]``）则保留标量。
        error: 失败时的可读错误信息（成功时为空串）。
        error_type: 失败分类，取值 ``Syntax/Security/Name/Type/Runtime``（成功时为 ``None``）。
        name: 结果名称（Series 名或列名，尽力推断，可能为空）。
    """

    ok: bool
    data: Any = None
    error: str = ""
    name: str = ""
    error_type: Optional[str] = None

    @property
    def series(self) -> Optional[pd.Series]:
        """成功且结果为 Series 时返回该 Series，否则返回 ``None``。"""
        if self.ok and isinstance(self.data, pd.Series):
            return self.data
        return None

    @classmethod
    def success(cls, data: Any, name: str = "") -> "IndicatorResult":
        """构造成功结果。"""
        return cls(ok=True, data=data, error="", name=name, error_type=None)

    @classmethod
    def failure(cls, error: str, error_type: Optional[str] = None) -> "IndicatorResult":
        """构造失败结果（绝不抛异常）。"""
        return cls(ok=False, data=None, error=error, name="", error_type=error_type)


class IndicatorEngine:
    """指标计算引擎（统一入口）。"""

    def __init__(self, adapter: Optional[TalibAdapter] = None) -> None:
        """初始化。

        Args:
            adapter: 指标适配器；``None`` 时使用默认 :class:`TalibAdapter`。
        """
        self.adapter: TalibAdapter = adapter or TalibAdapter()

    # ------------------------------------------------------------------ #
    # 对外 API
    # ------------------------------------------------------------------ #
    def calc(
        self,
        expr: Any,
        bars: Any,
        ctx: Optional[dict[str, Any]] = None,
    ) -> IndicatorResult:
        """计算指标表达式，返回统一 :class:`IndicatorResult`（不抛异常）。

        Args:
            expr: 表达式字符串，可引用 Bar 契约纯数据列
                （``datetime/open/high/low/close/volume/amount``）、计算库 ``pd``/``np``
                （**已收窄为纯计算函数白名单，不含 I/O 与反序列化**）、
                ``ta`` 指标函数命名空间（**已收窄为指标函数白名单代理，不含 ``core`` /
                ``_indicator_loader`` 等内部模块属性**；如
                ``ta.macd(close)['MACD_12_26_9']``）、
                内置指标（``MA/EMA/MACD/RSI/KDJ/BOLL``）、上下文（``symbol``/``period``/``n``）
                与 ``bars``（``bars`` 注入真实 DataFrame 的**深拷贝副本**用于
                ``MA(bars, 20)`` 等内置指标，表达式内原地修改不影响调用方数据；其
                ``.eval/.query/to_*/apply/pipe`` 等危险方法已在 AST 属性黑名单层被拦截，
                用户表达式无法触达字符串重入与写盘通道）。
            bars: Bar 契约 DataFrame（``datetime`` 升序）。
            ctx: 可选上下文，支持 ``symbol``/``period``/``n`` 三个键
                （默认 ``''``/``'d'``/``20``）。

        Returns:
            :class:`IndicatorResult`：成功时 ``ok=True`` 且 ``data`` 为与 bars 等长的
            Series（或 DataFrame / 标量）；失败时 ``ok=False``，``error`` 为可读信息，
            ``error_type`` 为 ``Syntax/Security/Name/Type/Runtime`` 之一。
        """
        # 1) 入参校验（不抛异常）。
        if not isinstance(expr, str) or not expr.strip():
            return IndicatorResult.failure(f"表达式必须为非空字符串，实际：{expr!r}")
        if not isinstance(bars, pd.DataFrame):
            return IndicatorResult.failure(
                f"bars 必须是 pandas.DataFrame，实际：{type(bars).__name__}"
            )
        if bars.empty:
            return IndicatorResult.failure("bars 为空，无法计算指标")
        missing = [c for c in BAR_COLUMNS if c not in bars.columns]
        if missing:
            return IndicatorResult.failure(f"bars 缺少 Bar 契约列：{missing}")

        # 2) 解析表达式 → 语法错误。
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as exc:
            return IndicatorResult.failure(
                f"表达式语法错误：{exc.msg}", error_type=ERROR_SYNTAX
            )

        # 3) 构建命名空间（其键即白名单名字集合）。
        namespace = self._build_namespace(bars, ctx)
        allowed_names = {
            k for k in namespace if not (k.startswith("__") and k.endswith("__"))
        }

        # 4) AST 白名单校验 → 安全违规 / 未知名字。
        try:
            validate_tree(tree, allowed_names)
        except IndicatorSecurityError as exc:
            return IndicatorResult.failure(str(exc), error_type=ERROR_SECURITY)
        except NameError as exc:
            return IndicatorResult.failure(str(exc), error_type=ERROR_NAME)

        # 5) 编译 + 执行 → 运行时错误分类。
        try:
            code = compile(tree, "<indicator>", "eval")
            raw = eval(code, namespace)  # noqa: S307 - 已通过 AST 白名单校验
        except NameError as exc:
            return IndicatorResult.failure(str(exc), error_type=ERROR_NAME)
        except TypeError as exc:
            return IndicatorResult.failure(str(exc), error_type=ERROR_TYPE)
        except ZeroDivisionError as exc:
            return IndicatorResult.failure(str(exc), error_type=ERROR_RUNTIME)
        except Exception as exc:  # noqa: BLE001 - 引擎层必须吞掉一切异常
            logger.debug("指标表达式求值失败：%r -> %s", expr, exc)
            return IndicatorResult.failure(str(exc), error_type=ERROR_RUNTIME)

        # 6) 结果归一化（复用 P3-S1 逻辑）。
        try:
            normalized = self._normalize(raw, bars)
        except TypeError as exc:
            return IndicatorResult.failure(str(exc), error_type=ERROR_TYPE)
        except Exception as exc:  # noqa: BLE001
            logger.debug("指标结果归一化失败：%r -> %s", expr, exc)
            return IndicatorResult.failure(str(exc), error_type=ERROR_RUNTIME)

        if self._has_inf(normalized):
            return IndicatorResult.failure(
                "表达式求值产生无穷值（疑似除零或数值溢出）",
                error_type=ERROR_RUNTIME,
            )

        name = self._infer_name(normalized)
        return IndicatorResult.success(normalized, name=name)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _build_namespace(
        self, bars: pd.DataFrame, ctx: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """构造受限 eval 命名空间。

        注入：Bar 契约纯数据列、``pd``/``np``/``ta``（三者均已收窄为白名单代理）、
        上下文 ``symbol``/``period``/``n``、内置指标函数、``bars``；``__builtins__`` 置空。
        ``bars`` 与各列均为**深拷贝副本**，表达式内的原地修改不会影响调用方 DataFrame。
        注入键的并集即白名单名字集合。
        """
        ctx = ctx or {}
        namespace: dict[str, Any] = {}
        for col in BAR_COLUMNS:
            if col in bars.columns:
                namespace[col] = bars[col].copy()
        namespace["pd"] = build_safe_pd()
        namespace["np"] = build_safe_np()
        namespace["ta"] = build_safe_ta()
        namespace["symbol"] = ctx.get("symbol", "")
        namespace["period"] = ctx.get("period", "d")
        namespace["n"] = ctx.get("n", 20)
        namespace["bars"] = bars.copy(deep=True)
        for name in _BUILTIN_NAMES:
            namespace[name] = getattr(_builtins, name)
        # 显式置空，避免 eval 自动注入完整内建。
        namespace["__builtins__"] = {}
        return namespace

    @staticmethod
    def _normalize(raw: Any, bars: pd.DataFrame) -> Any:
        """把求值结果统一到 bars 对齐形态。

        - Series / DataFrame：``reindex(bars.index)`` 保证长度一致；
        - 列表 / 元组 / ndarray：转 Series 并按 ``bars.index`` 对齐（长度不符则抛）；
        - 标量：原样返回。
        """
        if raw is None:
            raise ValueError("表达式结果为空（None）")
        if isinstance(raw, pd.Series):
            return raw.reindex(bars.index)
        if isinstance(raw, pd.DataFrame):
            return raw.reindex(bars.index)
        if isinstance(raw, (list, tuple, np.ndarray)):
            series = pd.Series(np.asarray(raw), index=bars.index)
            return series
        if isinstance(raw, (int, float, bool, np.number, np.bool_)):
            return raw
        raise TypeError(f"不支持的表达式结果类型：{type(raw).__name__}")

    @staticmethod
    def _has_inf(value: Any) -> bool:
        """检测数值结果中是否含 ±inf（除零 / 溢出的主要表征）。"""
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return False
        try:
            return bool(np.isinf(arr).any())
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _infer_name(value: Any) -> str:
        """尽力推断结果名称（Series 名 / DataFrame 列名）。"""
        if isinstance(value, pd.Series):
            return str(value.name) if value.name is not None else ""
        if isinstance(value, pd.DataFrame):
            return ",".join(str(c) for c in value.columns)
        return ""


# --------------------------------------------------------------------------- #
# 模块级单例
# --------------------------------------------------------------------------- #
_engine_singleton: Optional[IndicatorEngine] = None


def get_engine() -> IndicatorEngine:
    """获取全局指标引擎单例。"""
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = IndicatorEngine()
    return _engine_singleton


__all__ = ["IndicatorEngine", "IndicatorResult", "get_engine"]
