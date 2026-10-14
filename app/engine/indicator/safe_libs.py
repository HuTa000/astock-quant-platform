"""计算库白名单代理（P3-S2 收窄）：把 np/pd/ta 收窄为纯计算属性，堵死 I/O 与反序列化逃逸。

背景：design §2.2 最初注入完整 ``np``/``pd``/``ta`` 模块，导致 ``pd.read_pickle``
（反序列化即任意代码执行入口）、``np.load``/``np.save``、``ta.core``（re-export 完整
numpy/pandas）、``ta._indicator_loader``（re-export 完整 importlib，可
``import_module('os').system(...)``）等**普通方法（非 dunder）**绕过了 AST 白名单
（AST 只拦 ``__`` 属性与危险节点）。本模块提供 :func:`build_safe_np` / :func:`build_safe_pd` /
:func:`build_safe_ta`，返回只暴露白名单属性的代理对象，供 ``_build_namespace`` 注入，
从「注入对象」层面收窄。

白名单原则：**纯数学 / 聚合 / 常量 / 安全构造 / 指标函数，绝不含 I/O、反序列化、反射、
ndarray、模块内部对象（core / _indicator_loader / 类对象）**。
"""
from __future__ import annotations

import inspect
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from app.engine.indicator.talib_adapter import ta

# np 白名单：纯数学 / 聚合 / 序列 / 常量（绝不含 load/save/savez/fromfile/tofile/
# loadtxt/savetxt/genfromtxt/memmap/DataSource/ndarray 等 I/O、反序列化、泄露型对象）。
_NP_WHITELIST: tuple[str, ...] = (
    # 数学
    "abs", "sqrt", "log", "log10", "log2", "exp", "expm1", "log1p",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    "power", "square", "cbrt", "floor", "ceil", "round", "sign",
    "maximum", "minimum", "clip", "where",
    "isfinite", "isnan", "isinf", "nan_to_num",
    # 聚合 / 序列
    "cumsum", "cumprod", "diff", "sum", "prod", "mean", "std", "var",
    "min", "max", "ptp", "median", "percentile", "quantile",
    "nanmean", "nanstd", "nansum", "nanmin", "nanmax", "nanmedian",
    "average", "convolve",
    # 常量
    "pi", "e", "nan", "inf", "euler_gamma",
)

# pd 白名单：安全构造 / 聚合（绝不含 read_*/to_*/HDFStore/ExcelWriter/json_normalize
# 等 I/O、反序列化类属性）。
_PD_WHITELIST: tuple[str, ...] = (
    "Series", "DataFrame", "to_numeric", "to_datetime", "to_timedelta",
    "cut", "qcut", "concat", "merge", "isna", "notna", "unique", "factorize",
)


def build_safe_np() -> Any:
    """返回只暴露 np 纯计算白名单属性的代理对象（非白名单属性访问抛 ``AttributeError``）。"""
    return SimpleNamespace(**{name: getattr(np, name) for name in _NP_WHITELIST})


def build_safe_pd() -> Any:
    """返回只暴露 pd 安全构造/聚合白名单属性的代理对象（非白名单属性访问抛 ``AttributeError``）。"""
    return SimpleNamespace(**{name: getattr(pd, name) for name in _PD_WHITELIST})


@lru_cache(maxsize=1)
def build_safe_ta() -> Any:
    """返回只暴露 pandas-ta **指标函数** 的代理对象（堵死 ``ta.core`` / ``ta._indicator_loader`` 逃逸）。

    该代理只读、不可变，故以 ``lru_cache`` 缓存（避免每次 ``calc`` 都懒加载 193 个指标）。

    ``pandas_ta_classic`` 顶层模块 re-export 了 ``core``（完整 numpy/pandas）、
    ``_indicator_loader``（完整 importlib）、``Category`` / ``Strategy`` 等类对象，
    若直接注入整个模块，用户表达式可经 ``ta.core.np.load(...)`` 或
    ``ta._indicator_loader.importlib.import_module('os').system(...)`` 触发 I/O 与任意
    代码执行。故本函数只保留官方指标注册表 ``_INDICATOR_TO_CATEGORY`` 中的指标函数，
    以 ``inspect.isfunction`` 过滤（排除 class / module / 非函数属性），其余属性访问
    统一抛 ``AttributeError``（引擎层映射为 Runtime 失败）。

    Returns:
        仅含指标函数的代理对象（193 个指标：sma/ema/macd/rsi/bbands/kdj 等）。
    """
    registry: dict[str, Any] = getattr(
        getattr(ta, "_indicator_loader"), "_INDICATOR_TO_CATEGORY", {}
    )
    attrs: dict[str, Any] = {}
    for name in sorted(registry):
        if name.startswith("_"):
            continue
        try:
            value = getattr(ta, name)
        except Exception:  # noqa: BLE001 - 跳过无法懒加载的注册项
            continue
        if inspect.isfunction(value):
            attrs[name] = value
    return SimpleNamespace(**attrs)


__all__ = ["build_safe_np", "build_safe_pd", "build_safe_ta"]
