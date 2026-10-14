"""pandas-ta 适配层（pandas-ta-classic 0.8.32）。

本模块是项目内**唯一**的指标库导入点，屏蔽第三方库导入名细节：

    * 实际安装包名为 ``pandas-ta-classic``，但 **导入名是 ``pandas_ta_classic``**
      （不是 ``pandas_ta``，这是 R11 兼容性验证确认过的组合，勿改）；
    * 通过 :class:`TalibAdapter` 提供统一调用入口 ``call(name, *args, **kwargs)``
      与多列结果取单列工具 :func:`pick_column`；
    * 模块级别名 :data:`ta` 即 ``pandas_ta_classic``，被 :mod:`app.engine.indicator.safe_libs`
      用于构建指标函数白名单代理（``build_safe_ta``）。表达式命名空间注入的是该**收窄后的
      代理**（只含指标函数，不含 ``core`` / ``_indicator_loader`` 等内部模块属性），
      而非本模块的完整 :data:`ta` 模块。

注意：文件名沿用设计文档中的 ``talib_adapter``，但底层是 pandas-ta，并非真 TA-Lib。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# 关键：导入名是 pandas_ta_classic（不是 pandas_ta）。
import pandas_ta_classic as ta

__all__ = ["TalibAdapter", "ta", "pick_column"]


def pick_column(result: Any, preferred: str) -> pd.Series:
    """从 pandas-ta 的返回结果中取出「首选列」，统一返回单列 :class:`Series`。

    pandas-ta 多列指标（如 macd/bbands/kdj）返回 DataFrame，列名形如
    ``MACD_12_26_9`` / ``BBM_20_2.0`` / ``K_9_3``。本函数按以下规则匹配：

        1. 列名（忽略大小写）与 ``preferred`` 完全相等；
        2. 列名以 ``preferred + "_"`` 为前缀（用下划线消歧，避免 ``MACD`` 误匹配
           ``MACDh`` / ``MACDs``）。

    Args:
        result: pandas-ta 函数返回值（Series 或 DataFrame）。
        preferred: 首选列名或列前缀（如 ``"MACD"`` / ``"BBM"`` / ``"K"``）。

    Returns:
        选中的单列 Series。

    Raises:
        TypeError: ``result`` 既不是 Series 也不是 DataFrame。
        KeyError: 找不到匹配列。
    """
    if isinstance(result, pd.Series):
        return result
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"不支持的结果类型：{type(result).__name__}")

    target = str(preferred).upper()
    columns = [str(c) for c in result.columns]

    # 1) 完全相等（忽略大小写）
    for col in columns:
        if col.upper() == target:
            return result[col]

    # 2) 前缀 + "_"（消歧）
    for col in columns:
        if col.upper().startswith(target + "_"):
            return result[col]

    raise KeyError(
        f"未找到指标列 {preferred!r}，可用列：{list(result.columns)}"
    )


class TalibAdapter:
    """pandas-ta 适配器：统一调用入口 + 命名空间透传。"""

    def __init__(self) -> None:
        """初始化并固定底层指标命名空间。"""
        self._namespace = ta

    @property
    def namespace(self) -> Any:
        """返回原始 ``pandas_ta_classic`` 模块（含内部属性，仅供内部 ``call`` / 白名单构建使用）。

        注意：**不要**把本属性直接注入表达式命名空间——那会暴露 ``core`` /
        ``_indicator_loader`` 等内部模块对象。引擎层已改用
        :func:`app.engine.indicator.safe_libs.build_safe_ta` 收窄后的指标函数代理。
        """
        return self._namespace

    @property
    def ta(self) -> Any:
        """语义别名：等价于 :attr:`namespace`。"""
        return self._namespace

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """按名称调用 pandas-ta 指标函数。

        Args:
            name: 指标名（如 ``"sma"`` / ``"macd"`` / ``"bbands"``）。
            *args: 位置参数（通常是价格 Series）。
            **kwargs: 关键字参数（如 ``length`` / ``fast`` / ``signal``）。

        Returns:
            pandas-ta 原始返回值（Series 或 DataFrame）。

        Raises:
            AttributeError: 指标不存在。
            TypeError: 名字存在但不可调用。
            Exception: 指标函数自身抛出的错误（由上层统一转 Result）。
        """
        func: Any = getattr(self._namespace, name, None)
        if func is None:
            raise AttributeError(f"未知指标：{name!r}")
        if not callable(func):
            raise TypeError(f"{name!r} 不是可调用指标")
        return func(*args, **kwargs)

    def pick(self, result: Any, preferred: str) -> pd.Series:
        """便捷方法：等价于 :func:`pick_column`。"""
        return pick_column(result, preferred)

    def __getattr__(self, name: str) -> Any:
        """把未定义属性透传到底层 pandas_ta_classic（如 ``adapter.sma``）。"""
        return getattr(self._namespace, name)
