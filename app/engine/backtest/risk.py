"""策略风控与交易规则（P7-S6）。

定位：
    「选得好」不等于「拿得住」——本模块把两类**持仓期风控**固化成可回测规格：

    * **退出规则**（逐只持仓、每个交易日**收盘判定**、引擎在**次日开盘**成交）：
      止损（相对持仓成本回撤到阈值）、固定止盈（达到目标收益率落袋）、
      移动止盈（从持仓期最高收盘回撤到阈值止盈，锁住已有利润）；
    * **环境开关**（组合层）：基准转熊（收盘 vs 其 N 日均线）时**总仓位上限**降档
      （默认 50%），转熊**当日收盘判定、次日开盘立即防御减仓**；转牛不即时加仓，
      等下一个调仓日恢复满仓（避免「一天内来回打脸」的高换手）。

防前视口径（与全项目统一的三条纪律）：
    * 判定只用「≤ 当日」数据：止损/固定止盈用当日收盘与持仓成本，
      移动止盈用「截至当日的持仓期最高收盘」（``peak`` 逐日更新、只升不降）；
    * 成交一律在**次日开盘**（由引擎撮合），绝不用当日收盘价成交；
    * 环境标签「基准收盘 vs 其 N 日均线」的均线只含 ≤T 数据（rolling 因果），
      转熊当日收盘判定、次日开盘执行。

⚠️ 成本口径：持仓成本含买入费用（均价法，与引擎 ``basis`` 一致），
    因此触发止损时实际价格回撤**略小于**名义百分比（费用越重偏差越大）。

对外 API::

    from app.engine.backtest import RiskSpec, StrategySpec

    spec = StrategySpec(
        name="带风控的策略",
        conditions=[...],
        risk=RiskSpec(stop_loss_pct=0.10, trailing_stop_pct=0.15, regime_filter=True),
    )
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from app.engine.backtest.regime import DEFAULT_REGIME_WINDOW

logger = logging.getLogger(__name__)

# 退出原因常量（英文键，供程序判断）。
RISK_STOP_LOSS: str = "stop_loss"
RISK_TAKE_PROFIT: str = "take_profit"
RISK_TRAILING_STOP: str = "trailing_stop"

# 展示用中文标签。
RISK_REASON_LABELS: dict[str, str] = {
    RISK_STOP_LOSS: "止损",
    RISK_TAKE_PROFIT: "固定止盈",
    RISK_TRAILING_STOP: "移动止盈",
}

# 默认参数：止损 10%、触发后冷却 20 个交易日、环境均线 200 日（≈年线）、熊市仓位上限 50%。
DEFAULT_STOP_LOSS_PCT: float = 0.10
DEFAULT_COOLDOWN_DAYS: int = 20
DEFAULT_REGIME_EXPOSURE: float = 0.5


@dataclass
class RiskSpec:
    """风控规格（``StrategySpec.risk`` 的取值类型；``None`` 表示不启用任何风控）。

    Attributes:
        stop_loss_pct: 止损阈值（相对持仓成本的回撤，如 0.10 = 跌 10% 触发）；
            ``None`` 表示不启用止损。
        take_profit_pct: 固定止盈阈值（相对持仓成本的收益，如 0.20 = 涨 20% 触发）；
            ``None`` 表示不启用固定止盈。
        trailing_stop_pct: 移动止盈阈值（相对**持仓期最高收盘**的回撤）；
            ``None`` 表示不启用移动止盈。
        cooldown_days: 触发退出规则后的冷却期（交易日）——冷却期内该标的
            **不再买回**（防「卖了立刻买回」的手续费空转，含触发当日；
            0 表示最早次日信号即可重新买入）。
        regime_filter: 是否启用牛熊环境开关（基准转熊时总仓位降档）。
        regime_window: 环境判定均线窗口（交易日，默认 200 ≈ 年线）。
        regime_exposure: 熊市总仓位上限（占昨日收盘总资产比例，默认 0.5 = 50%）。

    Raises:
        ValueError: 比例不在 (0,1)（``None`` 放行）/ 冷却为负 / 窗口 < 1 /
            熊市仓位不在 (0,1]。
    """

    stop_loss_pct: Optional[float] = DEFAULT_STOP_LOSS_PCT
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS
    regime_filter: bool = False
    regime_window: int = DEFAULT_REGIME_WINDOW
    regime_exposure: float = DEFAULT_REGIME_EXPOSURE

    def __post_init__(self) -> None:
        """构造后严格校验（属调用方输入错误 → 抛 :class:`ValueError`）。"""
        for name in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} 必须是 (0, 1) 之间的比例：{value!r}") from exc
            if not 0.0 < parsed < 1.0:
                raise ValueError(f"{name} 必须在 (0, 1) 之间：{value!r}")
            setattr(self, name, parsed)

        try:
            cooldown = int(self.cooldown_days)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cooldown_days 必须是非负整数：{self.cooldown_days!r}") from exc
        if cooldown < 0:
            raise ValueError(f"cooldown_days 必须是非负整数：{self.cooldown_days!r}")
        self.cooldown_days = cooldown

        try:
            window = int(self.regime_window)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"regime_window 必须是正整数：{self.regime_window!r}") from exc
        if window < 1:
            raise ValueError(f"regime_window 必须是正整数：{self.regime_window!r}")
        self.regime_window = window

        try:
            exposure = float(self.regime_exposure)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"regime_exposure 必须在 (0, 1] 之间：{self.regime_exposure!r}"
            ) from exc
        if not 0.0 < exposure <= 1.0:
            raise ValueError(f"regime_exposure 必须在 (0, 1] 之间：{self.regime_exposure!r}")
        self.regime_exposure = exposure

        self.regime_filter = bool(self.regime_filter)

    @property
    def has_exit_rules(self) -> bool:
        """是否启用任一退出规则（止损 / 固定止盈 / 移动止盈）。"""
        return any(
            rule is not None
            for rule in (self.stop_loss_pct, self.take_profit_pct, self.trailing_stop_pct)
        )

    @property
    def enabled(self) -> bool:
        """是否启用任一风控逻辑（退出规则或环境开关）；全禁用时引擎按无风控回放。"""
        return self.has_exit_rules or self.regime_filter


def evaluate_exit(
    close: Optional[float],
    unit_cost: Optional[float],
    peak_close: Optional[float],
    spec: Optional[RiskSpec],
) -> Optional[str]:
    """收盘判定单只持仓是否触发退出规则（返回原因常量；未触发 / 不可判定 → ``None``）。

    优先级：**止损 → 固定止盈 → 移动止盈**（用户口径：保住本金优先；
    同时满足多个条件时记优先级更高的原因）。

    Args:
        close: 当日收盘价。
        unit_cost: 持仓单位成本（含买入费用的均价，来自引擎 ``basis``）。
        peak_close: 持仓期最高收盘价（含当日；调用方按 ``max(历史, 当日)`` 更新）。
        spec: 风控规格；``None`` / 未启用退出规则时直接返回 ``None``。

    Returns:
        :data:`RISK_STOP_LOSS` / :data:`RISK_TAKE_PROFIT` / :data:`RISK_TRAILING_STOP`
        之一；未触发或输入不可用（缺失 / 非有限 / 非正）返回 ``None``。
    """
    if spec is None or not spec.has_exit_rules:
        return None
    if close is None or unit_cost is None:
        return None
    try:
        close_value = float(close)
        cost_value = float(unit_cost)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(close_value) or close_value <= 0:
        return None
    if not math.isfinite(cost_value) or cost_value <= 0:
        return None

    # 边界比较统一用**乘法口径**（如 ``close <= cost × (1-pct)``）：避免除法浮点误差
    # 让「恰好 -10%」这类边界值漏判（9.0/10.0-1.0 = -0.099999…98 > -0.1）。
    if spec.stop_loss_pct is not None and close_value <= cost_value * (1.0 - spec.stop_loss_pct):
        return RISK_STOP_LOSS
    if spec.take_profit_pct is not None and close_value >= cost_value * (1.0 + spec.take_profit_pct):
        return RISK_TAKE_PROFIT
    if spec.trailing_stop_pct is not None and peak_close is not None:
        try:
            peak_value = float(peak_close)
        except (TypeError, ValueError):
            peak_value = 0.0
        if math.isfinite(peak_value) and peak_value > 0:
            if close_value <= peak_value * (1.0 - spec.trailing_stop_pct):
                return RISK_TRAILING_STOP
    return None


__all__ = [
    "RISK_STOP_LOSS",
    "RISK_TAKE_PROFIT",
    "RISK_TRAILING_STOP",
    "RISK_REASON_LABELS",
    "DEFAULT_STOP_LOSS_PCT",
    "DEFAULT_COOLDOWN_DAYS",
    "DEFAULT_REGIME_WINDOW",
    "DEFAULT_REGIME_EXPOSURE",
    "RiskSpec",
    "evaluate_exit",
]