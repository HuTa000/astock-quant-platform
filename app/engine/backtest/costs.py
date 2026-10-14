"""A 股交易成本模型（P7-S1）。

定位：
    把「回测里的成交」按 A 股真实费率扣减成本，并给出**含滑点**的成交价。
    回测不含成本 = 高估收益，成本模型是「结论可信」的前提。

费率口径（2023 年 8 月印花税减半后的现行标准，均可配置）：
    * 佣金 ``commission_rate``：默认 0.025%（万 2.5）**双边**，单笔最低 5 元
      （``min_commission``，小额交易被最低佣金显著惩罚——这正是现实）；
    * 印花税 ``stamp_tax_rate``：默认 0.05%（万 5）**仅卖出**；
    * 过户费 ``transfer_fee_rate``：默认 0.001%（十万分之一）**双边**；
    * 滑点 ``slippage_bps``：默认 0，单位「基点」（1bp = 0.01%）——买入按
      ``price × (1 + bps/10000)`` 成交、卖出按 ``price × (1 − bps/10000)``，
      用于模拟「开盘价拿不到、实际略差」的冲击成本。

典型用法::

    from app.engine.backtest import CostModel

    costs = CostModel()                      # 默认现实费率
    amounts = costs.fill_price(10.0, "buy") # 买入含滑点成交价
    fee = costs.trade_cost(10_000.0, "sell")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 买卖方向常量（与成交明细 ``side`` 列一致）。
BUY: str = "buy"
SELL: str = "sell"

# 基点换算：1 bp = 万分之一。
BPS_DENOMINATOR: float = 10_000.0


@dataclass(frozen=True)
class CostModel:
    """A 股交易成本模型（不可变，可安全共享）。

    Attributes:
        commission_rate: 佣金费率（双边，按成交金额计提）。
        min_commission: 单笔最低佣金（元）。
        stamp_tax_rate: 印花税率（仅卖出计提）。
        transfer_fee_rate: 过户费率（双边计提）。
        slippage_bps: 滑点（基点，双边按方向调整成交价）。
    """

    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 0.0

    def trade_cost(self, amount: float, side: str) -> float:
        """计算一笔成交的总费用（佣金 + 过户费 [+ 印花税]）。

        Args:
            amount: 成交金额（元，取绝对值）。
            side: ``"buy"`` 或 ``"sell"``。

        Returns:
            总费用（元）；``amount == 0`` 返回 0（负数按绝对值计提）。

        Raises:
            ValueError: ``side`` 非法（调用方编程错误）。
        """
        if side not in (BUY, SELL):
            raise ValueError(f"不支持的交易方向：{side!r}（仅支持 {BUY!r}/{SELL!r}）")
        value = abs(float(amount))
        if value <= 0:
            return 0.0
        commission = max(value * float(self.commission_rate), float(self.min_commission))
        transfer = value * float(self.transfer_fee_rate)
        stamp = value * float(self.stamp_tax_rate) if side == SELL else 0.0
        return float(commission + transfer + stamp)

    def fill_price(self, price: float, side: str) -> float:
        """按滑点方向调整成交价（买入上浮、卖出下调）。

        Args:
            price: 参考价（回测中为「信号次日开盘价」）。
            side: ``"buy"`` 或 ``"sell"``。

        Returns:
            含滑点的成交价。

        Raises:
            ValueError: ``price <= 0`` 或 ``side`` 非法。
        """
        if side not in (BUY, SELL):
            raise ValueError(f"不支持的交易方向：{side!r}（仅支持 {BUY!r}/{SELL!r}）")
        value = float(price)
        if value <= 0:
            raise ValueError(f"成交参考价必须为正：{price!r}")
        slip = float(self.slippage_bps) / BPS_DENOMINATOR
        if side == BUY:
            return value * (1.0 + slip)
        return value * (1.0 - slip)


# 零成本模型：单测「价格/数量」逻辑时使用（真实回测请用默认 :class:`CostModel`）。
ZERO_COST_MODEL: CostModel = CostModel(
    commission_rate=0.0,
    min_commission=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_bps=0.0,
)

# 默认成本模型（现实费率）。
DEFAULT_COST_MODEL: CostModel = CostModel()


__all__ = ["BUY", "SELL", "BPS_DENOMINATOR", "CostModel", "ZERO_COST_MODEL", "DEFAULT_COST_MODEL"]