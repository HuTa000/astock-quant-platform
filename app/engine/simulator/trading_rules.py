"""模拟训练：交易规则校验（P5-S2）。

职责（《细化设计》§3.4）：对一笔下单做**合法性校验**，返回 ``(ok, reason)``：

    | 校验 | 规则 | 拒绝提示 |
    |---|---|---|
    | 做多 | 仅 ``BUY``/``SELL`` 平仓，**禁卖空**（无持仓不能 SELL） | "仅支持做多，不能卖空" |
    | T+1 | ``SELL`` 时要求 ``position.buy_date < cursor 交易日`` | "T+1：当日买入不可卖出" |
    | 手数 | ``qty % 100 == 0`` 且 ``qty > 0`` | "数量须为 100 股整数倍" |
    | 资金 | ``BUY`` 时 ``price*qty <= cash`` | "现金不足" |
    | 持仓 | ``SELL`` 时 ``qty <= position.qty`` | "持仓不足" |

边界（本模块额外约定，防止静默错单）：
    * ``side`` 非 ``BUY/SELL`` → 拒（"方向非法"）；
    * ``price``/``qty`` 非数值 / ``price<=0`` / ``qty<=0`` → 拒（"价格非法"/"数量非法"）；
    * 回放尚未开始（``ctx.cursor_dt is None``）→ 拒（"回放尚未开始"）：T+1 判定
      必须依赖「下单当日」，无游标则无法判定，直接拒绝比给出错误判定更安全。

「当日」口径：``cursor_dt`` 的**日期**部分（``YYYY-MM-DD``）；T+1 判定用
``cursor_date > buy_date``（严格大于：同日在 legal_trading 语义上不可卖）。

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）；本模块纯 Python。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from app.engine.simulator.sim_state import SimContext

# 合法方向。
BUY: str = "BUY"
SELL: str = "SELL"

# A 股一手股数。
LOT_SIZE: int = 100


def _is_pos_number(value: object) -> bool:
    """是否为有限正数（False 对 bool 也成立，防 ``True`` 当作 1）。"""
    if isinstance(value, bool):
        return False
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(num) and num > 0


def _is_pos_int(value: object) -> bool:
    """是否为正整数（``bool`` 单独拒绝，防 ``True`` 当作 1）。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    # 容错：整数值的 float / 字符串（如 ``100.0``、``"100"``）。
    try:
        if value is None:
            return False
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(num) and num > 0 and num == int(num)


def max_buy_qty(cash: float, price: float, lot: int = LOT_SIZE) -> int:
    """指定现金与价格下的**最大可买股数**（纯函数，向下取整到一手）。

    供 UI「全仓 / 1/2 仓 / 1/4 仓」按钮换算股数；不改变任何状态。

    Args:
        cash: 可用现金（元）。
        price: 委托价（元）。
        lot: 一手股数（默认 :data:`LOT_SIZE`）。

    Returns:
        可买股数（``lot`` 的整数倍）；现金 / 价格非法或不足一手时返回 ``0``。
    """
    if lot <= 0 or not _is_pos_number(price):
        return 0
    try:
        amount = float(cash)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(amount) or amount <= 0:
        return 0
    return int(amount // (float(price) * lot)) * lot


def qty_for_amount(amount: float, price: float, lot: int = LOT_SIZE) -> int:
    """把「买入金额」换算为股数（纯函数，向下取整到一手）。

    Args:
        amount: 拟买入金额（元）。
        price: 委托价（元）。
        lot: 一手股数（默认 :data:`LOT_SIZE`）。

    Returns:
        股数（``lot`` 的整数倍）；金额 / 价格非法或不足一手时返回 ``0``。
    """
    if lot <= 0 or not _is_pos_number(price):
        return 0
    try:
        total = float(amount)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(total) or total <= 0:
        return 0
    return int(total // (float(price) * lot)) * lot


def validate_order(
    ctx: SimContext, side: str, price: float, qty: int, symbol: Optional[str] = None
) -> Tuple[bool, str]:
    """校验一笔订单是否合法。

    Args:
        ctx: 训练会话状态（含现金、持仓、回放游标）。
        side: ``'BUY'`` 或 ``'SELL'``。
        price: 拟成交价（正数）。
        qty: 拟成交股数（A 股：100 倍数）。
        symbol: 校验标的；``None`` 时用当前标的 ``ctx.symbol``（多标的训练：
            SELL 需按「该标的的持仓」判定，见 P6 优化）。

    Returns:
        ``(True, "")`` 合法；``(False, 拒绝原因中文文案)``。
    """
    # 0) 基本面参数校验（防静默错单）。
    if side not in (BUY, SELL):
        return False, "方向非法（仅支持 BUY / SELL）"
    if not _is_pos_number(price):
        return False, "价格非法（须为正数）"
    if not _is_pos_int(qty):
        return False, "数量非法（须为正整数）"
    qty_num = int(qty)

    # 1) 手数：100 整数倍。
    if qty_num % LOT_SIZE != 0:
        return False, "数量须为 100 股整数倍"

    # 2) T+1 判定需要「下单当日」＝回放游标的日期；尚未开始即拒绝。
    if ctx.cursor_dt is None:
        return False, "回放尚未开始"

    cursor_date = ctx.cursor_dt.date().isoformat()

    if side == BUY:
        # 3) 资金校验（现金买入，无杠杆）。
        if float(price) * qty_num > ctx.cash:
            return False, "现金不足"
        return True, ""

    # --- SELL ---
    key = str(symbol) if symbol else ctx.symbol
    pos = ctx.positions.get(key)
    # 4) 做多（禁卖空）。
    if pos is None or pos.qty <= 0:
        return False, "仅支持做多，不能卖空"
    # 5) 持仓数量。
    if qty_num > pos.qty:
        return False, "持仓不足"
    # 6) T+1：当日买入不可卖出（buy_date 必须严格早于 cursor 日期）。
    if pos.buy_date >= cursor_date:
        return False, "T+1：当日买入不可卖出"
    return True, ""


__all__ = [
    "BUY",
    "LOT_SIZE",
    "SELL",
    "max_buy_qty",
    "qty_for_amount",
    "validate_order",
]