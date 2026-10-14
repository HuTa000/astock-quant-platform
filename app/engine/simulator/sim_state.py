"""模拟训练：状态容器与交易记录（P5-S1；P6 优化起支持多标的）。

职责（《细化设计》§3.1）：
    * :class:`Position` —— 持仓（做多、T+1 语义由 :mod:`app.engine.simulator.trading_rules` 校验）；
    * :class:`Trade` —— 成交记录；
    * :class:`SimContext` —— 训练会话的**唯一状态容器与唯一数据出口**：
        - 持有行情（当前标的 ``_bars`` / 多标的 ``_frames``）、回放游标 ``cursor_dt``、
          资金、持仓、成交；
        - :meth:`SimContext.as_of` 是**唯一数据出口**，按 ``cursor_dt`` 截断，
          未来数据**物理不可见**（look-ahead 防护的结构性保证，P5 最高优先级设计）；
        - 多标的会话中现金为账户级共享，持仓按代码分别记账，回放中可随时
          :meth:`SimContext.set_symbol` 切换当前标的（账户/持仓不变）。

边界：
    * 引擎层（本模块）**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）；
    * 行情列为 Bar 契约（列 ``[datetime, open, high, low, close, volume, amount]``）；
    * 资金单位「元」，数量单位「股」。

⚠️ 数据私有化：``as_of`` 是唯一读口；调用方（交易规则 / 指标 / UI）**不得**直接触碰
``_bars`` / ``_frames``，否则未来数据泄露风险由调用方承担（测试门禁见
``tests/test_simulator.py``）。引擎内部推进/日开基准可用 :meth:`bars_for`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """持仓（单标的单方向，A 股仅支持做多）。

    Attributes:
        symbol: 标的代码。
        qty: 持仓股数（A 股为 100 的整数倍；合法性由交易规则校验）。
        cost: 加权买入均价（无手续费/滑点）。
        buy_date: 最近一次（最后）买入的交易日 ``YYYY-MM-DD``（T+1 判定依据）。
    """

    symbol: str
    qty: int
    cost: float
    buy_date: str


@dataclass
class Trade:
    """成交记录。

    Attributes:
        symbol: 标的代码。
        side: 方向，``'BUY' | 'SELL'``。
        price: 成交价。
        qty: 成交股数。
        dt: 成交时间（回放时间轴上的时间戳）。
        reason: 触发原因，``'manual' | 'condition'``。
    """

    symbol: str
    side: str
    price: float
    qty: int
    dt: datetime
    reason: str


@dataclass
class SimContext:
    """训练会话状态容器（单标的起步，P6 优化起支持**同时交易多只标的**）。

    Attributes:
        symbol: 当前标的代码（多标的会话中为「正在回放的标的」）。
        start_date: 回放起始日期（``YYYY-MM-DD``）；为 ``None`` 时取数据首日。
        cursor_dt: 回放游标；所有数据访问经 :meth:`as_of` 截断到该时刻。
        speed: 回放倍速（``1/2/5/10``），仅影响 UI 定时器间隔，不改变数据语义。
        cash: 可用现金（**账户级**，多标的共享）。
        positions: 持仓（键=代码；多标的可同时持有）。
        orders: 成交记录（按时间顺序追加）。
        _bars: **当前标的**全量 Bar（私有，仅经 :meth:`as_of` 访问）。
        _frames: 多标的会话中各标的的全量 Bar（键=代码；私有，供
            :meth:`as_of` / :meth:`bars_for` 取数；单标的会话可为空）。
    """

    symbol: str
    start_date: Optional[str] = None
    cursor_dt: Optional[datetime] = None
    speed: int = 1
    cash: float = 100_000.0
    initial_cash: Optional[float] = None  # 回放期初资金（供 PnL 计算；首笔成交时按当时 cash 记账）
    positions: dict[str, Position] = field(default_factory=dict)
    orders: list[Trade] = field(default_factory=list)
    _bars: Optional[pd.DataFrame] = None
    _frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # 多标的（P6 优化：回放中切换股票，账户共享）
    # ------------------------------------------------------------------ #
    def frames(self) -> dict[str, pd.DataFrame]:
        """本会话已加载的标的行情表（键=代码；返回浅拷贝，勿原地改内部帧）。"""
        return dict(self._frames)

    def set_symbol(self, symbol: str, frame: Optional[pd.DataFrame] = None) -> None:
        """切换当前标的（并登记其行情帧）。

        Args:
            symbol: 目标标的代码。
            frame: 该标的全量 Bar（与其它标的口径一致、已按起始日切片）；
                ``None`` 时沿用 :meth:`frames` 中已登记的同名帧（没有则视为空）。
        """
        key = str(symbol)
        if frame is not None:
            if not self._frames and self._bars is not None:
                # 单标的起步会话：先把初始标的的帧播种进多标的表，
                # 否则切走后初始标的持仓将无价可计（bars_for 返回空）。
                self._frames[self.symbol] = self._bars
            self._frames[key] = frame
        self.symbol = key
        if key in self._frames:
            self._bars = self._frames[key]

    def bars_for(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """返回某标的的**原始整段**行情（不做游标截断）。

        ⚠️ 仅供引擎内部（回放推进 / 日开基准）使用；交易逻辑与指标一律走
        :meth:`as_of`（未来数据防护）。返回值是内部帧，**勿原地变异**。
        """
        key = str(symbol) if symbol else self.symbol
        if self._frames:
            frame = self._frames.get(key)
            return frame if frame is not None else pd.DataFrame()
        if key == self.symbol and self._bars is not None:
            return self._bars
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 数据出口（唯一）
    # ------------------------------------------------------------------ #
    def as_of(
        self, dt: Optional[datetime] = None, symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """唯一数据出口：返回某标的 ``datetime <= min(dt, cursor_dt)`` 的 Bar。

        未来数据**物理不可见**：即使传入晚于 ``cursor_dt`` 的时间，也只会返回
        截至 ``cursor_dt`` 的数据（look-ahead 防护）。``cursor_dt`` 为 ``None``
        （尚未开始回放）时返回空表。

        Args:
            dt: 查询截止时间；``None`` 时取当前游标 ``cursor_dt``。
            symbol: 查询标的；``None`` 时取当前标的 ``ctx.symbol``。

        Returns:
            升序 Bar 契约 DataFrame 的**副本**（防调用方原地变异污染内部分析）。
        """
        bars = self.bars_for(symbol)
        if bars is None or bars.empty:
            return pd.DataFrame()
        if self.cursor_dt is None:
            return pd.DataFrame()
        limit = self.cursor_dt
        if dt is not None:
            limit = min(self.cursor_dt, dt)
        frame = bars[bars["datetime"] <= limit]
        return frame.copy()

    # ------------------------------------------------------------------ #
    # 行情便捷访问（均基于 as_of，无未来数据）
    # ------------------------------------------------------------------ #
    def last_price(
        self, dt: Optional[datetime] = None, symbol: Optional[str] = None
    ) -> Optional[float]:
        """某标的截至 ``dt``（默认当前游标）的**最新收盘价**；无数据返回 ``None``。"""
        frame = self.as_of(dt, symbol=symbol)
        if frame.empty:
            return None
        return float(frame.iloc[-1]["close"])

    def is_finished(self) -> bool:
        """回放是否已推进到**当前标的**数据末尾（游标已是最末一根的时间）。"""
        bars = self.bars_for()
        if bars is None or bars.empty:
            return True
        if self.cursor_dt is None:
            return False
        return self.cursor_dt >= bars["datetime"].iloc[-1]

    # ------------------------------------------------------------------ #
    # 资金 / 持仓
    # ------------------------------------------------------------------ #
    def total_asset(self, dt: Optional[datetime] = None) -> float:
        """总资产 = 现金 + 各持仓市值（**按各标的**截至 ``dt`` 的最新价计）。

        Args:
            dt: 估值时点；``None`` 用当前游标。

        Returns:
            总资产金额（元）。某持仓标的暂无可见价时按成本价估值（避免资产跳变）。
        """
        market_value = 0.0
        for sym, pos in self.positions.items():
            price = self.last_price(dt, symbol=sym)
            if price is None:
                price = float(pos.cost)
            market_value += float(pos.qty) * float(price)
        return float(self.cash) + market_value

    def position(self, symbol: str) -> Optional[Position]:
        """按代码查持仓；无返回 ``None``。"""
        return self.positions.get(symbol)

    def realized_pnl(self) -> float:
        """已实现盈亏 = 现金 - 期初现金（手续费/滑点为 0，故减去持仓市值即已实现）。

        Returns:
            已实现盈亏（元）；期初未记账（``initial_cash`` 为 ``None``）时返回 ``0.0``。
        """
        if self.initial_cash is None:
            return 0.0
        return float(self.cash) - self.initial_cash

    def unrealized_pnl(self, dt: Optional[datetime] = None) -> float:
        """未实现盈亏 = 各持仓市值 - 持仓成本（**按各标的**截至 ``dt`` 最新价计）。

        Returns:
            未实现盈亏（元）。
        """
        pnl = 0.0
        for sym, pos in self.positions.items():
            price = self.last_price(dt, symbol=sym)
            if price is None:
                price = float(pos.cost)
            pnl += float(pos.qty) * (float(price) - float(pos.cost))
        return pnl


__all__ = ["Position", "SimContext", "Trade"]