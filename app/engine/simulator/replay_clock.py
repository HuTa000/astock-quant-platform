"""模拟训练：1 分钟回放时钟（P5-S1）。

职责（《细化设计》§3.2 / §3.5）：
    * 按 1 分钟 Bar 序列**逐根推进**游标（每 tick = 下一根 1 分钟 Bar）；
    * 由数据序列推导交易日历（升序唯一日期）与每个交易日的首末时间；
    * 倍速 → 定时间隔（毫秒）纯函数 :func:`interval_for_speed`，读配置 ``sim.speed_map``；
    * 每推进一根发 ``SIM_TICK`` 事件（payload ``{"cursor_dt", "price"}``），
      并调用注入的 ``on_tick`` 回调；到达末尾发 ``SIM_FINISHED``（payload 空）；
    * :meth:`ReplayClock.tick` 是纯 Python 手动推进，**与 Qt 完全解耦**：
      UI 层（P5-S4）用 ``QTimer`` 按 ``interval_for_speed`` 驱动 ``tick``，
      本引擎层**严禁 import 任何 Qt 代码**。

look-ahead 防护：本时钟只负责「推进游标」；行情可见性由
:meth:`SimContext.as_of <app.engine.simulator.sim_state.SimContext.as_of>` 结构性保证
（未来数据物理不可见）。训练逻辑/指标一律走 ``as_of`` 取数。

事件常量：
    * :data:`SIM_TICK` == ``"SIM_TICK"``（payload ``{"cursor_dt", "price"}``）
    * :data:`SIM_FINISHED` == ``"SIM_FINISHED"``（payload ``{}``）
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd

from app.core.event_bus import get_bus
from app.core.config import get_config

logger = logging.getLogger(__name__)

# 事件名（《细化设计》§1.3 事件清单）。
SIM_TICK: str = "SIM_TICK"
SIM_FINISHED: str = "SIM_FINISHED"

# 配置键。
_CFG_SPEED_MAP: str = "sim.speed_map"
_CFG_SPEED_DEFAULT: str = "sim.speed_default"

# 内置倍速映射（配置缺失时兜底，与 §4.9 默认一致）。
# 2026-10-01（用户要求）：1x 由 200ms 调慢到 400ms（其余倍速等比例），让逐根回放更从容。
_DEFAULT_SPEED_MAP: dict[int, int] = {1: 400, 2: 200, 5: 80, 10: 40}


def interval_for_speed(speed: int, speed_map: Optional[dict] = None) -> int:
    """倍速 → 每 tick 定时器间隔（毫秒）纯函数。

    Args:
        speed: 倍速（``1/2/5/10``）。
        speed_map: 映射；``None`` 时读配置 ``sim.speed_map``，再兜底内置表。

    Returns:
        毫秒数。

    Raises:
        ValueError: 倍速不在映射中 / 非正整数。
    """
    mapping = speed_map
    if mapping is None:
        mapping = get_config().get(_CFG_SPEED_MAP, _DEFAULT_SPEED_MAP)
    try:
        key = int(speed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"倍速必须为映射中的整数：{speed!r}") from exc
    try:
        mapping = dict(mapping)
    except (TypeError, ValueError) as exc:  # pragma: no cover - 配置损坏防御
        raise ValueError(f"speed_map 配置非法：{mapping!r}") from exc
    if key not in mapping:
        raise ValueError(f"不支持的倍速：{speed!r}（可选 {sorted(mapping)}）")
    value = int(mapping[key])
    if value <= 0:
        raise ValueError(f"倍速映射值必须为正整数：{value}")
    return value


class ReplayClock:
    """1 分钟回放时钟（纯 Python，与 Qt 解耦）。

    用法示意（UI 层 P5-S4）::

        clock = ReplayClock(ctx)
        def drive():
            if not clock.tick():
                timer.stop()
        timer = QTimer(); timer.setInterval(interval_for_speed(ctx.speed)); ...
    """

    def __init__(
        self,
        ctx,
        on_tick: Optional[Callable[[dict], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        emit_events: bool = True,
    ) -> None:
        """构造回放时钟。

        Args:
            ctx: :class:`~app.engine.simulator.sim_state.SimContext`（含待回放 bars）。
            on_tick: 每推进一根的回调（收到 ``{"cursor_dt", "price"}``）。
            on_finished: 回放结束时回调（无参）。
            emit_events: 是否同时发 ``SIM_TICK`` / ``SIM_FINISHED`` 总线事件。
        """
        self._ctx = ctx
        self._on_tick = on_tick
        self._on_finished = on_finished
        self._emit_events = bool(emit_events)
        self._bars = self._normalize_bars(ctx._bars)
        self._done: bool = False
        self._initialized: bool = False

    # ------------------------------------------------------------------ #
    # 数据归一
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_bars(bars) -> pd.DataFrame:
        """规范为升序、仅 Bar 契约列、datetime 为 datetime64；None/空返回空表。"""
        if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
            return pd.DataFrame()
        frame = bars.copy()
        missing = [c for c in ("datetime", "open", "high", "low", "close", "volume", "amount")
                   if c not in frame.columns]
        if missing:
            logger.warning("ReplayClock 收到非法列：缺 %s", missing)
            return pd.DataFrame()
        frame = frame[["datetime", "open", "high", "low", "close", "volume", "amount"]].copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime", kind="stable")
        return frame.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 访问器
    # ------------------------------------------------------------------ #
    def has_data(self) -> bool:
        """是否存在可回放数据。"""
        return not self._bars.empty

    def is_done(self) -> bool:
        """回放是否已结束（推进到最末一根之后）。"""
        return self._done

    def trading_days(self) -> List[str]:
        """回放数据覆盖的交易日列表（升序 ``YYYY-MM-DD``）。"""
        if self._bars.empty:
            return []
        days = pd.to_datetime(self._bars["datetime"]).dt.date.unique()
        return [d.isoformat() for d in days]

    def count(self) -> int:
        """可回放 Bar 总数。"""
        return int(len(self._bars))

    def current_index(self) -> int:
        """当前游标对应 Bar 的索引；未开始 / 已完成返回 ``-1``。"""
        cursor = self._ctx.cursor_dt
        if cursor is None or self._bars.empty:
            return -1
        mask = self._bars["datetime"] <= cursor
        if not mask.any():
            return -1
        return int(mask.sum() - 1)

    def cursor_dt(self) -> Optional[datetime]:
        """当前游标时间（等价于 ``ctx.cursor_dt``）。"""
        return self._ctx.cursor_dt

    # ------------------------------------------------------------------ #
    # 推进
    # ------------------------------------------------------------------ #
    def tick(self) -> bool:
        """推进一根 1 分钟 Bar 并更新游标（纯 Python，可在任意线程调用）。

        Behavior：
            * 首次调用：游标落到**首根** Bar 时间；
            * 之后每次推进到**下一根**；
            * 已推进完最后一根（或已 done）→ 返回 ``False`` 且不再发事件；
            * 每成功推进发 ``SIM_TICK`` / 回调；越界发 ``SIM_FINISHED``。

        Returns:
            本次是否发生了推进（True）。回放结束返回 ``False``。
        """
        if self._bars.empty:
            self._finish()
            return False
        if self._done:
            return False

        next_idx = self.current_index() + 1
        if next_idx >= len(self._bars):
            self._finish()
            return False

        row = self._bars.iloc[next_idx]
        target = row["datetime"]
        price = float(row["close"])
        self._ctx.cursor_dt = target

        if self._emit_events:
            try:
                get_bus().publish(SIM_TICK, {"cursor_dt": target, "price": price})
            except Exception as exc:  # noqa: BLE001 - 事件隔离，推进不受影响
                logger.exception("发布 SIM_TICK 失败：%s", exc)
        if self._on_tick is not None:
            try:
                self._on_tick({"cursor_dt": target, "price": price})
            except Exception as exc:  # noqa: BLE001 - 回调异常不影响推进
                logger.exception("on_tick 回调异常：%s", exc)

        if next_idx == len(self._bars) - 1:
            self._finish()
        return True

    def _finish(self) -> None:
        """进入结束态：发 ``SIM_FINISHED`` 与结束回调（幂等）。"""
        if self._done:
            return
        self._done = True
        if self._emit_events:
            try:
                get_bus().publish(SIM_FINISHED, {})
            except Exception as exc:  # noqa: BLE001 - 事件隔离
                logger.exception("发布 SIM_FINISHED 失败：%s", exc)
        if self._on_finished is not None:
            try:
                self._on_finished()
            except Exception as exc:  # noqa: BLE001 - 回调异常隔离
                logger.exception("on_finished 回调异常：%s", exc)


__all__ = ["ReplayClock", "SIM_FINISHED", "SIM_TICK", "interval_for_speed"]