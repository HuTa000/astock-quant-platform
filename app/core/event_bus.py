"""轻量发布/订阅事件总线。

设计目标（遵循《细化设计》§1.3 事件清单）：
    - 极简 API：``subscribe(event, cb)`` / ``publish(event, payload)``；
    - 线程安全：内部使用可重入锁，允许在任意线程发布；
    - 订阅者异常隔离：单个订阅者抛错不影响其它订阅者，仅记录日志。

事件名与 payload 字段约定见《细化设计》§1.3，例如::

    DATA_UPDATE_PROGRESS -> {"symbol": str, "current": int, "total": int}

典型用法::

    from app.core.event_bus import get_bus
    bus = get_bus()
    bus.subscribe("KLINE_LOADED", lambda payload: print(payload))
    bus.publish("KLINE_LOADED", {"symbol": "000001", "period": "d"})
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# 订阅者回调签名：接收一个 payload dict，无返回值。
Callback = Callable[[dict], None]


class EventBus:
    """线程安全的轻量发布/订阅实现。"""

    def __init__(self) -> None:
        """初始化空的事件总线。"""
        self._subscribers: Dict[str, List[Callback]] = defaultdict(list)
        self._lock: threading.RLock = threading.RLock()

    def subscribe(self, event: str, cb: Callback) -> Callback:
        """订阅指定事件。

        Args:
            event: 事件名（建议使用 §1.3 中的大写常量，如 ``"KLINE_LOADED"``）。
            cb: 回调函数，签名 ``cb(payload: dict) -> None``。

        Returns:
            原样返回 ``cb``，便于链式使用/取消订阅。
        """
        with self._lock:
            self._subscribers[event].append(cb)
        return cb

    def unsubscribe(self, event: str, cb: Callback) -> None:
        """取消订阅。

        Args:
            event: 事件名。
            cb: 先前订阅的回调；不存在时静默忽略。
        """
        with self._lock:
            callbacks = self._subscribers.get(event)
            if callbacks and cb in callbacks:
                callbacks.remove(cb)
                if not callbacks:
                    self._subscribers.pop(event, None)

    def publish(self, event: str, payload: Optional[dict] = None) -> None:
        """发布事件，同步分发给所有订阅者。

        Args:
            event: 事件名。
            payload: 事件负载 dict；``None`` 视为空 dict ``{}``。

        Notes:
            为保证线程安全，发布前对订阅者列表做快照；单个订阅者异常会被
            捕获并记录日志，不影响其余订阅者。
        """
        data: dict = payload if payload is not None else {}
        with self._lock:
            callbacks = list(self._subscribers.get(event, []))
        for cb in callbacks:
            try:
                cb(data)
            except Exception:  # noqa: BLE001 - 故意兜底，隔离订阅者异常
                logger.exception("事件 %s 的订阅者 %r 执行异常", event, cb)

    def clear(self, event: Optional[str] = None) -> None:
        """清空订阅。

        Args:
            event: 指定事件则只清空该事件；``None`` 则清空全部。
        """
        with self._lock:
            if event is None:
                self._subscribers.clear()
            else:
                self._subscribers.pop(event, None)

    def subscriber_count(self, event: str) -> int:
        """返回指定事件的订阅者数量（主要用于测试/自检）。"""
        with self._lock:
            return len(self._subscribers.get(event, []))


# 模块级全局事件总线单例。
_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    """获取全局事件总线单例。"""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


# 便捷的模块级函数（等价于对全局单例的操作），便于简单场景直接调用。
def subscribe(event: str, cb: Callback) -> Callback:
    """订阅全局事件总线上的事件（``get_bus().subscribe`` 的语法糖）。"""
    return get_bus().subscribe(event, cb)


def publish(event: str, payload: Optional[dict] = None) -> None:
    """向全局事件总线发布事件（``get_bus().publish`` 的语法糖）。"""
    get_bus().publish(event, payload)
