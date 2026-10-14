"""事件总线单测（P1-S1 验收门禁）。"""
from __future__ import annotations

from app.core.event_bus import EventBus, get_bus, publish, subscribe


def test_subscribe_and_publish_receives_payload() -> None:
    """订阅者应能收到 publish 的 payload（核心验收）。"""
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("KLINE_LOADED", lambda payload: received.append(payload))

    bus.publish("KLINE_LOADED", {"symbol": "000001", "period": "d"})

    assert received == [{"symbol": "000001", "period": "d"}]


def test_publish_without_subscribers_does_not_raise() -> None:
    """无订阅者时发布不应抛异常。"""
    bus = EventBus()
    bus.publish("NO_SUBSCRIBER", {"a": 1})  # 不抛异常即通过


def test_publish_with_none_payload_becomes_empty_dict() -> None:
    """payload 为 None 时应归一化为空 dict。"""
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("THEME_CHANGED", received.append)
    bus.publish("THEME_CHANGED")
    assert received == [{}]


def test_multiple_subscribers_all_receive() -> None:
    """多个订阅者应全部收到同一 payload。"""
    bus = EventBus()
    first: list[dict] = []
    second: list[dict] = []
    bus.subscribe("SIM_TICK", first.append)
    bus.subscribe("SIM_TICK", second.append)

    payload = {"cursor_dt": "2026-09-21 09:31:00", "price": 10.2}
    bus.publish("SIM_TICK", payload)

    assert first == [payload]
    assert second == [payload]
    assert bus.subscriber_count("SIM_TICK") == 2


def test_unsubscribe_stops_delivery() -> None:
    """取消订阅后不再收到事件。"""
    bus = EventBus()
    received: list[dict] = []
    cb = bus.subscribe("WATCHLIST_CHANGED", received.append)
    bus.publish("WATCHLIST_CHANGED", {"action": "add"})
    bus.unsubscribe("WATCHLIST_CHANGED", cb)
    bus.publish("WATCHLIST_CHANGED", {"action": "remove"})

    assert received == [{"action": "add"}]


def test_subscriber_exception_is_isolated() -> None:
    """单个订阅者抛错不应影响其他订阅者。"""
    bus = EventBus()
    received: list[dict] = []

    def bad_subscriber(_payload: dict) -> None:
        raise RuntimeError("boom")

    bus.subscribe("DATA_UPDATE_FAILED", bad_subscriber)
    bus.subscribe("DATA_UPDATE_FAILED", received.append)

    bus.publish("DATA_UPDATE_FAILED", {"symbol": "600000"})

    assert received == [{"symbol": "600000"}]


def test_global_bus_singleton_and_helpers() -> None:
    """全局单例与模块级便捷函数应一致。"""
    assert get_bus() is get_bus()

    received: list[dict] = []
    subscribe("ALERT_TRIGGERED", received.append)
    publish("ALERT_TRIGGERED", {"symbol": "000001", "rule": "ma_cross"})

    assert received == [{"symbol": "000001", "rule": "ma_cross"}]
    # 清理，避免影响其它用例。
    get_bus().clear("ALERT_TRIGGERED")
