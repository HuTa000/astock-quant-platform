"""P4-S1 技术指标信号预警服务测试。

覆盖（对应《规格》逐条）：
    1. add_rule 落库 / 列表 / 按 symbol 过滤 / delete / 更新；
    2. ma_cross 金叉 / 死叉触发、非交叉不触发；direction 校验与向后兼容；
    3. macd_cross 金叉触发；
    4. volume_surge 触发 / 不触发；
    5. expr 自定义表达式触发 / 不触发；
    6. enabled 停用不检测；
    7. 容错（无 K 线 / 非法表达式 / 非法参数 / 非法 JSON / 非有限参数）跳过不抛；
    8. 事件 payload 六键契约（含 rule_id / kind / params）；
    9. 多规则独立判定、各自发事件；
    10. 事件驱动钩子 on_symbol_selected / on_data_updated。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.core.event_bus import EventBus
from app.data.store.db import Database
from app.engine.alert import (
    EVT_ALERT_TRIGGERED,
    AlertService,
    get_alert_service,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_bars(
    db: Database,
    symbol: str,
    closes,
    volumes=None,
    period: str = "d",
) -> None:
    """写入 n 根日线（``adjust=""``，与读库默认一致）。"""
    close = np.asarray(closes, dtype=float)
    n = len(close)
    volume = (
        np.full(n, 100.0) if volumes is None else np.asarray(volumes, dtype=float)
    )
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
            "amount": close * 1000.0,
        }
    )
    frame["symbol"] = symbol
    frame["period"] = period
    frame["adjust"] = ""
    db.upsert_bars(frame)


def _capture(bus: EventBus) -> list[dict]:
    """订阅 ``ALERT_TRIGGERED`` 并返回事件收集列表。"""
    captured: list[dict] = []
    bus.subscribe(EVT_ALERT_TRIGGERED, captured.append)
    return captured


# --------------------------------------------------------------------------- #
# 1. add_rule 落库 / 列表 / 过滤 / delete / 更新
# --------------------------------------------------------------------------- #
def test_add_rule_persists_and_filters(tmp_path) -> None:
    db = _open(tmp_path / "a1.db")
    svc = AlertService(db=db, bus=EventBus())

    rid1 = svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})
    rid2 = svc.add_rule("000001", "volume_surge", {"multiple": 2.0, "window": 5})

    assert rid1 > 0 and rid2 > 0 and rid1 != rid2

    all_rules = db.list_alert_rules()
    assert {r.symbol for r in all_rules} == {"600519", "000001"}

    only_519 = db.list_alert_rules(symbol="600519")
    assert [r.symbol for r in only_519] == ["600519"]
    assert only_519[0].kind == "ma_cross"
    assert only_519[0].enabled is True
    assert json.loads(only_519[0].params_json) == {"fast": 5, "slow": 20}

    db.delete_alert_rule(rid1)
    assert [r.symbol for r in db.list_alert_rules()] == ["000001"]


def test_add_rule_update_existing(tmp_path) -> None:
    db = _open(tmp_path / "a2.db")
    svc = AlertService(db=db, bus=EventBus())

    rid = svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})
    # 更新同一条规则（改 kind + 参数），应保持 id 不变。
    rid2 = svc.add_rule(
        "600519", "ma_cross", {"fast": 10, "slow": 30}, rule_id=rid
    )
    assert rid2 == rid
    rules = db.list_alert_rules(symbol="600519")
    assert len(rules) == 1
    assert json.loads(rules[0].params_json) == {"fast": 10, "slow": 30}


def test_add_rule_validation(tmp_path) -> None:
    db = _open(tmp_path / "a3.db")
    svc = AlertService(db=db, bus=EventBus())

    with pytest.raises(ValueError):
        svc.add_rule("", "ma_cross", {"fast": 5, "slow": 20})
    with pytest.raises(ValueError):
        svc.add_rule("600519", "unknown_kind", {})
    with pytest.raises(ValueError):
        svc.add_rule("600519", "ma_cross", {"fast": 0, "slow": 20})
    with pytest.raises(ValueError):
        svc.add_rule("600519", "volume_surge", {"multiple": 0, "window": 5})
    with pytest.raises(ValueError):
        svc.add_rule("600519", "expr", {"expr": ""})


def test_ma_cross_direction_validation(tmp_path) -> None:
    db = _open(tmp_path / "a4.db")
    svc = AlertService(db=db, bus=EventBus())

    with pytest.raises(ValueError):
        svc.add_rule(
            "600519", "ma_cross", {"fast": 5, "slow": 20, "direction": "sideways"}
        )
    # 合法 direction 可通过校验并落库。
    rid = svc.add_rule(
        "600519", "ma_cross", {"fast": 5, "slow": 20, "direction": "cross_down"}
    )
    rule = db.list_alert_rules(symbol="600519")[0]
    assert rule.id == rid
    assert json.loads(rule.params_json)["direction"] == "cross_down"


def test_as_int_non_finite_returns_none() -> None:
    from app.engine.alert.alert_service import _as_int

    assert _as_int(float("inf")) is None
    assert _as_int(float("-inf")) is None
    assert _as_int(float("nan")) is None


def test_inf_params_rejected(tmp_path) -> None:
    db = _open(tmp_path / "a5.db")
    svc = AlertService(db=db, bus=EventBus())

    with pytest.raises(ValueError):
        svc.add_rule("600519", "ma_cross", {"fast": float("inf"), "slow": 20})


# --------------------------------------------------------------------------- #
# 2. ma_cross 金叉
# --------------------------------------------------------------------------- #
def test_ma_cross_trigger(tmp_path) -> None:
    db = _open(tmp_path / "m1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    # 29 根横盘 10 + 1 根跳涨到 20：前一根 MA5=10≤MA20=10，当前根 MA5=12>MA20=10.5。
    _seed_bars(db, "600519", [10.0] * 29 + [20.0])
    rid = svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})

    result = svc.check_symbol("600519")

    assert len(result) == 1
    assert result[0]["rule_id"] == rid
    assert result[0]["kind"] == "ma_cross"
    assert result[0]["symbol"] == "600519"
    assert result[0]["value"] == pytest.approx(12.0)
    assert "金叉" in result[0]["message"]

    assert len(captured) == 1
    assert captured[0]["kind"] == "ma_cross"


def test_ma_cross_no_trigger(tmp_path) -> None:
    db = _open(tmp_path / "m2.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    _seed_bars(db, "600519", [10.0] * 30)   # 全程横盘，无上穿
    svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})

    result = svc.check_symbol("600519")

    assert result == []
    assert captured == []


def test_ma_cross_death_cross_trigger(tmp_path) -> None:
    db = _open(tmp_path / "m3.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    # 29 根横盘 20 + 1 根跳水到 10：前一根 MA5=20≥MA20=20，当前根 MA5=18<MA20=19.5。
    _seed_bars(db, "600519", [20.0] * 29 + [10.0])
    rid = svc.add_rule(
        "600519", "ma_cross", {"fast": 5, "slow": 20, "direction": "cross_down"}
    )

    result = svc.check_symbol("600519")

    assert len(result) == 1
    assert result[0]["rule_id"] == rid
    assert result[0]["kind"] == "ma_cross"
    assert result[0]["value"] == pytest.approx(18.0)
    assert "死叉" in result[0]["message"]

    assert len(captured) == 1
    assert captured[0]["kind"] == "ma_cross"
    assert captured[0]["params"]["direction"] == "cross_down"


def test_ma_cross_default_direction_is_cross_up(tmp_path) -> None:
    """不传 direction 的老规则仍按金叉处理（向后兼容）。"""
    db = _open(tmp_path / "m4.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    # 死叉形态（横盘 20 后跳水 10），不传 direction → 按金叉处理 → 不触发。
    _seed_bars(db, "600519", [20.0] * 29 + [10.0])
    svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})
    assert svc.check_symbol("600519") == []
    assert captured == []

    # 金叉形态（横盘 10 后跳涨 20）→ 不传 direction 仍触发。
    _seed_bars(db, "000001", [10.0] * 29 + [20.0])
    svc.add_rule("000001", "ma_cross", {"fast": 5, "slow": 20})
    assert len(svc.check_symbol("000001")) == 1
    assert len(captured) == 1


# --------------------------------------------------------------------------- #
# 3. macd_cross 金叉
# --------------------------------------------------------------------------- #
def test_macd_cross_trigger(tmp_path) -> None:
    db = _open(tmp_path / "mc1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    # 34 根自 25 阴跌到 5 + 1 根收平 5.0：DIF 在最后一根上穿 DEA（金叉）。
    closes = list(np.linspace(25.0, 5.0, 34)) + [5.0]
    _seed_bars(db, "600519", closes)
    rid = svc.add_rule(
        "600519", "macd_cross", {"fast": 12, "slow": 26, "signal": 9}
    )

    result = svc.check_symbol("600519")

    assert len(result) == 1
    assert result[0]["rule_id"] == rid
    assert result[0]["kind"] == "macd_cross"
    assert "金叉" in result[0]["message"]
    assert len(captured) == 1
    assert captured[0]["kind"] == "macd_cross"


def test_macd_cross_no_trigger(tmp_path) -> None:
    db = _open(tmp_path / "mc2.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    _seed_bars(db, "600519", [10.0] * 40)   # 横盘，DIF/DEA 均为 0，无上穿
    svc.add_rule("600519", "macd_cross", {"fast": 12, "slow": 26, "signal": 9})

    result = svc.check_symbol("600519")

    assert result == []
    assert captured == []


# --------------------------------------------------------------------------- #
# 4. volume_surge
# --------------------------------------------------------------------------- #
def test_volume_surge_trigger(tmp_path) -> None:
    db = _open(tmp_path / "v1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    # 前 6 根量 100，最后一根 500：500 > 前 5 日均量 100 × 2。
    _seed_bars(db, "600519", [10.0] * 7, volumes=[100.0] * 6 + [500.0])
    rid = svc.add_rule("600519", "volume_surge", {"multiple": 2.0, "window": 5})

    result = svc.check_symbol("600519")

    assert len(result) == 1
    assert result[0]["rule_id"] == rid
    assert result[0]["value"] == pytest.approx(500.0)
    assert len(captured) == 1
    assert captured[0]["kind"] == "volume_surge"


def test_volume_surge_no_trigger(tmp_path) -> None:
    db = _open(tmp_path / "v2.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    _seed_bars(db, "600519", [10.0] * 7, volumes=[100.0] * 7)  # 未放大
    svc.add_rule("600519", "volume_surge", {"multiple": 2.0, "window": 5})

    result = svc.check_symbol("600519")

    assert result == []
    assert captured == []


# --------------------------------------------------------------------------- #
# 5. expr 自定义表达式
# --------------------------------------------------------------------------- #
def test_expr_trigger(tmp_path) -> None:
    db = _open(tmp_path / "e1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    # 1..30 单调上涨：末位 close=30 > MA(close,20)=20.5。
    _seed_bars(db, "600519", list(range(1, 31)))
    rid = svc.add_rule("600519", "expr", {"expr": "close > MA(close, 20)"})

    result = svc.check_symbol("600519")

    assert len(result) == 1
    assert result[0]["rule_id"] == rid
    assert result[0]["kind"] == "expr"
    assert result[0]["value"] == pytest.approx(1.0)   # 布尔 True → 1.0
    assert len(captured) == 1
    assert captured[0]["kind"] == "expr"


def test_expr_no_trigger(tmp_path) -> None:
    db = _open(tmp_path / "e2.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    _seed_bars(db, "600519", [10.0] * 30)   # close=10 不 > MA=10
    svc.add_rule("600519", "expr", {"expr": "close > MA(close, 20)"})

    result = svc.check_symbol("600519")

    assert result == []
    assert captured == []


# --------------------------------------------------------------------------- #
# 6. enabled 停用
# --------------------------------------------------------------------------- #
def test_disabled_rule_skipped(tmp_path) -> None:
    db = _open(tmp_path / "d1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    _seed_bars(db, "600519", [10.0] * 29 + [20.0])   # 本应触发金叉
    svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20}, enabled=False)

    result = svc.check_symbol("600519")

    assert result == []
    assert captured == []


# --------------------------------------------------------------------------- #
# 7. 容错（跳过不抛）
# --------------------------------------------------------------------------- #
def test_no_bars_skipped(tmp_path) -> None:
    db = _open(tmp_path / "f1.db")
    svc = AlertService(db=db, bus=EventBus())

    svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})  # 不写任何 K 线

    assert svc.check_symbol("600519") == []


def test_illegal_expr_skipped(tmp_path) -> None:
    db = _open(tmp_path / "f2.db")
    svc = AlertService(db=db, bus=EventBus())

    _seed_bars(db, "600519", [10.0] * 30)
    svc.add_rule("600519", "expr", {"expr": "no_such_name > 1"})

    assert svc.check_symbol("600519") == []


def test_invalid_params_skipped(tmp_path) -> None:
    """参数非法（绕过 add_rule 校验直接落库）→ 求值跳过、不抛。"""
    db = _open(tmp_path / "f3.db")
    svc = AlertService(db=db, bus=EventBus())

    _seed_bars(db, "600519", [10.0] * 29 + [20.0])
    # fast=0 非法 → _eval_ma_cross 返回 (False, 0.0)
    db.save_alert_rule("600519", "ma_cross", json.dumps({"fast": 0, "slow": 20}))

    assert svc.check_symbol("600519") == []


def test_invalid_json_skipped(tmp_path) -> None:
    """params_json 非法 → 求值跳过、不抛。"""
    db = _open(tmp_path / "f4.db")
    svc = AlertService(db=db, bus=EventBus())

    _seed_bars(db, "600519", [10.0] * 29 + [20.0])
    db.save_alert_rule("600519", "ma_cross", "not-json")

    assert svc.check_symbol("600519") == []


# --------------------------------------------------------------------------- #
# 8. 事件 payload 契约（symbol/kind/rule_id/params/value/time）
# --------------------------------------------------------------------------- #
def test_event_payload_keys(tmp_path) -> None:
    db = _open(tmp_path / "p1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    _seed_bars(db, "600519", [10.0] * 29 + [20.0])
    rid = svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})

    svc.check_symbol("600519")

    assert len(captured) == 1
    payload = captured[0]
    assert set(payload.keys()) == {"symbol", "kind", "rule_id", "params", "value", "time"}
    assert payload["symbol"] == "600519"
    assert payload["kind"] == "ma_cross"
    assert payload["rule_id"] == rid
    assert payload["params"] == {"fast": 5, "slow": 20}
    assert isinstance(payload["value"], float)
    assert isinstance(payload["time"], str)


# --------------------------------------------------------------------------- #
# 9. 多规则独立判定
# --------------------------------------------------------------------------- #
def test_multiple_rules_independent(tmp_path) -> None:
    db = _open(tmp_path / "g1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    # 同时满足 ma_cross 金叉 + volume_surge 放大。
    _seed_bars(
        db,
        "600519",
        [10.0] * 29 + [20.0],
        volumes=[100.0] * 29 + [500.0],
    )
    svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})
    svc.add_rule("600519", "volume_surge", {"multiple": 2.0, "window": 5})

    result = svc.check_symbol("600519")

    kinds = sorted(r["kind"] for r in result)
    assert kinds == ["ma_cross", "volume_surge"]
    assert len(captured) == 2
    assert sorted(c["kind"] for c in captured) == ["ma_cross", "volume_surge"]


# --------------------------------------------------------------------------- #
# 10. 事件驱动钩子
# --------------------------------------------------------------------------- #
def test_on_symbol_selected_and_data_updated(tmp_path) -> None:
    db = _open(tmp_path / "h1.db")
    bus = EventBus()
    captured = _capture(bus)
    svc = AlertService(db=db, bus=bus)

    _seed_bars(db, "600519", [10.0] * 29 + [20.0])
    _seed_bars(db, "000001", [10.0] * 29 + [20.0])
    svc.add_rule("600519", "ma_cross", {"fast": 5, "slow": 20})
    svc.add_rule("000001", "ma_cross", {"fast": 5, "slow": 20})

    selected = svc.on_symbol_selected("600519")
    updated = svc.on_data_updated(["600519", "000001"])

    assert [r["symbol"] for r in selected] == ["600519"]
    assert sorted(r["symbol"] for r in updated) == ["000001", "600519"]
    # 两次检测共 3 次触发（600519 两次 + 000001 一次）。
    assert len(captured) == 3


def test_get_alert_service_singleton() -> None:
    a = get_alert_service()
    b = get_alert_service()
    assert isinstance(a, AlertService)
    assert a is b
