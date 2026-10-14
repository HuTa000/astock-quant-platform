"""P4-S2 预警弹窗 + 提示音 + 系统托盘 测试。

覆盖（对应任务验收）：
    1. ``ALERT_TRIGGERED`` 事件 → MainWindow 弹出弹窗（文案含 symbol 与信号中文名）；
    2. kind 中文映射（4 类 + 未知）；
    3. 托盘「退出」vs 窗口 X 关闭：X 关闭不退出、托盘退出真退出；
    4. 提示音：play 被调用（monkeypatch），异常时静默降级；
    5. 弹窗自动关闭定时器 + 可叠加（多条不互相覆盖）；
    6. ``assets/alert.wav`` 存在且可被 ``wave`` 读取（非空、格式正确）。
"""
from __future__ import annotations

import os
import wave

# 必须在导入任何 Qt 模块之前设置离屏平台。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QSystemTrayIcon  # noqa: E402

import app.ui.main_window as main_window_module  # noqa: E402
from app.core.event_bus import get_bus  # noqa: E402
from app.data.store.db import Database  # noqa: E402
from app.data.update_service import UpdateService  # noqa: E402
from app.engine.alert import EVT_ALERT_TRIGGERED  # noqa: E402
from app.ui.main_window import ALERT_WAV_PATH, MainWindow  # noqa: E402
from app.ui.widgets.alert_dialog import (  # noqa: E402
    AlertNotification,
    build_alert_lines,
    build_message,
    kind_label,
)


class _FakeSource:
    """最小假数据源（UI 测试不实际触发更新，仅需满足构造）。"""

    name = "fake"

    def get_kline(self, symbol, period, start, end, adjust=""):
        return pd.DataFrame()

    def get_minute(self, symbol, date_):
        return pd.DataFrame()

    def list_stocks(self):
        return pd.DataFrame()

    def get_spot(self):
        return pd.DataFrame()

    def get_adj_factor(self, symbol, start, end):
        return pd.DataFrame()

    def get_fundamental(self, symbol):
        return {}


class _FakeQSound:
    """默认假 QSound：静默、不真正发声。"""

    def __init__(self, path: str) -> None:
        self.path = path

    def play(self) -> None:
        pass


@pytest.fixture(scope="module")
def qapp():
    """提供（或创建）全局 QApplication。"""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp, monkeypatch):
    """构造离屏主窗口（注入内存库、假数据源、静音假 QSound）。"""
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    service = UpdateService(db=db, sources={"akshare": _FakeSource(), "baostock": _FakeSource()})
    win = MainWindow(db=db, service=service)
    monkeypatch.setattr(main_window_module, "QSound", _FakeQSound)
    yield win
    AlertNotification.close_all()
    win._really_quit = True
    win.close()


def _payload(**overrides) -> dict:
    """构造六键 payload 的辅助函数。"""
    data = {
        "symbol": "600519",
        "kind": "ma_cross",
        "rule_id": 1,
        "params": {"fast": 5, "slow": 20},
        "value": 12.34,
        "time": "2024-01-01T10:00:00",
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------- #
# kind 中文映射 / 文案纯函数
# --------------------------------------------------------------------------- #
def test_kind_label_maps_all_four_kinds() -> None:
    """四类 kind 中文名映射正确，未知 kind 原样返回、空串兜底。"""
    assert kind_label("ma_cross") == "均线交叉"
    assert kind_label("macd_cross") == "MACD 金叉"
    assert kind_label("volume_surge") == "放量"
    assert kind_label("expr") == "自定义表达式"
    assert kind_label("unknown_kind") == "unknown_kind"
    assert kind_label("") == "未知"


def test_build_alert_lines_contains_symbol_and_kind_label() -> None:
    """文案组装含 symbol 与信号中文名。"""
    lines = build_alert_lines(_payload())
    assert lines["title"] == "预警触发"
    assert lines["symbol"] == "600519"
    assert lines["kind_label"] == "均线交叉"
    assert "均线金叉" in lines["message"]
    assert lines["time"] == "2024-01-01T10:00:00"


def test_build_message_formats_volume_surge_value() -> None:
    """放量文案应带当前量数值。"""
    msg = build_message(_payload(kind="volume_surge", params={"multiple": 2.0, "window": 5}, value=12345))
    assert "12345" in msg


# --------------------------------------------------------------------------- #
# 事件 → 弹窗
# --------------------------------------------------------------------------- #
def test_alert_event_creates_notification(window, qapp) -> None:
    """``ALERT_TRIGGERED`` 事件经事件总线应触发 MainWindow 弹出弹窗。"""
    AlertNotification.close_all()
    get_bus().publish(
        EVT_ALERT_TRIGGERED,
        _payload(symbol="000001", kind="macd_cross", params={}, value=0.5),
    )
    qapp.processEvents()

    active = AlertNotification.active_notifications()
    assert len(active) == 1
    notif = active[0]
    assert notif._symbol_label.text() == "000001"
    assert notif._kind_label.text() == "MACD 金叉"


# --------------------------------------------------------------------------- #
# 弹窗可叠加 / 自动关闭
# --------------------------------------------------------------------------- #
def test_notifications_stack_without_overlap(qapp) -> None:
    """多条弹窗纵向堆叠（同 x、后弹在上方），不互相完全覆盖。"""
    AlertNotification.close_all()
    n1 = AlertNotification(_payload(symbol="A"))
    n2 = AlertNotification(_payload(symbol="B"))
    n1.show()
    n2.show()
    qapp.processEvents()

    assert n1.x() == n2.x()
    assert n2.y() < n1.y()  # 后弹的在上方（y 更小）
    AlertNotification.close_all()


def test_notification_auto_close_timer(qapp) -> None:
    """弹窗定时器为单次触发，到期后自动关闭并移出注册表。"""
    AlertNotification.close_all()
    notif = AlertNotification(_payload(), auto_close_ms=50)
    notif.show()
    assert notif._timer.isSingleShot()
    assert notif._timer.interval() == 50

    QTest.qWait(150)
    assert notif not in AlertNotification.active_notifications()


# --------------------------------------------------------------------------- #
# 托盘退出 vs 窗口 X 关闭
# --------------------------------------------------------------------------- #
def test_window_x_close_minimizes_not_quit(window, qapp, monkeypatch) -> None:
    """点 X 关闭应隐藏窗口、不真正退出（``_really_quit`` 保持 False）。

    离屏平台无系统托盘，需显式把托盘标为可用，才能走「缩托盘」分支。
    """
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: True)
    window.show()
    qapp.processEvents()

    assert window._really_quit is False
    result = window.close()  # 触发 closeEvent → ignore + hide
    assert result is False
    assert window._really_quit is False
    assert window.isVisible() is False


def test_tray_quit_really_quits(window, monkeypatch) -> None:
    """托盘「退出」应置位 ``_really_quit`` 并调用 ``QApplication.quit``。"""
    quit_calls: list[str] = []
    monkeypatch.setattr(QApplication, "quit", lambda self=None: quit_calls.append("quit"))

    assert window._tray is not None
    window._tray._request_quit()

    assert window._really_quit is True
    assert quit_calls == ["quit"]


def test_close_event_accepts_only_when_really_quit(window, monkeypatch) -> None:
    """closeEvent 在 ``_really_quit`` 为 True 时接受，否则忽略（托盘可用时）。

    离屏平台无系统托盘，需显式把托盘标为可用，否则会走「真退」分支。
    """
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: True)
    event = QCloseEvent()
    window._really_quit = False
    window.closeEvent(event)
    assert not event.isAccepted()

    event2 = QCloseEvent()
    window._really_quit = True
    window.closeEvent(event2)
    assert event2.isAccepted()


def test_close_event_accepts_when_system_tray_unavailable(window, monkeypatch) -> None:
    """系统托盘不可用（``isSystemTrayAvailable()`` 为 False）时，X 关闭应真退而非藏丢。"""
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", lambda: False)
    event = QCloseEvent()
    window._really_quit = False
    window.closeEvent(event)
    assert event.isAccepted()


def test_close_event_accepts_when_tray_is_none(window) -> None:
    """托盘初始化失败（``_tray is None``）时，X 关闭应真退而非藏丢。"""
    window._tray = None
    event = QCloseEvent()
    window._really_quit = False
    window.closeEvent(event)
    assert event.isAccepted()


# --------------------------------------------------------------------------- #
# 提示音
# --------------------------------------------------------------------------- #
def test_play_alert_sound_calls_play(window, monkeypatch) -> None:
    """提示音播放应调用 QSound.play，并传入内置 wav 路径。"""
    played: list[str] = []

    class FakeSound:
        def __init__(self, path: str) -> None:
            self.path = path

        def play(self) -> None:
            played.append(self.path)

    monkeypatch.setattr(main_window_module, "QSound", FakeSound)
    window._play_alert_sound()
    assert played == [str(ALERT_WAV_PATH)]


def test_play_alert_sound_silent_on_error(window, monkeypatch) -> None:
    """QSound 构造/播放抛异常时应静默降级，不向上抛出。"""

    class BoomSound:
        def __init__(self, path: str) -> None:
            pass

        def play(self) -> None:
            raise RuntimeError("no audio device")

    monkeypatch.setattr(main_window_module, "QSound", BoomSound)
    window._play_alert_sound()  # 不应抛出


# --------------------------------------------------------------------------- #
# wav 资源
# --------------------------------------------------------------------------- #
def test_alert_wav_exists_and_valid() -> None:
    """内置 wav 存在且可被标准库 wave 读取（非空、格式正确）。"""
    assert ALERT_WAV_PATH.exists()
    with wave.open(str(ALERT_WAV_PATH), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 44100
        assert w.getnframes() > 0
