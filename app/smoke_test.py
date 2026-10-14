"""无 GUI 自检脚本（P1-S1 验收用）。

用途：
    在无显示环境（CI / 无人值守）下验证应用骨架可正常构建：
    设置 ``QT_QPA_PLATFORM=offscreen``，构造 ``QApplication`` 与主窗口后立即退出，
    并校验配置加载与事件总线 ``subscribe`` / ``publish``。

运行方式::

    QT_QPA_PLATFORM=offscreen python -m app.smoke_test

成功时打印 ``SMOKE_TEST_OK`` 并以退出码 0 结束。
"""
from __future__ import annotations

import os

# 必须在导入 PySide6 / 构造 QApplication 之前设置，确保无显示环境可运行。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging  # noqa: E402
import sys  # noqa: E402

from app.core.config import get_config  # noqa: E402
from app.core.event_bus import EventBus  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402


def check_config() -> None:
    """校验配置加载与 dotted-key 读取。"""
    cfg = get_config()
    assert isinstance(cfg.as_dict(), dict), "配置应加载为 dict"
    primary = cfg.get("data.source.primary")
    assert primary == "baostock", f"data.source.primary 期望 'baostock'，实际 {primary!r}"
    assert cfg.get("data.source.fallback") == "akshare"
    assert cfg.get("data.source.minute") == "pytdx"
    assert cfg.get("not.exist.key", "fallback-value") == "fallback-value"


def check_event_bus() -> None:
    """校验事件总线 subscribe / publish。"""
    bus = EventBus()
    received: list[dict] = []
    bus.subscribe("DEMO_EVENT", lambda payload: received.append(payload))
    bus.publish("DEMO_EVENT", {"symbol": "000001", "price": 10.5})
    assert received == [{"symbol": "000001", "price": 10.5}], f"订阅者未收到正确 payload：{received!r}"


def check_window() -> None:
    """校验 QApplication 与主窗口可构建（offscreen）。"""
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import WINDOW_TITLE, MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    assert window.windowTitle() == WINDOW_TITLE, "主窗口标题不正确"
    window.close()
    assert app is not None  # 保持引用，避免被 GC


def main() -> int:
    """执行全部自检，成功返回 0。"""
    setup_logging()
    logger = logging.getLogger("astock.smoke_test")
    logger.info("开始无 GUI 自检（QT_QPA_PLATFORM=%s）", os.environ.get("QT_QPA_PLATFORM"))

    check_config()
    check_event_bus()
    check_window()

    logger.info("自检全部通过")
    print("SMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
