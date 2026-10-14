"""程序入口。

职责（P1-S1 + P6-S3）：
    1. 初始化配置与日志；
    2. 安装全局异常兜底（``sys.excepthook`` + Qt messageHandler，防崩 + 友好提示）；
    3. 创建 ``QApplication`` 与主窗口（可选启动页封面）；
    4. 进入事件循环。

运行方式（在 ``D:\\Stock\\astock`` 下）::

    python main.py
"""
from __future__ import annotations

import logging
import sys

from app import __app_name__
from app.core.config import get_config
from app.core.error_hook import install_excepthook, install_qt_message_guard
from app.core.event_bus import get_bus
from app.core.logging import setup_logging

logger = logging.getLogger("astock.main")


def main() -> int:
    """应用主函数。

    Returns:
        进程退出码（``QApplication.exec`` 的返回值）。
    """
    # 1) 初始化日志（输出到 data/app.log 与控制台）。
    setup_logging()
    # P6-S3：全局异常兜底（未捕获异常 → 日志 + 友好弹窗，不崩）。
    install_excepthook()
    install_qt_message_guard()

    # 2) 加载配置并打出关键项，便于运行时确认。
    cfg = get_config()
    logger.info(
        "配置加载完成：来源=%s，主数据源=%s，兜底=%s，指标引擎=%s",
        cfg.source_path or "内置默认值",
        cfg.get("data.source.primary"),
        cfg.get("data.source.fallback"),
        cfg.get("data.indicator_engine"),
    )

    # 3) 初始化全局事件总线（供后续模块共享）。
    bus = get_bus()
    logger.info("事件总线就绪：%s", type(bus).__name__)

    # 延迟导入 Qt，避免无 GUI 环境下导入即失败。
    from PySide6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(__app_name__)

    # 4) P6-S3：启动页（仅真实 GUI 平台展示；offscreen 跳过避免离屏阻塞）。
    splash = _create_splash(app)

    window = MainWindow()
    window.show()
    if splash is not None:
        splash.close()

    logger.info("进入事件循环")
    return app.exec()


def _create_splash(app) -> object:  # noqa: ANN001 - 返回 QSplashScreen 或 None
    """创建启动页（offscreen 平台返回 None 以跳过）。"""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QSplashScreen

        if str(app.platformName()).lower() == "offscreen":
            return None
        pixmap = QPixmap(420, 180)
        pixmap.fill(Qt.GlobalColor.transparent)
        splash = QSplashScreen(pixmap)
        splash.show()
        splash.showMessage(
            f"{__app_name__} 启动中…",
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
            Qt.GlobalColor.gray,
        )
        app.processEvents()
        return splash
    except Exception as exc:  # noqa: BLE001 - 启动页失败不影响主流程
        logger.warning("启动页创建失败（跳过）：%s", exc)
        return None


if __name__ == "__main__":
    sys.exit(main())
