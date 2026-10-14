"""日志配置模块。

职责：配置 Python ``logging``，将日志同时输出到控制台与 ``data/app.log``。
输出规范严格遵循《开发手册与交接文档》§4.8：
    - 使用 ``RotatingFileHandler``，单文件 5MB、保留 3 份；
    - 格式 ``%(asctime)s %(levelname)s %(name)s %(message)s``；
    - 默认级别 ``INFO``。

典型用法::

    from app.core.logging import setup_logging
    setup_logging()
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

from app.core.config import PROJECT_ROOT

# 日志格式（§4.8 规定）。
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"

# 单文件最大字节数：5MB。
MAX_BYTES: int = 5 * 1024 * 1024

# 滚动备份份数。
BACKUP_COUNT: int = 3

# 默认日志文件路径：<root>/data/app.log
DEFAULT_LOG_FILE: Path = PROJECT_ROOT / "data" / "app.log"

# 标记是否已初始化，避免重复添加 handler 导致日志重复输出。
_configured: bool = False


def get_log_file() -> Path:
    """返回日志文件路径（``<root>/data/app.log``）。"""
    return DEFAULT_LOG_FILE


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """初始化全局日志。

    Args:
        level: 日志级别，默认 ``logging.INFO``。
        log_file: 日志文件路径；默认 ``<root>/data/app.log``。
            父目录不存在时会自动创建。

    Returns:
        配置完成的根 logger。

    Notes:
        重复调用是幂等的：只有首次调用会真正添加 handler。
    """
    global _configured

    root_logger = logging.getLogger()
    if _configured:
        return root_logger

    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    # 文件处理器：RotatingFileHandler -> data/app.log（5MB × 3）。
    target: Path = Path(log_file) if log_file is not None else DEFAULT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        str(target),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 控制台处理器：便于前台运行时观察日志。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    _configured = True
    root_logger.info("日志系统初始化完成，输出文件：%s", target)
    return root_logger
