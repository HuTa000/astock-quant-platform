"""配置加载模块。

职责：
    1. 从 ``config/settings.yaml`` 读取配置为 dict；
    2. 支持 dotted-key 读取，例如 ``cfg.get("data.source.primary")``；
    3. 文件缺失或解析失败时，回退到内置默认值（见 ``DEFAULT_SETTINGS``）。

设计约定（遵循《开发手册与交接文档》§4.1 / §4.9）：
    - 配置键使用 dotted 形式（如 ``data.source.primary``）；
    - 所有键及其默认值以文档 §4.9 清单为准。

典型用法::

    from app.core.config import get_config
    cfg = get_config()
    assert cfg.get("data.source.primary") == "baostock"
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

# 项目根目录：本文件位于 <root>/app/core/config.py，故上溯两级即为项目根。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# 默认配置文件路径：<root>/config/settings.yaml
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"

# 内置默认配置（与《开发手册与交接文档》§4.9 配置键清单逐一对应）。
DEFAULT_SETTINGS: dict[str, Any] = {
    "data": {
        "source": {
            "primary": "baostock",  # 主数据源（2026-09-21 由 akshare 变更，见交接文档 R14）
            "fallback": "akshare",  # 兜底数据源
            "minute": "pytdx",      # 1 分钟历史数据源
        },
        "db_path": "data/app.db",           # SQLite 数据库文件（相对项目根）
        "cache_dir": "data/cache",          # 缓存目录
        "indicator_engine": "pandas_ta",    # 指标引擎（后续可切 talib_binary）
    },
    "ui": {
        "theme": "dark_red_green",          # 深色 + 红涨绿跌
        "red_color": "#e74c3c",             # 涨（红）
        "green_color": "#2ecc71",           # 跌（绿）
    },
    "sim": {
        "initial_cash": 100000,             # 模拟训练初始资金
        "speed_default": 1,                 # 默认回放倍速
        "speed_map": {1: 400, 2: 200, 5: 80, 10: 40},  # 倍速 -> ms/tick（1x=400ms）
    },
    "update": {
        "batch_size": 50,                   # 数据更新批大小
    },
    "export": {
        "excel_path": "data/exports",       # Excel 导出目录
        "image_dpi": 300,                   # 高清图导出 DPI
    },
    "backup": {
        "dir": "data/backups",              # 备份目录
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个 dict，``override`` 覆盖 ``base``，返回合并后的新 dict。

    Args:
        base: 基础配置（通常为内置默认值）。
        override: 覆盖配置（通常为从 YAML 读取的值）。

    Returns:
        合并后的全新 dict（不修改入参）。
    """
    result: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """配置容器，支持 dotted-key 读取。"""

    def __init__(
        self,
        data: dict[str, Any],
        source_path: Optional[Path] = None,
    ) -> None:
        """初始化配置容器。

        Args:
            data: 已合并完成的配置 dict。
            source_path: 实际读取的配置文件路径；若使用内置默认值则为 ``None``。
        """
        self._data: dict[str, Any] = copy.deepcopy(data)
        self._source_path: Optional[Path] = source_path

    @property
    def source_path(self) -> Optional[Path]:
        """实际加载的配置文件路径；使用内置默认值时返回 ``None``。"""
        return self._source_path

    def get(self, key: str, default: Any = None) -> Any:
        """按 dotted-key 读取配置值。

        Args:
            key: dotted 形式的键，例如 ``"data.source.primary"``。
                传入 ``""`` 或 ``None`` 时返回整份配置 dict。
            default: 键不存在时返回的默认值。

        Returns:
            命中则返回配置值，否则返回 ``default``。
        """
        if not key:
            return self.as_dict()
        node: Any = self._data
        for part in str(key).split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def as_dict(self) -> dict[str, Any]:
        """返回配置的深拷贝 dict，避免外部意外修改内部状态。"""
        return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        """支持 ``cfg["data.source.primary"]`` 下标访问（未命中抛 KeyError）。"""
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        """支持 ``"data.source.primary" in cfg`` 判断。"""
        sentinel = object()
        return self.get(key, sentinel) is not sentinel


def load_config(path: Optional[str | Path] = None) -> Config:
    """加载配置文件为 :class:`Config`。

    Args:
        path: 配置文件路径；默认使用 ``config/settings.yaml``。

    Returns:
        :class:`Config` 实例。若文件缺失或解析失败，则使用内置默认值，
        且 ``source_path`` 为 ``None``。文件存在时，其值会与内置默认值深合并，
        因而允许配置文件只写部分键。

    Notes:
        本函数不会抛出异常：文件不可读 / YAML 非法 / 顶层非 dict 时一律回退默认值。
    """
    config_path: Path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
    source_path: Optional[Path] = None

    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            loaded = None
        if isinstance(loaded, dict):
            data = _deep_merge(DEFAULT_SETTINGS, loaded)
            source_path = config_path

    return Config(data, source_path)


# 模块级单例，供全应用共享一份配置。
_config_singleton: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """获取全局配置单例。

    Args:
        reload: 为 ``True`` 时强制重新读取配置文件。

    Returns:
        全局共享的 :class:`Config` 实例。
    """
    global _config_singleton
    if _config_singleton is None or reload:
        _config_singleton = load_config()
    return _config_singleton
