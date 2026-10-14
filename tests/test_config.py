"""配置加载单测（P1-S1 验收补充）。"""
from __future__ import annotations

from pathlib import Path

from app.core.config import Config, DEFAULT_SETTINGS, get_config, load_config


def test_settings_yaml_loaded_as_dict_and_dotted_get() -> None:
    """settings.yaml 应加载为 dict，且 dotted-key 读取正确（核心验收）。"""
    cfg = load_config()
    assert isinstance(cfg.as_dict(), dict)
    assert cfg.get("data.source.primary") == "baostock"
    assert cfg.get("data.source.fallback") == "akshare"
    assert cfg.get("data.source.minute") == "pytdx"
    assert cfg.get("data.db_path") == "data/app.db"
    assert cfg.get("data.indicator_engine") == "pandas_ta"
    assert cfg.get("ui.theme") == "dark_red_green"
    assert cfg.get("ui.red_color") == "#e74c3c"
    assert cfg.get("ui.green_color") == "#2ecc71"
    assert cfg.get("sim.initial_cash") == 100000
    assert cfg.get("sim.speed_default") == 1
    assert cfg.get("sim.speed_map") == {1: 400, 2: 200, 5: 80, 10: 40}
    assert cfg.get("update.batch_size") == 50
    assert cfg.get("export.excel_path") == "data/exports"
    assert cfg.get("export.image_dpi") == 300
    assert cfg.get("backup.dir") == "data/backups"


def test_missing_file_falls_back_to_defaults() -> None:
    """文件缺失时应回退内置默认值，且 source_path 为 None。"""
    cfg = load_config(Path("this-file-does-not-exist.yaml"))
    assert cfg.source_path is None
    assert cfg.get("data.source.primary") == "baostock"
    assert cfg.as_dict() == DEFAULT_SETTINGS


def test_get_returns_default_for_unknown_key() -> None:
    """未知键应返回调用方提供的默认值。"""
    cfg = Config({"a": {"b": 1}})
    assert cfg.get("a.b") == 1
    assert cfg.get("a.c", "x") == "x"
    assert cfg.get("no.such.key", 0) == 0


def test_contains_and_getitem() -> None:
    """``in`` 与下标访问行为正确。"""
    cfg = Config({"a": {"b": 2}})
    assert "a.b" in cfg
    assert "a.c" not in cfg
    assert cfg["a.b"] == 2


def test_get_config_singleton() -> None:
    """get_config 应返回同一实例（除非 reload）。"""
    assert get_config() is get_config()
    assert get_config().get("data.source.primary") == "baostock"
