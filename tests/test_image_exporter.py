"""P4-S4 K 线高清图导出 —— 纯逻辑 + 写盘测试。

覆盖：
    1. ``target_size`` 纯函数：DPI=300 时宽高 = 原宽高 × (300/96)，取整正确；
    2. ``dpi_scale``：96 基准缩放系数；非法 DPI 抛 :class:`ImageExportError`；
    3. ``ensure_png_path``：扩展名缺 ``.png`` 自动补、已有 ``.png`` 保持不变；
    4. ``export_kline_image``：正常导出生成 PNG（读回存在、非空、可被 ``QPixmap`` 加载）；
    5. 写盘失败（目录不存在）→ 抛 :class:`ImageExportError`，不静默。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from app.export.image_exporter import (  # noqa: E402
    ImageExportError,
    dpi_scale,
    ensure_png_path,
    export_kline_image,
    scale_pixmap,
    target_size,
)

pytest.importorskip("PySide6")
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """提供（或创建）全局 QApplication（仅用于 QPixmap 构造/加载）。"""
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeWidget:
    """仅暴露 ``grab()`` 的抓图目标，使写盘逻辑与真实 QWidget 解耦。"""

    def __init__(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap

    def grab(self) -> QPixmap:
        return self._pixmap


def _pixmap(width: int = 200, height: int = 100) -> QPixmap:
    """生成一张非空纯色 pixmap。"""
    pm = QPixmap(width, height)
    pm.fill("#1e1e1e")
    return pm


# --------------------------------------------------------------------------- #
# 1. 纯函数：DPI 缩放系数
# --------------------------------------------------------------------------- #
def test_dpi_scale_300() -> None:
    assert dpi_scale(300) == pytest.approx(300 / 96.0)


def test_dpi_scale_96_is_identity() -> None:
    assert dpi_scale(96) == pytest.approx(1.0)


def test_dpi_scale_invalid_raises() -> None:
    for bad in (0, -1):
        with pytest.raises(ImageExportError):
            dpi_scale(bad)
    with pytest.raises(ImageExportError):
        dpi_scale("abc")


# --------------------------------------------------------------------------- #
# 2. 纯函数：目标尺寸
# --------------------------------------------------------------------------- #
def test_target_size_scales_by_300_dpi() -> None:
    # 800/96*300 = 2500、600/96*300 = 1875，均为精确整数。
    assert target_size(800, 600, 300) == (2500, 1875)


def test_target_size_96_dpi_unchanged() -> None:
    assert target_size(640, 480, 96) == (640, 480)


def test_target_size_rounds_correctly() -> None:
    # 100 × (300/96) = 312.5，Python round 采用银行家舍入 → 312。
    assert target_size(100, 100, 300) == (312, 312)


def test_target_size_clamps_to_min_one() -> None:
    assert target_size(0, 0, 300) == (1, 1)


# --------------------------------------------------------------------------- #
# 3. 纯函数：扩展名补全
# --------------------------------------------------------------------------- #
def test_ensure_png_path_appends_extension(tmp_path) -> None:
    assert ensure_png_path(tmp_path / "kline") == tmp_path / "kline.png"


def test_ensure_png_path_keeps_existing_png(tmp_path) -> None:
    assert ensure_png_path(tmp_path / "kline.png") == tmp_path / "kline.png"


# --------------------------------------------------------------------------- #
# 4. 导出写盘（真实 QPixmap 落盘）
# --------------------------------------------------------------------------- #
def test_export_writes_loadable_png(qapp, tmp_path) -> None:
    widget = _FakeWidget(_pixmap(200, 100))
    path = tmp_path / "kline.png"

    saved = export_kline_image(widget, path, dpi=300)

    assert saved == str(path)
    assert path.exists()
    assert path.stat().st_size > 0

    # 读回可被 QPixmap 加载，且尺寸符合 DPI 缩放。
    loaded = QPixmap(str(path))
    assert not loaded.isNull()
    assert (loaded.width(), loaded.height()) == target_size(200, 100, 300)


def test_export_appends_png_extension(qapp, tmp_path) -> None:
    widget = _FakeWidget(_pixmap(120, 80))
    saved = export_kline_image(widget, tmp_path / "kline_no_ext", dpi=96)

    assert saved == str(tmp_path / "kline_no_ext.png")
    assert (tmp_path / "kline_no_ext.png").exists()


def test_export_scale_override(qapp, tmp_path) -> None:
    """``scale`` 显式提供时覆盖 dpi 推导。"""
    widget = _FakeWidget(_pixmap(100, 50))
    saved = export_kline_image(widget, tmp_path / "scaled.png", scale=2)

    loaded = QPixmap(saved)
    assert (loaded.width(), loaded.height()) == (200, 100)


def test_scale_pixmap_uses_smooth_transformation(qapp) -> None:
    """放大结果尺寸正确（内部使用 SmoothTransformation，离屏可测尺寸）。"""
    scaled = scale_pixmap(_pixmap(96, 96), 300 / 96.0)
    assert (scaled.width(), scaled.height()) == (300, 300)


# --------------------------------------------------------------------------- #
# 5. 写盘失败 → 明确异常，不静默
# --------------------------------------------------------------------------- #
def test_export_invalid_directory_raises(tmp_path) -> None:
    widget = _FakeWidget(_pixmap())
    bad_path = tmp_path / "no_such_dir" / "x.png"

    with pytest.raises(ImageExportError):
        export_kline_image(widget, bad_path)


def test_export_none_widget_raises(tmp_path) -> None:
    with pytest.raises(ImageExportError):
        export_kline_image(None, tmp_path / "x.png")
