"""K 线高清图导出（P4-S4）。

职责：
    * 用 ``QWidget.grab()`` 抓取 K 线图表页（主图 + 副图 + 标注层 + 标题），
      按 DPI 放大后保存为高清 PNG，所见即所得；
    * **纯逻辑与 Qt 抓图解耦**：DPI 缩放系数（:func:`dpi_scale`）、目标尺寸
      （:func:`target_size`）、扩展名补全（:func:`ensure_png_path`）均为纯函数，
      不依赖 ``QApplication``，可无 GUI 单测；
    * 写盘失败归一为 :class:`ImageExportError`，绝不静默吞掉（复用 P4-S3 的
      :class:`~app.export.excel_exporter.ExcelExportError` 模式）。

DPI 约定：
    * Qt 逻辑 DPI 固定按 96 ppi 计；``dpi=300`` 即 ``300 / 96 = 3.125`` 倍放大；
    * 放大统一用 ``Qt.TransformationMode.SmoothTransformation``（双三次平滑），
      保证放大后清晰可读（禁用会锯齿的 ``FastTransformation``）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

logger = logging.getLogger(__name__)

# Qt 标准逻辑 DPI（96 ppi）。
QT_LOGICAL_DPI: float = 96.0

# 默认导出 DPI。
DEFAULT_DPI: int = 300


class ImageExportError(Exception):
    """图像导出失败统一异常（由 UI 层捕获并提示用户）。"""


def dpi_scale(dpi: int) -> float:
    """DPI → 相对 Qt 逻辑 DPI（96 ppi）的缩放系数（纯函数）。

    Args:
        dpi: 目标 DPI（正整数）。

    Returns:
        ``dpi / 96.0``。

    Raises:
        ImageExportError: ``dpi`` 非正整数。
    """
    try:
        value = int(dpi)
    except (TypeError, ValueError) as exc:
        raise ImageExportError(f"DPI 必须为正整数，收到：{dpi!r}") from exc
    if value <= 0:
        raise ImageExportError(f"DPI 必须为正整数，收到：{dpi!r}")
    return float(value) / QT_LOGICAL_DPI


def target_size(orig_w: int, orig_h: int, dpi: int) -> tuple[int, int]:
    """按 DPI 计算放大后的目标像素尺寸（纯函数，无 Qt 依赖）。

    Args:
        orig_w: 原始逻辑宽度（像素）。
        orig_h: 原始逻辑高度（像素）。
        dpi: 目标 DPI。

    Returns:
        ``(round(w * dpi/96), round(h * dpi/96))``，且每维至少为 1。
    """
    scale = dpi_scale(dpi)
    return (
        max(1, int(round(int(orig_w) * scale))),
        max(1, int(round(int(orig_h) * scale))),
    )


def ensure_png_path(path: Union[str, Path]) -> Path:
    """补全 / 归一 ``.png`` 扩展名（纯函数）。

    Args:
        path: 目标文件路径。

    Returns:
        扩展名为 ``.png`` 的 ``Path``（输入已为 ``.png`` 时保持原路径不变）。
    """
    target = Path(path)
    if target.suffix.lower() != ".png":
        target = target.with_suffix(".png")
    return target


def scale_pixmap(pixmap: QPixmap, scale: float) -> QPixmap:
    """按比例放大 ``QPixmap``（``SmoothTransformation`` 双三次平滑保清晰）。

    Args:
        pixmap: 原始 pixmap（屏幕逻辑像素）。
        scale: 放大系数（> 0）。

    Returns:
        放大后的 pixmap；``scale == 1`` 时原样返回（避免无谓拷贝）。
    """
    if scale <= 0:
        raise ImageExportError(f"缩放系数必须为正，收到：{scale}")
    width = pixmap.width()
    height = pixmap.height()
    target_w = max(1, int(round(width * scale)))
    target_h = max(1, int(round(height * scale)))
    if (target_w, target_h) == (width, height):
        return pixmap
    # 宽高均由同一系数缩放，比例天然保持，故用 IgnoreAspectRatio 精确落到
    # ``target_w × target_h``（KeepAspectRatio 会因取整把尺寸再压回比例，产生 1px 偏差）。
    return pixmap.scaled(
        target_w,
        target_h,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def export_kline_image(
    widget,
    path: Union[str, Path],
    dpi: int = DEFAULT_DPI,
    scale: Optional[float] = None,
) -> str:
    """抓取 K 线图 widget 并按 DPI 放大导出高清 PNG。

    Args:
        widget: 待抓图的 ``QWidget``（或其 ``grab()`` 返回 ``QPixmap`` 的等价对象）。
        path: 目标文件路径；扩展名缺失时自动补 ``.png``。
        dpi: 目标 DPI（默认 300）。
        scale: 可选缩放系数；为 ``None`` 时按 ``dpi / 96`` 推导。

    Returns:
        str: 实际写入的文件路径。

    Raises:
        ImageExportError: 目录不存在 / 抓图为空 / 写盘失败，异常信息含可读原因。
    """
    try:
        target = ensure_png_path(path)
        if not target.parent.exists():
            raise ImageExportError(f"保存目录不存在：{target.parent}")
        if widget is None:
            raise ImageExportError("抓图目标为空：K 线图尚未加载")
        try:
            pixmap = widget.grab()
        except Exception as exc:  # noqa: BLE001 - 抓图失败归一为明确异常
            raise ImageExportError(f"抓取 K 线图失败：{exc}") from exc
        if pixmap is None or pixmap.isNull():
            raise ImageExportError("抓取 K 线图为空（无数据或控件未显示）")

        factor = scale if scale is not None else dpi_scale(dpi)
        final = scale_pixmap(pixmap, factor)
        if not final.save(str(target), "PNG"):
            raise ImageExportError(f"PNG 写入失败：{target}")
    except ImageExportError:
        raise
    except Exception as exc:  # noqa: BLE001 - 统一归一为明确异常，绝不静默吞掉
        logger.exception("导出 K 线图失败")
        raise ImageExportError(f"导出失败：{exc}") from exc
    return str(target)


__all__ = [
    "ImageExportError",
    "DEFAULT_DPI",
    "QT_LOGICAL_DPI",
    "dpi_scale",
    "target_size",
    "ensure_png_path",
    "scale_pixmap",
    "export_kline_image",
]
