"""导出子包。

对外出口：
    * :func:`app.export.excel_exporter.export_screen_result`（选股结果导出 Excel）；
    * :func:`app.export.excel_exporter.build_workbook`（纯逻辑构造工作簿，不写盘）；
    * :class:`app.export.excel_exporter.ExcelExportError`（导出失败统一异常）；
    * :func:`app.export.image_exporter.export_kline_image`（K 线高清图导出 PNG）；
    * :class:`app.export.image_exporter.ImageExportError`（图像导出失败统一异常）。
"""
from __future__ import annotations

from app.export.excel_exporter import (
    ExcelExportError,
    build_workbook,
    export_screen_result,
)
from app.export.image_exporter import (
    ImageExportError,
    dpi_scale,
    ensure_png_path,
    export_kline_image,
    scale_pixmap,
    target_size,
)

__all__ = [
    "ExcelExportError",
    "build_workbook",
    "export_screen_result",
    "ImageExportError",
    "dpi_scale",
    "ensure_png_path",
    "export_kline_image",
    "scale_pixmap",
    "target_size",
]
