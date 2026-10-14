"""数据源子包。

对外出口：统一接口 :class:`DataSource`、异常 :class:`DataSourceError`、
工具 :func:`normalize_bars` / :func:`derive_market`、注册表 :func:`get_source`。

导入本包会顺带加载并注册 AkShare（P1-S3）、BaoStock（P1-S4）、pytdx（P1-S5）实现
（对应依赖不可用时仅告警，不阻断）。
"""
from __future__ import annotations

import logging

from app.data.datasource.base import (
    BAR_COLUMNS,
    SOURCES,
    DataSource,
    DataSourceError,
    derive_market,
    get_source,
    normalize_bars,
    register_source,
)

logger = logging.getLogger(__name__)

# 触发 AkShare 实现注册（heavy import 失败时降级为告警）。
try:
    from app.data.datasource import akshare_src as _akshare_src  # noqa: F401
except Exception as exc:  # pragma: no cover - 环境相关
    logger.warning("AkShare 数据源未能加载并注册：%s", exc)

# 触发 BaoStock 兜底实现注册（依赖不可用时降级为告警，不阻断）。
try:
    from app.data.datasource import baostock_src as _baostock_src  # noqa: F401
except Exception as exc:  # pragma: no cover - 环境相关
    logger.warning("BaoStock 数据源未能加载并注册：%s", exc)

# 触发 pytdx（1 分钟历史）实现注册（依赖不可用时降级为告警，不阻断）。
try:
    from app.data.datasource import pytdx_src as _pytdx_src  # noqa: F401
except Exception as exc:  # pragma: no cover - 环境相关
    logger.warning("pytdx 数据源未能加载并注册：%s", exc)

__all__ = [
    "DataSource",
    "DataSourceError",
    "SOURCES",
    "get_source",
    "register_source",
    "normalize_bars",
    "derive_market",
    "BAR_COLUMNS",
]
