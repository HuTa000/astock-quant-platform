"""存储子包：SQLite 建表脚本与存储封装。

对外主要出口：
    * :class:`~app.data.store.db.Database` —— 存储封装类；
    * :func:`~app.data.store.db.get_db` —— 全局数据库单例（自动建库建表）。
"""
from __future__ import annotations

from app.data.store.db import BAR_COLUMNS, Database, get_db

__all__ = ["Database", "get_db", "BAR_COLUMNS"]
