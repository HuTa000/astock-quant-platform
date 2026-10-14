"""pytest 全局配置。

确保项目根目录（本文件所在目录）位于 ``sys.path`` 中，
从而使测试可以直接 ``import app``。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
