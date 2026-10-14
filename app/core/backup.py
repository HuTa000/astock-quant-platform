"""一键备份 / 恢复（zip）—— P6-S1。

职责（《细化设计》§4「P6-S1」/ 需求方决策 2026-09-30）：
    * :func:`create_backup` —— 把「数据库 + 配置文件 + 缓存 + 导出产物」打成带时间戳的
      zip，存放于配置 ``backup.dir``（默认 ``data/backups``）；
    * :func:`restore_backup` —— 从指定 zip 还原用户数据；**恢复前先把当前
      app.db/settings.yaml 快照为 ``*.pre_restore``**（防误操作丢数据）；
    * 纯 stdlib（``zipfile/shutil``），无第三方依赖；切面可在 UI 层（P6-S4 菜单）调用。

备份内容（zip 内以「相对项目根」的路径存放）：
    * ``data/app.db``（SQLite 主库：自选/分组/备注/K线/模板/预警规则……）
    * ``config/settings.yaml``（配置）
    * ``data/cache/**``（Parquet 等分钟兜底缓存，若存在）
    * ``data/exports/**``（选股 Excel / K线 PNG 导出产物，若存在）

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）；本模块纯 Python
（UI 层负责弹 QFileDialog / 提示条）。
"""
from __future__ import annotations

import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from app.core.config import get_config

logger = logging.getLogger(__name__)

# 备份前缀（文件名：``astock_backup_YYYYMMDD_HHMMSS.zip``）。
_BACKUP_PREFIX: str = "astock_backup"
# 恢复前快照后缀。
_PRE_RESTORE_SUFFIX: str = ".pre_restore"


class BackupError(Exception):
    """备份 / 恢复失败统一异常（由 UI 层捕获并提示用户）。"""


# --------------------------------------------------------------------------- #
# 纯函数：路径解析
# --------------------------------------------------------------------------- #
def resolve_backup_dir(override: Optional[Union[str, Path]] = None) -> Path:
    """解析备份目录（读配置 ``backup.dir``；可被 ``override`` 覆盖）。

    Args:
        override: 显式目录；``None`` 读配置（默认 ``data/backups``）。

    Returns:
        绝对 ``Path``。
    """
    raw = override if override is not None else get_config().get("backup.dir", "data/backups")
    return Path(str(raw)).expanduser().resolve()


def backup_filename(now: Optional[datetime] = None) -> str:
    """生成备份文件名（带时间戳，纯函数便于测试）。

    Args:
        now: 时间；``None`` 取当前时间。

    Returns:
        形如 ``astock_backup_20260930_210512.zip``。
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{_BACKUP_PREFIX}_{stamp}.zip"


def default_backup_path(backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """默认备份目标路径 = 备份目录 + 时间戳文件名。"""
    return resolve_backup_dir(backup_dir) / backup_filename()


# --------------------------------------------------------------------------- #
# 备份范围构造（暴露为纯函数便于测试）
# --------------------------------------------------------------------------- #
def build_payload_paths(project_root: Optional[Union[str, Path]] = None) -> List[Path]:
    """构造待备份文件的绝对路径清单（存在者入列；目录整体打包）。

    Args:
        project_root: 项目根目录（默认 ``PROJECT_ROOT``）。

    Returns:
        已有路径列表：``data/app.db``、``config/settings.yaml``、``data/cache``、
        ``data/exports``。
    """
    root = Path(project_root) if project_root is not None else None
    if root is None:
        # 项目根：本文件位于 <root>/app/core/backup.py，故上溯两级。
        root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "data" / "app.db",
        root / "config" / "settings.yaml",
        root / "data" / "cache",
        root / "data" / "exports",
    ]
    return [p for p in candidates if p.exists()]


def make_archive(
    target: Union[str, Path],
    payload: List[Path],
    project_root: Optional[Union[str, Path]] = None,
    create_parent: bool = True,
) -> str:
    """把 ``payload`` 路径打成 zip（zip 内路径相对项目根）。

    Args:
        target: 目标 zip 路径。
        payload: 待打包的绝对路径清单（文件或目录）。
        project_root: 项目根目录（用于计算 zip 内相对路径）。
        create_parent: 目标父目录不存在时**自动创建**（一键备份语义）。

    Returns:
        str: 实际写入的 zip 路径。

    Raises:
        BackupError: 目标目录不可创建 / 打包失败。
    """
    target_p = Path(target)
    if create_parent:
        try:
            target_p.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise BackupError(f"无法创建备份目录：{target_p.parent}（{exc}）") from exc
    if not target_p.parent.exists():
        raise BackupError(f"备份目录不存在：{target_p.parent}")
    try:
        with zipfile.ZipFile(str(target_p), "w", zipfile.ZIP_DEFLATED) as zf:
            for item in payload:
                _add_to_zip(zf, item, project_root)
    except BackupError:
        raise
    except Exception as exc:  # noqa: BLE001 - 归一为明确异常
        logger.exception("打包备份失败")
        raise BackupError(f"备份失败：{exc}") from exc
    return str(target_p)


def _add_to_zip(zf: zipfile.ZipFile, item: Path, project_root: Optional[Path]) -> None:
    """把单个文件或目录递归写入 zip（zip 内路径相对 ``project_root``）。"""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    item = item.resolve()
    try:
        arcname = item.relative_to(root)
    except ValueError:
        # 越出项目根的文件：按顶层文件名存放（防御）。
        arcname = Path(item.name)
    if item.is_dir():
        for child in sorted(item.rglob("*")):
            if child.is_file():
                zf.write(str(child), str(arcname / child.relative_to(item)))
    else:
        zf.write(str(item), str(arcname))


# --------------------------------------------------------------------------- #
# 备份 / 恢复入口
# --------------------------------------------------------------------------- #
def create_backup(
    backup_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> str:
    """一键备份：DB + 配置 + 缓存 + 导出 打成时间戳 zip。

    Args:
        backup_dir: 备份目录；``None`` 读配置（默认 ``data/backups``）。
        project_root: 项目根（默认自动推导）。

    Returns:
        str: zip 路径。

    Raises:
        BackupError: 备份目录不可写 / 打包失败。
    """
    target = default_backup_path(backup_dir)
    payload = build_payload_paths(project_root)
    saved = make_archive(target, payload, project_root)
    logger.info("备份完成：%s（%d 项）", saved, len(payload))
    return saved


def restore_backup(
    archive: Union[str, Path],
    project_root: Optional[Union[str, Path]] = None,
) -> str:
    """从 zip 恢复数据；**恢复前先把当前 app.db/settings.yaml 快照为 pre_restore**。

    Args:
        archive: 待恢复的备份 zip 路径。
        project_root: 项目根（默认自动推导）。

    Returns:
        str: pre_restore 快照目录路径（空 List 时返回空串）。

    Raises:
        BackupError: 归档不存在 / 不是合法 zip / 覆盖失败。
    """
    archive_p = Path(archive)
    if not archive_p.is_file():
        raise BackupError(f"备份文件不存在：{archive_p}")
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 恢复前快照当前数据（防误操作）。
    snapshot_dir = root / "data" / f"pre_restore_{stamp}"
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name in ("app.db", "config"):
            src = root / "data" / name if name == "app.db" else root / name
            if src.exists():
                dest = snapshot_dir / src.name
                if src.is_dir():
                    shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
                else:
                    shutil.copy2(str(src), str(dest))
    except Exception as exc:  # noqa: BLE001 - 快照失败不阻塞恢复主流程
        logger.warning("恢复前快照失败（继续恢复）：%s", exc)

    # 2) 校验 zip 合法性与内容（防注入/损坏）。
    try:
        with zipfile.ZipFile(str(archive_p), "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise BackupError(f"备份文件损坏：{bad}")
            names = zf.namelist()
    except BackupError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise BackupError(f"无法读取备份文件：{exc}") from exc

    # 3) 解压覆盖（zip 内路径相对项目根；防路径穿越）。
    try:
        allowed = {n for n in names if not n.startswith(("/", "\\")) and ".." not in Path(n).parts}
        with zipfile.ZipFile(str(archive_p), "r") as zf:
            for name in allowed:
                zf.extract(name, str(root))
    except Exception as exc:  # noqa: BLE001
        logger.exception("恢复解压失败")
        raise BackupError(f"恢复失败：{exc}") from exc
    logger.info("恢复完成：%s（%d 项）", archive_p, len(allowed))
    return str(snapshot_dir)


__all__ = [
    "BackupError",
    "backup_filename",
    "build_payload_paths",
    "create_backup",
    "default_backup_path",
    "make_archive",
    "resolve_backup_dir",
    "restore_backup",
]