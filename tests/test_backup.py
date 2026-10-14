"""P6-S1 一键备份/恢复（zip）—— 纯逻辑测试。

覆盖（对齐《细化设计》§4「P6-S1」验收）：
    1. 文件名/路径纯函数（时间戳、目录解析、默认路径）；
    2. make_archive：文件与目录打包、zip 内相对路径、空 payload、目录不存在异常；
    3. create_backup：返回 zip 存在且包含 data/app.db + config/settings.yaml；
    4. restore_backup：完整往返 —— 造数据 → 备份 → 删除数据 → 恢复 → 数据一致；
    5. 恢复前快照（pre_restore 目录生成）；
    6. 边界：归档不存在 / 非 zip / 路径穿越（..）防御。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sqlite3  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from app.core.backup import (  # noqa: E402
    BackupError,
    backup_filename,
    build_payload_paths,
    create_backup,
    default_backup_path,
    make_archive,
    resolve_backup_dir,
    restore_backup,
)
from app.data.store.db import Database  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. 纯函数
# --------------------------------------------------------------------------- #
def test_backup_filename_timestamp() -> None:
    assert backup_filename(datetime(2026, 9, 30, 21, 5, 12)) == "astock_backup_20260930_210512.zip"


def test_backup_filename_now() -> None:
    name = backup_filename()
    assert name.startswith("astock_backup_")
    assert name.endswith(".zip")


def test_resolve_backup_dir_override() -> None:
    assert resolve_backup_dir(Path("x/y")) == Path("x/y").resolve()
    assert resolve_backup_dir("x/y") == Path("x/y").resolve()


def test_default_backup_path() -> None:
    path = default_backup_path()
    assert path.parent == resolve_backup_dir()
    assert path.name.startswith("astock_backup_")


# --------------------------------------------------------------------------- #
# 2. 打包
# --------------------------------------------------------------------------- #
def _build_tree(root: Path) -> None:
    """构造最小项目树：data/app.db(合法 SQLite) + config/settings.yaml + cache + exports。"""
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data" / "cache").mkdir(parents=True, exist_ok=True)
    (root / "data" / "exports").mkdir(parents=True, exist_ok=True)
    # 真实空 SQLite 库（否则 sqlite3 连接报 "file is not a database"）。
    conn = sqlite3.connect(str(root / "data" / "app.db"))
    conn.execute("CREATE TABLE seed (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("INSERT INTO seed(v) VALUES ('base')")
    conn.commit()
    conn.close()
    (root / "config" / "settings.yaml").write_text("data:\n  source:\n    primary: baostock\n")
    (root / "data" / "cache" / "a.pq").write_bytes(b"PARQUET")
    (root / "data" / "exports" / "x.png").write_bytes(b"PNG")


def test_build_payload_paths(tmp_path) -> None:
    _build_tree(tmp_path)
    payload = build_payload_paths(tmp_path)
    # 文件 + 目录均在列（目录整体打包）。
    assert any(p.name == "app.db" and p.is_file() for p in payload)
    assert any(p.name == "settings.yaml" and p.is_file() for p in payload)
    assert any(p.name == "cache" and p.is_dir() for p in payload)
    assert any(p.name == "exports" and p.is_dir() for p in payload)


def test_make_archive_flat_and_dir(tmp_path) -> None:
    _build_tree(tmp_path)
    target = tmp_path / "out.zip"
    payload = build_payload_paths(tmp_path)
    saved = make_archive(target, payload, tmp_path)
    assert saved == str(target)
    assert target.exists()

    import zipfile

    with zipfile.ZipFile(str(target)) as zf:
        names = sorted(zf.namelist())
    assert "data/app.db" in names
    assert "config/settings.yaml" in names
    assert "data/cache/a.pq" in names
    assert "data/exports/x.png" in names


def test_make_archive_dir_missing_auto_creates(tmp_path) -> None:
    """父目录不存在时一键备份自动创建（不报错）。"""
    saved = make_archive(tmp_path / "a" / "b" / "x.zip", [], tmp_path)
    assert Path(saved).exists()


def test_backup_dir_collision_raises(tmp_path) -> None:
    """目录创建失败路径：父路径被文件占据 → BackupError。"""
    blocker = tmp_path / "block"
    blocker.write_bytes(b"file blocks dir")
    with pytest.raises(BackupError):
        make_archive(blocker / "x.zip", [], tmp_path)


def test_make_archive_empty_payload(tmp_path) -> None:
    saved = make_archive(tmp_path / "empty.zip", [], tmp_path)
    assert Path(saved).exists()


# --------------------------------------------------------------------------- #
# 3. create_backup
# --------------------------------------------------------------------------- #
def test_create_backup_roundtrip_dir(tmp_path) -> None:
    _build_tree(tmp_path)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    saved = create_backup(backup_dir, tmp_path)
    assert Path(saved).is_file()
    assert Path(saved).suffix == ".zip"
    import zipfile

    with zipfile.ZipFile(saved) as zf:
        names = set(zf.namelist())
    assert {"data/app.db", "config/settings.yaml"} <= names


# --------------------------------------------------------------------------- #
# 4. 完整往返：备份 → 毁数据 → 恢复 → 一致
# --------------------------------------------------------------------------- #
def test_restore_roundtrip_recovers_data(tmp_path) -> None:
    _build_tree(tmp_path)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive = create_backup(backup_dir, tmp_path)

    # 用损坏数据覆盖（模拟误操作：文件还在但内容坏了）。
    (tmp_path / "data" / "app.db").write_bytes(b"CORRUPTED_DB")
    (tmp_path / "config" / "settings.yaml").write_text("data:\n  source:\n    primary: broken\n")

    # 恢复。
    snapshot = restore_backup(archive, tmp_path)

    # 数据恢复为备份时内容（合法 SQLite）。
    conn = sqlite3.connect(str(tmp_path / "data" / "app.db"))
    try:
        row = conn.execute("SELECT v FROM seed WHERE id=1").fetchone()
    finally:
        conn.close()
    assert row == ("base",)
    assert "baostock" in (tmp_path / "config" / "settings.yaml").read_text()

    # 恢复期间生成的 pre_restore 快照保留了「被覆盖前的坏数据」→ 证明预快照保护生效。
    snap_dir = Path(snapshot)
    assert snap_dir.is_dir()
    assert (snap_dir / "app.db").read_bytes() == b"CORRUPTED_DB"
    assert (snap_dir / "config" / "settings.yaml").exists()


def test_restore_archive_missing(tmp_path) -> None:
    with pytest.raises(BackupError):
        restore_backup(tmp_path / "nope.zip")


def test_restore_invalid_zip(tmp_path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip file at all " * 10)
    with pytest.raises(BackupError):
        restore_backup(bad)


def test_restore_blocked_traversal(tmp_path) -> None:
    """zip 内 ``../evil`` 不落地（路径穿越防御）。"""
    import zipfile

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(str(evil), "w") as zf:
        zf.writestr("../evil.txt", "x")
    restore_backup(evil, tmp_path)
    assert not (tmp_path.parent / "evil.txt").exists()


# --------------------------------------------------------------------------- #
# 5. 真实 DB 往返（把 SQLite 落进备份再恢复）
# --------------------------------------------------------------------------- #
def test_real_db_roundtrip(tmp_path) -> None:
    "备份 → 覆盖 → 恢复 → 自选股表数据完好。"
    _build_tree(tmp_path)
    db_path = tmp_path / "data" / "app.db"
    # 追加真实业务表数据。
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE watchlist (id INTEGER PRIMARY KEY, symbol TEXT)")
        conn.execute("INSERT INTO watchlist(symbol) VALUES ('000001')")
        conn.commit()
    finally:
        conn.close()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive = create_backup(backup_dir, tmp_path)
    (tmp_path / "data" / "app.db").write_bytes(b"OVERWRITTEN")
    restore_backup(archive, tmp_path)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT symbol FROM watchlist").fetchall()
    finally:
        conn.close()
    assert rows == [("000001",)]


# --------------------------------------------------------------------------- #
# 6. UI 层接线（MainWindow 菜单）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_menu_backup_restore_actions_exist(qapp) -> None:
    """「数据(D)」菜单含「备份数据」「恢复数据」，并置于「退出」之前。"""
    win = _make_window(qapp, _mem_db())
    try:
        texts = [a.text() for a in win._data_menu.actions() if a.text()]
        assert "备份数据(&B)" in texts
        assert "恢复数据(&R)" in texts
        assert texts.index("备份数据(&B)") < texts.index("恢复数据(&R)")
    finally:
        win.close()


def test_on_backup_creates_zip_and_status(qapp, monkeypatch, tmp_path) -> None:
    db = _mem_db()
    _seed_minute(db, "000001", 10)
    win = _make_window(qapp, db)
    try:
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        out: dict = {}
        monkeypatch.setattr(
            "app.ui.main_window.create_backup",
            lambda *a, **k: (out.update({"called": True}) or str(backup_dir / "x.zip")),
        )
        win._on_backup()
        assert out.get("called") is True
        assert "备份完成" in win.statusBar().currentMessage()
    finally:
        win.close()


def test_on_restore_cancel_no_dialog(qapp, monkeypatch) -> None:
    win = _make_window(qapp, _mem_db())
    try:
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getOpenFileName",
            lambda *a, **k: ("", ""),
        )
        called: list = []
        monkeypatch.setattr(
            "app.ui.main_window.restore_backup",
            lambda *a, **k: called.append(1),
        )
        win._on_restore()
        assert called == []
    finally:
        win.close()


def test_on_restore_yes_calls_restore(qapp, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    win = _make_window(qapp, _mem_db())
    try:
        monkeypatch.setattr(
            "app.ui.main_window.QFileDialog.getOpenFileName",
            lambda *a, **k: (str(Path(tmp_x()) / "b.zip"), ""),
        )
        # 用真实临时文件避免路径检查。
        fake = Path(tmp_x()) / "b.zip"
        fake.write_bytes(b"not a zip")
        monkeypatch.setattr(
            "app.ui.main_window.QMessageBox.question",
            staticmethod(
                lambda *a, **k: QMessageBox.StandardButton.Yes
            ),
        )
        called: list = []
        monkeypatch.setattr(
            "app.ui.main_window.restore_backup",
            lambda *a, **k: (called.append(a) or "snap_dir"),
        )
        win._on_restore()
        assert called and called[0][0] == str(fake)
        assert "恢复完成" in win.statusBar().currentMessage()
    finally:
        win.close()


def tmp_x() -> str:
    import tempfile

    return tempfile.gettempdir()


def _mem_db() -> "Database":
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


def _make_window(qapp, db):
    from app.data.update_service import UpdateService
    from app.ui.main_window import MainWindow

    class _FakeSource:
        name = "fake"

        def get_kline(self, *a, **k):
            return None

        def get_minute(self, *a, **k):
            return None

        def list_stocks(self):
            return None

        def get_spot(self):
            return None

        def get_adj_factor(self, *a, **k):
            return None

        def get_fundamental(self, *a, **k):
            return None

    service = UpdateService(db=db, sources={"baostock": _FakeSource(), "akshare": _FakeSource()})
    return MainWindow(db=db, service=service)


def _seed_minute(db, symbol: str, n: int) -> None:
    import numpy as np
    import pandas as pd

    base = pd.Timestamp("2026-01-05") + pd.Timedelta(hours=9, minutes=30)
    times = [base + pd.Timedelta(minutes=i) for i in range(n)]
    closes = [10.0 + i * 0.05 for i in range(n)]
    frame = pd.DataFrame(
        {
            "datetime": times,
            "open": [c - 0.02 for c in closes],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "amount": [1_000_000.0] * n,
        }
    )
    frame["symbol"] = symbol
    frame["period"] = "1m"
    frame["adjust"] = ""
    db.upsert_bars(frame)