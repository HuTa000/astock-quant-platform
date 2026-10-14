"""SQLite 存储封装。

职责：
    * 建库 / 建表（执行 ``schema.sql``，幂等）；
    * 提供 K线、复权因子、自选股、备注、指标模板、选股模板、配置、元数据的读写 API。

设计要点：
    * 单文件 SQLite，连接开启 ``PRAGMA foreign_keys=ON`` 以启用级联删除；
    * 时间列以 ISO 文本存储，读写时统一转换；
    * 内部使用可重入锁 + ``check_same_thread=False``，保证跨线程调用安全
      （后续数据更新线程 / 事件回调可能来自非 UI 线程）；
    * ``get_bars`` 返回全局 Bar 契约列 ``[datetime, open, high, low, close, volume, amount]``，
      ``datetime`` 升序。

典型用法::

    from app.data.store.db import get_db
    db = get_db()                       # 取 config.data.db_path 建库并建表
    db.upsert_bars([bar1, bar2])
    df = db.get_bars("000001", "d")
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import pandas as pd

from app.core.config import PROJECT_ROOT, get_config
from app.data.models import (
    AlertRule,
    Bar,
    IndexConstituent,
    IndicatorTemplate,
    Note,
    ScreenerTemplate,
    StockInfo,
    WatchlistGroup,
    WatchlistItem,
)

logger = logging.getLogger(__name__)

# 建表脚本路径。
SCHEMA_PATH: Path = Path(__file__).resolve().parent / "schema.sql"

# 时间文本格式（ISO）。
DT_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# get_bars 返回的 Bar 契约列。
BAR_COLUMNS: list[str] = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


# --------------------------------------------------------------------------- #
# 时间转换工具
# --------------------------------------------------------------------------- #
def _parse_dt(text: str) -> datetime:
    """把 ISO 文本解析为 :class:`datetime`。

    Args:
        text: 形如 ``"2026-09-21 09:30:00"`` / ``"2026-09-21T09:30:00"`` / ``"2026-09-21"``。

    Returns:
        解析得到的 datetime。

    Raises:
        ValueError: 文本无法解析为时间。
    """
    if isinstance(text, datetime):
        return text
    raw = str(text).strip()
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间文本：{text!r}")


def _as_datetime(value: Any) -> datetime:
    """把多种时间表示（datetime/date/str/pandas.Timestamp）统一为 datetime。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return _parse_dt(value)
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime()
    raise TypeError(f"不支持的时间类型：{type(value)!r}")


def _dt_text(value: Any) -> str:
    """把时间值格式化为 ISO 文本（用于写库 / 作 where 边界）。"""
    return _as_datetime(value).strftime(DT_FORMAT)


def _none_if_nan(value: Any) -> Any:
    """把 NaN / NaT 归一化为 ``None``，其余原样返回。"""
    try:
        if value is None:
            return None
        # pandas 标量判断
        if value is pd.NaT:
            return None
        result = pd.isna(value)
        if isinstance(result, bool) and result:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _as_float(value: Any) -> Optional[float]:
    """尽力转换为 float；无法转换或为 NaN 时返回 ``None``。"""
    value = _none_if_nan(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bare_symbol(value: Any) -> str:
    """把任意写法的股票代码归一化为裸 6 位码（P7-S4）。

    数据源返回的是带市场前缀的 ``sh.600000``，而本地库 ``kline.symbol`` 用裸码
    ``600000``；两者混用会导致「明明拉过数据却判定为缺失」。无法提取 6 位数字时
    返回空串（由调用方跳过该行）。

    Args:
        value: 形如 ``"sh.600000"`` / ``"600000"`` / ``"600000.SH"`` 的值。

    Returns:
        裸 6 位码；无法识别时返回 ``""``。
    """
    if value is None:
        return ""
    match = re.search(r"(\d{6})", str(value))
    return match.group(1) if match else ""


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
class Database:
    """SQLite 存储封装。"""

    def __init__(self, db_path: Optional[Union[Path, str]] = None) -> None:
        """初始化（不建立连接）。

        Args:
            db_path: 数据库文件路径。``None`` 时取配置 ``data.db_path``
                （相对项目根 ``PROJECT_ROOT`` 解析）。特殊值 ``":memory:"`` 表示内存库。
        """
        self._lock: threading.RLock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

        if db_path is None:
            rel = get_config().get("data.db_path", "data/app.db")
            self.db_path: Union[Path, str] = (PROJECT_ROOT / str(rel)).resolve()
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path).resolve()

    # ---------------------------- 连接管理 ---------------------------- #
    @property
    def connection(self) -> sqlite3.Connection:
        """返回当前连接，必要时自动连接。"""
        return self._ensure_conn()

    def _ensure_conn(self) -> sqlite3.Connection:
        """确保连接已建立并返回。"""
        if self._conn is None:
            self.connect()
        assert self._conn is not None  # for type checkers
        return self._conn

    def connect(self) -> None:
        """建立数据库连接并开启外键约束。"""
        with self._lock:
            if self._conn is not None:
                return
            if isinstance(self.db_path, Path):
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON;")
            self._conn = conn
            logger.info("已连接数据库：%s", self.db_path)

    def close(self) -> None:
        """关闭连接（幂等）。"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                logger.info("已关闭数据库：%s", self.db_path)

    def init_schema(self) -> None:
        """执行 ``schema.sql`` 建表建索引（幂等）。"""
        with self._lock:
            conn = self._ensure_conn()
            script = SCHEMA_PATH.read_text(encoding="utf-8")
            conn.executescript(script)
            conn.commit()
            logger.info("建表完成（幂等）：%s", self.db_path)

    def __enter__(self) -> "Database":
        """支持 ``with Database(...) as db:``。"""
        self.connect()
        self.init_schema()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """退出上下文时关闭连接。"""
        self.close()

    # ---------------------------- 内部工具 ---------------------------- #
    @staticmethod
    def _where_range(
        column: str,
        start: Optional[Any],
        end: Optional[Any],
    ) -> tuple[list[str], list[Any]]:
        """构造 ``column >= start AND column <= end`` 片段。"""
        clauses: list[str] = []
        params: list[Any] = []
        if start is not None:
            clauses.append(f"{column} >= ?")
            params.append(_dt_text(start))
        if end is not None:
            clauses.append(f"{column} <= ?")
            params.append(_dt_text(end))
        return clauses, params

    # ------------------------------- K线 ------------------------------ #
    def upsert_bars(self, bars: Union[Sequence[Bar], pd.DataFrame]) -> int:
        """批量写入 / 更新 K线。

        Args:
            bars: :class:`Bar` 列表，或包含所需列的 ``DataFrame``。
                DataFrame 需含 ``symbol, period`` 与时间列（``dt`` 或 ``datetime``），
                以及 ``open, high, low, close, volume, amount``；``adjust`` 可选。

        Returns:
            写入的行数。

        Raises:
            TypeError: ``bars`` 类型不受支持。
            ValueError: ``DataFrame`` 缺少必需列。
        """
        rows = self._normalize_bars(bars)
        if not rows:
            return 0
        sql = (
            "INSERT OR REPLACE INTO kline "
            "(symbol, period, dt, open, high, low, close, volume, amount, adjust) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(sql, rows)
            conn.commit()
        logger.debug("upsert_bars 写入 %d 行", len(rows))
        return len(rows)

    @staticmethod
    def _normalize_bars(
        bars: Union[Sequence[Bar], pd.DataFrame],
    ) -> list[tuple[Any, ...]]:
        """把入参统一为待写入的元组列表。"""
        rows: list[tuple[Any, ...]] = []
        if isinstance(bars, pd.DataFrame):
            frame = bars
            time_col = "dt" if "dt" in frame.columns else (
                "datetime" if "datetime" in frame.columns else None
            )
            if time_col is None:
                raise ValueError("DataFrame 缺少时间列（dt 或 datetime）")
            required = ["symbol", "period", "open", "high", "low", "close", "volume", "amount"]
            missing = [c for c in required if c not in frame.columns]
            if missing:
                raise ValueError(f"DataFrame 缺少必需列：{missing}")
            has_adjust = "adjust" in frame.columns
            for record in frame.to_dict("records"):
                adjust = record.get("adjust", "") if has_adjust else ""
                rows.append(
                    (
                        record["symbol"],
                        record["period"],
                        _dt_text(record[time_col]),
                        _as_float(record["open"]),
                        _as_float(record["high"]),
                        _as_float(record["low"]),
                        _as_float(record["close"]),
                        _as_float(record["volume"]),
                        _as_float(record["amount"]),
                        str(adjust) if adjust is not None else "",
                    )
                )
        elif isinstance(bars, (list, tuple)):
            for bar in bars:
                if not isinstance(bar, Bar):
                    raise TypeError(f"bars 元素须为 Bar，实际为 {type(bar)!r}")
                rows.append(
                    (
                        bar.symbol,
                        bar.period,
                        _dt_text(bar.dt),
                        _as_float(bar.open),
                        _as_float(bar.high),
                        _as_float(bar.low),
                        _as_float(bar.close),
                        _as_float(bar.volume),
                        _as_float(bar.amount),
                        bar.adjust or "",
                    )
                )
        else:
            raise TypeError(f"不支持的 bars 类型：{type(bars)!r}")
        return rows

    def get_bars(
        self,
        symbol: str,
        period: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        adjust: str = "",
    ) -> pd.DataFrame:
        """读取 K线，返回 Bar 契约 DataFrame（datetime 升序）。

        Args:
            symbol: 标的代码。
            period: 周期。
            start: 起始时间（含），``datetime`` 或 ISO 文本。
            end: 结束时间（含），``datetime`` 或 ISO 文本。
            adjust: 复权口径，默认 ``''``。

        Returns:
            列 ``[datetime, open, high, low, close, volume, amount]`` 的 DataFrame；
            无数据时返回同列空表。
        """
        clauses = ["symbol = ?", "period = ?", "adjust = ?"]
        params: list[Any] = [symbol, period, adjust or ""]
        range_clauses, range_params = self._where_range("dt", start, end)
        clauses.extend(range_clauses)
        params.extend(range_params)
        sql = (
            "SELECT dt, open, high, low, close, volume, amount FROM kline "
            f"WHERE {' AND '.join(clauses)} ORDER BY dt ASC"
        )
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(sql, params).fetchall()
        if not raw:
            return pd.DataFrame(columns=BAR_COLUMNS)
        frame = pd.DataFrame(
            [
                {
                    "datetime": _parse_dt(row["dt"]),
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "amount": row["amount"],
                }
                for row in raw
            ]
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        for col in ("open", "high", "low", "close", "volume", "amount"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame[BAR_COLUMNS].reset_index(drop=True)

    def count_bars(
        self,
        symbol: Optional[str] = None,
        period: Optional[str] = None,
    ) -> int:
        """统计 K线行数（可按股 / 周期过滤）。"""
        clauses: list[str] = []
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if period is not None:
            clauses.append("period = ?")
            params.append(period)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(f"SELECT COUNT(*) AS n FROM kline{where}", params).fetchone()
        return int(row["n"]) if row else 0

    def list_kline_symbols(self, period: str = "d", adjust: str = "") -> list[str]:
        """列出库内指定周期 / 复权口径下**有 K 线数据**的标的代码（升序去重）。

        回测引擎（P7-S1）用它确定「股票池 = 本地库现有标的」；也便于数据
        健康检查 / 覆盖率统计。

        Args:
            period: 周期，默认 ``"d"``。
            adjust: 复权口径，默认 ``""``（未复权）。

        Returns:
            标的代码列表（升序）；无数据时返回空列表。
        """
        sql = (
            "SELECT DISTINCT symbol FROM kline WHERE period = ? AND adjust = ? "
            "ORDER BY symbol ASC"
        )
        with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(sql, [period, adjust or ""]).fetchall()
        return [str(row["symbol"]) for row in rows]

    # ---------------------------- 复权因子 ---------------------------- #
    def upsert_adj_factors(self, symbol: str, rows: Iterable[tuple[Any, Any]]) -> int:
        """写入 / 更新复权因子。

        Args:
            symbol: 标的代码。
            rows: ``[(dt, factor), ...]`` 可迭代序列。

        Returns:
            写入行数。
        """
        data = [(_dt_text(dt), float(factor)) for dt, factor in rows]
        if not data:
            return 0
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(
                "INSERT OR REPLACE INTO adj_factor (symbol, dt, factor) VALUES (?, ?, ?)",
                [(symbol, dt, factor) for dt, factor in data],
            )
            conn.commit()
        logger.debug("upsert_adj_factors(%s) 写入 %d 行", symbol, len(data))
        return len(data)

    def get_adj_factors(
        self,
        symbol: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> pd.DataFrame:
        """读取复权因子序列，返回列 ``[datetime, factor]``（datetime 升序）。"""
        clauses = ["symbol = ?"]
        params: list[Any] = [symbol]
        range_clauses, range_params = self._where_range("dt", start, end)
        clauses.extend(range_clauses)
        params.extend(range_params)
        sql = (
            "SELECT dt, factor FROM adj_factor "
            f"WHERE {' AND '.join(clauses)} ORDER BY dt ASC"
        )
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(sql, params).fetchall()
        if not raw:
            return pd.DataFrame(columns=["datetime", "factor"])
        frame = pd.DataFrame(
            [{"datetime": _parse_dt(r["dt"]), "factor": r["factor"]} for r in raw]
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
        return frame.reset_index(drop=True)

    # ----------------------------- 自选股 ----------------------------- #
    def create_group(self, name: str, sort_order: int = 0) -> int:
        """新建自选股分组，返回新分组 id。"""
        with self._lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                "INSERT INTO watchlist_group (name, sort_order) VALUES (?, ?)",
                (name, int(sort_order)),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_groups(self) -> list[WatchlistGroup]:
        """列出所有分组（按 sort_order, id 升序）。"""
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(
                "SELECT id, name, sort_order FROM watchlist_group "
                "ORDER BY sort_order ASC, id ASC"
            ).fetchall()
        return [
            WatchlistGroup(id=r["id"], name=r["name"], sort_order=r["sort_order"])
            for r in raw
        ]

    def delete_group(self, group_id: int) -> None:
        """删除分组（其下条目经外键级联删除）。"""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM watchlist_group WHERE id = ?", (group_id,))
            conn.commit()

    def rename_group(self, group_id: int, name: str) -> None:
        """重命名分组（P2-S6 新增）。

        Args:
            group_id: 分组 id。
            name: 新名称。

        Notes:
            仅更新 ``watchlist_group.name``；不存在的 id 不产生副作用（``UPDATE`` 影响 0 行）。
        """
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE watchlist_group SET name = ? WHERE id = ?", (str(name), int(group_id))
            )
            conn.commit()

    def add_watch_item(self, group_id: int, symbol: str) -> int:
        """向分组添加自选股，返回条目 id（已存在则返回既有 id）。"""
        with self._lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                "INSERT OR IGNORE INTO watchlist_item (group_id, symbol, added_at) "
                "VALUES (?, ?, ?)",
                (group_id, symbol, datetime.now().strftime(DT_FORMAT)),
            )
            conn.commit()
            if cur.lastrowid and cur.rowcount:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM watchlist_item WHERE group_id = ? AND symbol = ?",
                (group_id, symbol),
            ).fetchone()
            return int(row["id"])

    def remove_watch_item(self, group_id: int, symbol: str) -> None:
        """从分组移除自选股。"""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "DELETE FROM watchlist_item WHERE group_id = ? AND symbol = ?",
                (group_id, symbol),
            )
            conn.commit()

    def list_watch_items(self, group_id: Optional[int] = None) -> list[WatchlistItem]:
        """列出条目（可按分组过滤），按 id 升序。"""
        if group_id is None:
            sql = (
                "SELECT id, group_id, symbol, added_at, note_mark FROM watchlist_item "
                "ORDER BY id ASC"
            )
            params: list[Any] = []
        else:
            sql = (
                "SELECT id, group_id, symbol, added_at, note_mark FROM watchlist_item "
                "WHERE group_id = ? ORDER BY id ASC"
            )
            params = [group_id]
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(sql, params).fetchall()
        return [
            WatchlistItem(
                id=r["id"],
                group_id=r["group_id"],
                symbol=r["symbol"],
                added_at=_parse_dt(r["added_at"]),
                note_mark=bool(r["note_mark"]),
            )
            for r in raw
        ]

    def set_note_mark(self, group_id: int, symbol: str, mark: bool) -> None:
        """设置条目的备注标记。"""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE watchlist_item SET note_mark = ? WHERE group_id = ? AND symbol = ?",
                (1 if mark else 0, group_id, symbol),
            )
            conn.commit()

    # ---------------------------- 股票名录 ---------------------------- #
    def upsert_stock_list(self, rows: Union[Sequence[StockInfo], pd.DataFrame]) -> int:
        """批量写入 / 更新全 A 股名录。

        Args:
            rows: :class:`StockInfo` 序列，或包含 ``code`` / ``symbol`` 与 ``name``
                列的 ``DataFrame``（``market`` 可选）；入参为 ``DataFrame`` 时逐行取值，
                空 ``symbol`` 行跳过。

        Returns:
            写入的行数。

        Raises:
            TypeError: ``rows`` 类型不受支持（非序列 / 非 ``DataFrame``）。
        """
        data = self._normalize_stock_list(rows)
        if not data:
            return 0
        stamp = datetime.now().strftime(DT_FORMAT)
        prepared = [(symbol, name, market, stamp) for symbol, name, market in data]
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(
                "INSERT OR REPLACE INTO stock_list (symbol, name, market, updated_at) "
                "VALUES (?, ?, ?, ?)",
                prepared,
            )
            conn.commit()
        logger.debug("upsert_stock_list 写入 %d 行", len(prepared))
        return len(prepared)

    @staticmethod
    def _normalize_stock_list(
        rows: Union[Sequence[StockInfo], pd.DataFrame],
    ) -> list[tuple[str, str, str]]:
        """把名录入参统一为 ``[(symbol, name, market), ...]``（空 symbol 行跳过）。"""
        data: list[tuple[str, str, str]] = []
        if isinstance(rows, pd.DataFrame):
            frame = rows
            for record in frame.to_dict("records"):
                raw_symbol = record.get("code")
                if raw_symbol is None:
                    raw_symbol = record.get("symbol")
                symbol = str(raw_symbol).strip() if raw_symbol is not None else ""
                if not symbol or symbol.lower() == "nan":
                    continue
                name = str(record.get("name", "") or "").strip()
                market = str(record.get("market", "") or "").strip()
                data.append((symbol, name, market))
        elif isinstance(rows, (list, tuple)):
            for info in rows:
                if not isinstance(info, StockInfo):
                    raise TypeError(f"rows 元素须为 StockInfo，实际为 {type(info)!r}")
                symbol = str(info.symbol).strip()
                if not symbol:
                    continue
                data.append((symbol, str(info.name).strip(), str(info.market or "").strip()))
        else:
            raise TypeError(f"不支持的 rows 类型：{type(rows)!r}")
        return data

    def list_stocks(self, keyword: Optional[str] = None) -> list[StockInfo]:
        """列出全 A 股名录（按 ``symbol`` 升序）。

        Args:
            keyword: 关键词；非空时按 ``symbol LIKE %kw%`` 或 ``name LIKE %kw%`` 过滤。

        Returns:
            :class:`StockInfo` 列表。
        """
        clauses: list[str] = []
        params: list[Any] = []
        if keyword:
            like = f"%{keyword}%"
            clauses.append("(symbol LIKE ? OR name LIKE ?)")
            params.extend([like, like])
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT symbol, name, market, updated_at FROM stock_list "
            f"{where} ORDER BY symbol ASC"
        )
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(sql, params).fetchall()
        return [
            StockInfo(
                symbol=r["symbol"],
                name=r["name"],
                market=r["market"],
                updated_at=_parse_dt(r["updated_at"]),
            )
            for r in raw
        ]

    def count_stocks(self) -> int:
        """统计名录内的股票数量。"""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute("SELECT COUNT(*) AS n FROM stock_list").fetchone()
        return int(row["n"]) if row else 0

    # -------------------------- 指数成分股快照 -------------------------- #
    def upsert_index_constituents(
        self, rows: Union[Sequence[IndexConstituent], pd.DataFrame]
    ) -> int:
        """批量写入 / 更新指数成分股名单快照（P7-S4，幂等）。

        Args:
            rows: :class:`IndexConstituent` 序列，或含 ``index_code`` / ``symbol`` /
                ``name``（可选）/ ``effective_date`` 列的 ``DataFrame``；
                空 ``symbol`` / 空生效日的行跳过。``symbol`` 会自动归一化为裸 6 位码。

        Returns:
            写入的行数（``INSERT OR REPLACE``，重复快照不新增行）。

        Raises:
            TypeError: ``rows`` 类型不受支持（非序列 / 非 ``DataFrame``）。
        """
        data = self._normalize_index_constituents(rows)
        if not data:
            return 0
        with self._lock:
            conn = self._ensure_conn()
            conn.executemany(
                "INSERT OR REPLACE INTO index_constituent "
                "(index_code, symbol, name, effective_date) VALUES (?, ?, ?, ?)",
                data,
            )
            conn.commit()
        logger.debug("upsert_index_constituents 写入 %d 行", len(data))
        return len(data)

    @staticmethod
    def _normalize_index_constituents(
        rows: Union[Sequence[IndexConstituent], pd.DataFrame],
    ) -> list[tuple[str, str, str, str]]:
        """把入参统一为 ``[(index_code, symbol, name, effective_date_text), ...]``。

        ``effective_date`` 一律落为 ``YYYY-MM-DD`` 文本（PIT 比较只需日期粒度）。
        """
        data: list[tuple[str, str, str, str]] = []
        if isinstance(rows, pd.DataFrame):
            frame = rows
            for record in frame.to_dict("records"):
                raw_symbol = record.get("symbol")
                if raw_symbol is None:
                    raw_symbol = record.get("code")
                symbol = _bare_symbol(raw_symbol)
                index_code = str(record.get("index_code", "") or "").strip()
                raw_date = record.get("effective_date")
                if not symbol or not index_code or raw_date is None:
                    continue
                try:
                    eff = _as_datetime(raw_date).strftime("%Y-%m-%d")
                except (TypeError, ValueError):
                    continue
                name = str(record.get("name", "") or "").strip()
                if name.lower() == "nan":
                    name = ""
                data.append((index_code, symbol, name, eff))
        elif isinstance(rows, (list, tuple)):
            for item in rows:
                if not isinstance(item, IndexConstituent):
                    raise TypeError(f"rows 元素须为 IndexConstituent，实际为 {type(item)!r}")
                symbol = _bare_symbol(item.symbol)
                index_code = str(item.index_code or "").strip()
                if not symbol or not index_code:
                    continue
                eff = _as_datetime(item.effective_date).strftime("%Y-%m-%d")
                data.append((index_code, symbol, str(item.name or "").strip(), eff))
        else:
            raise TypeError(f"不支持的 rows 类型：{type(rows)!r}")
        return data

    def list_index_constituents(
        self, index_code: Optional[str] = None
    ) -> list[IndexConstituent]:
        """列出已落库的成分股快照（按 ``index_code`` / ``effective_date`` 升序）。

        Args:
            index_code: 指数代码；``None`` 返回所有指数。

        Returns:
            :class:`IndexConstituent` 列表；无数据返回空列表。
        """
        sql = (
            "SELECT index_code, symbol, name, effective_date FROM index_constituent"
        )
        params: list[Any] = []
        if index_code:
            sql += " WHERE index_code = ?"
            params.append(index_code)
        sql += " ORDER BY index_code ASC, effective_date ASC, symbol ASC"
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(sql, params).fetchall()
        return [
            IndexConstituent(
                index_code=str(r["index_code"]),
                symbol=str(r["symbol"]),
                name=str(r["name"] or ""),
                effective_date=_parse_dt(r["effective_date"]),
            )
            for r in raw
        ]

    def list_index_codes(self) -> list[str]:
        """列出库内已有成分股快照的指数代码（升序去重）。"""
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(
                "SELECT DISTINCT index_code FROM index_constituent ORDER BY index_code ASC"
            ).fetchall()
        return [str(r["index_code"]) for r in raw]

    def count_index_constituents(self, index_code: Optional[str] = None) -> int:
        """统计成分股快照行数（可按指数过滤）。"""
        sql = "SELECT COUNT(*) AS n FROM index_constituent"
        params: list[Any] = []
        if index_code:
            sql += " WHERE index_code = ?"
            params.append(index_code)
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------ 备注 ------------------------------ #
    def get_note(self, symbol: str) -> Optional[Note]:
        """读取个股备注；不存在返回 ``None``。"""
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT symbol, text, updated_at FROM stock_note WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return Note(
            symbol=row["symbol"],
            text=row["text"],
            updated_at=_parse_dt(row["updated_at"]),
        )

    def set_note(self, symbol: str, text: str) -> None:
        """写入 / 更新个股备注。"""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO stock_note (symbol, text, updated_at) "
                "VALUES (?, ?, ?)",
                (symbol, text, datetime.now().strftime(DT_FORMAT)),
            )
            conn.commit()

    def delete_note(self, symbol: str) -> None:
        """删除个股备注。"""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM stock_note WHERE symbol = ?", (symbol,))
            conn.commit()

    # --------------------------- 指标模板 ----------------------------- #
    def save_indicator_template(
        self,
        name: str,
        category: str,
        expr: str,
        template_id: Optional[int] = None,
    ) -> int:
        """保存指标模板（``template_id`` 为 ``None`` 时新增，否则更新），返回 id。"""
        now = datetime.now().strftime(DT_FORMAT)
        with self._lock:
            conn = self._ensure_conn()
            if template_id is None:
                cur = conn.execute(
                    "INSERT INTO indicator_template "
                    "(name, category, expr, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (name, category or "", expr, now, now),
                )
                conn.commit()
                return int(cur.lastrowid)
            conn.execute(
                "UPDATE indicator_template "
                "SET name = ?, category = ?, expr = ?, updated_at = ? WHERE id = ?",
                (name, category or "", expr, now, template_id),
            )
            conn.commit()
            return int(template_id)

    def list_indicator_templates(
        self, category: Optional[str] = None
    ) -> list[IndicatorTemplate]:
        """列出指标模板（可按分类过滤），按 id 升序。"""
        if category is None:
            sql = (
                "SELECT id, name, category, expr, created_at, updated_at "
                "FROM indicator_template ORDER BY id ASC"
            )
            params: list[Any] = []
        else:
            sql = (
                "SELECT id, name, category, expr, created_at, updated_at "
                "FROM indicator_template WHERE category = ? ORDER BY id ASC"
            )
            params = [category]
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(sql, params).fetchall()
        return [
            IndicatorTemplate(
                id=r["id"],
                name=r["name"],
                category=r["category"],
                expr=r["expr"],
                created_at=_parse_dt(r["created_at"]),
                updated_at=_parse_dt(r["updated_at"]),
            )
            for r in raw
        ]

    def delete_indicator_template(self, template_id: int) -> None:
        """删除指标模板。"""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM indicator_template WHERE id = ?", (template_id,))
            conn.commit()

    # --------------------------- 选股模板 ----------------------------- #
    def save_screener_template(
        self,
        name: str,
        conditions_json: str,
        template_id: Optional[int] = None,
    ) -> int:
        """保存选股模板（``template_id`` 为 ``None`` 时新增，否则更新），返回 id。"""
        now = datetime.now().strftime(DT_FORMAT)
        with self._lock:
            conn = self._ensure_conn()
            if template_id is None:
                cur = conn.execute(
                    "INSERT INTO screener_template (name, conditions_json, created_at) "
                    "VALUES (?, ?, ?)",
                    (name, conditions_json, now),
                )
                conn.commit()
                return int(cur.lastrowid)
            conn.execute(
                "UPDATE screener_template SET name = ?, conditions_json = ? WHERE id = ?",
                (name, conditions_json, template_id),
            )
            conn.commit()
            return int(template_id)

    def list_screener_templates(self) -> list[ScreenerTemplate]:
        """列出选股模板，按 id 升序。"""
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(
                "SELECT id, name, conditions_json, created_at FROM screener_template "
                "ORDER BY id ASC"
            ).fetchall()
        return [
            ScreenerTemplate(
                id=r["id"],
                name=r["name"],
                conditions_json=r["conditions_json"],
                created_at=_parse_dt(r["created_at"]),
            )
            for r in raw
        ]

    # --------------------------- 预警规则 ----------------------------- #
    def save_alert_rule(
        self,
        symbol: str,
        kind: str,
        params_json: str,
        enabled: bool = True,
        rule_id: Optional[int] = None,
    ) -> int:
        """保存预警规则（``rule_id`` 为 ``None`` 时新增，否则更新），返回规则 id。

        Args:
            symbol: 标的代码。
            kind: 规则类型（``ma_cross/macd_cross/volume_surge/expr``）。
            params_json: 参数 JSON 文本。
            enabled: 是否启用（默认启用）。
            rule_id: 更新目标 id；``None`` 表示新增。
        """
        now = datetime.now().strftime(DT_FORMAT)
        flag = 1 if enabled else 0
        with self._lock:
            conn = self._ensure_conn()
            if rule_id is None:
                cur = conn.execute(
                    "INSERT INTO alert_rule "
                    "(symbol, kind, params_json, enabled, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol, kind, params_json, flag, now, now),
                )
                conn.commit()
                return int(cur.lastrowid)
            conn.execute(
                "UPDATE alert_rule "
                "SET symbol = ?, kind = ?, params_json = ?, enabled = ?, updated_at = ? "
                "WHERE id = ?",
                (symbol, kind, params_json, flag, now, rule_id),
            )
            conn.commit()
            return int(rule_id)

    def list_alert_rules(self, symbol: Optional[str] = None) -> list[AlertRule]:
        """列出预警规则（可按 ``symbol`` 过滤），按 id 升序。"""
        if symbol is None:
            sql = (
                "SELECT id, symbol, kind, params_json, enabled, created_at, updated_at "
                "FROM alert_rule ORDER BY id ASC"
            )
            params: list[Any] = []
        else:
            sql = (
                "SELECT id, symbol, kind, params_json, enabled, created_at, updated_at "
                "FROM alert_rule WHERE symbol = ? ORDER BY id ASC"
            )
            params = [symbol]
        with self._lock:
            conn = self._ensure_conn()
            raw = conn.execute(sql, params).fetchall()
        return [
            AlertRule(
                id=r["id"],
                symbol=r["symbol"],
                kind=r["kind"],
                params_json=r["params_json"],
                enabled=bool(r["enabled"]),
                created_at=_parse_dt(r["created_at"]),
                updated_at=_parse_dt(r["updated_at"]),
            )
            for r in raw
        ]

    def delete_alert_rule(self, rule_id: int) -> None:
        """删除预警规则（不存在的 id 静默忽略）。"""
        with self._lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM alert_rule WHERE id = ?", (rule_id,))
            conn.commit()

    def set_alert_rule_enabled(self, rule_id: int, enabled: bool) -> None:
        """启用 / 停用预警规则。"""
        now = datetime.now().strftime(DT_FORMAT)
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                "UPDATE alert_rule SET enabled = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, now, rule_id),
            )
            conn.commit()

    # --------------------------- 配置 / 元数据 ------------------------- #
    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """读取配置项（DB 内 config 表）；不存在返回 ``default``。"""
        return self._get_kv("config", key, default)

    def set_config(self, key: str, value: str) -> None:
        """写入配置项。"""
        self._set_kv("config", key, value)

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """读取运行时元数据；不存在返回 ``default``。"""
        return self._get_kv("meta", key, default)

    def set_meta(self, key: str, value: str) -> None:
        """写入运行时元数据。"""
        self._set_kv("meta", key, value)

    def _get_kv(self, table: str, key: str, default: Optional[str]) -> Optional[str]:
        with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                f"SELECT value FROM {table} WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row is not None else default

    def _set_kv(self, table: str, key: str, value: str) -> None:
        with self._lock:
            conn = self._ensure_conn()
            conn.execute(
                f"INSERT OR REPLACE INTO {table} (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
            conn.commit()


# --------------------------------------------------------------------------- #
# 模块级单例
# --------------------------------------------------------------------------- #
_db_singleton: Optional[Database] = None


def get_db(reload: bool = False) -> Database:
    """获取全局数据库单例。

    Args:
        reload: 为 ``True`` 时重建（先关闭旧连接）。

    Returns:
        已连接并完成建表的 :class:`Database` 实例。
    """
    global _db_singleton
    if _db_singleton is None or reload:
        if _db_singleton is not None:
            _db_singleton.close()
        _db_singleton = Database()
        _db_singleton.connect()
        _db_singleton.init_schema()
    return _db_singleton
