"""指数成分股 PIT（Point-In-Time）工具（P7-S4）。

业务背景（为什么需要这个模块）
    「拿**今天**还在沪深300 里的股票，回溯到 2015 年做因子 / 回测研究」= 幸存者偏差：
    样本天然只包含「活到今天」的公司，退市、被剔除、被吸并的股票全部缺席（阶段 4 实测
    2015 年至今累计出现过 667 只成分股，而「当前成分」只有 300 只），收益因此被系统性
    高估。本模块把名单变成**当时可得（point-in-time）**的口径——某交易日只允许使用
    「该日之前最近一份生效名单」中的股票，是量化并消除该偏差的基础设施。

设计约定
    * **零外部依赖**：只用标准库 + pandas；不 import Qt、不 import 数据库
      （数据来源由调用方传入），因此可直接单测；
    * **输入统一**：名单记录 = :class:`~app.data.models.IndexConstituent` 序列、
      等价的 ``(index_code, symbol, name, effective_date)`` 元组序列，或含
      ``index_code`` / ``symbol`` / ``effective_date`` 列的 ``DataFrame``；
    * **不抛业务异常**：非法 / 空输入一律降级为「空结果」，不阻断调用链；
    * 时间一律按**日期粒度**比较（成分名单按日生效，无盘中粒度）。

术语
    * **快照（snapshot）**：某一生效日对应的完整成分名单（300 只）。
    * **PIT 口径**：某交易日「当时能看到」的名单 = 生效日 ≤ 该日 的最近一份快照。
    * **并集（union）**：某区间内所有快照出现过的标的（补数据用；不补必有幸存者偏差）。
"""
from __future__ import annotations

import bisect
import re
from datetime import date, datetime
from typing import Any, Iterator, Optional, Sequence

import pandas as pd

# 日期文本格式。
DATE_FORMAT: str = "%Y-%m-%d"

# 默认指数：沪深300（阶段 1~3 的研究池口径）。
DEFAULT_INDEX_CODE: str = "sh.000300"

__all__ = [
    "DATE_FORMAT",
    "DEFAULT_INDEX_CODE",
    "membership_matrix",
    "normalize_symbol",
    "pit_members",
    "pit_snapshot",
    "quarter_ends",
    "snapshot_dates",
    "to_date_str",
    "universe_union",
]


def normalize_symbol(value: Any) -> str:
    """把任意写法的股票代码归一化为裸 6 位码。

    Args:
        value: 形如 ``"sh.600000"`` / ``"600000.SH"`` / ``"600000"`` / ``600000`` 的值。

    Returns:
        裸 6 位码；无法提取时返回 ``""``（调用方应跳过该行）。
    """
    if value is None:
        return ""
    match = re.search(r"(\d{6})", str(value))
    return match.group(1) if match else ""


def to_date_str(value: Any) -> str:
    """把日期值统一为 ``YYYY-MM-DD`` 文本（仅日期粒度）。

    Args:
        value: ``datetime`` / ``date`` / ``pandas.Timestamp`` / ``"20260930"`` /
            ``"2026-09-30"`` / ``"2026/09/30"`` 等。

    Returns:
        ``YYYY-MM-DD`` 文本；空 / NaT / 无法识别时返回 ``""``。
    """
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        try:
            if pd.isna(value):  # 覆盖 pandas.NaT（NaT 是 datetime 的子类）
                return ""
        except (TypeError, ValueError):
            pass
        return value.strftime(DATE_FORMAT)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    compact = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", text)
    if compact:
        return f"{compact.group(1)}-{compact.group(2)}-{compact.group(3)}"
    return text[:10].replace("/", "-")


def quarter_ends(start: Any, end: Any = None) -> list[str]:
    """生成 ``start`` ~ ``end`` 之间的**季末日期**列表（成分股快照采样点）。

    沪深300 每年 6 月 / 12 月定期调仓，季末采样足以覆盖每一份名单——数据源返回的是
    「查询日当天或之前最近一份生效名单」，同一份名单会被多次命中，落库时按
    ``(index_code, symbol, effective_date)`` 主键自动去重，因此多采样无副作用。

    Args:
        start: 起始日期（含）。
        end: 结束日期（含）；``None`` 取今天。

    Returns:
        ``YYYY-MM-DD`` 列表（升序）；起止非法或区间为空时返回空列表。
    """
    start_s = to_date_str(start)
    end_s = to_date_str(end) or date.today().strftime(DATE_FORMAT)
    if not start_s or start_s > end_s:
        return []
    out: list[str] = []
    for year in range(int(start_s[:4]), int(end_s[:4]) + 1):
        for month_day in ("03-31", "06-30", "09-30", "12-31"):
            day = f"{year}-{month_day}"
            if start_s <= day <= end_s:
                out.append(day)
    return out


def _normalize_records(records: Any) -> list[tuple[str, str, str]]:
    """把入参统一为 ``[(index_code, symbol, effective_date_text), ...]``（丢弃非法行）。"""
    out: list[tuple[str, str, str]] = []
    if records is None:
        return out
    if isinstance(records, pd.DataFrame):
        for rec in records.to_dict("records"):
            raw_symbol = rec.get("symbol")
            if raw_symbol is None:
                raw_symbol = rec.get("code")
            symbol = normalize_symbol(raw_symbol)
            index_code = str(rec.get("index_code", "") or "").strip()
            eff = to_date_str(rec.get("effective_date"))
            if symbol and index_code and eff:
                out.append((index_code, symbol, eff))
        return out
    for item in records:
        if isinstance(item, (tuple, list)) and len(item) >= 4:
            index_code = str(item[0] or "").strip()
            symbol = normalize_symbol(item[1])
            eff = to_date_str(item[3])
        else:
            index_code = str(getattr(item, "index_code", "") or "").strip()
            symbol = normalize_symbol(getattr(item, "symbol", None))
            eff = to_date_str(getattr(item, "effective_date", None))
        if symbol and index_code and eff:
            out.append((index_code, symbol, eff))
    return out


def _snapshots(
    records: Any, index_code: Optional[str] = None
) -> list[tuple[str, frozenset[str]]]:
    """归并为「生效日 → 成员集合」的升序快照序列（内部工具）。"""
    buckets: dict[str, set[str]] = {}
    for code, symbol, eff in _normalize_records(records):
        if index_code and code != index_code:
            continue
        buckets.setdefault(eff, set()).add(symbol)
    return [(day, frozenset(buckets[day])) for day in sorted(buckets)]


def snapshot_dates(records: Any, index_code: Optional[str] = None) -> list[str]:
    """列出已有的快照生效日（升序去重）。"""
    return [
        day
        for day, _ in _snapshots(records, index_code)
    ]


def pit_snapshot(
    records: Any, as_of: Any, index_code: Optional[str] = None
) -> tuple[str, list[str]]:
    """取 ``as_of`` 当日**可用**的名单（生效日 ≤ as_of 的最近一份快照）。

    **严格 PIT**：``as_of`` 早于首份名单时返回 ``("", [])``——宁可承认「当时没有
    样本」，也绝不用未来名单（那是未来函数）。

    Args:
        records: 名单记录（见模块 docstring 的输入约定）。
        as_of: 目标交易日。
        index_code: 指数代码；``None`` 表示不区分指数。

    Returns:
        ``(effective_date, members)``；无可用名单时为 ``("", [])``。
    """
    snaps = _snapshots(records, index_code)
    target = to_date_str(as_of)
    if not snaps or not target:
        return "", []
    days = [day for day, _ in snaps]
    pos = bisect.bisect_right(days, target) - 1
    if pos < 0:
        return "", []
    day, members = snaps[pos]
    return day, sorted(members)


def pit_members(records: Any, as_of: Any, index_code: Optional[str] = None) -> list[str]:
    """``pit_snapshot`` 的薄封装：只要成员列表（升序）。"""
    return pit_snapshot(records, as_of, index_code)[1]


def universe_union(
    records: Any,
    index_code: Optional[str] = None,
    start: Any = None,
    end: Any = None,
) -> list[str]:
    """某区间内所有快照出现过的成分股**并集**（升序去重）。

    用途：补数据清单——只补「当前成分」必然漏掉退市 / 被剔除的股票，补全并集才可能
    真正消除幸存者偏差。

    Args:
        records: 名单记录。
        index_code: 指数代码；``None`` 表示不区分。
        start: 只统计生效日 ≥ ``start`` 的快照；``None`` 不限。
        end: 只统计生效日 ≤ ``end`` 的快照；``None`` 不限。

    Returns:
        裸 6 位码列表（升序）。
    """
    start_s = to_date_str(start) if start is not None else ""
    end_s = to_date_str(end) if end is not None else ""
    seen: set[str] = set()
    for code, symbol, eff in _normalize_records(records):
        if index_code and code != index_code:
            continue
        if start_s and eff < start_s:
            continue
        if end_s and eff > end_s:
            continue
        seen.add(symbol)
    return sorted(seen)


def membership_matrix(
    dates: Sequence[Any],
    symbols: Sequence[str],
    records: Any,
    index_code: Optional[str] = None,
) -> pd.DataFrame:
    """构造 ``dates × symbols`` 的**布尔成分掩码**（``True`` = 当日为该指数成分）。

    实现按快照**分段赋值**（47 份快照 → 47 次赋值），不做「逐日 × 逐股」循环，
    因此可直接用于 2857 个交易日 × 667 只标的的量级。

    Args:
        dates: 交易日序列（作为行索引）。
        symbols: 标的序列（作为列；内部归一化为裸码并去重）。
        records: 名单记录。
        index_code: 指数代码；``None`` 表示不区分。

    Returns:
        布尔 ``DataFrame``：行 = 交易日，列 = 标的；无可用快照时全为 ``False``。
        早于首份名单的交易日严格保持 ``False``（PIT 口径）。
    """
    index = pd.Index([to_date_str(d) for d in dates], name="date")
    columns: list[str] = []
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized and normalized not in columns:
            columns.append(normalized)
    frame = pd.DataFrame(False, index=index, columns=columns)
    if frame.empty or not columns:
        return frame
    snaps = _snapshots(records, index_code)
    if not snaps:
        return frame
    days = [day for day, _ in snaps]
    for position, (effective, members) in enumerate(snaps):
        upper = days[position + 1] if position + 1 < len(days) else None
        if upper is None:
            mask = index >= effective
        else:
            mask = (index >= effective) & (index < upper)
        if not bool(mask.any()):
            continue
        hit = [symbol for symbol in columns if symbol in members]
        if hit:
            frame.loc[mask, hit] = True
    return frame
