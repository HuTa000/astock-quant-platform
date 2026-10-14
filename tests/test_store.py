"""存储层单测（P1-S2 验收门禁）。

全部用例使用 ``tmp_path`` 下的临时数据库，绝不触碰真实库 ``data/app.db``。
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.data.models import Bar
from app.data.store.db import BAR_COLUMNS, Database


@pytest.fixture()
def db(tmp_path):
    """提供基于临时文件的已建表 :class:`Database`，用后关闭。"""
    database = Database(tmp_path / "test.db")
    database.init_schema()
    yield database
    database.close()


# --------------------------------------------------------------------------- #
# 建表
# --------------------------------------------------------------------------- #
def test_init_schema_creates_all_tables(db: Database) -> None:
    """init_schema 后 9 张表应全部就绪。"""
    rows = db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r["name"] for r in rows}
    expected = {
        "watchlist_group",
        "watchlist_item",
        "stock_note",
        "kline",
        "adj_factor",
        "indicator_template",
        "screener_template",
        "config",
        "meta",
    }
    assert expected.issubset(names), f"缺少表：{expected - names}"

    # 索引也应创建
    idx = {r["name"] for r in db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()}
    assert "idx_kline_symbol_period_dt" in idx


def test_init_schema_is_idempotent(db: Database) -> None:
    """重复执行 init_schema 不应报错、不应丢表。"""
    db.init_schema()
    db.init_schema()
    assert db.count_bars() == 0


# --------------------------------------------------------------------------- #
# K线
# --------------------------------------------------------------------------- #
def test_kline_upsert_get_roundtrip(db: Database) -> None:
    """kline 写入后可原样读回（Bar 契约列 + OHLCV + dt 相等）。"""
    dt1 = datetime(2026, 9, 21, 9, 30)
    dt2 = datetime(2026, 9, 21, 9, 31)
    bars = [
        Bar("000001", "1m", dt2, 10.5, 10.8, 10.4, 10.7, 1200.0, 12800.0),
        Bar("000001", "1m", dt1, 10.0, 10.6, 9.9, 10.5, 1000.0, 10500.0),
    ]
    written = db.upsert_bars(bars)
    assert written == 2

    df = db.get_bars("000001", "1m")
    assert list(df.columns) == BAR_COLUMNS
    assert len(df) == 2
    # 升序
    assert df["datetime"].is_monotonic_increasing
    assert df.loc[0, "datetime"] == pd.Timestamp(dt1)
    assert float(df.loc[0, "open"]) == 10.0
    assert float(df.loc[0, "high"]) == 10.6
    assert float(df.loc[0, "low"]) == 9.9
    assert float(df.loc[0, "close"]) == 10.5
    assert float(df.loc[0, "volume"]) == 1000.0
    assert float(df.loc[0, "amount"]) == 10500.0
    assert df.dtypes["datetime"].kind == "M"  # datetime64


def test_kline_upsert_same_key_overwrites(db: Database) -> None:
    """同 (symbol, period, dt, adjust) 重复写入应覆盖而非重复插入。"""
    dt = datetime(2026, 9, 21, 9, 30)
    db.upsert_bars([Bar("000001", "d", dt, 1, 1, 1, 1, 1, 1)])
    db.upsert_bars([Bar("000001", "d", dt, 2, 2, 2, 2, 2, 2)])
    df = db.get_bars("000001", "d")
    assert len(df) == 1
    assert float(df.loc[0, "close"]) == 2.0


def test_kline_different_adjust_not_overwrite(db: Database) -> None:
    """同 (symbol, period, dt) 不同 adjust 的数据互不覆盖。"""
    dt = datetime(2026, 9, 21)
    db.upsert_bars([Bar("000001", "d", dt, 10, 10, 10, 10, 10, 10, adjust="")])
    db.upsert_bars([Bar("000001", "d", dt, 8, 8, 8, 8, 8, 8, adjust="qfq")])

    raw = db.get_bars("000001", "d", adjust="")
    qfq = db.get_bars("000001", "d", adjust="qfq")
    assert len(raw) == 1 and float(raw.loc[0, "close"]) == 10.0
    assert len(qfq) == 1 and float(qfq.loc[0, "close"]) == 8.0
    # 总数统计应包含两种口径
    assert db.count_bars("000001", "d") == 2


def test_kline_range_filter_and_count(db: Database) -> None:
    """start/end 区间过滤与 count_bars 过滤正确。"""
    base = datetime(2026, 9, 1)
    bars = [Bar("600000", "d", base.replace(day=d), d, d, d, d, d, d) for d in range(1, 6)]
    db.upsert_bars(bars)
    assert db.count_bars() == 5
    assert db.count_bars("600000") == 5
    assert db.count_bars("600000", "d") == 5
    assert db.count_bars("000001") == 0

    sub = db.get_bars("600000", "d", start="2026-09-02", end="2026-09-04")
    assert len(sub) == 3
    assert sub.loc[0, "datetime"] == pd.Timestamp(datetime(2026, 9, 2))


def test_get_bars_empty_returns_contract_columns(db: Database) -> None:
    """无数据时仍返回契约列的空 DataFrame。"""
    df = db.get_bars("300999", "d")
    assert list(df.columns) == BAR_COLUMNS
    assert df.empty


def test_list_kline_symbols(db: Database) -> None:
    """list_kline_symbols 按 (period, adjust) 列出升序去重标的（P7-S1 回测股票池）。"""
    base = datetime(2026, 9, 1)
    db.upsert_bars(
        [
            Bar("600000", "d", base.replace(day=1), 1, 1, 1, 1, 1, 1, adjust=""),
            Bar("000001", "d", base.replace(day=1), 1, 1, 1, 1, 1, 1, adjust=""),
            Bar("000001", "d", base.replace(day=2), 1, 1, 1, 1, 1, 1, adjust=""),
            Bar("000001", "d", base.replace(day=1), 1, 1, 1, 1, 1, 1, adjust="qfq"),
            Bar("sh.000300", "d", base.replace(day=1), 1, 1, 1, 1, 1, 1, adjust=""),
            Bar("600000", "5m", base.replace(day=1), 1, 1, 1, 1, 1, 1, adjust=""),
        ]
    )

    # 默认 (period="d", adjust="")：升序去重（含指数代码，ASCII 下数字 < 字母）。
    assert db.list_kline_symbols() == ["000001", "600000", "sh.000300"]
    assert db.list_kline_symbols("d", "qfq") == ["000001"]
    assert db.list_kline_symbols("5m") == ["600000"]
    assert db.list_kline_symbols("d", "hfq") == []
    assert db.list_kline_symbols("w") == []


def test_upsert_bars_from_dataframe(db: Database) -> None:
    """支持直接写入符合契约的 DataFrame。"""
    frame = pd.DataFrame(
        {
            "symbol": ["000002", "000002"],
            "period": ["d", "d"],
            "datetime": pd.to_datetime(["2026-09-01", "2026-09-02"]),
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.05, 2.05],
            "volume": [100, 200],
            "amount": [105, 410],
            "adjust": ["qfq", "qfq"],
        }
    )
    assert db.upsert_bars(frame) == 2
    out = db.get_bars("000002", "d", adjust="qfq")
    assert len(out) == 2
    assert float(out.loc[1, "close"]) == 2.05


def test_upsert_bars_dataframe_missing_columns(db: Database) -> None:
    """DataFrame 缺列时应抛 ValueError。"""
    with pytest.raises(ValueError):
        db.upsert_bars(pd.DataFrame({"symbol": ["x"], "period": ["d"]}))


# --------------------------------------------------------------------------- #
# 复权因子
# --------------------------------------------------------------------------- #
def test_adj_factor_roundtrip(db: Database) -> None:
    """复权因子写入 / 读取一致，列名 [datetime, factor]。"""
    rows = [
        ("2026-01-02", 1.0),
        ("2026-06-15", 1.25),
        ("2026-09-01", 1.5),
    ]
    assert db.upsert_adj_factors("000001", rows) == 3
    df = db.get_adj_factors("000001")
    assert list(df.columns) == ["datetime", "factor"]
    assert len(df) == 3
    assert df["datetime"].is_monotonic_increasing
    assert float(df.loc[2, "factor"]) == 1.5
    # 区间过滤
    sub = db.get_adj_factors("000001", start="2026-06-01", end="2026-07-01")
    assert len(sub) == 1


# --------------------------------------------------------------------------- #
# 自选股
# --------------------------------------------------------------------------- #
def test_watchlist_group_and_items_flow(db: Database) -> None:
    """分组增删查 + 条目增删 + 备注标记。"""
    gid = db.create_group("核心池", sort_order=1)
    gid2 = db.create_group("观察池", sort_order=2)
    groups = db.list_groups()
    assert [g.name for g in groups] == ["核心池", "观察池"]
    assert groups[0].sort_order == 1

    assert db.add_watch_item(gid, "000001") > 0
    assert db.add_watch_item(gid, "600000") > 0
    # 重复添加返回既有 id，不新增
    same = db.add_watch_item(gid, "000001")
    assert same == db.list_watch_items(gid)[0].id
    assert len(db.list_watch_items(gid)) == 2
    assert len(db.list_watch_items(gid2)) == 0
    assert len(db.list_watch_items()) == 2

    db.set_note_mark(gid, "000001", True)
    marked = {i.symbol: i.note_mark for i in db.list_watch_items(gid)}
    assert marked["000001"] is True
    assert marked["600000"] is False

    db.remove_watch_item(gid, "600000")
    assert [i.symbol for i in db.list_watch_items(gid)] == ["000001"]


def test_delete_group_cascades_items(db: Database) -> None:
    """删除分组应级联删除其条目（依赖 PRAGMA foreign_keys=ON）。"""
    gid = db.create_group("临时")
    db.add_watch_item(gid, "000001")
    assert len(db.list_watch_items()) == 1
    db.delete_group(gid)
    assert db.list_watch_items() == []
    assert db.list_groups() == []


# --------------------------------------------------------------------------- #
# 备注
# --------------------------------------------------------------------------- #
def test_note_crud(db: Database) -> None:
    """备注读取 / 写入 / 删除。"""
    assert db.get_note("000001") is None
    db.set_note("000001", "关注年报")
    note = db.get_note("000001")
    assert note is not None
    assert note.symbol == "000001"
    assert note.text == "关注年报"
    assert isinstance(note.updated_at, datetime)

    db.set_note("000001", "已更新")
    assert db.get_note("000001").text == "已更新"

    db.delete_note("000001")
    assert db.get_note("000001") is None


# --------------------------------------------------------------------------- #
# 指标模板
# --------------------------------------------------------------------------- #
def test_indicator_template_crud(db: Database) -> None:
    """指标模板新增 / 更新 / 分类过滤 / 删除。"""
    tid = db.save_indicator_template("MA20", "均线", "close.rolling(20).mean()")
    assert tid > 0
    db.save_indicator_template("MACD", "动量", "ta.macd(close)")

    all_templates = db.list_indicator_templates()
    assert len(all_templates) == 2
    assert all_templates[0].name == "MA20"
    assert all_templates[0].category == "均线"
    assert all_templates[0].expr == "close.rolling(20).mean()"
    assert isinstance(all_templates[0].created_at, datetime)

    only_ma = db.list_indicator_templates(category="均线")
    assert [t.name for t in only_ma] == ["MA20"]

    # 更新（同 id 覆盖）
    db.save_indicator_template("MA20改", "均线", "close.rolling(10).mean()", template_id=tid)
    updated = {t.id: t for t in db.list_indicator_templates()}[tid]
    assert updated.name == "MA20改"
    assert updated.expr == "close.rolling(10).mean()"
    assert len(db.list_indicator_templates()) == 2

    db.delete_indicator_template(tid)
    assert [t.name for t in db.list_indicator_templates()] == ["MACD"]


# --------------------------------------------------------------------------- #
# 选股模板
# --------------------------------------------------------------------------- #
def test_screener_template_crud(db: Database) -> None:
    """选股模板新增 / 更新 / 列表。"""
    sid = db.save_screener_template("低PE多头", '{"conditions": []}')
    assert sid > 0
    db.save_screener_template("放量突破", '{"conditions": [1]}')

    templates = db.list_screener_templates()
    assert len(templates) == 2
    assert templates[0].name == "低PE多头"
    assert templates[0].conditions_json == '{"conditions": []}'
    assert isinstance(templates[0].created_at, datetime)

    db.save_screener_template("低PE多头v2", '{"conditions": [2]}', template_id=sid)
    updated = {t.id: t for t in db.list_screener_templates()}[sid]
    assert updated.name == "低PE多头v2"
    assert updated.conditions_json == '{"conditions": [2]}'


# --------------------------------------------------------------------------- #
# 配置 / 元数据
# --------------------------------------------------------------------------- #
def test_config_and_meta(db: Database) -> None:
    """config / meta 键值存取与默认值。"""
    assert db.get_config("theme") is None
    assert db.get_config("theme", "dark") == "dark"
    db.set_config("theme", "dark_red_green")
    assert db.get_config("theme") == "dark_red_green"
    db.set_config("theme", "light")
    assert db.get_config("theme") == "light"

    assert db.get_meta("last_update_time") is None
    db.set_meta("last_update_time", "2026-09-21 15:00:00")
    assert db.get_meta("last_update_time") == "2026-09-21 15:00:00"


def test_context_manager_creates_schema(tmp_path) -> None:
    """with Database(...) 应自动连接并建表。"""
    with Database(tmp_path / "ctx.db") as database:
        assert database.count_bars() == 0
    # 关闭后再次查询应自动重连
    assert database.count_bars() == 0
    database.close()
