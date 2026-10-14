"""P2-S4 K线辅助标注（涨跌停 / 跳空缺口 / 除权）测试。

离屏（``QT_QPA_PLATFORM=offscreen``），**全离线合成数据**。

覆盖：
    1. :func:`limit_pct`（各板块涨跌停幅度）；
    2. :func:`limit_prices`（ROUND_HALF_UP 精确向量）；
    3. :func:`detect_limit_events`（命中 / 容差外不命中）；
    4. :func:`detect_gaps`（上跳 / 下跳 / 幅度不足）；
    5. :func:`detect_ex_dividend`（因子变化日 + 索引对齐）；
    6. 集成：**判定用未复权序列**（显示序列为被故意扭曲的 qfq）；
    7. :meth:`set_annotations_enabled` 开关；
    8. 回归：全量 pytest 无回归（外层运行）。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from app.data.store.db import Database  # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.themes import colors  # noqa: E402
from app.ui.widgets.kline_annotations import (  # noqa: E402
    detect_ex_dividend,
    detect_gaps,
    detect_limit_events,
    limit_pct,
    limit_prices,
)
from app.ui.widgets.kline_view import KLineView  # noqa: E402


# --------------------------------------------------------------------------- #
# 数据构造工具
# --------------------------------------------------------------------------- #
_DATES_6 = [
    "2024-01-01", "2024-01-02", "2024-01-03",
    "2024-01-04", "2024-01-05", "2024-01-08",
]


def _frame(dates, open_, high, low, close) -> pd.DataFrame:
    """按行构造 Bar 契约 DataFrame（volume/amount 占位）。"""
    n = len(dates)
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(dates),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100.0] * n,
            "amount": [10000.0] * n,
        }
    )


def _mem_db() -> Database:
    db = Database(":memory:")
    db.connect()
    db.init_schema()
    return db


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def view(qapp):
    v = KLineView(db=_mem_db())
    yield v
    v.close()


# --------------------------------------------------------------------------- #
# 1. limit_pct
# --------------------------------------------------------------------------- #
def test_limit_pct_by_board() -> None:
    assert limit_pct("600519") == 0.10
    assert limit_pct("000001") == 0.10
    assert limit_pct("002594") == 0.10
    assert limit_pct("300750") == 0.20
    assert limit_pct("301111") == 0.20
    assert limit_pct("688981") == 0.20
    assert limit_pct("689009") == 0.20
    assert limit_pct("830799") == 0.30
    assert limit_pct("430047") == 0.30
    # 带板块前缀亦可
    assert limit_pct("sz300750") == 0.20
    assert limit_pct("bj830799") == 0.30


# --------------------------------------------------------------------------- #
# 2. limit_prices（钉死 ROUND_HALF_UP）
# --------------------------------------------------------------------------- #
def test_limit_prices_half_up_vectors() -> None:
    up, down = limit_prices(10.00, 0.10)
    assert up == pytest.approx(11.00)
    assert down == pytest.approx(9.00)


def test_limit_prices_half_up_925() -> None:
    """10.25 跌停 9.225 → 必须 9.23（银行家舍入会给 9.22）。"""
    up, down = limit_prices(10.25, 0.10)
    assert up == pytest.approx(11.28)
    assert down == pytest.approx(9.23)


def test_limit_prices_half_up_1003() -> None:
    up, down = limit_prices(10.03, 0.10)
    assert up == pytest.approx(11.03)
    assert down == pytest.approx(9.03)


# --------------------------------------------------------------------------- #
# 3. detect_limit_events
# --------------------------------------------------------------------------- #
def test_detect_limit_up_hit() -> None:
    df = _frame(
        ["2024-01-01", "2024-01-02"],
        open_=[10.00, 10.50],
        high=[10.10, 11.00],
        low=[9.90, 10.50],
        close=[10.00, 11.00],  # 前收 10.00 → 涨停价 11.00
    )
    events = detect_limit_events(df, "600519")
    assert len(events) == 1
    assert events[0]["kind"] == "limit_up"
    assert events[0]["index"] == 1
    assert events[0]["price"] == pytest.approx(11.00)


def test_detect_limit_up_below_tolerance_no_hit() -> None:
    df = _frame(
        ["2024-01-01", "2024-01-02"],
        open_=[10.00, 10.50],
        high=[10.10, 10.99],
        low=[9.90, 10.40],
        close=[10.00, 10.94],  # 与 11.00 差 0.06 > tol(0.005) → 不命中
    )
    assert detect_limit_events(df, "600519") == []


def test_detect_limit_down_hit() -> None:
    df = _frame(
        ["2024-01-01", "2024-01-02"],
        open_=[10.00, 9.50],
        high=[10.10, 9.50],
        low=[9.90, 9.00],
        close=[10.00, 9.00],  # 前收 10.00 → 跌停价 9.00
    )
    events = detect_limit_events(df, "000001")
    assert len(events) == 1
    assert events[0]["kind"] == "limit_down"
    assert events[0]["price"] == pytest.approx(9.00)


# --------------------------------------------------------------------------- #
# 4. detect_gaps
# --------------------------------------------------------------------------- #
def test_detect_gap_up() -> None:
    df = _frame(
        ["2024-01-01", "2024-01-02"],
        open_=[10.00, 10.30],
        high=[10.20, 10.60],
        low=[10.00, 10.30],  # 10.30 > 前高 10.20，幅度 0.98% ≥ 0.5%
        close=[10.10, 10.50],
    )
    events = detect_gaps(df)
    assert len(events) == 1
    assert events[0]["kind"] == "gap_up"
    assert events[0]["index"] == 1
    assert events[0]["lower"] == pytest.approx(10.20)
    assert events[0]["upper"] == pytest.approx(10.30)


def test_detect_gap_down() -> None:
    df = _frame(
        ["2024-01-01", "2024-01-02"],
        open_=[10.00, 9.70],
        high=[10.20, 9.80],  # 9.80 < 前低 10.00，幅度 2% ≥ 0.5%
        low=[10.00, 9.50],
        close=[10.10, 9.60],
    )
    events = detect_gaps(df)
    assert len(events) == 1
    assert events[0]["kind"] == "gap_down"
    assert events[0]["lower"] == pytest.approx(9.80)
    assert events[0]["upper"] == pytest.approx(10.00)


def test_detect_gap_below_min_pct_no_hit() -> None:
    df = _frame(
        ["2024-01-01", "2024-01-02"],
        open_=[10.00, 10.20],
        high=[10.20, 10.25],
        low=[10.00, 10.205],  # 10.205 > 前高 10.20，但幅度仅 0.05% < 0.5%
        close=[10.10, 10.22],
    )
    assert detect_gaps(df) == []


def test_detect_gap_too_small_min_pct_param() -> None:
    """调小 min_pct 后，同一微小缺口即可被识别（验证阈值语义）。"""
    df = _frame(
        ["2024-01-01", "2024-01-02"],
        open_=[10.00, 10.20],
        high=[10.20, 10.25],
        low=[10.00, 10.205],
        close=[10.10, 10.22],
    )
    events = detect_gaps(df, min_pct=0.0001)
    assert len(events) == 1
    assert events[0]["kind"] == "gap_up"


# --------------------------------------------------------------------------- #
# 5. detect_ex_dividend
# --------------------------------------------------------------------------- #
def test_detect_ex_dividend_factor_change() -> None:
    factors = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "factor": [1.0, 1.0, 1.05, 1.05],
        }
    )
    events = detect_ex_dividend(factors)
    assert len(events) == 1
    assert events[0]["datetime"] == pd.Timestamp("2024-01-03")
    assert events[0]["factor_from"] == pytest.approx(1.0)
    assert events[0]["factor_to"] == pytest.approx(1.05)
    assert events[0]["index"] is None  # 未提供 bars


def test_detect_ex_dividend_index_alignment() -> None:
    factors = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "factor": [1.0, 1.0, 1.05, 1.05],
        }
    )
    bars = pd.DataFrame({"datetime": pd.to_datetime(_DATES_6[:4])})
    events = detect_ex_dividend(factors, bars)
    assert len(events) == 1
    assert events[0]["index"] == 2


# --------------------------------------------------------------------------- #
# 6. 集成：判定必须用未复权序列
# --------------------------------------------------------------------------- #
def _raw_daily() -> pd.DataFrame:
    """未复权日线：含 1 根真涨停 + 1 个向上跳空（仅此一个缺口）。"""
    return _frame(
        _DATES_6,
        open_=[10.00, 10.50, 11.30, 11.35, 11.40, 11.45],
        high=[10.10, 11.00, 11.60, 11.50, 11.55, 11.60],
        low=[9.90, 10.05, 11.20, 11.30, 11.35, 11.40],  # low1=10.05 ≤ high0，故无多余缺口
        close=[10.00, 11.00, 11.40, 11.45, 11.50, 11.55],
    )


def _qfq_daily() -> pd.DataFrame:
    """前复权日线：被故意扭曲为平滑序列，使涨跌停/跳空**看不出来**。"""
    return _frame(
        _DATES_6,
        open_=[9.00, 9.00, 9.05, 9.10, 9.15, 9.20],
        high=[9.05, 9.10, 9.15, 9.20, 9.25, 9.30],
        low=[8.95, 8.95, 9.00, 9.05, 9.10, 9.15],
        close=[9.00, 9.05, 9.10, 9.15, 9.20, 9.25],
    )


def _seed_full(db: Database, symbol: str) -> None:
    """预置未复权日线（adjust=""）+ 前复权日线（adjust="qfq"）+ 一次因子变化。"""
    raw = _raw_daily()
    raw["symbol"] = symbol
    raw["period"] = "d"
    raw["adjust"] = ""
    db.upsert_bars(raw)

    qfq = _qfq_daily()
    qfq["symbol"] = symbol
    qfq["period"] = "d"
    qfq["adjust"] = "qfq"
    db.upsert_bars(qfq)

    db.upsert_adj_factors(
        symbol,
        [(pd.Timestamp(d), f) for d, f in zip(_DATES_6, [1.0, 1.0, 1.05, 1.05, 1.05, 1.05])],
    )


def test_qfq_series_alone_has_no_events(view: KLineView) -> None:
    """反证：扭曲后的 qfq 序列自身既无涨跌停也无跳空。"""
    qfq = view._normalize(_qfq_daily())
    assert detect_limit_events(qfq, "600519") == []
    assert detect_gaps(qfq) == []


def test_integration_detects_on_unadjusted_not_display(qapp) -> None:
    """显示 qfq、判定未复权：事件应来自未复权序列，并精确对齐到显示索引。"""
    symbol = "600519"
    db = _mem_db()
    _seed_full(db, symbol)

    view = KLineView(db=db)
    try:
        view.load(symbol, "d", adjust="qfq")
        counts = view.annotation_counts()
        events = view.annotation_events()

        # 事件来自未复权序列（qfq 自身无事件）
        assert counts["limit_up"] >= 1
        assert counts["gap_up"] >= 1
        assert counts["ex_div"] >= 1

        # 判定价来自未复权（11.00 / 11.20），但 y 值已按「显示/判定」收盘比例换算到显示刻度：
        # D1 比例 = 9.05/11.00 → 11.00 × 比例 = 9.05（= 显示该根收盘）
        assert events["limit_up"][0]["price"] == pytest.approx(9.05)
        assert events["limit_up"][0]["index"] == 1  # 对齐到显示序列索引
        # D2 比例 = 9.10/11.40 → 缺口界 11.00/11.20 换算为 8.7807/8.9404
        assert events["gap_up"][0]["lower"] == pytest.approx(11.00 * 9.10 / 11.40)
        assert events["gap_up"][0]["upper"] == pytest.approx(11.20 * 9.10 / 11.40)
        assert events["gap_up"][0]["index"] == 2
        assert events["ex_div"][0]["index"] == 2
        assert events["ex_div"][0]["factor_from"] == pytest.approx(1.0)
        assert events["ex_div"][0]["factor_to"] == pytest.approx(1.05)

        # 标注图元确实渲染到主图
        assert view.annotation_item_count() >= 3
    finally:
        view.close()


def test_annotation_y_rescaled_to_display_scale(qapp) -> None:
    """高分红股：未复权与显示刻度差 2 倍时，标注 y 必须换算到显示刻度（不浮高）。

    回归用户真机实测（601398 日 K）：未复权涨停价 11.00 直接画在前复权轴上 → 标注浮高
    1.7 倍；换算后应落在显示蜡烛附近（该根显示收盘 5.50）。
    """
    symbol = "601398"
    db = _mem_db()
    dates = _DATES_6[:4]
    raw = _frame(
        dates,
        open_=[10.00, 10.20, 11.05, 11.10],
        high=[10.10, 11.00, 11.25, 11.35],
        low=[9.90, 10.05, 11.15, 11.25],  # low1=10.05 ≤ 前高 10.10，D1 不构成缺口
        close=[10.00, 11.00, 11.20, 11.30],
    )
    raw["symbol"] = symbol
    raw["period"] = "d"
    raw["adjust"] = ""
    db.upsert_bars(raw)

    qfq = raw.copy()
    for col in ("open", "high", "low", "close"):
        qfq[col] = qfq[col] * 0.5  # 显示刻度 = 未复权的一半（模拟高分红复权系数）
    qfq["adjust"] = "qfq"
    db.upsert_bars(qfq)

    view = KLineView(db=db)
    try:
        view.load(symbol, "d", adjust="qfq")
        counts = view.annotation_counts()
        events = view.annotation_events()

        assert counts["limit_up"] == 1  # D1：10.00 → 11.00（涨停，未复权口径）
        assert counts["gap_up"] == 1  # D2：low 11.15 > 前高 11.00

        # 标注 y 在显示刻度：涨停价 11.00 × 0.5 = 5.50（= 显示该根收盘）
        assert events["limit_up"][0]["price"] == pytest.approx(5.50)
        # 缺口界同样换算：11.00 × 0.5 / 11.15 × 0.5
        assert events["gap_up"][0]["lower"] == pytest.approx(5.50)
        assert events["gap_up"][0]["upper"] == pytest.approx(5.575)
    finally:
        view.close()


def test_fallback_to_display_when_no_unadjusted(qapp) -> None:
    """无未复权数据时回退用显示序列判定（不抛异常）。"""
    symbol = "000001"
    db = _mem_db()
    qfq = _raw_daily()  # 复用事件丰富序列当作显示序列
    qfq["symbol"] = symbol
    qfq["period"] = "d"
    qfq["adjust"] = "qfq"
    db.upsert_bars(qfq)

    view = KLineView(db=db)
    try:
        view.load(symbol, "d", adjust="qfq")  # adjust="" 无数据 → 回退
        counts = view.annotation_counts()
        assert counts["limit_up"] >= 1
        assert counts["gap_up"] >= 1
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 7. 标注开关
# --------------------------------------------------------------------------- #
def test_annotations_toggle(qapp) -> None:
    symbol = "600519"
    db = _mem_db()
    _seed_full(db, symbol)

    view = KLineView(db=db)
    try:
        view.load(symbol, "d", adjust="qfq")
        assert view.annotations_enabled() is True
        assert view.annotation_item_count() >= 1

        view.set_annotations_enabled(False)
        assert view.annotations_enabled() is False
        assert all(not item.isVisible() for item in view._annotation_items)

        view.set_annotations_enabled(True)
        assert view.annotations_enabled() is True
        assert view.annotation_item_count() >= 1
        assert any(item.isVisible() for item in view._annotation_items)
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 7b. 涨跌停蜡烛着色（2026-10-07：替代原黄色文字标注）
# --------------------------------------------------------------------------- #
def test_limit_candles_colored_gold_and_deep_green(qapp) -> None:
    """涨停蜡烛 = 金色 / 跌停 = 深绿，且不再绘制文字标注；随开关联动恢复红绿。"""
    symbol = "000001"
    db = _mem_db()
    raw = _frame(
        _DATES_6[:3],
        open_=[10.00, 10.50, 10.40],
        high=[10.10, 11.00, 10.60],
        low=[9.90, 10.05, 9.90],
        close=[10.00, 11.00, 9.90],  # D1 +10% 涨停；D2 -10% 跌停
    )
    raw["symbol"] = symbol
    raw["period"] = "d"
    raw["adjust"] = ""
    db.upsert_bars(raw)

    view = KLineView(db=db)
    try:
        view.load(symbol, "d", adjust="")
        assert view.annotation_counts()["limit_up"] == 1
        assert view.annotation_counts()["limit_down"] == 1
        assert view.annotation_item_count() == 0  # 无文字标注（无缺口 / 除权）

        data = view._candle_item._data
        assert data[0][5] == colors.UP_COLOR  # 普通上涨仍是红
        assert data[1][5] == colors.LIMIT_UP_COLOR
        assert data[2][5] == colors.LIMIT_DOWN_COLOR

        view.set_annotations_enabled(False)  # 关闭 → 恢复普通红 / 绿
        data = view._candle_item._data
        assert data[1][5] == colors.UP_COLOR
        assert data[2][5] == colors.DOWN_COLOR

        view.set_annotations_enabled(True)  # 重开 → 金 / 深绿
        assert view._candle_item._data[1][5] == colors.LIMIT_UP_COLOR
        assert view._candle_item._data[2][5] == colors.LIMIT_DOWN_COLOR
    finally:
        view.close()


# --------------------------------------------------------------------------- #
# 附加：标注不影响 P2-S2 既有行为
# --------------------------------------------------------------------------- #
def test_set_bars_clears_annotations(view: KLineView) -> None:
    view.set_bars(_raw_daily())  # 直接 set_bars（无 symbol）不应产生标注
    assert view.annotation_counts() == {
        "limit_up": 0, "limit_down": 0, "touch_up": 0,
        "touch_down": 0, "gap_up": 0, "gap_down": 0, "ex_div": 0,
    }
    assert view.annotation_item_count() == 0
    assert view.bar_count() == 6
    assert not view.is_placeholder_visible()
