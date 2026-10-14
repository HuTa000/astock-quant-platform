"""P7-S3 多因子分析编排 + 面板组装单测（行为对拍）。

覆盖：
    1. 面板前瞻收益口径：``forward(h)[T] = open[T+1+h] / open[T+1] − 1``
       （T+1 开盘买入，绝不使用当日收盘价）；
    2. 停牌（无 bar）⇒ 端点缺失记 NaN，横截面自动跳过；
    3. 因子面板：手算对拍、惰性缓存（同一列返回同一对象）、未知因子全 NaN；
    4. 编排成功路径：IC 表契约列 / 因子顺序 / 窗口过滤 / 分层 / 共线性；
    5. 自动股票池排除指数；显式股票池缺 qfq ⇒ 剔除并降级；
    6. 降级路径：空库 / 窗口无交易日 / 样本不足 / 无效因子名 / 非法参数
       （TypeError、ValueError）；
    7. ic_decay_table 旋转手算；_resolve_* 参数规范化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.store.db import Database
from app.engine.factor import (
    DEFAULT_FACTOR_NAMES,
    DEFAULT_HORIZONS,
    IC_TABLE_COLUMNS,
    FactorAnalysisSpec,
    FactorAnalyzer,
    build_panel,
    ic_decay_table,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _biz_days(count: int, start: str = "2024-01-01") -> list[str]:
    """生成 count 个「交易日」日期串（工作日口径）。"""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=count)]


def _seed(
    db: Database, symbol: str, spec: dict, adjusts=("qfq",), volume: float = 1_000_000.0
) -> None:
    """按 ``{日期: (o, h, l, c[, volume])}`` 写入日线（默认仅前复权）。"""
    for adjust in adjusts:
        frame = pd.DataFrame(
            [
                {
                    "datetime": pd.Timestamp(day),
                    "open": float(values[0]),
                    "high": float(values[1]),
                    "low": float(values[2]),
                    "close": float(values[3]),
                    "volume": float(values[4]) if len(values) > 4 else volume,
                    "amount": float(values[3])
                    * (float(values[4]) if len(values) > 4 else volume),
                }
                for day, values in spec.items()
            ]
        )
        frame["symbol"] = symbol
        frame["period"] = "d"
        frame["adjust"] = adjust
        db.upsert_bars(frame)


def _linear_spec(days: list[str], start_price: float = 10.0) -> dict:
    """线性上涨行情：每日 +1 元（open = close，便于手算前瞻收益）。"""
    return {
        day: (start_price + index, start_price + index + 0.5, start_price + index - 0.5,
              start_price + index)
        for index, day in enumerate(days)
    }


def _momentum_spec(days: list[str], phase: int, start_price: float, seed: int) -> dict:
    """阶段性趋势行情 + 确定性噪声（60 日周期：前 30 日 +0.5%/日、后 30 日 −0.5%/日）。

    各标的相位与噪声种子不同 ⇒ 每个因子在横截面上都有区分度（IC / VIF 可算）；
    块内动量延续 ⇒ 过去 5 日涨幅与未来数日收益同向（IC 均值为正的确定性构造）。
    噪声幅度（0.3%/日）小于趋势幅度（0.5%/日），不会淹没动量结构。
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.003, len(days))
    volume_noise = rng.uniform(0.8, 1.2, len(days))
    base_volume = 1_000_000.0 * (1.0 + 0.05 * phase / 60.0)
    spec: dict[str, tuple[float, float, float, float, float]] = {}
    price = start_price
    for index, day in enumerate(days):
        position = (index + phase) % 60
        drift = 0.005 if position < 30 else -0.005
        price *= 1.0 + drift + float(noise[index])
        band = 0.01 + abs(float(noise[index]))
        spec[day] = (
            price,
            price * (1.0 + band),
            price * (1.0 - band),
            price,
            base_volume * float(volume_noise[index]),
        )
    return spec


def _seed_momentum_universe(db: Database, count: int = 12, days: int = 200) -> list[str]:
    """播种 12 只相位错开的动量标的，返回标的清单。"""
    calendar = _biz_days(days)
    symbols = [f"0000{i:02d}" for i in range(1, count + 1)]
    for index, symbol in enumerate(symbols):
        _seed(
            db,
            symbol,
            _momentum_spec(
                calendar, phase=index * 5, start_price=10.0 + index * 0.1, seed=20261011 + index
            ),
        )
    return symbols


# --------------------------------------------------------------------------- #
# 1~3. 面板口径
# --------------------------------------------------------------------------- #
def test_panel_forward_t_plus_one_semantics(tmp_path) -> None:
    """前瞻收益手算：open 逐日 10..17 ⇒ forward(1)/forward(2) 逐点对拍。"""
    db = _open(tmp_path / "panel.db")
    days = _biz_days(8)
    _seed(db, "000001", _linear_spec(days, start_price=10.0))

    panel, warnings = build_panel(db, ["000001"])

    assert warnings == []
    assert panel is not None and not panel.empty
    assert panel.symbols == ["000001"]
    assert list(panel.dates) == [pd.Timestamp(day) for day in days]

    one_day = panel.forward(1)["000001"]
    assert float(one_day.loc[days[0]]) == pytest.approx(12.0 / 11.0 - 1.0)
    assert float(one_day.loc[days[5]]) == pytest.approx(17.0 / 16.0 - 1.0)
    assert pd.isna(one_day.loc[days[6]])                 # T+2 无 bar（末日）
    assert pd.isna(one_day.loc[days[7]])
    two_day = panel.forward(2)["000001"]
    assert float(two_day.loc[days[0]]) == pytest.approx(13.0 / 11.0 - 1.0)
    assert pd.isna(two_day.loc[days[6]])
    # 非法预测期 ⇒ 全 NaN（不抛异常）；面板本身仍可用
    assert panel.forward(0).isna().all().all()
    assert panel.forward("x").isna().all().all()

    # 因子手算：ret_5 = close / close.shift(5) − 1（第 8 日 = 17/12 − 1）
    factor = panel.factor("ret_5")
    assert float(factor["000001"].loc[days[7]]) == pytest.approx(17.0 / 12.0 - 1.0)
    assert pd.isna(factor["000001"].loc[days[4]])        # 预热期（不足 5 个样本）
    # 惰性缓存：同一因子重复调用返回同一对象；未知因子全 NaN
    assert panel.factor("ret_5") is panel.factor("ret_5")
    assert panel.factor("nope").isna().all().all()


def test_panel_suspension_yields_nan(tmp_path) -> None:
    """停牌（缺 bar）⇒ 跨越缺口的端点记 NaN；因子窗口按个股自身交易日序列。"""
    db = _open(tmp_path / "suspended.db")
    days = _biz_days(8)
    spec = _linear_spec(days, start_price=10.0)
    suspended = dict(spec)
    del suspended[days[3]]                               # 第 4 日停牌
    _seed(db, "000001", suspended)
    _seed(db, "000002", spec)                            # 全勤标的撑起全局日历

    panel, _ = build_panel(db, ["000001", "000002"])

    forward = panel.forward(1)["000001"]
    assert float(forward.loc[days[0]]) == pytest.approx(12.0 / 11.0 - 1.0)
    assert pd.isna(forward.loc[days[1]])                 # T+2 = 停牌日
    assert pd.isna(forward.loc[days[2]])                 # T+1 = 停牌日（买不进）
    assert pd.isna(panel.factor("ret_5")["000001"].loc[days[3]])   # 停牌日无因子值
    # 因子按个股自身 7 根 bar 计算：第 8 日 ret_5 = 17/11 − 1（跳过停牌日）
    assert float(panel.factor("ret_5")["000001"].loc[days[7]]) == pytest.approx(
        17.0 / 11.0 - 1.0
    )


# --------------------------------------------------------------------------- #
# 4. 编排成功路径
# --------------------------------------------------------------------------- #
def test_run_success_contract_and_window(tmp_path) -> None:
    """成功路径：契约列 / 因子顺序 / 窗口过滤 / 分层 / 共线性 / IC 为正。"""
    db = _open(tmp_path / "run.db")
    calendar = _biz_days(200)
    symbols = _seed_momentum_universe(db)
    spec = FactorAnalysisSpec(
        symbols=symbols,
        factor_names=["ma_dev_20", "ret_5"],
        horizons=[3, 5],
        start=calendar[40],
        end=calendar[120],
        n_groups=4,
        min_stocks=6,
    )

    report = FactorAnalyzer(db=db).run(spec)

    assert report.ok
    assert [factor.name for factor in report.factors] == ["ret_5", "ma_dev_20"]  # 注册序
    assert report.universe == sorted(symbols)
    assert report.evaluation_days == 81                   # calendar[40..120] 含两端
    assert list(report.ic_table.columns) == IC_TABLE_COLUMNS
    assert len(report.ic_table) == 4                      # 2 因子 × 2 预测期
    # 窗口尾部的前瞻收益由窗口外数据补齐（面板全历史计算）⇒ 各行满样本
    assert report.ic_table["n_days"].tolist() == [81, 81, 81, 81]
    assert set(report.ic_series) == {
        ("ret_5", 3), ("ret_5", 5), ("ma_dev_20", 3), ("ma_dev_20", 5)
    }
    # 阶段性趋势数据中「过去 5 日涨幅」与未来数日收益同向 ⇒ IC 均值为正
    ret5_short = report.ic_table[
        (report.ic_table["factor"] == "ret_5") & (report.ic_table["horizon"] == 3)
    ]
    assert float(ret5_short["ic_mean"].iloc[0]) > 0.0
    # 分层：窗口内每日都能建仓
    assert report.layered["ret_5"].ok and report.layered["ret_5"].days == 81
    assert report.layered["ma_dev_20"].ok and report.layered["ma_dev_20"].days == 81
    assert report.layered["ret_5"].n_groups == 4
    assert report.layered["ret_5"].monotonicity is not None
    # 共线性：两因子相关矩阵 + VIF 均为有限正数
    assert report.collinearity is not None and report.collinearity.ok
    assert report.collinearity.corr.shape == (2, 2)
    assert float(report.collinearity.vif["ret_5"]) > 0.0
    assert report.collinearity.n_obs > 0


def test_run_default_factors_and_horizons(tmp_path) -> None:
    """默认参数：全部 12 个因子 × 默认预测期一次跑通（含 250 日窗口因子）。"""
    db = _open(tmp_path / "defaults.db")
    symbols = _seed_momentum_universe(db, count=12, days=300)

    report = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(symbols=symbols, horizons=DEFAULT_HORIZONS[:1], min_stocks=6)
    )

    assert report.ok
    assert [factor.name for factor in report.factors] == DEFAULT_FACTOR_NAMES
    assert len(report.ic_table) == len(DEFAULT_FACTOR_NAMES)
    assert set(report.layered) == set(DEFAULT_FACTOR_NAMES)
    assert report.collinearity is not None and report.collinearity.ok
    assert report.collinearity.corr.shape == (len(DEFAULT_FACTOR_NAMES),) * 2


def test_auto_pool_excludes_index(tmp_path) -> None:
    """自动股票池（symbols=None）排除指数标的。"""
    db = _open(tmp_path / "pool.db")
    days = _biz_days(40)
    _seed(db, "000001", _linear_spec(days))
    _seed(db, "sh.000300", _linear_spec(days, start_price=1000.0))

    report = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(factor_names=["ret_5"], horizons=[2])
    )

    assert report.universe == ["000001"]                  # 指数被排除
    assert not report.ok                                  # 单标的样本不足（默认闸门 10）


# --------------------------------------------------------------------------- #
# 5~6. 降级路径
# --------------------------------------------------------------------------- #
def test_run_empty_database_degrades(tmp_path) -> None:
    """空库 / 显式标的无 qfq 数据 ⇒ ok=False 且提示补数据（不抛异常）。"""
    db = _open(tmp_path / "empty.db")
    empty = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(factor_names=["ret_5"], horizons=[1])
    )
    assert not empty.ok
    assert "前复权" in empty.error

    # 只有未复权数据：显式股票池仍按 qfq 读取 → 剔除后无可用标的
    _seed(db, "000001", _linear_spec(_biz_days(30)), adjusts=("",))
    unadjusted = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(symbols=["000001"], factor_names=["ret_5"], horizons=[1])
    )
    assert not unadjusted.ok
    assert "前复权" in unadjusted.error


def test_run_window_out_of_range(tmp_path) -> None:
    """评估窗口落在数据范围之外 ⇒ ok=False（不抛异常）。"""
    db = _open(tmp_path / "window.db")
    _seed_momentum_universe(db, count=12, days=60)

    report = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(start="2030-01-01", factor_names=["ret_5"], horizons=[2])
    )

    assert not report.ok
    assert "没有任何交易日" in report.error


def test_run_min_stocks_gate_degrades(tmp_path) -> None:
    """横截面样本不足（min_stocks 过大）⇒ 全部因子无有效样本 ⇒ ok=False。"""
    db = _open(tmp_path / "gate.db")
    symbols = _seed_momentum_universe(db, count=12, days=80)

    report = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(
            symbols=symbols, factor_names=["ret_5"], horizons=[2], min_stocks=50
        )
    )

    assert not report.ok
    assert "无有效样本" in report.error
    assert report.ic_series == {}


def test_run_unknown_factor_names(tmp_path) -> None:
    """无效因子名被忽略并记警告；全部无效 ⇒ ok=False。"""
    db = _open(tmp_path / "unknown.db")
    symbols = _seed_momentum_universe(db, count=12, days=120)

    partial = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(
            symbols=symbols, factor_names=["nope", "ret_5"], horizons=[2], min_stocks=6
        )
    )
    assert partial.ok
    assert [factor.name for factor in partial.factors] == ["ret_5"]
    assert any("无效因子名" in message for message in partial.warnings)

    all_bad = FactorAnalyzer(db=db).run(
        FactorAnalysisSpec(symbols=symbols, factor_names=["nope"], horizons=[2])
    )
    assert not all_bad.ok
    assert "没有可分析的因子" in all_bad.error


def test_run_invalid_arguments_raise(tmp_path) -> None:
    """spec 类型错误 / 日期非法 / 起止倒置 ⇒ TypeError / ValueError。"""
    db = _open(tmp_path / "args.db")
    analyzer = FactorAnalyzer(db=db)

    with pytest.raises(TypeError):
        analyzer.run("bad")
    with pytest.raises(ValueError):
        analyzer.run(FactorAnalysisSpec(start="not-a-date"))
    with pytest.raises(ValueError):
        analyzer.run(FactorAnalysisSpec(start="2024-06-01", end="2024-01-01"))


# --------------------------------------------------------------------------- #
# 7. 结果表与参数规范化
# --------------------------------------------------------------------------- #
def test_ic_decay_table_pivot() -> None:
    """宽表旋转：行 = 因子展示名、列 = IC@N日；非法/空输入返回空表。"""
    table = pd.DataFrame(
        [
            {"factor": "ret_5", "label": "5日涨幅", "horizon": 5, "ic_mean": 0.03},
            {"factor": "ret_5", "label": "5日涨幅", "horizon": 20, "ic_mean": 0.01},
            {"factor": "ret_20", "label": "20日涨幅", "horizon": 5, "ic_mean": -0.02},
            {"factor": "ret_20", "label": "20日涨幅", "horizon": 20, "ic_mean": -0.01},
        ]
    )

    wide = ic_decay_table(table)

    assert list(wide.columns) == ["IC@5日", "IC@20日"]
    assert float(wide.loc["5日涨幅（ret_5）", "IC@5日"]) == pytest.approx(0.03)
    assert float(wide.loc["20日涨幅（ret_20）", "IC@20日"]) == pytest.approx(-0.01)
    assert ic_decay_table(pd.DataFrame()).empty
    assert ic_decay_table(pd.DataFrame({"x": [1]})).empty
    assert ic_decay_table("bad").empty


def test_resolve_helpers_normalize() -> None:
    """预测期规范化（去重/升序/剔除非法）+ 因子名解析（注册序/去重/未知警告）。"""
    assert FactorAnalyzer._resolve_horizons([5, "5", 10, 0, -3, "x", 2.7]) == [2, 5, 10]
    assert FactorAnalyzer._resolve_horizons(None) == []
    assert FactorAnalyzer._resolve_horizons(7) == []

    warnings: list[str] = []
    resolved = FactorAnalyzer._resolve_factors(
        ["ma_dev_20", "ret_5", "ret_5", "nope"], warnings
    )
    assert [factor.name for factor in resolved] == ["ret_5", "ma_dev_20"]
    assert warnings == ["无效因子名已忽略：nope"]

    full = FactorAnalyzer._resolve_factors(None, [])
    assert [factor.name for factor in full] == DEFAULT_FACTOR_NAMES