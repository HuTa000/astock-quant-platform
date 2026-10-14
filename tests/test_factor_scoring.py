"""P7-S5 多因子打分合成单测（行为对拍）。

覆盖：
    1. ``cross_section_zscore``    —— 横截面标准化手算、常量行 NaN、±3σ 截断开关；
    2. ``dynamic_icir_weights``    —— **兑现滞后**（无未来函数）、扩展窗口 ICIR、
       双因子归一化（Σ|w|=1）、负 IC 反向、样本不足归零、非法输入降级；
    3. ``combine_scores``          —— 加权合成手算、缺失严格口径（权重非零缺值
       ⇒ 该股当日 NaN）、全零权重当日 NaN、非法输入降级；
    4. ``top_picks``               —— 得分降序 + 同分代码升序、全 NaN 日不产出、
       非法 N 返回空；
    5. ``FactorScorer`` E2E        —— 动量宇宙：预热期全 NaN、scored_days、
       权重求和口径、latest_weights 契约列、默认核心三因子顺序；
    6. PIT 口径 E2E                —— 掩码生效：非成分日只保留当时名单成员；
    7. 降级路径：空库 / 窗口无交易日 / 全预热（ok=False）、非法 universe
       （ValueError）、spec 类型错误（TypeError）、无效因子名。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.models import IndexConstituent
from app.data.store.db import Database
from app.engine.factor import (
    DEFAULT_SCORING_FACTORS,
    WEIGHT_TABLE_COLUMNS,
    FactorScorer,
    ScoreReport,
    ScoringSpec,
    combine_scores,
    cross_section_zscore,
    dynamic_icir_weights,
    top_picks,
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


def _calendar(count: int) -> pd.DatetimeIndex:
    """生成 count 个交易日的 DatetimeIndex（与 _biz_days 同口径）。"""
    return pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=count))


def _seed(db: Database, symbol: str, spec: dict) -> None:
    """按 ``{日期: (o, h, l, c[, volume])}`` 写入前复权日线。"""
    frame = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp(day),
                "open": float(values[0]),
                "high": float(values[1]),
                "low": float(values[2]),
                "close": float(values[3]),
                "volume": float(values[4]) if len(values) > 4 else 1_000_000.0,
                "amount": float(values[3])
                * (float(values[4]) if len(values) > 4 else 1_000_000.0),
            }
            for day, values in spec.items()
        ]
    )
    frame["symbol"] = symbol
    frame["period"] = "d"
    frame["adjust"] = "qfq"
    db.upsert_bars(frame)


def _momentum_spec(days: list[str], phase: int, start_price: float, seed: int) -> dict:
    """阶段性趋势行情 + 确定性噪声（60 日周期：前 30 日 +0.5%/日、后 30 日 −0.5%/日）。

    与 P7-S3 编排测试同款构造：各标的相位与噪声种子不同 ⇒ 横截面上有区分度。
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
                calendar, phase=index * 5, start_price=10.0 + index * 0.1, seed=20261012 + index
            ),
        )
    return symbols


def _seed_constituents(
    db: Database, snapshots: dict[str, list[str]], index_code: str = "sh.000300"
) -> None:
    """按 ``{生效日: [标的, ...]}`` 写入成分股快照。"""
    rows = [
        IndexConstituent(
            index_code=index_code,
            symbol=symbol,
            name="",
            effective_date=pd.Timestamp(day),
        )
        for day, members in snapshots.items()
        for symbol in members
    ]
    db.upsert_index_constituents(rows)


# --------------------------------------------------------------------------- #
# 1) cross_section_zscore
# --------------------------------------------------------------------------- #
def test_zscore_hand_calc_and_nan_passthrough() -> None:
    """手算对拍：行 [1,2,3] ⇒ z = [−1.2247, 0, +1.2247]（ddof=0）；缺失保持 NaN。"""
    panel = pd.DataFrame(
        {"a": [1.0, 5.0], "b": [2.0, 5.0], "c": [3.0, 5.0]},
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
    )
    panel.loc["2024-01-02", "c"] = np.nan  # 第二行只留 [5, 5]
    z = cross_section_zscore(panel, clip=None)

    expected = 1.0 / np.sqrt(2.0 / 3.0)
    assert z.loc["2024-01-01", "a"] == pytest.approx(-expected)
    assert z.loc["2024-01-01", "b"] == pytest.approx(0.0)
    assert z.loc["2024-01-01", "c"] == pytest.approx(expected)
    # 第二行 [5,5]：标准差为 0 ⇒ 全 NaN（无区分度）
    assert z.loc["2024-01-02"].isna().all()


def test_zscore_constant_row_is_nan() -> None:
    """常量行（全同值）⇒ 标准差 0 ⇒ 该日全 NaN。"""
    panel = pd.DataFrame(
        {"a": [5.0], "b": [5.0], "c": [5.0]}, index=pd.DatetimeIndex(["2024-01-01"])
    )
    z = cross_section_zscore(panel)
    assert z.iloc[0].isna().all()


def test_zscore_clip_boundary() -> None:
    """±3σ 截断：极端值 z 被夹到 ±3；关闭截断时不夹。

    单离群点的最大 z = √(n−1)，故需 n ≥ 11 才能越过 3σ（此处 12 个样本
    ⇒ z = √11 ≈ 3.317）。
    """
    panel = pd.DataFrame(
        {
            **{name: [0.0] for name in "abcdefghijk"},
            "l": [100.0],
        },
        index=pd.DatetimeIndex(["2024-01-01"]),
    )
    clipped = cross_section_zscore(panel, clip=3.0)
    raw = cross_section_zscore(panel, clip=None)
    assert raw.iloc[0]["l"] == pytest.approx(np.sqrt(11.0))
    assert clipped.iloc[0]["l"] == pytest.approx(3.0)
    # 零值样本的 z（未越界）不被夹：−(1/n) / (√(n−1)/n) = −1/√11
    assert clipped.iloc[0]["a"] == pytest.approx(-1.0 / np.sqrt(11.0))


def test_zscore_invalid_input_degrades() -> None:
    """非法输入（非 DataFrame / 空表）⇒ 空表，不抛异常。"""
    assert cross_section_zscore("not-a-frame").empty
    assert cross_section_zscore(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# 2) dynamic_icir_weights
# --------------------------------------------------------------------------- #
def test_weights_no_lookahead_availability_lag() -> None:
    """兑现滞后：IC 在「评估日 + 前瞻期 + 1」个交易日后才参与权重估计。

    构造：pos0 IC=+0.5、pos1 IC=+0.3、pos6 IC=−9（剧烈翻空）。
    horizon=1 ⇒ 可用位置 2/3/8；min_ic_days=2 ⇒ pos0~2 权重为 0；
    pos3 起 ICIR>0 ⇒ 单因子归一化权重 = +1；pos8 起 ICIR<0 ⇒ −1。
    关键断言：pos6 的「未来」IC 不影响 pos3~7 的权重（无未来函数）。
    """
    calendar = _calendar(12)
    ic = pd.Series([0.5, 0.3, np.nan, np.nan, np.nan, np.nan, -9.0], index=calendar[:7])
    weights = dynamic_icir_weights({"f": ic}, calendar, horizon=1, min_ic_days=2)

    series = weights["f"]
    assert series.iloc[:3].eq(0.0).all()  # 样本不足 / 尚无可用 IC
    assert series.iloc[3:8].eq(1.0).all()  # 只有两个正 IC 可用 → ICIR>0
    assert series.iloc[8] == pytest.approx(-1.0)  # 第三个（负）IC 生效后翻向
    assert series.iloc[9:].eq(-1.0).all()


def test_weights_two_factors_normalized_ratio() -> None:
    """双因子：权重比 = ICIR 比，且 Σ|w| = 1（归一化口径）。

    A：IC = [0.1, 0.2]（可用位置 2、3）⇒ mean=0.15、std(ddof=1)=0.07071；
    B：IC = [0.1, 0.2, 0.4]（可用位置 2、3、4）⇒ mean=0.23333、std=0.15275。
    """
    calendar = _calendar(8)
    ic_a = pd.Series([0.1, 0.2], index=calendar[:2])
    ic_b = pd.Series([0.1, 0.2, 0.4], index=calendar[:3])
    weights = dynamic_icir_weights({"a": ic_a, "b": ic_b}, calendar, horizon=1, min_ic_days=2)

    mean_a, std_a = np.mean([0.1, 0.2]), np.std([0.1, 0.2], ddof=1)
    mean_b, std_b = np.mean([0.1, 0.2, 0.4]), np.std([0.1, 0.2, 0.4], ddof=1)
    icir_a, icir_b = mean_a / std_a, mean_b / std_b
    total = abs(icir_a) + abs(icir_b)

    last = weights.iloc[-1]
    assert abs(last["a"]) + abs(last["b"]) == pytest.approx(1.0)
    assert last["a"] == pytest.approx(icir_a / total)
    assert last["b"] == pytest.approx(icir_b / total)
    # pos4 之前 B 只有 2 个样本可用，A 有 1 个（不足 2）⇒ A 权重仍为 0
    assert weights["a"].iloc[2] == pytest.approx(0.0)


def test_weights_negative_ic_yields_negative_weight() -> None:
    """负 IC 因子 ⇒ 权重为负（自动反向：因子值越低得分越高）。"""
    calendar = _calendar(6)
    ic = pd.Series([-0.08, -0.12], index=calendar[:2])
    weights = dynamic_icir_weights({"f": ic}, calendar, horizon=1, min_ic_days=2)
    assert weights["f"].iloc[3] == pytest.approx(-1.0)
    assert weights["f"].iloc[2] == pytest.approx(0.0)


def test_weights_insufficient_samples_are_zero() -> None:
    """有效样本 < min_ic_days ⇒ 权重恒为 0（预热期不建仓）。"""
    calendar = _calendar(8)
    ic = pd.Series([0.1, 0.2, 0.3], index=calendar[:3])
    weights = dynamic_icir_weights({"f": ic}, calendar, horizon=1, min_ic_days=5)
    assert weights["f"].eq(0.0).all()


def test_weights_empty_ic_series_zero_column() -> None:
    """某因子无任何 IC 样本 ⇒ 该列权重全 0（其余因子照常）。"""
    calendar = _calendar(6)
    ic = pd.Series([0.1, 0.2], index=calendar[:2])
    weights = dynamic_icir_weights({"ok": ic, "empty": pd.Series(dtype=float)},
                                   calendar, horizon=1, min_ic_days=2)
    assert weights["empty"].eq(0.0).all()
    assert weights["ok"].iloc[-1] == pytest.approx(1.0)


def test_weights_invalid_input_degrades() -> None:
    """非法输入：空字典 / 空日历 ⇒ 空表。"""
    assert dynamic_icir_weights({}, _calendar(5)).empty
    assert dynamic_icir_weights({"f": pd.Series(dtype=float)},
                                pd.DatetimeIndex([])).empty


# --------------------------------------------------------------------------- #
# 3) combine_scores
# --------------------------------------------------------------------------- #
def _weights_frame(index: pd.DatetimeIndex, columns: list[str], values: list[list[float]]):
    """构造权重矩阵（测试内快捷方式）。"""
    return pd.DataFrame(values, index=index, columns=columns)


def test_combine_hand_calc() -> None:
    """合成手算：score = w1·z1 + w2·z2（逐格相乘再相加）。"""
    index = pd.DatetimeIndex(["2024-01-01"])
    z1 = pd.DataFrame({"a": [1.0], "b": [-2.0]}, index=index)
    z2 = pd.DataFrame({"a": [0.5], "b": [0.5]}, index=index)
    weights = _weights_frame(index, ["f1", "f2"], [[0.6, -0.4]])

    score = combine_scores({"f1": z1, "f2": z2}, weights)
    assert score.loc["2024-01-01", "a"] == pytest.approx(0.6 * 1.0 + (-0.4) * 0.5)
    assert score.loc["2024-01-01", "b"] == pytest.approx(0.6 * -2.0 + (-0.4) * 0.5)


def test_combine_missing_is_strict() -> None:
    """缺失严格口径：权重非零但因子缺值 ⇒ 该股当日 NaN；权重为 0 缺值不受影响。"""
    index = pd.DatetimeIndex(["2024-01-01"])
    z1 = pd.DataFrame({"a": [np.nan], "b": [1.0]}, index=index)
    z2 = pd.DataFrame({"a": [0.5], "b": [np.nan]}, index=index)
    weights = _weights_frame(index, ["f1", "f2"], [[0.6, 0.0]])

    score = combine_scores({"f1": z1, "f2": z2}, weights)
    # a：f1 权重非零却缺值 ⇒ NaN（即使 f2 有值、但权重为 0 不参与）
    assert np.isnan(score.loc["2024-01-01", "a"])
    # b：f1 有值（f2 权重为 0 缺值可忽略）⇒ 正常合成
    assert score.loc["2024-01-01", "b"] == pytest.approx(0.6 * 1.0)


def test_combine_zero_weight_day_all_nan() -> None:
    """当日全部因子权重为 0（预热）⇒ 该日全 NaN（不建仓）。"""
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    z1 = pd.DataFrame({"a": [1.0, 2.0]}, index=index)
    weights = _weights_frame(index, ["f1"], [[0.0], [1.0]])
    score = combine_scores({"f1": z1}, weights)
    assert score.loc["2024-01-01"].isna().all()
    assert score.loc["2024-01-02", "a"] == pytest.approx(2.0)


def test_combine_invalid_input_degrades() -> None:
    """非法输入：空因子字典 / 空权重 ⇒ 空表。"""
    index = pd.DatetimeIndex(["2024-01-01"])
    assert combine_scores({}, _weights_frame(index, ["f"], [[1.0]])).empty
    assert combine_scores({"f": pd.DataFrame({"a": [1.0]}, index=index)},
                          pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# 4) top_picks
# --------------------------------------------------------------------------- #
def test_top_picks_descending_with_tie_break() -> None:
    """得分降序取前 N；同分按代码升序（确定性）。全 NaN 日不产出。"""
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    score = pd.DataFrame(
        {"000002": [0.5, np.nan], "000001": [0.5, np.nan], "000003": [0.9, np.nan]},
        index=index,
    )
    picks = top_picks(score, 2)
    assert list(picks) == [index[0]]  # 第二日全 NaN ⇒ 不产出
    assert picks[index[0]] == ["000003", "000001"]  # 0.9 第一；0.5 同分按代码升序


def test_top_picks_invalid_n_returns_empty() -> None:
    """非法 N（0 / 负数 / 非数字）⇒ 空 dict。"""
    index = pd.DatetimeIndex(["2024-01-01"])
    score = pd.DataFrame({"a": [1.0]}, index=index)
    assert top_picks(score, 0) == {}
    assert top_picks(score, -3) == {}
    assert top_picks(score, "abc") == {}
    assert top_picks(pd.DataFrame(), 5) == {}


# --------------------------------------------------------------------------- #
# 5) FactorScorer E2E
# --------------------------------------------------------------------------- #
def test_scorer_e2e_momentum_universe(tmp_path) -> None:
    """动量宇宙 E2E：预热期全 NaN、之后打分有效、权重口径与结果表契约。"""
    db = _open(tmp_path / "scorer.db")
    symbols = _seed_momentum_universe(db, count=12, days=200)

    spec = ScoringSpec(factor_names=["ret_5"], ic_horizon=1, min_ic_days=30, min_stocks=10)
    report = FactorScorer(db).run(spec)

    assert report.ok, report.error
    assert report.universe_mode == "current"
    assert report.universe == symbols
    assert len(report.dates) == 200
    assert 0 < report.scored_days < 200
    # 预热期（权重为 0）应出现在窗口前部，且全是 NaN
    first_valid = report.score.notna().any(axis=1).idxmax()
    warmup_rows = report.score.loc[report.score.index < first_valid]
    assert warmup_rows.isna().all().all()
    # 有分数的日子每只股票都应有值（单因子 + 齐全数据）
    tail = report.score.loc[first_valid:].dropna(how="all")
    assert tail.notna().all(axis=1).all()
    # 权重口径：非零日 |w| = 1（单因子归一化）
    weights = report.weights["ret_5"]
    nonzero = weights[weights != 0.0]
    assert not nonzero.empty
    assert np.allclose(nonzero.abs().to_numpy(dtype=float), 1.0)
    # latest_weights 契约列与样本数
    assert list(report.latest_weights.columns) == WEIGHT_TABLE_COLUMNS
    row = report.latest_weights.iloc[0]
    assert row["factor"] == "ret_5"
    assert int(row["n_ic_days"]) >= 30
    assert float(row["weight"]) == pytest.approx(1.0) or float(row["weight"]) == pytest.approx(-1.0)


def test_scorer_defaults_to_core_three_factors(tmp_path) -> None:
    """默认因子 = 核心三因子（按注册顺序），dist_high_250 预热较长。"""
    db = _open(tmp_path / "scorer_defaults.db")
    _seed_momentum_universe(db, count=12, days=320)

    report = FactorScorer(db).run(ScoringSpec(ic_horizon=1, min_ic_days=30, min_stocks=10))

    assert report.ok, report.error
    names = [definition.name for definition in report.factors]
    assert names == ["ret_5", "log_amt_20", "dist_high_250"]
    assert set(names) == set(DEFAULT_SCORING_FACTORS)
    assert report.scored_days > 0
    # dist_high_250 的 IC 样本数应远小于 ret_5（250 日预热 + 兑现滞后）
    rows = report.latest_weights.set_index("factor")
    assert int(rows.loc["dist_high_250", "n_ic_days"]) < int(rows.loc["ret_5", "n_ic_days"])


def test_scorer_pit_mask_limits_cross_section(tmp_path) -> None:
    """PIT 口径：非成分日只保留「当时生效名单」成员，晚加入的标的不得提前出现。"""
    db = _open(tmp_path / "scorer_pit.db")
    days = _biz_days(200)
    symbols = _seed_momentum_universe(db, count=12, days=200)
    # 快照语义 = 「该日生效的完整名单」：第二份名单包含原 6 只 + 新 6 只。
    _seed_constituents(
        db,
        {days[0]: symbols[:6], days[80]: symbols},
    )

    spec = ScoringSpec(
        factor_names=["ret_5"],
        ic_horizon=1,
        min_ic_days=20,
        min_stocks=3,
        universe="pit",
    )
    report = FactorScorer(db).run(spec)

    assert report.ok, report.error
    assert report.universe_mode == "pit"
    assert report.universe == sorted(symbols)  # 并集 = 全部 12 只
    scored = report.score.dropna(how="all")
    assert not scored.empty
    # 中段（第二份名单生效前）：只有第一批 6 只有分
    mid = scored.loc[scored.index[5]]
    assert set(mid.dropna().index) == set(symbols[:6])
    # 末日：两批都在名单内，全部 12 只都有分
    last = report.score.iloc[-1].dropna()
    assert set(last.index) == set(symbols)


def test_scorer_warmup_only_returns_error(tmp_path) -> None:
    """窗口内全为预热期（IC 样本不足）⇒ ok=False 且错误信息含「预热」。"""
    db = _open(tmp_path / "scorer_warmup.db")
    _seed_momentum_universe(db, count=12, days=40)

    report = FactorScorer(db).run(
        ScoringSpec(factor_names=["ret_5"], ic_horizon=1, min_ic_days=120)
    )
    assert report.ok is False
    assert "预热" in report.error


def test_scorer_empty_db_degrades(tmp_path) -> None:
    """空库 ⇒ ok=False（错误信息说明无可用数据）。"""
    db = _open(tmp_path / "scorer_empty.db")
    report = FactorScorer(db).run(ScoringSpec())
    assert report.ok is False
    assert report.error


def test_scorer_window_without_days_degrades(tmp_path) -> None:
    """输出窗口与数据无交集 ⇒ ok=False。"""
    db = _open(tmp_path / "scorer_window.db")
    _seed_momentum_universe(db, count=12, days=60)
    report = FactorScorer(db).run(
        ScoringSpec(factor_names=["ret_5"], min_ic_days=5, start="2030-01-01")
    )
    assert report.ok is False
    assert "交易日" in report.error


def test_scorer_invalid_factor_names_degrades(tmp_path) -> None:
    """因子名全部无效 ⇒ ok=False 且警告记录无效名。"""
    db = _open(tmp_path / "scorer_names.db")
    _seed_momentum_universe(db, count=12, days=60)
    report = FactorScorer(db).run(ScoringSpec(factor_names=["nope"]))
    assert report.ok is False
    assert any("nope" in message for message in report.warnings)


def test_scorer_invalid_universe_raises(tmp_path) -> None:
    """非法 universe 取值 ⇒ ValueError（调用方输入错误）。"""
    db = _open(tmp_path / "scorer_mode.db")
    with pytest.raises(ValueError):
        FactorScorer(db).run(ScoringSpec(universe="wrong"))


def test_scorer_type_error(tmp_path) -> None:
    """spec 类型错误 ⇒ TypeError。"""
    db = _open(tmp_path / "scorer_type.db")
    with pytest.raises(TypeError):
        FactorScorer(db).run({"factor_names": ["ret_5"]})


def test_report_dataclass_shape() -> None:
    """ScoreReport 默认构造可直接用（空产出不抛异常）。"""
    report = ScoreReport(ok=False)
    assert report.score.empty
    assert report.latest_weights.empty
    assert list(report.latest_weights.columns) == WEIGHT_TABLE_COLUMNS