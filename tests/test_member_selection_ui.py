"""P7-S8 策略成员筛选对话框测试（离屏 ``offscreen``）。

覆盖：
    1. 构造 + 默认参数（样本内 2015-01-01 ~ 2020-12-31 / 样本外止 = 今天 / 门槛默认
       跑赢 + 回撤 + 夏普 0.5 + 正年过半 / 候选池 15 个 = 4 窗口 × 2 调仓 + 3 因子
       非空子集 / Top-N / 总资金）；
    2. ``current_criteria()``：勾选与取值透传（取消 / 改值 / 回撤上限启用）；
    3. ``current_split()``：默认口径与「顺序非法 → ``None`` + 状态提示」；
    4. ``current_pool()``：默认 15 候选与命名；取消窗口 / 调仓 / 因子后候选数变化、
       全取消 → ``None`` + 提示；PIT 无快照 → ``None`` + 提示，切「本地全库」→
       均线族 ``symbols=None``；
    5. 同步 ``run_selection()``（注入 fake runner）：runner 收到的池 / 切分 / 门槛
       与界面一致，明细表 / 对比表 / 四线曲线 / 状态刷新；
    6. ``ok=False`` 报告（fake runner）：不抛异常、视图清空、状态含失败原因；
    7. 同步 ``run_selection()``（真实临时库 + 合成动量行情 + 缩减候选池）：报告
       ``ok``、两张表与四线曲线有数据、状态含「成员筛选完成」；
    8. 后台线程路径（fake runner）：完成后按钮恢复、不残留运行中的线程。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QDate, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.data.store.db import Database  # noqa: E402
from app.engine.backtest import (  # noqa: E402
    DEFAULT_INITIAL_CASH,
    DEFAULT_MA_TOP_N,
    DEFAULT_MA_WINDOWS,
    DEFAULT_SCORED_TOP_N,
    VARIANT_ALL,
    VARIANT_BEST_SINGLE,
    VARIANT_SELECTED_FULL,
    VARIANT_SELECTED_IN,
    MemberCandidate,
    MemberEvaluation,
    PortfolioVariant,
    SampleSplit,
    SelectionComparison,
    SelectionReport,
    StrategySpec,
)
from app.engine.backtest.metrics import compute_metrics  # noqa: E402
from app.engine.factor import DEFAULT_SCORING_FACTORS  # noqa: E402
from app.engine.screener import Condition  # noqa: E402
from app.ui.widgets.member_selection import (  # noqa: E402
    MemberSelectionDialog,
    _COMPARISON_HEADERS,
    _SELECTION_HEADERS,
)

# 合成动量标的清单（播种与断言共用）。
_MOMENTUM_SYMBOLS: list[str] = [f"0000{index:02d}" for index in range(1, 13)]

# 落库的成分股快照生效日（落在默认切分内，供 PIT 并集取池）。
_SNAPSHOT_EFFECTIVE: str = "2020-01-01"


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_momentum_universe(db: Database, days_count: int = 480) -> list[str]:
    """播种 12 只相位错开的动量标的（qfq 日线），口径与既有 UI 测试一致。"""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=days_count)]
    for index, symbol in enumerate(_MOMENTUM_SYMBOLS):
        rng = np.random.default_rng(20261012 + index)
        noise = rng.normal(0.0, 0.003, len(days))
        volume_noise = rng.uniform(0.8, 1.2, len(days))
        price = 10.0 + index * 0.1
        records: list[dict] = []
        for position, day in enumerate(days):
            phase = (position + index * 5) % 60
            drift = 0.005 if phase < 30 else -0.005
            price *= 1.0 + drift + float(noise[position])
            band = 0.01 + abs(float(noise[position]))
            volume = 1_000_000.0 * float(volume_noise[position])
            records.append(
                {
                    "datetime": pd.Timestamp(day),
                    "open": price,
                    "high": price * (1.0 + band),
                    "low": price * (1.0 - band),
                    "close": price,
                    "volume": volume,
                    "amount": price * volume,
                    "symbol": symbol,
                    "period": "d",
                    "adjust": "qfq",
                }
            )
        db.upsert_bars(pd.DataFrame(records))
    return list(_MOMENTUM_SYMBOLS)


def _seed_constituents(db: Database) -> None:
    """播种一份覆盖全部 12 只动量标的的沪深300成分股快照（PIT 并集取池用）。"""
    frame = pd.DataFrame(
        {
            "index_code": ["sh.000300"] * len(_MOMENTUM_SYMBOLS),
            "symbol": list(_MOMENTUM_SYMBOLS),
            "name": [f"合成{index:02d}" for index in range(1, len(_MOMENTUM_SYMBOLS) + 1)],
            "effective_date": [_SNAPSHOT_EFFECTIVE] * len(_MOMENTUM_SYMBOLS),
        }
    )
    db.upsert_index_constituents(frame)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory) -> Database:
    """模块级共享的合成行情库（含成分股快照；筛选只读，可安全复用）。"""
    path = tmp_path_factory.mktemp("member_selection_ui") / "seed.db"
    db = _open(path)
    _seed_momentum_universe(db)
    _seed_constituents(db)
    return db


@pytest.fixture(scope="module")
def plain_db(tmp_path_factory) -> Database:
    """无成分股快照的合成行情库（PIT 失败路径用）。"""
    path = tmp_path_factory.mktemp("member_selection_ui_plain") / "seed.db"
    db = _open(path)
    _seed_momentum_universe(db)
    return db


def _make_dialog(db=None, runner=None) -> MemberSelectionDialog:
    return MemberSelectionDialog(
        db=db if db is not None else object(), runner=runner
    )


def _select_factors(dialog: MemberSelectionDialog, names) -> None:
    """按名字批量设置打分因子勾选状态（其余取消）。"""
    wanted = set(names)
    for name, check in dialog.factor_checks.items():
        check.setChecked(name in wanted)


def _set_windows(dialog: MemberSelectionDialog, windows) -> None:
    """按值批量设置均线窗口勾选状态（其余取消）。"""
    wanted = set(int(value) for value in windows)
    for window, check in dialog.window_checks.items():
        check.setChecked(window in wanted)


def _set_rebalances(dialog: MemberSelectionDialog, codes) -> None:
    """按代码批量设置调仓频率勾选状态（其余取消）。"""
    wanted = set(codes)
    for code, check in dialog.rebalance_checks.items():
        check.setChecked(code in wanted)


def _set_all_criteria_off(dialog: MemberSelectionDialog) -> None:
    """把合格门槛全部关闭（判定恒通过；真实回测路径缩短判定链）。"""
    dialog.beat_check.setChecked(False)
    dialog.dd_vs_bench_check.setChecked(False)
    dialog.dd_limit_check.setChecked(False)
    dialog.sharpe_check.setChecked(False)
    dialog.win_ratio_check.setChecked(False)


def _set_dates(dialog: MemberSelectionDialog, in_start: str, in_end: str, out_end: str) -> None:
    """设置样本切分日期（字符串 ``YYYY-MM-DD``）。"""
    dialog.in_start_edit.setDate(QDate.fromString(in_start, "yyyy-MM-dd"))
    dialog.in_end_edit.setDate(QDate.fromString(in_end, "yyyy-MM-dd"))
    dialog.out_end_edit.setDate(QDate.fromString(out_end, "yyyy-MM-dd"))


def _make_equity(periods: int = 30) -> pd.DataFrame:
    """合成一段样本外净值（nav 上行 / benchmark_nav 缓行，段首 1.0）。"""
    days = pd.bdate_range("2021-01-04", periods=periods)
    nav = np.linspace(1.0, 1.20, periods)
    bench = np.linspace(1.0, 1.08, periods)
    return pd.DataFrame({"date": days, "nav": nav, "benchmark_nav": bench})


class _FakeResult:
    """鸭子类型的最小结果对象（``slice_equity`` / 绘图只取 ``equity``）。"""

    def __init__(self, equity: pd.DataFrame) -> None:
        self.equity = equity
        self.trades = None


def _make_metrics_out() -> object:
    """用合成净值算出样本外绩效（供状态提示读取关键指标）。"""
    frame = _make_equity()
    nav = pd.Series(
        frame["nav"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame["date"])),
    )
    bench = pd.Series(
        frame["benchmark_nav"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(frame["date"])),
    )
    return compute_metrics(nav, bench, None)


def _make_report(split: SampleSplit, criteria, ok: bool = True) -> SelectionReport:
    """合成一份筛选报告（两个候选：其一 样本内合格 / 其二 样本内被淘汰）。"""
    if not ok:
        return SelectionReport(
            ok=False,
            error="全部候选回测失败，筛选报告无效",
            split=split,
            criteria=criteria,
            warnings=["[候选·站上5日线·周频] 回测失败：无有效数据"],
        )

    def _candidate(name: str, window: int) -> MemberCandidate:
        return MemberCandidate(
            name=name,
            spec=StrategySpec(
                name=name,
                conditions=[Condition(f"close - close.rolling({window}).mean()", ">", 0)],
                rebalance="weekly",
                top_n=3,
            ),
        )

    passed = MemberEvaluation(
        candidate=_candidate("站上5日线·周频", 5),
        ok=True,
        passed_in=True,
        passed_full=False,
        reasons_in=[],
        reasons_full=["全样本未跑赢基准（+10.00% < +20.00%）"],
    )
    rejected = MemberEvaluation(
        candidate=_candidate("站上10日线·月频", 10),
        ok=True,
        passed_in=False,
        passed_full=False,
        reasons_in=["样本内未跑赢基准（+5.00% < +43.10%）"],
        reasons_full=["全样本未跑赢基准（+8.00% < +20.00%）"],
    )
    return SelectionReport(
        ok=True,
        split=split,
        criteria=criteria,
        evaluations=[passed, rejected],
        passed_in=[passed.candidate.name],
        passed_full=[],
    )


def _make_comparison(split: SampleSplit) -> SelectionComparison:
    """合成四线对比（全候选 / 筛选后成功；上界参考失败；最优单策略成功）。"""
    equity = _make_equity()
    metrics_out = _make_metrics_out()
    return SelectionComparison(
        ok=True,
        split=split,
        variants=[
            PortfolioVariant(
                key=VARIANT_ALL,
                name="筛选前·全候选等权",
                members=["站上5日线·周频"],
                ok=True,
                result=_FakeResult(equity),
                metrics_out=metrics_out,
            ),
            PortfolioVariant(
                key=VARIANT_SELECTED_IN,
                name="筛选后·样本内合格等权",
                members=["站上5日线·周频"],
                ok=True,
                result=_FakeResult(equity),
                metrics_out=metrics_out,
            ),
            PortfolioVariant(
                key=VARIANT_SELECTED_FULL,
                name="上界参考·全样本合格等权",
                members=[],
                ok=False,
                error="无可用成员（筛选结果为空）",
            ),
            PortfolioVariant(
                key=VARIANT_BEST_SINGLE,
                name="对照·最优单策略",
                members=["站上5日线·周频"],
                ok=True,
                result=_FakeResult(equity),
                metrics_out=metrics_out,
            ),
        ],
        warnings=["全样本无合格成员，「上界参考」组合无法构建"],
    )


class _FakeRunner:
    """记录调用参数并返回预置报告 / 对比的假编排器（无线程、无库、无网络）。"""

    def __init__(self, report: SelectionReport, comparison: SelectionComparison) -> None:
        self.report = report
        self.comparison = comparison
        self.calls: list[tuple] = []

    def run(self, pool, split, criteria):
        self.calls.append(("run", pool, split, criteria))
        return self.report

    def compare(self, report, split):
        self.calls.append(("compare", report, split))
        return self.comparison


# --------------------------------------------------------------------------- #
# 1. 构造 + 默认参数
# --------------------------------------------------------------------------- #
def test_construct_defaults(qapp) -> None:
    dialog = _make_dialog()

    # 样本切分：默认口径 = 引擎默认常量；样本外自样本内截止次日。
    split = dialog.current_split()
    assert split is not None
    assert split.in_start == pd.Timestamp("2015-01-01")
    assert split.in_end == pd.Timestamp("2020-12-31")
    assert split.out_start == pd.Timestamp("2021-01-01")
    assert split.out_end == pd.Timestamp(QDate.currentDate().toString("yyyy-MM-dd"))

    # 门槛默认：跑赢 + 回撤不深于基准 + 夏普 0.5 + 正年过半；回撤上限不启用。
    criteria = dialog.current_criteria()
    assert criteria.beat_benchmark is True
    assert criteria.drawdown_not_worse_than_benchmark is True
    assert criteria.max_drawdown_limit is None
    assert criteria.min_sharpe == pytest.approx(0.5)
    assert criteria.min_win_year_ratio == pytest.approx(0.5)

    # 候选池默认：4 窗口 × 2 调仓 + 3 因子非空子集 = 15。
    assert dialog.selected_ma_windows() == list(DEFAULT_MA_WINDOWS)
    assert dialog.selected_ma_rebalances() == ["weekly", "monthly"]
    assert dialog.selected_factors() == list(DEFAULT_SCORING_FACTORS)
    assert dialog.ma_top_n_spin.value() == DEFAULT_MA_TOP_N
    assert dialog.scored_top_n_spin.value() == DEFAULT_SCORED_TOP_N
    assert dialog.total_cash_spin.value() == pytest.approx(DEFAULT_INITIAL_CASH)
    assert dialog.current_pool_choice() == "pit"
    assert "候选数：15" in dialog.pool_hint_label.text()

    # 尚未运行：结果视图为空。
    assert dialog.last_report() is None
    assert dialog.last_comparison() is None
    assert dialog.selection_table.rowCount() == 0
    assert dialog.comparison_table.rowCount() == 0
    for curve in dialog.variant_curves.values():
        x_values, _ = curve.getData()
        assert x_values is None or len(x_values) == 0
    dialog.close()


# --------------------------------------------------------------------------- #
# 2. current_criteria 透传
# --------------------------------------------------------------------------- #
def test_current_criteria_from_ui(qapp) -> None:
    dialog = _make_dialog()

    dialog.beat_check.setChecked(False)
    dialog.dd_vs_bench_check.setChecked(False)
    dialog.dd_limit_check.setChecked(True)
    dialog.dd_limit_spin.setValue(30.0)
    dialog.sharpe_check.setChecked(False)
    dialog.win_ratio_check.setChecked(True)
    dialog.win_ratio_spin.setValue(60.0)

    criteria = dialog.current_criteria()

    assert criteria.beat_benchmark is False
    assert criteria.drawdown_not_worse_than_benchmark is False
    assert criteria.max_drawdown_limit == pytest.approx(0.30)
    assert criteria.min_sharpe is None
    assert criteria.min_win_year_ratio == pytest.approx(0.60)
    dialog.close()


# --------------------------------------------------------------------------- #
# 3. current_split 非法顺序
# --------------------------------------------------------------------------- #
def test_current_split_invalid_order(qapp) -> None:
    dialog = _make_dialog()

    _set_dates(dialog, "2025-01-01", "2024-01-01", "2025-06-30")
    assert dialog.current_split() is None
    assert "样本切分不合法" in dialog.status_label.text()

    _set_dates(dialog, "2024-01-01", "2024-06-30", "2024-06-30")
    assert dialog.current_split() is None
    assert "样本外截止必须晚于样本内截止" in dialog.status_label.text()

    _set_dates(dialog, "2024-01-01", "2024-06-30", "2025-06-30")
    assert dialog.current_split() is not None
    dialog.close()


# --------------------------------------------------------------------------- #
# 4. current_pool：默认 / 勾选变化 / 全取消 / PIT 失败降级
# --------------------------------------------------------------------------- #
def test_current_pool_defaults_and_toggles(qapp, seeded_db) -> None:
    dialog = MemberSelectionDialog(db=seeded_db)

    pool = dialog.current_pool()
    assert pool is not None and len(pool) == 15
    names = [candidate.name for candidate in pool]
    assert "站上5日线·周频" in names
    assert "站上60日线·月频" in names
    assert "打分·log_amt_20+ret_5+dist_high_250" in names
    # 均线族候选带 PIT 并集标的池（快照覆盖全部 12 只合成标的）。
    ma_candidate = next(c for c in pool if c.family == "ma")
    assert ma_candidate.spec.symbols == sorted(_MOMENTUM_SYMBOLS)

    _set_windows(dialog, [5])
    pool = dialog.current_pool()
    assert pool is not None and len(pool) == 2 + 7
    _set_rebalances(dialog, ["monthly"])
    pool = dialog.current_pool()
    assert pool is not None and len(pool) == 1 + 7
    _select_factors(dialog, ["ret_5", "log_amt_20"])
    pool = dialog.current_pool()
    assert pool is not None and len(pool) == 1 + 3
    assert "候选数：4" in dialog.pool_hint_label.text()

    # 逐类全取消 → None + 对应提示（按窗口 → 调仓 → 因子的检查顺序）。
    _set_windows(dialog, [])
    assert dialog.current_pool() is None
    assert "请至少勾选一个均线窗口" in dialog.status_label.text()

    _set_windows(dialog, [5])
    _set_rebalances(dialog, [])
    assert dialog.current_pool() is None
    assert "请至少勾选一个调仓频率" in dialog.status_label.text()

    _set_rebalances(dialog, ["weekly"])
    _select_factors(dialog, [])
    assert dialog.current_pool() is None
    assert "请至少勾选一个打分因子" in dialog.status_label.text()
    dialog.close()


def test_current_pool_pit_missing_then_local(qapp, plain_db) -> None:
    """库内无成分股快照：PIT 并集取池失败提示；切「本地全库」→ 均线族 ``symbols=None``。"""
    dialog = MemberSelectionDialog(db=plain_db)

    assert dialog.current_pool() is None
    assert "成分股快照" in dialog.status_label.text()

    dialog.pool_combo.setCurrentIndex(dialog.pool_combo.findData("local"))
    pool = dialog.current_pool()
    assert pool is not None and len(pool) == 15
    ma_candidate = next(c for c in pool if c.family == "ma")
    assert ma_candidate.spec.symbols is None  # 引擎回退「本地全部前复权标的」
    dialog.close()


# --------------------------------------------------------------------------- #
# 5. 同步 run_selection（注入 fake runner）
# --------------------------------------------------------------------------- #
def test_run_selection_sync_with_fake_runner(qapp, seeded_db) -> None:
    probe = MemberSelectionDialog(db=seeded_db)
    split = probe.current_split()
    assert split is not None
    fake = _FakeRunner(_make_report(split, probe.current_criteria()), _make_comparison(split))
    probe.close()
    dialog = MemberSelectionDialog(db=seeded_db, runner=fake)

    report = dialog.run_selection()

    assert report is fake.report
    assert dialog.last_report() is fake.report
    assert dialog.last_comparison() is fake.comparison

    # runner 收到的池 / 切分 / 门槛与界面一致。
    kind, pool, received_split, received_criteria = fake.calls[0]
    assert kind == "run" and len(pool) == 15
    assert received_split.in_start == pd.Timestamp("2015-01-01")
    assert received_criteria == dialog.current_criteria()
    assert fake.calls[1][0] == "compare"

    # 明细表：13 列契约 + 2 行；合格列显示「是 / 否」；淘汰原因列有文本（样本内口径）。
    assert dialog.selection_table.rowCount() == 2
    assert dialog.selection_table.columnCount() == len(_SELECTION_HEADERS)
    assert dialog.selection_table.item(0, 0).text() == "站上5日线·周频"
    assert dialog.selection_table.item(0, 7).text() == "是"
    assert dialog.selection_table.item(0, 11).text() == "否"
    assert dialog.selection_table.item(0, 12).text() == ""  # 合格候选无淘汰原因
    assert dialog.selection_table.item(1, 0).text() == "站上10日线·月频"
    assert dialog.selection_table.item(1, 7).text() == "否"
    assert "未跑赢基准" in dialog.selection_table.item(1, 12).text()

    # 对比表：4 行契约列。
    assert dialog.comparison_table.rowCount() == 4
    assert dialog.comparison_table.columnCount() == len(_COMPARISON_HEADERS)
    assert dialog.comparison_table.item(0, 0).text() == VARIANT_ALL

    # 曲线：三条成功变体有数据（30 点）+ 基准；失败的上界参考为空。
    x_all, y_all = dialog.variant_curves[VARIANT_ALL].getData()
    assert len(x_all) == 30 and len(y_all) == 30
    x_in, _ = dialog.variant_curves[VARIANT_SELECTED_IN].getData()
    assert len(x_in) == 30
    x_full, _ = dialog.variant_curves[VARIANT_SELECTED_FULL].getData()
    assert x_full is None or len(x_full) == 0
    x_best, _ = dialog.variant_curves[VARIANT_BEST_SINGLE].getData()
    assert len(x_best) == 30
    x_bench, _ = dialog.benchmark_curve.getData()
    assert len(x_bench) == 30
    assert dialog.nav_axis.label_count() == 30
    assert "样本外段" in dialog.nav_scope_label.text()

    # 状态与警告：含完成口径与筛选后组合样本外指标；警告透传对比级警告。
    status = dialog.status_label.text()
    assert "成员筛选完成" in status and "样本内合格 1" in status
    assert "筛选后组合（样本外自 2021-01-01）" in status
    assert not dialog.warnings_label.isHidden()
    assert "上界参考" in dialog.warnings_label.text()
    dialog.close()


# --------------------------------------------------------------------------- #
# 6. ok=False 报告（fake runner）
# --------------------------------------------------------------------------- #
def test_run_selection_failure_report(qapp, seeded_db) -> None:
    probe = MemberSelectionDialog(db=seeded_db)
    split = probe.current_split()
    assert split is not None
    fake = _FakeRunner(
        _make_report(split, probe.current_criteria(), ok=False),
        _make_comparison(split),
    )
    probe.close()
    dialog = MemberSelectionDialog(db=seeded_db, runner=fake)

    report = dialog.run_selection()

    assert report is not None and report.ok is False
    assert "成员筛选未完成" in dialog.status_label.text()
    assert "全部候选回测失败" in dialog.status_label.text()
    # 失败时视图清空；compare 不被调用（报告无效无法对比）。
    assert dialog.selection_table.rowCount() == 0
    assert dialog.comparison_table.rowCount() == 0
    for curve in dialog.variant_curves.values():
        x_values, _ = curve.getData()
        assert x_values is None or len(x_values) == 0
    assert [call[0] for call in fake.calls] == ["run"]
    assert not dialog.warnings_label.isHidden()
    dialog.close()


# --------------------------------------------------------------------------- #
# 7. 同步 run_selection（真实小型库 + 缩减候选池）
# --------------------------------------------------------------------------- #
def test_run_selection_real_small_pool(qapp, seeded_db) -> None:
    """缩减到 2 个候选（5 日线·周频 + 打分·ret_5），门槛全关，真实引擎全链路。"""
    dialog = MemberSelectionDialog(db=seeded_db)
    _set_dates(dialog, "2024-06-01", "2025-06-30", "2025-12-31")
    _set_windows(dialog, [5])
    _set_rebalances(dialog, ["weekly"])
    _select_factors(dialog, ["ret_5"])
    _set_all_criteria_off(dialog)

    report = dialog.run_selection()

    assert report is not None, dialog.status_label.text()
    assert report.ok is True
    assert len(report.evaluations) == 2
    assert len(report.passed_in) == 2  # 门槛全关 → 恒合格

    assert dialog.selection_table.rowCount() == 2
    assert dialog.comparison_table.rowCount() == 4
    assert dialog.last_comparison() is not None
    assert dialog.last_comparison().ok is True

    # 四线变体均有成员（2 个候选全合格）→ 曲线全部有数据；筛选后与上界同集合。
    for key in (VARIANT_ALL, VARIANT_SELECTED_IN, VARIANT_SELECTED_FULL, VARIANT_BEST_SINGLE):
        x_values, _ = dialog.variant_curves[key].getData()
        assert x_values is not None and len(x_values) > 0, key
    x_bench, _ = dialog.benchmark_curve.getData()
    assert len(x_bench) > 0

    status = dialog.status_label.text()
    assert "成员筛选完成" in status and "样本内合格 2" in status
    dialog.close()


# --------------------------------------------------------------------------- #
# 8. 后台线程路径 + 关闭不残留线程
# --------------------------------------------------------------------------- #
def test_async_run_with_fake_runner(qapp, seeded_db) -> None:
    probe = MemberSelectionDialog(db=seeded_db)
    split = probe.current_split()
    assert split is not None
    fake = _FakeRunner(_make_report(split, probe.current_criteria()), _make_comparison(split))
    probe.close()
    dialog = MemberSelectionDialog(db=seeded_db, runner=fake)

    dialog._on_run()
    assert dialog._worker is not None
    assert dialog.run_button.isEnabled() is False

    worker = dialog._worker
    assert worker.wait(10_000)
    for _ in range(200):
        qapp.processEvents()
        if dialog._worker is None:
            break

    assert dialog._worker is None
    assert dialog.run_button.isEnabled() is True
    assert dialog.last_report() is fake.report
    assert dialog.selection_table.rowCount() == 2

    dialog.close()
    dialog.deleteLater()
    assert dialog._worker is None