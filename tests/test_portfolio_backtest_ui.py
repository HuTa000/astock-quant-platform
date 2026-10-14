"""P7-S7 多策略组合回测对话框测试（离屏 ``offscreen``）。

覆盖：
    1. 构造 + 默认参数（成员权重 1:1 / 20 日线 Top-3 周频 / 打分 Top-N 月频 PIT /
       前瞻 20 / 最少 IC 120 / 总资金 100 万 / 起始 2015-01-01）；
    2. ``current_spec()``：默认值透传（成员名 / 类型 / 归一化权重 / 标的池 /
       因子勾选）；取消全部因子 → ``None`` 且状态提示，恢复勾选可运行；
    3. 非等权（3:1）→ 归一化 0.75 / 0.25，成员名随均线窗口变化；
    4. 条件式标的池：库内无成分股快照 → ``None`` + 状态提示；切「本地全库」→
       ``symbols=None``（引擎回退）；
    5. 同步 ``run_portfolio_backtest()``（真实临时 SQLite 库 + 合成动量行情）：
       结果 ``ok``、绩效表 / 成员对照 / 相关性矩阵 / 贡献归因 / 成交明细有数据、
       净值曲线有数据、状态含「组合回测完成」与 Σ贡献对账；
    6. 非等权组合：回测警告含分账口径说明，归因表权重 0.75 / 0.25；
    7. ``ok=False`` 结果（打分成员 PIT 口径但库内无快照）：不抛异常、状态含错误
       原因、视图清空；
    8. 对话框销毁后不残留运行中的后台线程。
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
    DEFAULT_SCORED_REBALANCE,
    DEFAULT_SCORED_TOP_N,
    DEFAULT_SCORED_UNIVERSE,
    PORTFOLIO_TRADE_COLUMNS,
)
from app.engine.factor import (  # noqa: E402
    DEFAULT_IC_HORIZON,
    DEFAULT_MIN_IC_DAYS,
    DEFAULT_SCORING_FACTORS,
)
from app.ui.widgets.portfolio_backtest import (  # noqa: E402
    PortfolioBacktestDialog,
    _CONTRIBUTION_HEADERS,
    _MEMBER_HEADERS,
)

# 合成动量标的清单（播种与断言共用）。
_MOMENTUM_SYMBOLS: list[str] = [f"0000{index:02d}" for index in range(1, 13)]

# 落库的成分股快照生效日（落在对话框默认窗口内，供 PIT 并集取池）。
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
    """播种 12 只相位错开的动量标的（qfq 日线），口径与打分 UI 测试一致。"""
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
    """模块级共享的合成行情库（含成分股快照；组合回测只读，可安全复用）。"""
    path = tmp_path_factory.mktemp("portfolio_ui") / "seed.db"
    db = _open(path)
    _seed_momentum_universe(db)
    _seed_constituents(db)
    return db


@pytest.fixture(scope="module")
def plain_db(tmp_path_factory) -> Database:
    """无成分股快照的合成行情库（PIT 失败路径用）。"""
    path = tmp_path_factory.mktemp("portfolio_ui_plain") / "seed.db"
    db = _open(path)
    _seed_momentum_universe(db)
    return db


def _make_dialog(db=None) -> PortfolioBacktestDialog:
    return PortfolioBacktestDialog(db=db if db is not None else object())


def _select_factors(dialog: PortfolioBacktestDialog, names) -> None:
    """按名字批量设置打分因子勾选状态（其余取消）。"""
    wanted = set(names)
    for index in range(dialog.factor_list.count()):
        item = dialog.factor_list.item(index)
        name = str(item.data(Qt.ItemDataRole.UserRole))
        item.setCheckState(
            Qt.CheckState.Checked if name in wanted else Qt.CheckState.Unchecked
        )


def _prepare_for_run(dialog: PortfolioBacktestDialog) -> None:
    """把对话框调到「快速可跑」口径：打分 current 池 / 周频 / 小预热门槛 / 2 因子 / Top-3。"""
    dialog.scored_universe_combo.setCurrentIndex(
        dialog.scored_universe_combo.findData("current")
    )
    dialog.scored_rebalance_combo.setCurrentIndex(
        dialog.scored_rebalance_combo.findData("weekly")
    )
    dialog.ic_horizon_spin.setValue(5)
    dialog.min_ic_days_spin.setValue(30)
    dialog.scored_top_n_spin.setValue(3)
    _select_factors(dialog, ["ret_5", "log_amt_20"])


_DETAIL_TABLES = (
    "metrics_table",
    "yearly_table",
    "regime_table",
    "members_table",
    "correlation_table",
    "contribution_table",
    "trades_table",
)


# --------------------------------------------------------------------------- #
# 1. 构造 + 默认参数
# --------------------------------------------------------------------------- #
def test_construct_defaults(qapp) -> None:
    dialog = _make_dialog()

    assert dialog.cond_window_spin.value() == 20
    assert dialog.cond_top_n_spin.value() == 3
    assert dialog.current_cond_rebalance() == "weekly"
    assert dialog.current_cond_pool() == "pit"
    assert dialog.cond_weight_spin.value() == pytest.approx(1.0)
    assert dialog.scored_top_n_spin.value() == DEFAULT_SCORED_TOP_N
    assert dialog.current_scored_rebalance() == DEFAULT_SCORED_REBALANCE
    assert dialog.current_scored_universe() == DEFAULT_SCORED_UNIVERSE
    assert dialog.scored_weight_spin.value() == pytest.approx(1.0)
    assert dialog.ic_horizon_spin.value() == DEFAULT_IC_HORIZON
    assert dialog.min_ic_days_spin.value() == DEFAULT_MIN_IC_DAYS
    assert len(dialog.selected_factors()) == 3
    assert set(dialog.selected_factors()) == set(DEFAULT_SCORING_FACTORS)
    assert dialog.total_cash_spin.value() == pytest.approx(DEFAULT_INITIAL_CASH)
    assert dialog.current_dates() == (
        "2015-01-01",
        QDate.currentDate().toString("yyyy-MM-dd"),
    )
    assert dialog.last_result() is None
    for name in _DETAIL_TABLES:
        assert getattr(dialog, name).rowCount() == 0
    x_nav, _ = dialog.nav_strategy_curve.getData()
    assert x_nav is None or len(x_nav) == 0


# --------------------------------------------------------------------------- #
# 2. current_spec 默认值 / 无因子 / 恢复勾选
# --------------------------------------------------------------------------- #
def test_current_spec_defaults(qapp, seeded_db) -> None:
    dialog = PortfolioBacktestDialog(db=seeded_db)

    spec = dialog.current_spec()

    assert spec is not None
    assert [member.name for member in spec.members] == ["站上20日线", "多因子打分"]
    assert [member.kind for member in spec.members] == ["condition", "scored"]
    assert [member.weight for member in spec.members] == pytest.approx([0.5, 0.5])
    assert spec.initial_cash == pytest.approx(DEFAULT_INITIAL_CASH)
    assert spec.start == "2015-01-01"

    condition_spec = spec.members[0].spec
    assert condition_spec.symbols == sorted(_MOMENTUM_SYMBOLS)
    assert condition_spec.top_n == 3
    assert condition_spec.rebalance == "weekly"
    assert "rolling(20)" in str(condition_spec.conditions[0].field)

    scored_spec = spec.members[1].spec
    assert set(scored_spec.factors) == set(DEFAULT_SCORING_FACTORS)
    assert scored_spec.top_n == DEFAULT_SCORED_TOP_N
    assert scored_spec.universe == "pit"
    dialog.close()


def test_current_spec_no_factors_then_restore(qapp) -> None:
    dialog = _make_dialog()

    _select_factors(dialog, [])  # 取消全部勾选
    assert dialog.current_spec() is None
    assert "请至少勾选一个打分因子" in dialog.status_label.text()

    # 无真实库时默认 PIT 并集取池不可用 → 切「本地全库」验证恢复勾选后可构建。
    dialog.cond_pool_combo.setCurrentIndex(dialog.cond_pool_combo.findData("local"))
    _select_factors(dialog, DEFAULT_SCORING_FACTORS)  # 恢复核心三因子
    spec = dialog.current_spec()
    assert spec is not None
    assert set(spec.members[1].spec.factors) == set(DEFAULT_SCORING_FACTORS)


def test_current_spec_weights_and_window_name(qapp, seeded_db) -> None:
    """非等权按输入归一化；成员名随均线窗口变化；总资金透传。"""
    dialog = PortfolioBacktestDialog(db=seeded_db)

    dialog.cond_weight_spin.setValue(3.0)
    dialog.scored_weight_spin.setValue(1.0)
    dialog.cond_window_spin.setValue(60)
    dialog.total_cash_spin.setValue(500_000.0)

    spec = dialog.current_spec()

    assert spec is not None
    assert spec.members[0].name == "站上60日线"
    assert "rolling(60)" in str(spec.members[0].spec.conditions[0].field)
    assert [member.weight for member in spec.members] == pytest.approx([0.75, 0.25])
    assert spec.initial_cash == pytest.approx(500_000.0)
    dialog.close()


def test_current_spec_pit_without_snapshots_then_local(qapp, plain_db) -> None:
    """库内无成分股快照：PIT 并集取池失败提示；切「本地全库」→ ``symbols=None``。"""
    dialog = PortfolioBacktestDialog(db=plain_db)

    assert dialog.current_spec() is None
    assert "成分股快照" in dialog.status_label.text()

    dialog.cond_pool_combo.setCurrentIndex(
        dialog.cond_pool_combo.findData("local")
    )
    spec = dialog.current_spec()
    assert spec is not None
    assert spec.members[0].spec.symbols is None  # 引擎回退「本地全部前复权标的」
    dialog.close()


# --------------------------------------------------------------------------- #
# 3. run_portfolio_backtest 成功路径（真实小规模组合回测）
# --------------------------------------------------------------------------- #
def test_run_portfolio_backtest_success(qapp, seeded_db) -> None:
    dialog = PortfolioBacktestDialog(db=seeded_db)
    _prepare_for_run(dialog)

    result = dialog.run_portfolio_backtest()

    assert result is not None, dialog.status_label.text()
    assert result.ok is True, result.error
    assert dialog.last_result() is result

    # 绩效表（15 行）+ 逐年表 + 牛熊表 + 成员对照（2 行）+ 归因（2 行）+ 成交明细。
    assert dialog.metrics_table.rowCount() == 15
    assert dialog.yearly_table.rowCount() >= 1
    assert dialog.regime_table.rowCount() >= 1
    assert dialog.members_table.rowCount() == 2
    assert dialog.members_table.columnCount() == len(_MEMBER_HEADERS)
    assert dialog.contribution_table.rowCount() == 2
    assert dialog.contribution_table.columnCount() == len(_CONTRIBUTION_HEADERS)
    assert dialog.trades_table.columnCount() == len(PORTFOLIO_TRADE_COLUMNS)
    assert dialog.trades_table.rowCount() >= 1

    # 相关性矩阵：2×2、对角 1.0，表头为成员名。
    assert result.correlation.shape == (2, 2)
    assert float(result.correlation.iloc[0, 0]) == pytest.approx(1.0)
    assert dialog.correlation_table.rowCount() == 2
    assert dialog.correlation_table.columnCount() == 3
    assert dialog.correlation_table.horizontalHeaderItem(0).text() == "成员"
    assert dialog.correlation_table.horizontalHeaderItem(1).text() == "站上20日线"
    assert dialog.correlation_table.item(0, 0).text() != "—"

    # 贡献归因对账：Σ contribution = 组合总收益。
    contribution_sum = float(result.contribution["contribution"].sum())
    assert contribution_sum == pytest.approx(result.metrics.total_return, abs=1e-9)
    # 成交明细末列为成员名且落在成员集合内。
    member_names = {member.name for member in result.members}
    assert dialog.trades_table.item(0, 8).text() in member_names

    # 净值曲线有数据（组合 + 基准 + 两条成员曲线），轴标签数量与数据点一致。
    x_nav, y_nav = dialog.nav_strategy_curve.getData()
    assert len(x_nav) == len(y_nav) and len(x_nav) > 0
    assert dialog.nav_axis.label_count() == len(x_nav)
    x_member, _ = dialog.nav_member1_curve.getData()
    assert x_member is not None and len(x_member) > 0
    assert "等权" in dialog.nav_member1_label.text() or "站上20日线" in dialog.nav_member1_label.text()

    # 状态汇总含完成口径与 Σ贡献对账。
    status = dialog.status_label.text()
    assert "组合回测完成" in status and "Σ成员贡献" in status
    assert "最大回撤" in status

    # 合成库无沪深300指数 → 等权代理基准，警告如实展示。
    assert not dialog.warnings_label.isHidden()
    assert "代理" in dialog.warnings_label.text()

    # 同步路径不创建后台线程。
    assert dialog._worker is None
    dialog.close()


def test_run_weighted_backtest_reports_split(qapp, seeded_db) -> None:
    """3:1 非等权组合：回测警告含分账口径说明，归因表权重 0.75 / 0.25。"""
    dialog = PortfolioBacktestDialog(db=seeded_db)
    _prepare_for_run(dialog)
    dialog.cond_weight_spin.setValue(3.0)
    dialog.scored_weight_spin.setValue(1.0)

    result = dialog.run_portfolio_backtest()

    assert result is not None, dialog.status_label.text()
    assert result.ok is True, result.error
    assert [member.weight for member in result.members] == pytest.approx([0.75, 0.25])
    assert any("分账" in warning for warning in result.warnings)
    assert dialog.contribution_table.item(0, 1).text() == "75.00%"
    dialog.close()


# --------------------------------------------------------------------------- #
# 4. ok=False 路径（打分成员 PIT 失败）
# --------------------------------------------------------------------------- #
def test_run_scored_pit_failure_is_reported(qapp, plain_db) -> None:
    """打分成员 PIT 口径但库内无快照：``ok=False`` 不抛异常，状态含原因，视图清空。"""
    dialog = PortfolioBacktestDialog(db=plain_db)
    _prepare_for_run(dialog)
    # 条件式成员切「本地全库」保证其本身可跑，失败点定位在打分成员 PIT。
    dialog.cond_pool_combo.setCurrentIndex(dialog.cond_pool_combo.findData("local"))
    dialog.scored_universe_combo.setCurrentIndex(
        dialog.scored_universe_combo.findData("pit")
    )

    result = dialog.run_portfolio_backtest()

    assert result is not None
    assert result.ok is False
    assert "多因子打分" in result.error
    assert "成分股快照" in result.error
    assert "组合回测未完成" in dialog.status_label.text()
    assert "成分股快照" in dialog.status_label.text()
    # 失败时视图清空（成员对照保留现场由 result.members 承载，界面不误导）。
    for name in _DETAIL_TABLES:
        assert getattr(dialog, name).rowCount() == 0
    x_nav, _ = dialog.nav_strategy_curve.getData()
    assert x_nav is None or len(x_nav) == 0
    dialog.close()


# --------------------------------------------------------------------------- #
# 5. 销毁不残留线程
# --------------------------------------------------------------------------- #
def test_dialog_close_leaves_no_running_worker(qapp, seeded_db) -> None:
    dialog = PortfolioBacktestDialog(db=seeded_db)
    _prepare_for_run(dialog)
    dialog.run_portfolio_backtest()

    worker = dialog._worker
    assert worker is None or not worker.isRunning()

    dialog.close()
    dialog.deleteLater()
    assert dialog._worker is None