"""P7-S5 多因子打分选股对话框测试（离屏 ``offscreen``）。

覆盖：
    1. 构造 + 默认参数（核心三因子默认勾选 / Top-30 / 月频 / PIT / 前瞻 20 / 最少 IC 120）；
    2. ``current_spec()`` 默认值与透传；取消全部因子 → ``None`` 且状态提示；恢复勾选可运行；
    3. 同步 ``run_scored_backtest()``（真实临时 SQLite 库 + 合成动量行情，current 口径 /
       周频 / 小预热门槛）：结果 ``ok``、绩效表 / 权重表 / 名单表 / 成交明细有行、
       净值曲线有数据、状态含「打分 / 回测」；
    4. ``ok=False`` 结果（PIT 口径但库内无成分股快照）：不抛异常、状态含错误原因、视图清空；
    5. 对话框销毁后不残留运行中的后台线程；
    6. 风控参数区（P7-S6）：默认全部关闭 → ``risk=None``；参数控件启用态随勾选联动；
       勾选后 ``current_spec().risk`` 正确透传（含「仅环境开关」不泄漏止损默认值）；
       勾选风控经真实引擎回放仍 ``ok``。
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
from app.engine.backtest import TRADE_COLUMNS  # noqa: E402
from app.engine.factor import DEFAULT_SCORING_FACTORS, WEIGHT_TABLE_COLUMNS  # noqa: E402
from app.ui.widgets.score_backtest import ScoreBacktestDialog  # noqa: E402


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _open(path) -> Database:
    """打开文件库并建表。"""
    db = Database(path)
    db.connect()
    db.init_schema()
    return db


def _seed_momentum_universe(db: Database, count: int = 12, days_count: int = 480) -> list[str]:
    """播种 count 只相位错开的动量标的（qfq 日线），返回标的清单。

    与 ``tests/test_backtest_scored.py`` 的合成口径一致：60 日为一个周期，
    前 30 日 +0.5%/日、后 30 日 −0.5%/日，并叠加确定性噪声——保证存在可被
    IC 捕捉的横截面动量结构，且各标的相位错开。
    """
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=days_count)]
    symbols = [f"0000{index:02d}" for index in range(1, count + 1)]
    for index, symbol in enumerate(symbols):
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
    return symbols


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory) -> Database:
    """模块级共享的合成行情库（打分回测只读，可安全复用）。"""
    path = tmp_path_factory.mktemp("score_ui") / "seed.db"
    db = _open(path)
    _seed_momentum_universe(db)
    return db


def _make_dialog(db=None) -> ScoreBacktestDialog:
    return ScoreBacktestDialog(db=db if db is not None else object())


def _select_factors(dialog: ScoreBacktestDialog, names) -> None:
    """按名字批量设置因子勾选状态（其余取消）。"""
    wanted = set(names)
    for index in range(dialog.factor_list.count()):
        item = dialog.factor_list.item(index)
        name = str(item.data(Qt.ItemDataRole.UserRole))
        item.setCheckState(
            Qt.CheckState.Checked if name in wanted else Qt.CheckState.Unchecked
        )


def _prepare_for_run(dialog: ScoreBacktestDialog) -> None:
    """把对话框调到「快速可跑」口径：current 池 / 周频 / 小预热门槛 / 2 因子 / Top-3。"""
    dialog.universe_combo.setCurrentIndex(dialog.universe_combo.findData("current"))
    dialog.rebalance_combo.setCurrentIndex(dialog.rebalance_combo.findData("weekly"))
    dialog.ic_horizon_spin.setValue(5)
    dialog.min_ic_days_spin.setValue(30)
    dialog.top_n_spin.setValue(3)
    _select_factors(dialog, ["ret_5", "log_amt_20"])


_DETAIL_TABLES = (
    "metrics_table",
    "yearly_table",
    "regime_table",
    "weights_table",
    "picks_table",
    "trades_table",
)


# --------------------------------------------------------------------------- #
# 1. 构造 + 默认参数
# --------------------------------------------------------------------------- #
def test_construct_defaults(qapp) -> None:
    dialog = _make_dialog()

    assert dialog.top_n_spin.value() == 30
    assert dialog.current_rebalance() == "monthly"
    assert dialog.current_universe() == "pit"
    assert dialog.ic_horizon_spin.value() == 20
    assert dialog.min_ic_days_spin.value() == 120
    assert len(dialog.selected_factors()) == 3
    assert set(dialog.selected_factors()) == set(DEFAULT_SCORING_FACTORS)
    assert dialog.last_result() is None
    for name in _DETAIL_TABLES:
        assert getattr(dialog, name).rowCount() == 0
    x_nav, _ = dialog.nav_strategy_curve.getData()
    assert x_nav is None or len(x_nav) == 0


# --------------------------------------------------------------------------- #
# 2. current_spec 默认值 / 无因子 / 恢复勾选
# --------------------------------------------------------------------------- #
def test_current_spec_defaults(qapp) -> None:
    dialog = _make_dialog()

    spec = dialog.current_spec()

    assert spec is not None
    assert spec.top_n == 30
    assert spec.rebalance == "monthly"
    assert spec.universe == "pit"
    assert len(spec.factors) == 3
    assert set(spec.factors) == set(DEFAULT_SCORING_FACTORS)
    assert spec.ic_horizon == 20
    assert spec.min_ic_days == 120
    assert spec.start == "2015-01-01"
    assert spec.end == QDate.currentDate().toString("yyyy-MM-dd")


def test_current_spec_no_factors_then_restore(qapp) -> None:
    dialog = _make_dialog()

    _select_factors(dialog, [])  # 取消全部勾选
    assert dialog.current_spec() is None
    assert "请至少勾选一个因子" in dialog.status_label.text()

    _select_factors(dialog, DEFAULT_SCORING_FACTORS)  # 恢复核心三因子
    spec = dialog.current_spec()
    assert spec is not None
    assert set(spec.factors) == set(DEFAULT_SCORING_FACTORS)


# --------------------------------------------------------------------------- #
# 3. run_scored_backtest 成功路径（真实小规模回测）
# --------------------------------------------------------------------------- #
def test_run_scored_backtest_success(qapp, seeded_db) -> None:
    dialog = ScoreBacktestDialog(db=seeded_db)
    _prepare_for_run(dialog)

    result = dialog.run_scored_backtest()

    assert result is not None, dialog.status_label.text()
    assert result.ok is True, result.error
    assert dialog.last_result() is result
    # 引擎回显口径。
    assert result.scoring is not None
    assert result.scoring.scored_days > 0
    assert result.backtest is not None

    # 绩效表（15 行）+ 逐年表 + 权重表（2 因子）+ 名单表 + 成交明细列契约。
    assert dialog.metrics_table.rowCount() == 15
    assert dialog.yearly_table.rowCount() >= 1
    assert dialog.weights_table.rowCount() == 2
    assert dialog.weights_table.columnCount() == len(WEIGHT_TABLE_COLUMNS)
    assert dialog.picks_table.rowCount() >= 1
    assert dialog.trades_table.columnCount() == len(TRADE_COLUMNS)
    assert dialog.trades_table.rowCount() >= 1

    # 净值曲线有数据，轴标签数量与数据点一致。
    x_nav, y_nav = dialog.nav_strategy_curve.getData()
    assert len(x_nav) == len(y_nav) and len(x_nav) > 0
    assert dialog.nav_axis.label_count() == len(x_nav)
    assert "策略净值" in dialog.nav_strategy_label.text()
    assert "基准" in dialog.nav_benchmark_info.text()

    # 状态汇总含「打分 / 回测」与关键指标。
    status = dialog.status_label.text()
    assert "打分" in status and "回测" in status
    assert "研究池" in status and "最大回撤" in status
    # 合成库无沪深300指数 → 等权代理基准，警告如实展示。
    assert not dialog.warnings_label.isHidden()
    assert "代理" in dialog.warnings_label.text()

    # 同步路径不创建后台线程。
    assert dialog._worker is None
    dialog.close()


def test_run_scored_backtest_pit_without_snapshots(qapp, seeded_db) -> None:
    """PIT 口径但库内无成分股快照：``ok=False`` 结果不抛异常，状态含错误原因。"""
    dialog = ScoreBacktestDialog(db=seeded_db)
    _prepare_for_run(dialog)
    # 切回 PIT 口径（库内无成分股快照 → 打分显式失败）。
    dialog.universe_combo.setCurrentIndex(dialog.universe_combo.findData("pit"))

    result = dialog.run_scored_backtest()

    assert result is not None
    assert result.ok is False
    assert "成分股快照" in result.error
    assert "回测未完成" in dialog.status_label.text()
    assert "成分股快照" in dialog.status_label.text()
    # 失败时视图清空。
    for name in _DETAIL_TABLES:
        assert getattr(dialog, name).rowCount() == 0
    x_nav, _ = dialog.nav_strategy_curve.getData()
    assert x_nav is None or len(x_nav) == 0
    dialog.close()


# --------------------------------------------------------------------------- #
# 4. 销毁不残留线程
# --------------------------------------------------------------------------- #
def test_dialog_close_leaves_no_running_worker(qapp, seeded_db) -> None:
    dialog = ScoreBacktestDialog(db=seeded_db)
    _prepare_for_run(dialog)
    dialog.run_scored_backtest()

    worker = dialog._worker
    assert worker is None or not worker.isRunning()

    dialog.close()
    dialog.deleteLater()
    assert dialog._worker is None


# --------------------------------------------------------------------------- #
# 5. 风控参数区（P7-S6）
# --------------------------------------------------------------------------- #
_RISK_CHECKBOXES = (
    "stop_loss_check",
    "take_profit_check",
    "trailing_stop_check",
    "regime_filter_check",
)


def test_risk_defaults_all_closed(qapp) -> None:
    """默认全部不勾选：不启用风控，参数控件禁用（如实披露 A/B 结论）。"""
    dialog = _make_dialog()

    assert dialog.current_risk_spec() is None
    assert dialog.current_spec().risk is None
    for name in _RISK_CHECKBOXES:
        assert getattr(dialog, name).isChecked() is False

    # 起始建议值（止损 10% 取引擎默认；固定止盈 20% / 移动止盈 15%）。
    assert dialog.stop_loss_spin.value() == pytest.approx(10.0)
    assert dialog.take_profit_spin.value() == pytest.approx(20.0)
    assert dialog.trailing_stop_spin.value() == pytest.approx(15.0)
    assert dialog.cooldown_spin.value() == 20
    assert dialog.regime_window_spin.value() == 200
    assert dialog.regime_exposure_spin.value() == 50

    # 未勾选 → 对应参数控件禁用。
    for name in (
        "stop_loss_spin",
        "take_profit_spin",
        "trailing_stop_spin",
        "cooldown_spin",
        "regime_window_spin",
        "regime_exposure_spin",
    ):
        assert getattr(dialog, name).isEnabled() is False


def test_risk_controls_follow_checkboxes(qapp) -> None:
    """参数控件启用态随勾选联动：退出规则任一勾选才放开冷却期。"""
    dialog = _make_dialog()

    dialog.stop_loss_check.setChecked(True)
    assert dialog.stop_loss_spin.isEnabled() is True
    assert dialog.cooldown_spin.isEnabled() is True

    dialog.stop_loss_check.setChecked(False)
    assert dialog.stop_loss_spin.isEnabled() is False
    assert dialog.cooldown_spin.isEnabled() is False

    dialog.trailing_stop_check.setChecked(True)
    assert dialog.trailing_stop_spin.isEnabled() is True
    assert dialog.cooldown_spin.isEnabled() is True
    assert dialog.take_profit_spin.isEnabled() is False

    dialog.regime_filter_check.setChecked(True)
    assert dialog.regime_window_spin.isEnabled() is True
    assert dialog.regime_exposure_spin.isEnabled() is True

    dialog.regime_filter_check.setChecked(False)
    assert dialog.regime_window_spin.isEnabled() is False
    assert dialog.regime_exposure_spin.isEnabled() is False


def test_current_risk_spec_checked_params(qapp) -> None:
    """勾选后各参数正确转成比例 / 指数（含「仅环境开关」不泄漏止损默认值）。"""
    dialog = _make_dialog()

    dialog.stop_loss_check.setChecked(True)
    dialog.stop_loss_spin.setValue(8.0)
    dialog.cooldown_spin.setValue(5)
    spec = dialog.current_risk_spec()
    assert spec is not None
    assert spec.stop_loss_pct == pytest.approx(0.08)
    assert spec.take_profit_pct is None
    assert spec.trailing_stop_pct is None
    assert spec.cooldown_days == 5
    assert spec.regime_filter is False

    dialog.take_profit_check.setChecked(True)
    dialog.take_profit_spin.setValue(25.0)
    dialog.trailing_stop_check.setChecked(True)
    dialog.trailing_stop_spin.setValue(12.5)
    dialog.regime_filter_check.setChecked(True)
    dialog.regime_window_spin.setValue(250)
    dialog.regime_exposure_spin.setValue(40)
    spec = dialog.current_risk_spec()
    assert spec is not None
    assert spec.take_profit_pct == pytest.approx(0.25)
    assert spec.trailing_stop_pct == pytest.approx(0.125)
    assert spec.regime_filter is True
    assert spec.regime_window == 250
    assert spec.regime_exposure == pytest.approx(0.40)

    # 仅环境开关：退出规则显式 None，不泄漏 RiskSpec 的止损默认值。
    dialog.stop_loss_check.setChecked(False)
    dialog.take_profit_check.setChecked(False)
    dialog.trailing_stop_check.setChecked(False)
    spec = dialog.current_risk_spec()
    assert spec is not None
    assert spec.stop_loss_pct is None
    assert spec.take_profit_pct is None
    assert spec.trailing_stop_pct is None
    assert spec.regime_filter is True


def test_current_spec_passes_risk(qapp) -> None:
    """``current_spec().risk`` 与风控参数区一致（默认 None / 勾选透传）。"""
    dialog = _make_dialog()

    assert dialog.current_spec().risk is None

    dialog.stop_loss_check.setChecked(True)
    dialog.stop_loss_spin.setValue(15.0)
    spec = dialog.current_spec()
    assert spec is not None
    assert spec.risk is not None
    assert spec.risk.stop_loss_pct == pytest.approx(0.15)
    assert spec.risk.take_profit_pct is None


def test_run_scored_backtest_with_risk_enabled(qapp, seeded_db) -> None:
    """勾选风控经真实引擎回放仍 ``ok``（UI → ScoredBacktestSpec → 引擎风控钩子链路通）。"""
    dialog = ScoreBacktestDialog(db=seeded_db)
    _prepare_for_run(dialog)
    dialog.stop_loss_check.setChecked(True)
    dialog.stop_loss_spin.setValue(2.0)  # 紧止损：确保合成行情下确实触发退出

    result = dialog.run_scored_backtest()

    assert result is not None, dialog.status_label.text()
    assert result.ok is True, result.error
    assert dialog.trades_table.rowCount() >= 1
    dialog.close()