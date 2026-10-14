"""P7-S2 策略分析对话框测试（离屏 ``offscreen``）。

覆盖：
    1. 构造 + 默认参数（一行条件 / 周频 / Top3 / 表格为空）；
    2. ``current_conditions()`` 收集与非法行跳过（空表达式 / 非数值阈值）；
    3. ``build_spec()``：名称 / 调仓 / 上限 / 排序表达式 / 股票池透传；
       空条件 → ``None`` 且状态提示；
    4. ``run_backtest()``（注入 fake 引擎）：曲线有点、5 张表填满、
       状态与警告标签正确、引擎收到正确的起止日期；
    5. ``run_backtest()``：引擎返回 ``ok=False`` → 视图清空 + 失败提示；
    6. ``run_backtest()``：引擎抛异常 → 不崩溃 + 失败提示；
    7. 格式化纯函数（``format_pct`` / ``format_num`` / ``format_count`` / ``format_date``）。
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtCore import QDate  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.engine.backtest import (  # noqa: E402
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    BacktestResult,
    PerformanceMetrics,
    drawdown_series,
)
from app.ui.widgets.strategy_analysis import (  # noqa: E402
    StrategyAnalysisDialog,
    format_count,
    format_date,
    format_num,
    format_pct,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
class FakeEngine:
    """注入用 fake 回测引擎（返回固定结果或抛指定异常，并记录调用）。"""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple] = []

    def run(self, spec, start=None, end=None):
        self.calls.append((spec, start, end))
        if self._error is not None:
            raise self._error
        return self._result


def _make_result(ok: bool = True) -> BacktestResult:
    """构造一份合成回测结果（300 个交易日、跨 2021~2022 两年）。"""
    dates = pd.date_range("2021-06-01", periods=300, freq="B")
    # 净值含一段 8% 回撤（i=150~219 期间打折），用于验证回撤区间表非空。
    nav_values = [1.0009**i * (0.92 if 150 <= i < 220 else 1.0) for i in range(300)]
    nav = pd.Series(nav_values, index=dates)
    benchmark = pd.Series([1.0004**i for i in range(300)], index=dates)
    equity = pd.DataFrame(
        {
            "date": dates,
            "nav": nav.to_numpy(),
            "benchmark_nav": benchmark.to_numpy(),
            "drawdown": drawdown_series(nav).to_numpy(),
        }
    )
    assert list(equity.columns) == EQUITY_COLUMNS
    trades = pd.DataFrame(
        [
            ["2021-07-01", "000001", "buy", 10.0, 100, 1000.0, 5.0, None],
            ["2021-08-02", "000001", "sell", 11.0, 100, 1100.0, 5.6, 94.4],
        ],
        columns=TRADE_COLUMNS,
    )
    regime = pd.DataFrame(
        {
            "regime": ["牛市", "熊市"],
            "days": [180, 120],
            "share": [0.6, 0.4],
            "strategy_return": [0.20, 0.05],
            "benchmark_return": [0.10, 0.02],
            "excess_return": [0.10, 0.03],
        }
    )
    metrics = PerformanceMetrics(
        total_return=0.30,
        annual_return=0.25,
        annual_volatility=0.20,
        sharpe=1.20,
        max_drawdown=-0.15,
        calmar=1.60,
        alpha=0.10,
        beta=0.90,
        tracking_error=0.05,
        information_ratio=0.80,
        benchmark_total_return=0.12,
        excess_return=0.18,
        win_rate=0.55,
        profit_loss_ratio=1.80,
        trade_count=7,
    )
    if not ok:
        return BacktestResult(ok=False, error="本地库内没有可用于回测的日线数据")
    return BacktestResult(
        ok=True,
        equity=equity,
        trades=trades,
        metrics=metrics,
        regime=regime,
        universe=["000001", "600000", "600519", "601398", "601869"],
        benchmark_symbol="sh.000300",
        benchmark_kind="INDEX",
        warnings=["600000：区间内部分日期停牌，估值沿用前收"],
    )


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_dialog(engine=None, symbols=None) -> StrategyAnalysisDialog:
    return StrategyAnalysisDialog(
        db=object(),  # 回测引擎已注入，db 仅占位（对话框不写库）
        engine=engine if engine is not None else FakeEngine(_make_result()),
        symbols=symbols,
    )


# --------------------------------------------------------------------------- #
# 1. 构造 + 默认参数
# --------------------------------------------------------------------------- #
def test_construct_defaults(qapp) -> None:
    dialog = _make_dialog()
    assert len(dialog.condition_rows) == 1
    assert dialog.current_rebalance() == "weekly"
    assert dialog.current_top_n() == 3
    assert dialog.last_result() is None
    for table in (
        dialog.metrics_table,
        dialog.yearly_table,
        dialog.regime_table,
        dialog.drawdown_table,
        dialog.decomposition_table,
    ):
        assert table.rowCount() == 0


# --------------------------------------------------------------------------- #
# 2. current_conditions 收集 / 跳过非法行
# --------------------------------------------------------------------------- #
def test_current_conditions_collects_and_skips_invalid(qapp) -> None:
    dialog = _make_dialog()
    # 默认行：预置「站上 20 日均线」+ op ">" + 阈值 "0" → 合法。
    conds = dialog.current_conditions()
    assert len(conds) == 1
    assert conds[0].field == "close - close.rolling(20).mean()"
    assert conds[0].op == ">"
    assert conds[0].value == 0.0

    # 第二行：自由输入表达式 + 非数值阈值 → 跳过。
    dialog.add_condition_row()
    row1 = dialog.condition_rows[1]
    row1.field_combo.setCurrentText("close")
    row1.op_combo.setCurrentText(">=")
    row1.value_edit.setText("abc")
    # 第三行：合法自由输入。
    dialog.add_condition_row()
    row2 = dialog.condition_rows[2]
    row2.field_combo.setCurrentText("volume")
    row2.op_combo.setCurrentText("<")
    row2.value_edit.setText("1000")

    conds = dialog.current_conditions()
    assert len(conds) == 2
    assert [c.field for c in conds] == ["close - close.rolling(20).mean()", "volume"]
    assert conds[1].op == "<"
    assert conds[1].value == 1000.0


# --------------------------------------------------------------------------- #
# 3. build_spec
# --------------------------------------------------------------------------- #
def test_build_spec_transfers_params(qapp) -> None:
    dialog = _make_dialog(symbols=["000001", "600519"])
    dialog.strategy_name_edit.setText("站上20日线 · 周频 Top3")
    dialog.top_n_spin.setValue(3)
    dialog.rank_by_edit.setText("close")

    spec = dialog.build_spec()

    assert spec is not None
    assert spec.name == "站上20日线 · 周频 Top3"
    assert spec.rebalance == "weekly"
    assert spec.top_n == 3
    assert spec.rank_by == "close"
    assert spec.symbols == ["000001", "600519"]
    assert len(spec.conditions) == 1


def test_build_spec_top_n_unlimited_and_no_symbols(qapp) -> None:
    dialog = _make_dialog(symbols=None)
    dialog.top_n_spin.setValue(0)  # 界面「不限制」→ None

    spec = dialog.build_spec()

    assert spec is not None
    assert spec.top_n is None
    assert spec.symbols is None


def test_build_spec_no_valid_conditions(qapp) -> None:
    dialog = _make_dialog()
    dialog.condition_rows[0].value_edit.setText("")  # 阈值空 → 无有效条件

    assert dialog.build_spec() is None
    assert "没有有效条件" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# 4. run_backtest 成功路径
# --------------------------------------------------------------------------- #
def test_run_backtest_fills_views(qapp) -> None:
    engine = FakeEngine(_make_result())
    dialog = _make_dialog(engine=engine, symbols=["000001", "600519"])

    result = dialog.run_backtest()

    assert result is not None
    assert result.ok is True
    assert dialog.last_result() is result
    # 引擎收到正确的股票池与起止日期。
    spec, start, end = engine.calls[0]
    assert spec.symbols == ["000001", "600519"]
    assert start == "2015-01-01"
    assert end == QDate.currentDate().toString("yyyy-MM-dd")

    # 表格填满。
    assert dialog.metrics_table.rowCount() == 15
    assert dialog.yearly_table.rowCount() >= 1
    assert dialog.regime_table.rowCount() == 2
    assert dialog.regime_table.item(0, 0).text() == "牛市"
    assert dialog.drawdown_table.rowCount() >= 1
    assert dialog.decomposition_table.rowCount() == 8
    # 指标卡首行：区间总收益 30.00%。
    assert dialog.metrics_table.item(0, 0).text() == "区间总收益"
    assert dialog.metrics_table.item(0, 1).text() == "30.00%"

    # 曲线有数据（300 个点）。
    x_strategy, y_strategy = dialog.nav_strategy_curve.getData()
    x_benchmark, _ = dialog.nav_benchmark_curve.getData()
    x_dd, _ = dialog.dd_curve.getData()
    assert len(x_strategy) == 300 and len(y_strategy) == 300
    assert len(x_benchmark) == 300
    assert len(x_dd) == 300
    assert dialog.nav_axis.label_count() == 300
    # 图上末值标签已刷新。
    assert "策略净值" in dialog.nav_strategy_label.text()
    assert "1." in dialog.nav_strategy_label.text()

    # 状态与警告。
    status = dialog.status_label.text()
    assert "回测完成" in status
    assert "股票池 5 只" in status
    assert "成交 2 笔" in status
    assert not dialog.warnings_label.isHidden()  # 对话框未 show，用 isHidden 判显式隐藏
    assert "停牌" in dialog.warnings_label.text()


# --------------------------------------------------------------------------- #
# 5. run_backtest 失败路径（ok=False）
# --------------------------------------------------------------------------- #
def test_run_backtest_failed_result(qapp) -> None:
    dialog = _make_dialog(engine=FakeEngine(_make_result(ok=False)))

    result = dialog.run_backtest()

    assert result is not None
    assert result.ok is False
    assert dialog.metrics_table.rowCount() == 0
    assert dialog.yearly_table.rowCount() == 0
    x_strategy, _ = dialog.nav_strategy_curve.getData()
    assert x_strategy is None or len(x_strategy) == 0
    assert "回测未完成" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# 6. run_backtest 失败路径（引擎抛异常）
# --------------------------------------------------------------------------- #
def test_run_backtest_engine_exception(qapp) -> None:
    dialog = _make_dialog(engine=FakeEngine(error=RuntimeError("库文件损坏")))

    result = dialog.run_backtest()

    assert result is None
    assert dialog.last_result() is None
    assert dialog.metrics_table.rowCount() == 0
    assert "回测失败" in dialog.status_label.text()
    assert "库文件损坏" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# 7. 格式化纯函数
# --------------------------------------------------------------------------- #
def test_format_helpers() -> None:
    assert format_pct(None) == "—"
    assert format_pct(float("nan")) == "—"
    assert format_pct(0.1234) == "12.34%"
    assert format_pct(-0.15) == "-15.00%"
    assert format_num(None) == "—"
    assert format_num(1.5) == "1.5000"
    assert format_num(1.5, 2) == "1.50"
    assert format_count(None) == "—"
    assert format_count(3.0) == "3"
    assert format_date(None) == "—"
    assert format_date("2021-01-04") == "2021-01-04"
    assert format_date(pd.Timestamp("2021-01-04")) == "2021-01-04"