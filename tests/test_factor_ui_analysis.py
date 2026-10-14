"""P7-S3 因子分析对话框测试（离屏 ``offscreen``）。

覆盖：
    1. 构造 + 默认参数（12 因子全选 / 预测期 5,10,20,60 / 5 层 / 10 标的 / 表格为空）；
    2. ``selected_factors()`` 勾选变更、``current_horizons()`` 与 ``parse_horizons`` 解析；
    3. ``current_spec()``：股票池 / 因子 / 预测期 / 层数 / 闸门 / 起止日期透传，
       无勾选因子或预测期非法 → ``None`` 且状态提示；
    4. ``run_analysis()``（注入 fake 分析器）：IC / 衰减 / 分层 / 共线性表填满、
       IC 曲线与分层曲线有点、状态与警告标签正确、分析器收到正确 spec；
    5. ``run_analysis()``：``ok=False`` → 视图清空 + 未完成提示；
    6. ``run_analysis()``：分析器抛异常 → 不崩溃 + 失败提示；
    7. 格式化纯函数（``format_ic`` / ``format_pvalue`` / ``format_flag``）。
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

from app.engine.factor import (  # noqa: E402
    IC_TABLE_COLUMNS,
    FactorReport,
    get_factor,
)
from app.engine.factor.collinearity import CollinearityResult  # noqa: E402
from app.engine.factor.layered import GROUP_STATS_COLUMNS, LayeredResult  # noqa: E402
from app.ui.widgets.factor_analysis import (  # noqa: E402
    FactorAnalysisDialog,
    format_flag,
    format_ic,
    format_pvalue,
    parse_horizons,
)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
class FakeAnalyzer:
    """注入用 fake 分析器（返回固定结果或抛指定异常，并记录收到的 spec）。"""

    def __init__(self, report=None, error: Exception | None = None) -> None:
        self._report = report
        self._error = error
        self.calls: list = []

    def run(self, spec):
        self.calls.append(spec)
        if self._error is not None:
            raise self._error
        return self._report


# 合成报告用：3 个因子 × 2 个预测期。
_FACTOR_META: tuple[tuple[str, str, str], ...] = (
    ("ret_5", "近5日涨幅", "动量"),
    ("ret_20", "近20日涨幅", "动量"),
    ("ma_dev_20", "20日均线乖离", "趋势"),
)


def _ic_rows() -> list[dict]:
    """合成 IC 长表行：短预测期显著为正、长预测期不显著（IC 衰减）。"""
    rows: list[dict] = []
    for name, label, category in _FACTOR_META:
        rows.append(
            {
                "factor": name,
                "label": label,
                "category": category,
                "horizon": 5,
                "n_days": 100,
                "ic_mean": 0.0500,
                "ic_std": 0.1200,
                "ic_ir": 0.4167,
                "ic_win_rate": 0.5600,
                "t_stat": 4.1667,
                "t_stat_nw": 3.2000,
                "p_value": 0.0014,
                "nw_lags": 4,
                "significant": True,
            }
        )
        rows.append(
            {
                "factor": name,
                "label": label,
                "category": category,
                "horizon": 20,
                "n_days": 100,
                "ic_mean": 0.0200,
                "ic_std": 0.1000,
                "ic_ir": 0.2000,
                "ic_win_rate": 0.5100,
                "t_stat": 2.0000,
                "t_stat_nw": 1.2500,
                "p_value": 0.2100,
                "nw_lags": 19,
                "significant": False,
            }
        )
    return rows


def _make_report(ok: bool = True) -> FactorReport:
    """构造一份合成因子分析结果（3 因子 × 2 预测期 + 5 层分层 + 3×3 共线性）。"""
    if not ok:
        return FactorReport(ok=False, error="评估窗口内没有任何交易日")

    ic_table = pd.DataFrame(_ic_rows(), columns=IC_TABLE_COLUMNS)

    dates = pd.date_range("2021-06-01", periods=30, freq="B")
    ic_series = {
        (name, horizon): pd.Series(
            np.linspace(-0.05, 0.12, 30), index=dates, name=f"{name}@{horizon}"
        )
        for name, _, _ in _FACTOR_META
        for horizon in (5, 20)
    }

    group_names = [f"Q{index}" for index in range(1, 6)]
    group_nav = pd.DataFrame(
        {
            group: (1.0 + 0.001 * order) ** np.arange(1, 31)
            for order, group in enumerate(group_names)
        },
        index=dates,
    )
    group_stats = pd.DataFrame(
        [
            {
                "group": group,
                "days": 30,
                "mean_daily_return": 0.001 * order,
                "total_return": float(group_nav[group].iloc[-1]) - 1.0,
                "annual_return": 0.10 + 0.02 * order,
                "annual_vol": 0.20,
                "sharpe": 0.50 + 0.10 * order,
                "max_drawdown": -0.15,
            }
            for order, group in enumerate(group_names)
        ],
        columns=GROUP_STATS_COLUMNS,
    )
    layered = {
        name: LayeredResult(
            ok=True,
            n_groups=5,
            group_names=list(group_names),
            group_nav=group_nav,
            group_stats=group_stats,
            long_short_nav=pd.Series(
                (1.0 + 0.002) ** np.arange(1, 31).astype(float), index=dates
            ),
            long_short_annual=0.1234,
            long_short_sharpe=1.5000,
            monotonicity=0.9000,
            days=30,
            warnings=["3 个交易日因样本不足或无区分度未建仓（净值持平）"],
        )
        for name, _, _ in _FACTOR_META
    }

    symbols = ["ret_5", "ret_20", "ma_dev_20"]
    corr = pd.DataFrame(
        [
            [1.0, 0.65, 0.30],
            [0.65, 1.0, 0.40],
            [0.30, 0.40, 1.0],
        ],
        index=symbols,
        columns=symbols,
    )
    vif = pd.Series([1.2000, 6.5000, 12.0000], index=symbols)
    collinearity = CollinearityResult(
        ok=True, corr=corr, vif=vif, high_corr_pairs=[], n_obs=100, warnings=[]
    )

    return FactorReport(
        ok=True,
        ic_table=ic_table,
        ic_series=ic_series,
        layered=layered,
        collinearity=collinearity,
        factors=[get_factor(name) for name, _, _ in _FACTOR_META],
        universe=["000001", "600000", "600519", "601398"],
        evaluation_days=123,
        n_groups=5,
        warnings=["600000：本地库无前复权日线，已剔除"],
    )


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_dialog(analyzer=None, symbols=None) -> FactorAnalysisDialog:
    return FactorAnalysisDialog(
        db=object(),  # 分析器已注入，db 仅占位（对话框不写库）
        analyzer=analyzer if analyzer is not None else FakeAnalyzer(_make_report()),
        symbols=symbols,
    )


# --------------------------------------------------------------------------- #
# 1. 构造 + 默认参数
# --------------------------------------------------------------------------- #
def test_construct_defaults(qapp) -> None:
    dialog = _make_dialog()

    # 12 个注册因子全部列出且默认勾选。
    from app.engine.factor import DEFAULT_FACTOR_NAMES

    assert dialog.factor_list.count() == len(DEFAULT_FACTOR_NAMES) == 12
    assert dialog.selected_factors() == list(DEFAULT_FACTOR_NAMES)
    assert dialog.current_horizons() == [5, 10, 20, 60]
    assert dialog.groups_spin.value() == 5
    assert dialog.min_stocks_spin.value() == 10
    assert dialog.last_report() is None
    # 表格 / 曲线均空。
    for table in (
        dialog.ic_table,
        dialog.decay_table,
        dialog.layer_stats_table,
        dialog.corr_table,
        dialog.vif_table,
    ):
        assert table.rowCount() == 0
    assert dialog.ic_factor_combo.count() == 0
    assert dialog.layer_factor_combo.count() == 0
    x_ic, _ = dialog.ic_curve.getData()
    assert x_ic is None or len(x_ic) == 0


# --------------------------------------------------------------------------- #
# 2. 勾选 / 预测期解析
# --------------------------------------------------------------------------- #
def test_selected_factors_follows_check_state(qapp) -> None:
    dialog = _make_dialog()
    # 取消前两个因子的勾选。
    for index in (0, 1):
        dialog.factor_list.item(index).setCheckState(Qt.CheckState.Unchecked)

    from app.engine.factor import DEFAULT_FACTOR_NAMES

    assert dialog.selected_factors() == list(DEFAULT_FACTOR_NAMES[2:])


def test_parse_horizons_variants(qapp) -> None:
    assert parse_horizons("5, 10,20") == [5, 10, 20]
    assert parse_horizons("20，5；10 60") == [5, 10, 20, 60]
    assert parse_horizons("7, 7, 5") == [5, 7]  # 去重升序
    assert parse_horizons("abc, 0, -3") == []  # 非法项忽略
    assert parse_horizons("") == []


# --------------------------------------------------------------------------- #
# 3. current_spec
# --------------------------------------------------------------------------- #
def test_current_spec_transfers_params(qapp) -> None:
    dialog = _make_dialog(symbols=["000001", "600519"])
    dialog.horizons_edit.setText("20, 5")
    dialog.groups_spin.setValue(3)
    dialog.min_stocks_spin.setValue(8)
    dialog.start_date_edit.setDate(QDate(2020, 1, 1))
    dialog.end_date_edit.setDate(QDate(2024, 12, 31))

    spec = dialog.current_spec()

    assert spec is not None
    assert spec.symbols == ["000001", "600519"]
    assert spec.horizons == [5, 20]
    assert len(spec.factor_names) == 12
    assert spec.n_groups == 3
    assert spec.min_stocks == 8
    assert spec.start == "2020-01-01"
    assert spec.end == "2024-12-31"


def test_current_spec_no_factors(qapp) -> None:
    dialog = _make_dialog()
    for index in range(dialog.factor_list.count()):
        dialog.factor_list.item(index).setCheckState(Qt.CheckState.Unchecked)

    assert dialog.current_spec() is None
    assert "请至少勾选一个因子" in dialog.status_label.text()


def test_current_spec_invalid_horizons(qapp) -> None:
    dialog = _make_dialog()
    dialog.horizons_edit.setText("abc")

    assert dialog.current_spec() is None
    assert "预测期非法" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# 4. run_analysis 成功路径
# --------------------------------------------------------------------------- #
def test_run_analysis_fills_views(qapp) -> None:
    analyzer = FakeAnalyzer(_make_report())
    dialog = _make_dialog(analyzer=analyzer, symbols=["000001", "600519", "600000"])

    report = dialog.run_analysis()

    assert report is not None
    assert report.ok is True
    assert dialog.last_report() is report
    # 分析器收到正确的股票池与起止日期。
    spec = analyzer.calls[0]
    assert spec.symbols == ["000001", "600519", "600000"]
    assert spec.start == "2015-01-01"
    assert spec.end == QDate.currentDate().toString("yyyy-MM-dd")

    # (a) IC 汇总表：3 因子 × 2 预测期 = 6 行 × 12 列，首行逐格对拍。
    assert dialog.ic_table.rowCount() == 6
    assert dialog.ic_table.columnCount() == 12
    assert [dialog.ic_table.horizontalHeaderItem(i).text() for i in range(4)] == [
        "因子",
        "中文名",
        "类别",
        "预测期",
    ]
    assert dialog.ic_table.item(0, 0).text() == "ret_5"
    assert dialog.ic_table.item(0, 1).text() == "近5日涨幅"
    assert dialog.ic_table.item(0, 3).text() == "5 日"
    assert dialog.ic_table.item(0, 4).text() == "100"
    assert dialog.ic_table.item(0, 5).text() == "0.0500"
    assert dialog.ic_table.item(0, 8).text() == "56.00%"
    assert dialog.ic_table.item(0, 9).text() == "3.2000"
    assert dialog.ic_table.item(0, 10).text() == "0.0014"
    assert dialog.ic_table.item(0, 11).text() == "是"
    # 第 2 行（同因子、预测期 20）不显著。
    assert dialog.ic_table.item(1, 3).text() == "20 日"
    assert dialog.ic_table.item(1, 11).text() == "否"

    # (b) IC 衰减表：行 = 3 因子（展示名）、列 = 因子 + 2 个预测期。
    assert dialog.decay_table.rowCount() == 3
    assert dialog.decay_table.columnCount() == 3
    assert dialog.decay_table.horizontalHeaderItem(1).text() == "IC@5日"
    assert dialog.decay_table.horizontalHeaderItem(2).text() == "IC@20日"
    decay_names = {dialog.decay_table.item(row, 0).text() for row in range(3)}
    assert "近5日涨幅（ret_5）" in decay_names

    # (c) 分层统计表：5 层 × 8 列；摘要含多空年化 / 单调性。
    assert dialog.layer_stats_table.rowCount() == 5
    assert dialog.layer_stats_table.columnCount() == 8
    assert dialog.layer_stats_table.item(0, 0).text() == "Q1"
    assert dialog.layer_stats_table.item(4, 0).text() == "Q5"
    summary = dialog.layer_summary_label.text()
    assert "12.34%" in summary
    assert "1.5000" in summary
    assert "0.9000" in summary

    # (d) 共线性：相关矩阵 3×3（+首列）与 VIF 判定分级。
    assert dialog.corr_table.rowCount() == 3
    assert dialog.corr_table.columnCount() == 4
    assert dialog.corr_table.item(0, 1).text() == "1.0000"
    assert dialog.corr_table.item(0, 2).text() == "0.6500"
    assert dialog.vif_table.rowCount() == 3
    assert dialog.vif_table.item(0, 2).text() == "1.2000"
    assert dialog.vif_table.item(0, 3).text() == "正常"
    assert dialog.vif_table.item(1, 3).text() == "警戒（谨慎同用）"
    assert dialog.vif_table.item(2, 3).text() == "严重共线（建议剔除或合成）"
    # VIF 表第二列是中文名（取自注册表）。
    assert dialog.vif_table.item(2, 1).text() == "20日均线乖离"

    # (e) 选择器：因子 × 预测期。
    assert dialog.ic_factor_combo.count() == 3
    assert dialog.ic_horizon_combo.count() == 2
    assert dialog.layer_factor_combo.count() == 3
    assert dialog.ic_factor_combo.currentData() == "ret_5"
    assert dialog.ic_horizon_combo.currentData() == 5

    # (f) 曲线有点：IC 时序 30 点、5 条分层曲线 + 1 条多空曲线。
    x_ic, y_ic = dialog.ic_curve.getData()
    assert len(x_ic) == 30 and len(y_ic) == 30
    assert dialog.ic_axis.label_count() == 30
    assert len(dialog.layer_curves) == 5
    assert set(dialog.layer_curves) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    x_ls, y_ls = dialog.long_short_curve.getData()
    assert len(x_ls) == 30 and len(y_ls) == 30
    assert dialog.layer_axis.label_count() == 30
    assert "IC 时序" in dialog.ic_axis_label.text()
    assert "分层净值" in dialog.layer_axis_label.text()

    # (g) 状态与警告。
    status = dialog.status_label.text()
    assert "分析完成" in status
    assert "因子 3 个" in status
    assert "标的 4 只" in status
    assert "评估交易日 123 天" in status
    assert not dialog.warnings_label.isHidden()
    assert "已剔除" in dialog.warnings_label.text()


def test_switch_factor_refreshes_views(qapp) -> None:
    """切换图表的因子 / 预测期选择器后，曲线与分层表随之刷新。"""
    dialog = _make_dialog()
    dialog.run_analysis()

    # 切到第三个因子（ma_dev_20）。
    dialog.layer_factor_combo.setCurrentIndex(2)
    assert dialog.layer_stats_table.rowCount() == 5
    assert "20日均线乖离" in dialog.layer_summary_label.text()

    # IC 图切到 20 日预测期：均值标签随之更新。
    dialog.ic_horizon_combo.setCurrentIndex(1)
    assert "20 日" in dialog.ic_axis_label.text()
    x_ic, _ = dialog.ic_curve.getData()
    assert len(x_ic) == 30


# --------------------------------------------------------------------------- #
# 5. run_analysis 失败路径（ok=False）
# --------------------------------------------------------------------------- #
def test_run_analysis_failed_result(qapp) -> None:
    dialog = _make_dialog(analyzer=FakeAnalyzer(_make_report(ok=False)))

    report = dialog.run_analysis()

    assert report is not None
    assert report.ok is False
    assert dialog.last_report() is report
    assert dialog.ic_table.rowCount() == 0
    assert dialog.layer_stats_table.rowCount() == 0
    assert dialog.corr_table.rowCount() == 0
    x_ic, _ = dialog.ic_curve.getData()
    assert x_ic is None or len(x_ic) == 0
    assert "分析未完成" in dialog.status_label.text()
    assert "评估窗口内没有任何交易日" in dialog.status_label.text()


def test_run_analysis_clears_previous_result(qapp) -> None:
    """先成功后失败：视图必须清空，不能残留上一次结果。"""
    dialog = _make_dialog()
    dialog.run_analysis()
    assert dialog.ic_table.rowCount() == 6

    dialog._analyzer = FakeAnalyzer(_make_report(ok=False))
    dialog.run_analysis()

    assert dialog.ic_table.rowCount() == 0
    assert dialog.layer_stats_table.rowCount() == 0
    assert len(dialog.layer_curves) == 0
    assert dialog.ic_factor_combo.count() == 0
    assert dialog.warnings_label.isHidden()


# --------------------------------------------------------------------------- #
# 6. run_analysis 失败路径（分析器抛异常）
# --------------------------------------------------------------------------- #
def test_run_analysis_analyzer_exception(qapp) -> None:
    dialog = _make_dialog(analyzer=FakeAnalyzer(error=RuntimeError("库文件损坏")))

    result = dialog.run_analysis()

    assert result is None
    assert dialog.last_report() is None
    assert dialog.ic_table.rowCount() == 0
    assert "分析失败" in dialog.status_label.text()
    assert "库文件损坏" in dialog.status_label.text()


# --------------------------------------------------------------------------- #
# 7. 格式化纯函数
# --------------------------------------------------------------------------- #
def test_format_helpers() -> None:
    assert format_ic(None) == "—"
    assert format_ic(float("nan")) == "—"
    assert format_ic(0.05) == "0.0500"
    assert format_ic(-0.12345) == "-0.1235"
    assert format_pvalue(None) == "—"
    assert format_pvalue(0.05) == "0.0500"
    assert format_pvalue(0.00001) == "<0.0001"
    assert format_flag(True) == "是"
    assert format_flag(False) == "否"