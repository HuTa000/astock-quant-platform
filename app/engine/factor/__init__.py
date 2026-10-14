"""多因子有效性分析引擎（P7-S3）。

对外出口：
    * :data:`FACTOR_REGISTRY` / :class:`FactorDef` / :func:`factor_names` /
      :func:`compute_factor` / :func:`compute_factors` / :data:`DEFAULT_FACTOR_NAMES`：
      12 个价量因子计算库（动量 / 趋势 / 波动 / 量能 / 位置 / 超买超卖，
      :mod:`app.engine.factor.factors`）；
    * :class:`FactorPanel` / :func:`build_panel`：横截面面板组装
      （因子值 + T+1 开盘前瞻收益，
      :mod:`app.engine.factor.panel`）；
    * :func:`rank_ic` / :func:`summarize_ic` / :class:`ICSummary` /
      :func:`newey_west_mean_se` / :func:`t_pvalue_two_sided`：Rank IC 与
      统计检验（Newey-West 修正 t 值、精确双侧 p 值，
      :mod:`app.engine.factor.ic`）；
    * :func:`run_layered` / :class:`LayeredResult` / :func:`spearman_rank_corr`：
      分层回测与单调性（:mod:`app.engine.factor.layered`）；
    * :func:`compute_collinearity` / :class:`CollinearityResult` /
      :func:`standardize_cross_section`：相关矩阵与 VIF 共线性诊断
      （:mod:`app.engine.factor.collinearity`）；
    * :class:`FactorAnalyzer` / :class:`FactorAnalysisSpec` / :class:`FactorReport` /
      :func:`ic_decay_table`：编排入口与结果表（:mod:`app.engine.factor.analysis`）；
    * :func:`cross_section_zscore` / :func:`dynamic_icir_weights` /
      :func:`combine_scores` / :func:`top_picks` / :class:`FactorScorer` /
      :class:`ScoringSpec` / :class:`ScoreReport`：多因子打分合成与 Top-N 选股
      （ICIR 动态加权，P7-S5，:mod:`app.engine.factor.scoring`）。

典型用法::

    from app.engine.factor import FactorAnalyzer, FactorAnalysisSpec

    spec = FactorAnalysisSpec(horizons=(5, 20), start="2020-01-01")
    report = FactorAnalyzer().run(spec)
    print(report.ic_table.head())              # IC 汇总（因子 × 预测期）
    print(report.layered["ret_20"].group_stats)  # 分层统计
    print(report.collinearity.vif)             # VIF

⚠️ 引擎层**严禁 import 任何 Qt 代码**（《开发手册》§3 分层红线）。
"""
from __future__ import annotations

from app.engine.factor.analysis import (
    DEFAULT_HORIZONS,
    IC_TABLE_COLUMNS,
    FactorAnalysisSpec,
    FactorAnalyzer,
    FactorReport,
    ic_decay_table,
)
from app.engine.factor.collinearity import (
    HIGH_CORR_THRESHOLD,
    VIF_SEVERE_THRESHOLD,
    VIF_WARNING_THRESHOLD,
    CollinearityResult,
    compute_collinearity,
    standardize_cross_section,
)
from app.engine.factor.factors import (
    DEFAULT_FACTOR_NAMES,
    FACTOR_REGISTRY,
    FactorDef,
    compute_factor,
    compute_factors,
    factor_names,
    get_factor,
)
from app.engine.factor.ic import (
    IC_SUMMARY_COLUMNS,
    ICSummary,
    newey_west_mean_se,
    rank_ic,
    summarize_ic,
    t_pvalue_two_sided,
)
from app.engine.factor.layered import (
    GROUP_STATS_COLUMNS,
    LayeredResult,
    run_layered,
    spearman_rank_corr,
)
from app.engine.factor.panel import FactorPanel, build_panel
from app.engine.factor.scoring import (
    DEFAULT_IC_HORIZON,
    DEFAULT_MIN_IC_DAYS,
    DEFAULT_SCORING_FACTORS,
    DEFAULT_Z_CLIP,
    WEIGHT_TABLE_COLUMNS,
    FactorScorer,
    ScoreReport,
    ScoringSpec,
    combine_scores,
    cross_section_zscore,
    dynamic_icir_weights,
    top_picks,
)

__all__ = [
    # 因子计算
    "FACTOR_REGISTRY",
    "DEFAULT_FACTOR_NAMES",
    "FactorDef",
    "factor_names",
    "get_factor",
    "compute_factor",
    "compute_factors",
    # 面板
    "FactorPanel",
    "build_panel",
    # IC 与统计检验
    "IC_SUMMARY_COLUMNS",
    "ICSummary",
    "rank_ic",
    "summarize_ic",
    "newey_west_mean_se",
    "t_pvalue_two_sided",
    # 分层
    "GROUP_STATS_COLUMNS",
    "LayeredResult",
    "run_layered",
    "spearman_rank_corr",
    # 共线性
    "HIGH_CORR_THRESHOLD",
    "VIF_WARNING_THRESHOLD",
    "VIF_SEVERE_THRESHOLD",
    "CollinearityResult",
    "compute_collinearity",
    "standardize_cross_section",
    # 编排
    "DEFAULT_HORIZONS",
    "IC_TABLE_COLUMNS",
    "FactorAnalysisSpec",
    "FactorReport",
    "FactorAnalyzer",
    "ic_decay_table",
    # 打分合成（P7-S5）
    "DEFAULT_SCORING_FACTORS",
    "DEFAULT_IC_HORIZON",
    "DEFAULT_MIN_IC_DAYS",
    "DEFAULT_Z_CLIP",
    "WEIGHT_TABLE_COLUMNS",
    "ScoringSpec",
    "ScoreReport",
    "FactorScorer",
    "cross_section_zscore",
    "dynamic_icir_weights",
    "combine_scores",
    "top_picks",
]