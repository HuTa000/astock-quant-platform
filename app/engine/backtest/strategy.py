"""策略规格定义（P7-S1）。

定位：
    把用户「脑子里的选股思路」固化成可回测的规格对象：**选什么**（条件）、
    **多久调一次仓**（rebalance）、**最多持几只**（top_n/rank_by）、
    **成本与基准口径**。回测引擎只认 :class:`StrategySpec`，与 UI 解耦。

两种选股入口（二选一）：
    * **条件表达式**（默认）：``conditions``（+ ``rank_by`` 排序取前 N）；
    * **预计算名单**：``daily_picks = {信号日: [标的, ...]}``——由
      :class:`~app.engine.factor.scoring.FactorScorer` 提前算好的「信号日 →
      当日 Top-N 名单」，引擎在信号日的**次一交易日开盘**照单成交（名单本身
      已满足「只用 ≤ 信号日数据」的因果约束，见 scoring 模块的三道闸门）。

条件表达（复用选股引擎的 :class:`~app.engine.screener.Condition`）：
    * 条件形态为「**指标表达式 vs 常数**」——``field`` 是表达式、``value`` 是阈值；
    * 由于阈值只能是常数，跨指标比较要写成**差值表达式**。例如：
      「收盘价高于 20 日均线」→ ``Condition("close - close.rolling(20).mean()", ">", 0)``；
      「放量（量 > 2 倍 5 日均量）」→ ``Condition("volume / volume.rolling(5).mean()", ">", 2)``；
    * 多个条件按 ``logic``（``AND``/``OR``）叠加。

⚠️ 为什么**拒绝**基本面条件（``pe/pb/roe/...``）：
    基本面数据源只提供**最新一期**快照（无「财报公布日」时间戳），若历史回测直接
    用它，就等于「2020 年时就知道 2023 年的业绩」——典型**前视偏差（未来函数）**。
    正确做法需要 PIT（Point-In-Time，按公告日可得的）财务数据库；本项目暂无该数据，
    故在规格层**直接拒绝**并抛出 :class:`ValueError`（阶段 4 数据质量审计会再讨论
    「财报延迟」的量化处理）。

典型用法::

    from app.engine.backtest import CostModel, StrategySpec
    from app.engine.screener import Condition

    spec = StrategySpec(
        name="站上20日线的强势股",
        conditions=[Condition("close - close.rolling(20).mean()", ">", 0)],
        rebalance="weekly",
        top_n=3,
        rank_by="close.rolling(20).mean() / close",   # 偏离均线更小者优先
        costs=CostModel(),
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from app.engine.backtest.costs import CostModel
from app.engine.backtest.risk import RiskSpec
from app.engine.screener import FUNDAMENTAL_FIELDS, Condition

logger = logging.getLogger(__name__)

# 调仓频率取值。
REBALANCE_DAILY: str = "daily"
REBALANCE_WEEKLY: str = "weekly"
REBALANCE_MONTHLY: str = "monthly"
REBALANCE_CHOICES: tuple[str, ...] = (REBALANCE_DAILY, REBALANCE_WEEKLY, REBALANCE_MONTHLY)

# 默认基准：沪深300（库内无该指数数据时，引擎自动降级为「股票池等权」代理并记警告）。
DEFAULT_BENCHMARK: str = "sh.000300"

# 默认初始资金（元）。
DEFAULT_INITIAL_CASH: float = 1_000_000.0


@dataclass
class StrategySpec:
    """策略规格（回测引擎的唯一输入契约）。

    Attributes:
        name: 策略名称（可为空，空则记「未命名策略」）。
        conditions: 选股条件列表（:class:`Condition` 或等价 dict）；**至少一个**
            （走预计算名单 ``daily_picks`` 时除外，两者互斥）。
        logic: 条件叠加方式，``"AND"``（默认）/``"OR"``。
        symbols: 股票池；``None`` 表示「用本地库内全部有日线的标的」（默认）；
            提供 ``daily_picks`` 时，``None`` 自动取**名单并集**（受限池）。
        rebalance: 调仓频率，``daily``（默认）/``weekly``/``monthly``；
            周频/月频以「每周/每月最后一个交易日收盘」出信号、次一交易日开盘成交。
        top_n: 最多持有只数；``None`` 表示选中即持有（不限制数量）。
        rank_by: 排序表达式（**值大者优先**），仅当 ``top_n`` 生效时用于取前 N；
            排序值缺失的标的会被剔除（无法排名即不参与）。
        daily_picks: **预计算每日名单** ``{信号日: [标的, ...]}``（多因子打分选股的
            入口）：提供时 ``conditions`` 必须为空、``top_n``/``rank_by`` 不适用
            （名单已由打分排序确定）；日期键自动归一化为 ``Timestamp``，
            名单去重保序。引擎只认名单里的标的，**不在名单上的日子按空仓处理**。
        costs: 交易成本模型（默认现实费率，见 :class:`CostModel`）。
        benchmark: 基准代码（默认 :data:`DEFAULT_BENCHMARK` 沪深300）。
        initial_cash: 初始资金（元，默认 100 万）。
        risk: 风控规格（止损 / 止盈 / 移动止盈 / 冷却期 / 牛熊环境仓位开关，
            :class:`~app.engine.backtest.risk.RiskSpec`）；``None``（默认）表示
            不启用任何风控——引擎行为与此前完全一致。

    Raises:
        ValueError: 条件为空且无名单 / 含基本面字段 / 名单与条件或 top_n 冲突 /
            名单为空或日期非法 / 逻辑或调仓频率非法 / 阈值结构非法 /
            ``risk`` 类型错误。
    """

    name: str
    conditions: list[Any] = field(default_factory=list)
    logic: str = "AND"
    symbols: Optional[list[str]] = None
    rebalance: str = REBALANCE_DAILY
    top_n: Optional[int] = None
    rank_by: Optional[str] = None
    daily_picks: Optional[dict[Any, list[str]]] = None
    costs: CostModel = field(default_factory=CostModel)
    benchmark: str = DEFAULT_BENCHMARK
    initial_cash: float = DEFAULT_INITIAL_CASH
    risk: Optional[RiskSpec] = None

    def __post_init__(self) -> None:
        """构造后严格校验（属调用方输入错误 → 抛 :class:`ValueError`）。"""
        self.name = str(self.name).strip() or "未命名策略"

        # 预计算名单归一化：日期键 → Timestamp（当日零点），名单去重保序、空日丢弃。
        picks: Optional[dict[pd.Timestamp, list[str]]] = None
        if self.daily_picks is not None:
            if not isinstance(self.daily_picks, dict):
                raise ValueError(
                    "daily_picks 必须是 {信号日: [标的, ...]} 字典，"
                    f"实际：{type(self.daily_picks).__name__}"
                )
            picks = {}
            for raw_day, raw_symbols in self.daily_picks.items():
                try:
                    day = pd.Timestamp(raw_day)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"daily_picks 的信号日无法解析：{raw_day!r}") from exc
                if pd.isna(day):
                    raise ValueError(f"daily_picks 的信号日无法解析：{raw_day!r}")
                picked: list[str] = []
                seen: set[str] = set()
                for symbol in raw_symbols or []:
                    key = str(symbol).strip()
                    if key and key not in seen:
                        seen.add(key)
                        picked.append(key)
                if picked:
                    picks[day.normalize()] = picked
            if not picks:
                raise ValueError("daily_picks 为空（没有任何信号日带候选标的）")
        self.daily_picks = picks

        # 条件归一化：dict → Condition，并拒绝基本面条件（前视偏差，见模块 docstring）。
        normalized: list[Condition] = []
        for raw in self.conditions or []:
            if isinstance(raw, Condition):
                normalized.append(raw)
            elif isinstance(raw, dict):
                normalized.append(Condition.from_dict(raw))
            else:
                raise ValueError(f"无法识别的条件类型：{type(raw).__name__}")
        if picks is not None and normalized:
            raise ValueError("daily_picks 与 conditions 互斥：预计算名单策略无需再写选股条件")
        if picks is None and not normalized:
            raise ValueError("策略至少需要一个选股条件（conditions 不能为空）")
        fundamental = [c.field for c in normalized if c.field in FUNDAMENTAL_FIELDS]
        if fundamental:
            raise ValueError(
                "回测不支持基本面条件（当前数据无公告日时间戳，直接用会造成前视偏差）："
                f"{fundamental}；请改用技术面表达式条件"
            )
        self.conditions = normalized

        # 逻辑叠加方式。
        logic_text = str(self.logic).strip().upper() if self.logic is not None else "AND"
        if logic_text not in ("AND", "OR"):
            raise ValueError(f"logic 必须是 'AND' 或 'OR'，实际：{self.logic!r}")
        self.logic = logic_text

        # 调仓频率。
        rebalance_text = str(self.rebalance).strip().lower()
        if rebalance_text not in REBALANCE_CHOICES:
            raise ValueError(
                f"rebalance 必须是 {list(REBALANCE_CHOICES)} 之一，实际：{self.rebalance!r}"
            )
        self.rebalance = rebalance_text

        # 持有上限与排序表达式（预计算名单已自带排序与数量，不允许再叠加）。
        if self.top_n is not None:
            try:
                top_n_value = int(self.top_n)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"top_n 必须是正整数：{self.top_n!r}") from exc
            if top_n_value <= 0:
                raise ValueError(f"top_n 必须为正整数：{self.top_n!r}")
            self.top_n = top_n_value
        if self.rank_by is not None:
            rank_text = str(self.rank_by).strip()
            self.rank_by = rank_text or None
        if picks is not None and (self.top_n is not None or self.rank_by):
            raise ValueError(
                "daily_picks 与 top_n/rank_by 互斥：名单的数量与排序已由打分确定"
            )

        # 股票池：去重保序；空列表等价于「未指定」（用库内全部）；
        # 预计算名单且未显式给池 → 用名单并集作受限池（名单外的标的不必读库）。
        if self.symbols is not None:
            deduped: list[str] = []
            seen: set[str] = set()
            for symbol in self.symbols:
                key = str(symbol).strip()
                if key and key not in seen:
                    seen.add(key)
                    deduped.append(key)
            self.symbols = deduped or None
        if picks is not None and self.symbols is None:
            self.symbols = sorted({s for day_picks in picks.values() for s in day_picks}) or None

        # 基准与初始资金。
        self.benchmark = str(self.benchmark).strip()
        try:
            cash = float(self.initial_cash)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"initial_cash 必须是数值：{self.initial_cash!r}") from exc
        if cash <= 0:
            raise ValueError(f"initial_cash 必须为正数：{self.initial_cash!r}")
        self.initial_cash = cash

        # 风控规格：None = 不启用（默认）；类型错误属调用方输入错误。
        if self.risk is not None and not isinstance(self.risk, RiskSpec):
            raise ValueError(
                f"risk 必须是 RiskSpec 或 None，实际：{type(self.risk).__name__}"
            )

    @property
    def condition_fields(self) -> list[str]:
        """条件表达式清单（便于日志 / UI 展示）。"""
        return [c.field for c in self.conditions]


__all__ = [
    "REBALANCE_DAILY",
    "REBALANCE_WEEKLY",
    "REBALANCE_MONTHLY",
    "REBALANCE_CHOICES",
    "DEFAULT_BENCHMARK",
    "DEFAULT_INITIAL_CASH",
    "StrategySpec",
]