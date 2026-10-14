"""数据模型（dataclass），与 SQLite 表一一对应。

字段定义严格遵循《细化设计》§1.2 与《开发手册与交接文档》§4.4。
时间统一使用 :class:`datetime.datetime`；持久化时由存储层转换为 ISO 文本。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Bar:
    """K 线单根数据（全局 Bar 契约的面向对象形态）。

    Attributes:
        symbol: 标的代码，如 ``"000001"``。
        period: 周期，取值 ``{'1m','5m','15m','30m','60m','d','w','m'}``。
        dt: 该根 K 线的时间戳。
        open/high/low/close: OHLC 价格。
        volume: 成交量。
        amount: 成交额。
        adjust: 复权口径，取值 ``''``（不复权）/``'qfq'``/``'hfq'``。
    """

    symbol: str
    period: str
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    adjust: str = ""


@dataclass
class WatchlistGroup:
    """自选股分组。"""

    id: int
    name: str
    sort_order: int = 0


@dataclass
class WatchlistItem:
    """自选股条目（隶属某个分组）。"""

    id: int
    group_id: int
    symbol: str
    added_at: datetime = field(default_factory=datetime.now)
    note_mark: bool = False


@dataclass
class StockInfo:
    """全 A 股名录条目（P6 优化：股票查找与加载）。

    Attributes:
        symbol: 标的代码，6 位字符串，如 ``"000001"``。
        name: 股票名称。
        market: 交易市场，取值 ``"sh"/"sz"/"bj"/"unknown"``；缺省为空串。
        updated_at: 该条名录的最后更新时间。
    """

    symbol: str
    name: str
    market: str = ""
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class IndexConstituent:
    """指数成分股名单快照条目（P7-S4：历史成分股 PIT 口径）。

    用途：阶段 1~3 的研究池是「**今天**仍在指数里的股票」，等于提前知道谁活了下来
    （幸存者偏差）。本表把名单变成**当时可得**的口径，供 PIT 回溯与偏差量化。

    Attributes:
        index_code: 指数代码，如 ``"sh.000300"``。
        symbol: 成分股代码，裸 6 位（如 ``"600000"``，与 ``kline.symbol`` 同口径）。
        name: 快照时点的股票名称；缺省为空串。
        effective_date: 名单**生效日**（非查询日）；仅日期部分有意义。
    """

    index_code: str
    symbol: str
    name: str = ""
    effective_date: datetime = field(default_factory=datetime.now)


@dataclass
class Note:
    """个股备注。"""

    symbol: str
    text: str
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class IndicatorTemplate:
    """自定义指标模板。"""

    id: int
    name: str
    category: str
    expr: str
    created_at: datetime
    updated_at: datetime


@dataclass
class ScreenerTemplate:
    """选股条件模板（条件以 JSON 文本保存，schema 见《细化设计》§3）。"""

    id: int
    name: str
    conditions_json: str
    created_at: datetime


@dataclass
class AlertRule:
    """技术指标信号预警规则（P4-S1）。

    Attributes:
        id: 规则主键。
        symbol: 标的代码。
        kind: 规则类型，取值 ``ma_cross/macd_cross/volume_surge/expr``。
        params_json: 参数 JSON 文本（按 ``kind`` 区分，schema 见
            :mod:`app.engine.alert.alert_service`）。
        enabled: 是否启用。
        created_at: 创建时间。
        updated_at: 更新时间。
    """

    id: int
    symbol: str
    kind: str
    params_json: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class ConfigEntry:
    """配置项（统一以字符串存，读取时按类型解析）。"""

    key: str
    value: str


@dataclass
class Fundamental:
    """公司基本面摘要（供基本面弹窗使用；本步仅定义结构）。"""

    symbol: str
    name: str
    industry: str
    list_date: str
    total_market_cap: float
    float_market_cap: float
    pe: float
    pb: float
    roe: float
    revenue_yoy: float
    report_period: str
    summary: str
