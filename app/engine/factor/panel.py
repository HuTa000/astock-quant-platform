"""横截面面板组装（P7-S3）。

定位：
    把本地库中「多只标的的日线」拼装成因子分析需要的**两张对齐面板**：
    * **因子面板**（dates × symbols）：每只标的的因子值按日期对齐；
    * **前瞻收益面板**（dates × symbols）：从 T+1 开盘买入、持有 N 个交易日后
      开盘卖出的收益（与 P7-S1 回测「T 日收盘出信号 → T+1 开盘成交」同口径）。

设计要点（与金融数据四大坑的对应处理）：
    * **前视偏差**：因子值只用 ≤ 当日数据（见 :mod:`factors`）；前瞻收益从
      **T+1 开盘**起算——信号日在 T、买入在 T+1、卖出在 T+1+N，任一端点无
      bar（停牌）即置 NaN 不参与，绝不「用当日收盘价假装当日成交」；
    * **幸存者偏差**：股票池 = 调用方给定的标的（默认本地库现有 qfq 标的，
      含停止更新的）；已退市股票在库中缺失——结论只代表「当前仍可见的样本」，
      量化属阶段 4「数据质量审计」；
    * **复权口径**：统一前复权（qfq）；**无 qfq 数据的标的直接剔除并记警告**
      （混用未复权会因除权缺口产生假动量）；
    * **数据缺失 / 停牌**：某日无 bar → 该股当日因子值与前瞻收益均为 NaN，
      横截面自动跳过该股（缺失不填 0、不前向填充）。

面板采用**惰性计算 + 缓存**：因子与前瞻收益在被首次请求时才计算并缓存，
避免一次性持有 12 个因子 × 4 个预测期的大矩阵。

典型用法::

    from app.data.store.db import get_db
    from app.engine.factor.panel import build_panel

    panel, warnings = build_panel(get_db())
    factor = panel.factor("ret_20")     # 因子面板（DataFrame）
    forward = panel.forward(5)          # 5 日前瞻收益面板（DataFrame）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from app.data.datasource.base import is_index_symbol
from app.data.store.db import Database, get_db
from app.engine.factor.factors import compute_factor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 面板容器
# --------------------------------------------------------------------------- #
@dataclass
class FactorPanel:
    """多标的对齐面板（惰性计算 + 缓存的因子值 / 前瞻收益）。

    Attributes:
        dates: 全局交易日历（所有标的日期并集，升序）。
        symbols: 参与面板的标的（列顺序）。
        open: 开盘价宽表（dates × symbols，停牌 / 未上市为 NaN）——前瞻收益与
            分层组合的日收益都从它派生。
        warnings: 组装过程中的警告（如个别标的无 qfq 数据被剔除）。
    """

    dates: pd.DatetimeIndex
    symbols: list[str]
    open: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    # PIT 成分掩码（dates × symbols 布尔矩阵，P7-S4）：``None`` = 不启用
    # （默认「当前可见标的」口径，行为与 P7-S3 完全一致）；非 None 时，
    # 非成分日 / 非成分股的一切取值（因子与前瞻收益）都会被置 NaN，
    # 从而实现「某交易日只用当时生效的成分名单」的严格 PIT 回溯。
    mask: Optional[pd.DataFrame] = None
    # 内部：各标的原始日线与其日期索引（因子计算用；不参与序列化展示）。
    _bars: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _stamps: dict[str, pd.DatetimeIndex] = field(default_factory=dict, repr=False)
    _factor_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _forward_cache: dict[int, pd.DataFrame] = field(default_factory=dict, repr=False)

    @property
    def empty(self) -> bool:
        """是否无任何可用标的 / 交易日。"""
        return not self.symbols or len(self.dates) == 0

    def factor(self, name: str) -> pd.DataFrame:
        """取因子面板（首次调用计算并缓存）。

        Args:
            name: 因子名（:data:`~app.engine.factor.factors.FACTOR_REGISTRY` 键）。

        Returns:
            dates × symbols 的因子值矩阵；未知因子名返回全 NaN 空壳（不抛异常）。
            启用 PIT 掩码时（见 :attr:`mask`），非成分日 / 非成分股置 NaN。
        """
        if name not in self._factor_cache:
            self._factor_cache[name] = self._build_factor(name)
        return self._apply_mask(self._factor_cache[name])

    def forward(self, horizon: int) -> pd.DataFrame:
        """取前瞻收益面板（首次调用计算并缓存）。

        口径：信号日 T 收盘算因子 → **T+1 开盘买入 → T+1+horizon 开盘卖出**，
        收益 = ``open[T+1+horizon] / open[T+1] − 1``；任一端点无 bar（停牌 /
        未上市 / 数据中断）→ NaN（不可成交，不参与横截面）。

        Args:
            horizon: 持有交易日数（≥1）；非法值返回全 NaN 面板（不抛异常）。

        Returns:
            dates × symbols 的前瞻收益矩阵。
        """
        try:
            key = int(horizon)
        except (TypeError, ValueError):
            logger.warning("非法预测期：%r，返回空面板", horizon)
            return pd.DataFrame(index=self.dates, columns=self.symbols, dtype=float)
        if key < 1:
            logger.warning("预测期必须 ≥1，实际 %s，返回空面板", key)
            return pd.DataFrame(index=self.dates, columns=self.symbols, dtype=float)
        if key not in self._forward_cache:
            entry = self.open.shift(-1)
            exit_ = self.open.shift(-(1 + key))
            forward = exit_ / entry - 1.0
            self._forward_cache[key] = forward.replace([np.inf, -np.inf], np.nan)
        return self._apply_mask(self._forward_cache[key])

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _apply_mask(self, frame: pd.DataFrame) -> pd.DataFrame:
        """按 PIT 成分掩码把非成分日 / 非成分股置 NaN（未启用时原样返回）。

        ``mask`` 按日期 / 列对齐：日期索引统一解析为日期粒度（掩码由
        :func:`~app.data.universe.membership_matrix` 生成，索引为日期文本），
        掩码缺失的格子视为 False（非成分）。掩码结构异常时降级为不启用
        （记警告，不抛异常）。
        """
        if self.mask is None:
            return frame
        try:
            aligned = pd.DataFrame(
                self.mask.to_numpy(),
                index=pd.DatetimeIndex(pd.to_datetime(list(self.mask.index))),
                columns=list(self.mask.columns),
            ).reindex(index=frame.index, columns=frame.columns)
        except Exception:  # noqa: BLE001 - 掩码结构异常时不启用掩码
            logger.exception("PIT 掩码与面板不匹配，本次忽略掩码")
            return frame
        return frame.where(aligned.fillna(False).astype(bool).to_numpy())

    def _build_factor(self, name: str) -> pd.DataFrame:
        """逐标的计算因子并按其日期对齐到全局日历（缺失留 NaN）。"""
        columns: dict[str, pd.Series] = {}
        for symbol in self.symbols:
            bars = self._bars.get(symbol)
            stamps = self._stamps.get(symbol)
            if bars is None or stamps is None or bars.empty:
                continue
            values = compute_factor(name, bars)
            if values.empty or len(values) != len(stamps):
                continue
            series = pd.Series(values.to_numpy(dtype=float), index=stamps)
            series = series[~series.index.isna()]
            columns[symbol] = series.reindex(self.dates)
        if not columns:
            return pd.DataFrame(index=self.dates, columns=self.symbols, dtype=float)
        return pd.DataFrame(columns, index=self.dates)


# --------------------------------------------------------------------------- #
# 组装入口
# --------------------------------------------------------------------------- #
def build_panel(
    db: Optional[Database] = None,
    symbols: Optional[Sequence[str]] = None,
    mask: Optional[pd.DataFrame] = None,
) -> tuple[Optional[FactorPanel], list[str]]:
    """从本地库组装因子分析面板（**不抛业务异常**）。

    Args:
        db: 存储实例；默认全局单例 :func:`~app.data.store.db.get_db`。
        symbols: 标的清单；``None`` = 本地库现有 qfq 日线标的（自动排除指数——
            指数不可买卖）。显式传入时不再自动排除指数（由调用方负责）。
        mask: PIT 成分掩码（dates × symbols 布尔矩阵，P7-S4）；``None`` = 不启用
            （默认「当前可见标的」口径）。启用后非成分日 / 非成分股的因子值与
            前瞻收益均置 NaN，实现严格 PIT 成分回溯。

    Returns:
        ``(panel, warnings)``：面板与警告清单；库中无任何可用数据时返回
        ``(None, [...])``。无 qfq 数据的标的会被剔除并计入警告。
    """
    warnings: list[str] = []
    store = db if db is not None else get_db()
    try:
        if symbols is None:
            universe = [
                symbol
                for symbol in store.list_kline_symbols("d", "qfq")
                if not is_index_symbol(symbol)
            ]
        else:
            universe = []
            seen: set[str] = set()
            for item in symbols:
                key = str(item)
                if key and key not in seen:
                    seen.add(key)
                    universe.append(key)
    except Exception:  # noqa: BLE001 - 库异常按空池处理，不向调用方抛
        logger.exception("读取标的清单失败")
        return None, ["读取本地标的清单失败，请检查数据库连接"]

    if not universe:
        return None, ["本地库中没有可用的前复权日线标的（请先更新数据）"]

    bars_map: dict[str, pd.DataFrame] = {}
    stamps_map: dict[str, pd.DatetimeIndex] = {}
    missing: list[str] = []
    for symbol in universe:
        try:
            bars = store.get_bars(symbol, "d", adjust="qfq")
        except Exception:  # noqa: BLE001 - 单标的读取失败只跳过
            logger.exception("读取 %s 日线失败（跳过）", symbol)
            missing.append(symbol)
            continue
        if bars is None or bars.empty:
            missing.append(symbol)
            continue
        stamps = pd.DatetimeIndex(pd.to_datetime(bars["datetime"], errors="coerce"))
        valid = ~stamps.isna()
        if not valid.any():
            missing.append(symbol)
            continue
        bars_map[symbol] = bars.loc[valid].reset_index(drop=True)
        stamps_map[symbol] = stamps[valid]

    if not bars_map:
        return None, ["本地库中没有可用的前复权日线标的（请先更新数据）"]

    if missing:
        warnings.append(
            f"{len(missing)} 只标的无 qfq 日线数据已剔除：{'、'.join(missing[:5])}"
            + ("…" if len(missing) > 5 else "")
        )

    usable = [symbol for symbol in universe if symbol in bars_map]
    all_dates = pd.DatetimeIndex(
        sorted({stamp for stamps in stamps_map.values() for stamp in stamps})
    )
    open_frame = pd.DataFrame(index=all_dates, columns=usable, dtype=float)
    for symbol in usable:
        bars = bars_map[symbol]
        series = pd.Series(
            pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float),
            index=stamps_map[symbol],
        )
        open_frame[symbol] = series.reindex(all_dates)

    panel = FactorPanel(
        dates=all_dates,
        symbols=usable,
        open=open_frame,
        warnings=list(warnings),
        mask=mask,
        _bars=bars_map,
        _stamps=stamps_map,
    )
    return panel, warnings