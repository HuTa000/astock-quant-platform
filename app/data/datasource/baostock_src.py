"""BaoStock 兜底数据源实现。

本实现基于 **baostock 0.9.3**（dist 版本；``baostock.__version__`` 报 ``00.9.30``）。
经 venv 内自省确认以下接口存在：
    * ``login(user_id, password)`` / ``logout(user_id)``
    * ``query_history_k_data_plus(code, fields, start_date, end_date, frequency, adjustflag)``
    * ``query_all_stock(day)``
    * ``query_stock_basic(code, code_name)`` / ``query_stock_industry(code, date)``
    * ``query_dupont_data(code, year, quarter)``（含 ``dupontROE``）
    * ``query_growth_data(code, year, quarter)``（含 ``YOYNI`` 等同比指标）
    * ``query_adjust_factor(code, start_date, end_date)``（``foreAdjustFactor``/``backAdjustFactor``）

⚠️ 跨源一致性约定（重要）：
    1. **成交量单位统一为「手」**。AkShare 东财的 ``volume`` 单位本就是「手」；
       BaoStock ``volume`` 单位是「股」。故本实现把 BaoStock 的 volume **除以 100 转为「手」**，
       保证与 AkShare 同口径。``amount``（成交额）两源均为「元」，无需转换。
    2. **基本面 ``roe`` / ``revenue_yoy`` 统一为百分比数值**（如 ``4.69`` 表示 4.69%）。
       BaoStock ``dupontROE`` 与 ``YOYNI`` 为**小数比率**，本实现 **×100** 转为百分比数值，
       与 AkShare（``stock_financial_analysis_indicator`` 返回的百分比数值）口径一致
       （如 ``0.046746`` → ``4.6746``）。

能力边界：
    * 无 1 分钟 → ``get_minute`` 返回 Bar 契约空表并提示用 pytdx（P1-S5）；
    * 无实时 spot → ``get_spot`` 返回空表并说明；
    * 复权因子 → 使用 ``query_adjust_factor``（前复权因子 ``foreAdjustFactor``）。

错误处理：所有对外方法内部异常统一包装为 :class:`DataSourceError`（保留异常链）。
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any, Iterator, Optional

import pandas as pd

from app.data.datasource.base import (
    BAR_COLUMNS,
    DataSourceError,
    derive_market,
    is_index_symbol,
    normalize_bars,
    register_source,
)

logger = logging.getLogger(__name__)

# ---- 导入 baostock（失败时给出可读错误，不影响本模块被 import） ---- #
try:  # pragma: no cover - 环境相关
    import baostock as bs

    BAOSTOCK_VERSION: str = str(getattr(bs, "__version__", "unknown"))
    _BS_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # pragma: no cover - 环境相关
    bs = None  # type: ignore[assignment]
    BAOSTOCK_VERSION = "unavailable"
    _BS_IMPORT_ERROR = _exc

# 周期编码 -> BaoStock frequency
_FREQ_MAP: dict[str, str] = {
    "d": "d",
    "w": "w",
    "m": "m",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "60m": "60",
}

# 复权编码 -> BaoStock adjustflag（1 后复权 / 2 前复权 / 3 不复权）
_ADJUST_MAP: dict[str, str] = {"": "3", "qfq": "2", "hfq": "1"}

# 各频率请求字段（首次尝试）；失败时回退到最小字段集。
_DAILY_FIELDS: str = "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
_WEEK_MONTH_FIELDS: str = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
_MINUTE_FIELDS: str = "date,time,code,open,high,low,close,volume,amount,adjustflag"
_MINIMAL_FIELDS: str = "date,code,open,high,low,close,volume,amount,adjustflag"
# 指数专用字段集（P7-S1：回测基准 sh.000300）：指数不提供 adjustflag / turn /
# tradestatus / pctChg / isST，且最小字段集（含 adjustflag）对指数同样不被接受，
# 故指数**不做**最小字段集回退（见 get_kline）。
_INDEX_FIELDS: str = "date,code,open,high,low,close,volume,amount"

# 指数代码 -> BaoStock 成分股查询接口名（P7-S4：历史成分股 PIT 口径）。
# BaoStock 的这三个接口均支持 ``date`` 参数并返回「该日期当天或之前最近一份生效」的名单，
# 返回首列 ``date`` 即名单生效日 —— 这正是可回溯的历史快照来源。
_INDEX_CONSTITUENT_QUERIES: dict[str, str] = {
    "sh.000300": "query_hs300_stocks",  # 沪深300
    "sh.000016": "query_sz50_stocks",   # 上证50
    "sh.000905": "query_zz500_stocks",  # 中证500
}

# 成分股快照契约列（P7-S4）：失败 / 空名单时也返回**带列名**的空表，
# 与 ``get_minute`` / ``get_spot`` 的空表约定一致，保持调用方 schema 稳定。
_CONSTITUENT_COLUMNS: tuple[str, ...] = (
    "index_code", "symbol", "name", "effective_date",
)

# Fundamental 全部字段
_FUNDAMENTAL_KEYS: tuple[str, ...] = (
    "symbol",
    "name",
    "industry",
    "list_date",
    "total_market_cap",
    "float_market_cap",
    "pe",
    "pb",
    "roe",
    "revenue_yoy",
    "report_period",
    "summary",
)


def _to_float(value: Any) -> Optional[float]:
    """尽力把值转为 float；空/非法返回 ``None``。"""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_pct(value: Any) -> Optional[float]:
    """把「小数比率」转为「百分比数值」（×100）；无效返回 ``None``。

    Args:
        value: 形如 ``"0.046746"`` 的小数比率。

    Returns:
        ``4.6746``（百分比数值）；``None`` 表示无效值。
    """
    num = _to_float(value)
    return None if num is None else num * 100.0


def _parse_bs_time(text: Any) -> Any:
    """解析 BaoStock 分钟 ``time`` 字段（``YYYYMMDDHHMMSSsss``）为 datetime。"""
    raw = str(text).strip()
    if not raw:
        return pd.NaT
    if len(raw) >= 14 and raw[:14].isdigit():
        return datetime.strptime(raw[:14], "%Y%m%d%H%M%S")
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return pd.NaT


def _to_bs_date(value: Any) -> str:
    """把日期值格式化为 BaoStock 所需 ``YYYY-MM-DD``。"""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip().replace("/", "-")
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _to_bs_code(symbol: str) -> str:
    """把 ``000001`` 转为 BaoStock 代码 ``sz.000001``。"""
    market = derive_market(symbol)
    match = re.search(r"(\d{6})", str(symbol))
    core = match.group(1) if match else str(symbol)
    return f"{market}.{core}"


def _to_bare_code(value: Any) -> str:
    """把 ``sh.600000`` 归一化为裸 6 位码 ``600000``（P7-S4）。

    BaoStock 成分股接口返回带市场前缀的代码，而本地库 ``kline.symbol`` 用裸码；
    不归一化会导致「本地明明有数据却判定为缺失」。无法提取时返回空串。

    Args:
        value: 形如 ``"sh.600000"`` / ``"600000.SH"`` / ``"600000"`` 的值。

    Returns:
        裸 6 位码；无法识别时返回 ``""``。
    """
    if value is None:
        return ""
    match = re.search(r"(\d{6})", str(value))
    return match.group(1) if match else ""


def _recent_days(count: int = 7) -> Iterator[str]:
    """由今天向前生成 ``YYYY-MM-DD`` 日期串（用于 ``query_all_stock``）。"""
    today = date.today()
    for offset in range(count):
        yield (today - timedelta(days=offset)).strftime("%Y-%m-%d")


def _iter_quarters(count: int = 8) -> Iterator[tuple[int, int]]:
    """由当前季度向前生成 ``(year, quarter)`` 序列（用于财务接口回退查找）。"""
    today = date.today()
    year, quarter = today.year, (today.month - 1) // 3 + 1
    for _ in range(count):
        yield year, quarter
        quarter -= 1
        if quarter == 0:
            quarter, year = 4, year - 1


def _result_to_df(result: Any) -> pd.DataFrame:
    """把 BaoStock ``ResultData`` 转为 DataFrame（列名取 ``fields``）。"""
    raw_fields = list(getattr(result, "fields", []) or [])
    fields = [f for f in raw_fields if f != ""]
    rows: list[list[Any]] = []
    while result.next():
        row = list(result.get_row_data())
        if len(row) > len(fields):
            row = row[: len(fields)]
        rows.append(row)
    if not fields:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=fields)


class BaoStockSource:
    """基于 BaoStock 的兜底数据源实现（日/周/月线 + 财务）。"""

    name: str = "baostock"

    def __init__(self) -> None:
        """初始化（懒登录，首次调用接口时才 ``login()``）。"""
        self._bs = bs
        self._version = BAOSTOCK_VERSION
        self._logged_in: bool = False

    # ------------------------------------------------------------------ #
    # 连接管理
    # ------------------------------------------------------------------ #
    def _require_bs(self) -> Any:
        """确保 baostock 可用。

        Raises:
            DataSourceError: baostock 未成功导入。
        """
        if self._bs is None:
            raise DataSourceError(
                self.name,
                _BS_IMPORT_ERROR or RuntimeError("baostock 不可用"),
                "baostock 未能导入，请检查安装",
            )
        return self._bs

    def _ensure_login(self) -> None:
        """懒登录并复用连接。

        Raises:
            DataSourceError: 登录失败（``error_code != '0'``）。
        """
        if self._logged_in:
            return
        result = self._require_bs().login()
        if str(getattr(result, "error_code", "1")) != "0":
            raise DataSourceError(
                self.name,
                RuntimeError(str(getattr(result, "error_msg", "login failed"))),
                "baostock 登录失败",
            )
        self._logged_in = True
        logger.debug("baostock 登录成功（版本 %s）", self._version)

    def close(self) -> None:
        """登出（幂等）。"""
        if self._logged_in and self._bs is not None:
            try:
                self._bs.logout()
            except Exception as exc:  # noqa: BLE001
                logger.warning("baostock 登出异常：%s", exc)
            finally:
                self._logged_in = False

    def __enter__(self) -> "BaoStockSource":
        """支持 ``with`` 用法（自动登录）。"""
        self._ensure_login()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """退出 ``with`` 时登出。"""
        self.close()

    # ------------------------------------------------------------------ #
    # 股票列表
    # ------------------------------------------------------------------ #
    def list_stocks(self) -> pd.DataFrame:
        """全量 A 股列表，返回列 ``[code, name, market]``。"""
        try:
            akapi = self._require_bs()
            self._ensure_login()
            result = None
            for day in _recent_days():
                result = akapi.query_all_stock(day=day)
                if str(getattr(result, "error_code", "1")) == "0":
                    break
            if result is None or str(getattr(result, "error_code", "1")) != "0":
                raise DataSourceError(
                    self.name,
                    RuntimeError(str(getattr(result, "error_msg", ""))),
                    "query_all_stock 失败",
                )
            frame = _result_to_df(result)
            name_col = "code_name" if "code_name" in frame.columns else None
            codes = frame["code"].astype(str).str.replace(r"^(sh|sz|bj)\.", "", regex=True)
            out = pd.DataFrame(
                {
                    "code": codes.str.zfill(6),
                    "name": frame[name_col].astype(str) if name_col else "",
                }
            )
            out["market"] = out["code"].map(derive_market)
            return out[["code", "name", "market"]].reset_index(drop=True)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("baostock.list_stocks 失败")
            raise DataSourceError(self.name, exc) from exc

    # ------------------------------------------------------------------ #
    # 指数成分股（P7-S4）
    # ------------------------------------------------------------------ #
    def get_index_constituents(
        self, index_code: str = "sh.000300", date_: Any = None
    ) -> pd.DataFrame:
        """拉取指数成分股名单**快照**（P7-S4：历史成分股 PIT 口径）。

        与 :meth:`get_kline` 的差异：成分股名单是**离散快照**（随指数调仓生效），
        因此按日期逐个查询，返回的 ``effective_date`` 是名单生效日而非查询日。

        Args:
            index_code: 指数代码，支持 ``sh.000300``（沪深300）/``sh.000016``
                （上证50）/``sh.000905``（中证500）。
            date_: 查询日期（``str``/``date``/``datetime``）；BaoStock 返回该日期
                当天或之前**最近一份生效**的名单。``None`` 取最新一份。

        Returns:
            列 ``[index_code, symbol, name, effective_date]`` 的 DataFrame，其中
            ``symbol`` 为**裸 6 位码**（与本地库 ``kline.symbol`` 同口径）、
            ``effective_date`` 为 ``YYYY-MM-DD`` 文本；不支持的指数 / 查询失败 /
            空名单一律返回**空表**（不抛异常，便于上层「跳过该快照」）。
        """
        query_name = _INDEX_CONSTITUENT_QUERIES.get(index_code)
        if query_name is None:
            logger.warning("baostock 不支持该指数的成分股查询：%s", index_code)
            return pd.DataFrame(columns=_CONSTITUENT_COLUMNS)
        try:
            akapi = self._require_bs()
            self._ensure_login()
            query = getattr(akapi, query_name, None)
            if query is None:  # pragma: no cover - 依赖 baostock 版本
                logger.warning("baostock 缺少接口 %s", query_name)
                return pd.DataFrame(columns=_CONSTITUENT_COLUMNS)
            result = query() if date_ is None else query(date=_to_bs_date(date_))
            if str(getattr(result, "error_code", "1")) != "0":
                logger.warning(
                    "成分股查询 %s(%s) 失败：%s",
                    query_name, date_, getattr(result, "error_msg", ""),
                )
                return pd.DataFrame(columns=_CONSTITUENT_COLUMNS)
            frame = _result_to_df(result)
            if frame.empty:
                return pd.DataFrame(columns=_CONSTITUENT_COLUMNS)
            code_col = "code" if "code" in frame.columns else frame.columns[1]
            name_col = "code_name" if "code_name" in frame.columns else None
            date_col = frame.columns[0]
            out = pd.DataFrame(
                {
                    "index_code": index_code,
                    "symbol": frame[code_col].map(_to_bare_code),
                    "name": frame[name_col].astype(str) if name_col else "",
                    "effective_date": frame[date_col].astype(str),
                }
            )
            out = out[out["symbol"] != ""]
            return out[list(_CONSTITUENT_COLUMNS)].reset_index(drop=True)
        except Exception:  # noqa: BLE001 - 成分股失败按空表降级，不中断调用方
            logger.exception("baostock.get_index_constituents(%s) 失败", index_code)
            return pd.DataFrame(columns=_CONSTITUENT_COLUMNS)

    # ------------------------------------------------------------------ #
    # K 线
    # ------------------------------------------------------------------ #
    def get_kline(
        self,
        symbol: str,
        period: str,
        start: Any,
        end: Any,
        adjust: str = "",
    ) -> pd.DataFrame:
        """获取日/周/月/5m/15m/30m/60m K 线，返回 Bar 契约（volume 已转为「手」）。

        Args:
            symbol: 标的代码。
            period: ``'d'|'w'|'m'|'5m'|'15m'|'30m'|'60m'``。
            start: 起始日期。
            end: 结束日期。
            adjust: ``''|'qfq'|'hfq'``。

        Returns:
            Bar 契约 DataFrame。

        Raises:
            DataSourceError: 周期/复权不支持或接口调用失败。
        """
        try:
            akapi = self._require_bs()
            self._ensure_login()
            if period not in _FREQ_MAP:
                raise ValueError(f"不支持的周期：{period!r}")
            if adjust not in _ADJUST_MAP:
                raise ValueError(f"不支持的复权口径：{adjust!r}")

            frequency = _FREQ_MAP[period]
            adjustflag = _ADJUST_MAP[adjust]
            code = _to_bs_code(symbol)
            start_date = _to_bs_date(start)
            end_date = _to_bs_date(end)

            # 起止日期颠倒（如增量起始已越过结束日期）⇒ 无数据可拉，直接返回空表，
            # 避免 baostock 报「起始日期大于终止日期，请修改。」（编排层已先 this 场景跳过）。
            if start_date > end_date:
                logger.warning(
                    "baostock 查询 %s 起止日期无效（%s 晚于 %s），返回空表",
                    code, start_date, end_date,
                )
                return pd.DataFrame(columns=BAR_COLUMNS)

            # 指数（如 sh.000300）字段集与股票不同：无 adjustflag 等字段，
            # 且失败时不回退最小字段集（见 _INDEX_FIELDS 注释）。
            index_mode = is_index_symbol(symbol)
            if frequency in ("5", "15", "30", "60"):
                fields = _MINUTE_FIELDS
            elif index_mode:
                fields = _INDEX_FIELDS
            elif frequency in ("w", "m"):
                fields = _WEEK_MONTH_FIELDS
            else:
                fields = _DAILY_FIELDS

            result = akapi.query_history_k_data_plus(
                code, fields, start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag=adjustflag,
            )
            if str(getattr(result, "error_code", "1")) != "0":
                if index_mode:
                    raise DataSourceError(
                        self.name,
                        RuntimeError(str(getattr(result, "error_msg", ""))),
                        f"query_history_k_data_plus({code}) 失败（指数字段集）",
                    )
                # 字段集不被支持时回退到最小字段集
                logger.warning(
                    "baostock 查询 %s 失败（%s），回退最小字段集重试",
                    code, getattr(result, "error_msg", ""),
                )
                result = akapi.query_history_k_data_plus(
                    code, _MINIMAL_FIELDS, start_date=start_date, end_date=end_date,
                    frequency=frequency, adjustflag=adjustflag,
                )
                if str(getattr(result, "error_code", "1")) != "0":
                    raise DataSourceError(
                        self.name,
                        RuntimeError(str(getattr(result, "error_msg", ""))),
                        f"query_history_k_data_plus({code}) 失败",
                    )

            frame = _result_to_df(result)
            if frame.empty:
                return pd.DataFrame(columns=BAR_COLUMNS)

            # 构造 datetime：分钟用 time 字段，日/周/月用 date 字段。
            if frequency in ("5", "15", "30", "60") and "time" in frame.columns:
                datetimes = frame["time"].map(_parse_bs_time)
            else:
                datetimes = pd.to_datetime(frame["date"], errors="coerce")

            # 成交量：股 -> 手（除以 100），与 AkShare 东财口径统一。
            volume_lots = pd.to_numeric(frame["volume"], errors="coerce") / 100.0

            prepared = pd.DataFrame(
                {
                    "datetime": datetimes,
                    "open": pd.to_numeric(frame["open"], errors="coerce"),
                    "high": pd.to_numeric(frame["high"], errors="coerce"),
                    "low": pd.to_numeric(frame["low"], errors="coerce"),
                    "close": pd.to_numeric(frame["close"], errors="coerce"),
                    "volume": volume_lots,
                    "amount": pd.to_numeric(frame["amount"], errors="coerce"),
                }
            )
            return normalize_bars(prepared)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("baostock.get_kline(%s, %s) 失败", symbol, period)
            raise DataSourceError(self.name, exc) from exc

    # ------------------------------------------------------------------ #
    # 不支持的 1 分钟
    # ------------------------------------------------------------------ #
    def get_minute(self, symbol: str, date_: Any) -> pd.DataFrame:
        """BaoStock 无 1 分钟数据，返回 Bar 契约空表。"""
        logger.warning(
            "baostock 无 1 分钟数据：symbol=%s date=%s；请使用 pytdx 数据源（P1-S5）。",
            symbol,
            date_,
        )
        return pd.DataFrame(columns=BAR_COLUMNS)

    # ------------------------------------------------------------------ #
    # 不支持的实时 spot
    # ------------------------------------------------------------------ #
    def get_spot(self) -> pd.DataFrame:
        """BaoStock 无实时 spot 快照，返回空表。"""
        logger.warning("baostock 无实时 spot 数据，返回空表。")
        return pd.DataFrame(columns=["code", "name", "price", "pct_change"])

    # ------------------------------------------------------------------ #
    # 复权因子
    # ------------------------------------------------------------------ #
    def get_adj_factor(self, symbol: str, start: Any, end: Any) -> pd.DataFrame:
        """复权因子序列（前复权因子 ``foreAdjustFactor``），返回 ``[datetime, factor]``。

        Notes:
            BaoStock ``query_adjust_factor`` 返回 ``dividOperateDate`` 与
            ``foreAdjustFactor``/``backAdjustFactor``；本实现取 ``foreAdjustFactor``。
            若该接口在本版本不存在，则返回空表并说明。
        """
        try:
            akapi = self._require_bs()
            if not hasattr(akapi, "query_adjust_factor"):
                logger.warning("baostock 无 query_adjust_factor，返回空表。")
                return pd.DataFrame(columns=["datetime", "factor"])
            self._ensure_login()
            result = akapi.query_adjust_factor(
                _to_bs_code(symbol), start_date=_to_bs_date(start), end_date=_to_bs_date(end)
            )
            if str(getattr(result, "error_code", "1")) != "0":
                raise DataSourceError(
                    self.name,
                    RuntimeError(str(getattr(result, "error_msg", ""))),
                    f"query_adjust_factor({symbol}) 失败",
                )
            frame = _result_to_df(result)
            if frame.empty:
                return pd.DataFrame(columns=["datetime", "factor"])
            out = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(frame["dividOperateDate"], errors="coerce"),
                    "factor": pd.to_numeric(frame["foreAdjustFactor"], errors="coerce"),
                }
            ).dropna(subset=["datetime", "factor"])
            out = out.sort_values("datetime", kind="stable")
            return out.reset_index(drop=True)
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("baostock.get_adj_factor(%s) 失败", symbol)
            raise DataSourceError(self.name, exc) from exc

    # ------------------------------------------------------------------ #
    # 基本面
    # ------------------------------------------------------------------ #
    def get_fundamental(self, symbol: str) -> dict:
        """拼装个股基本面摘要（缺字段填 ``None``/``''``，不抛异常）。

        Returns:
            键与 :class:`app.data.models.Fundamental` 一致的 dict。

        Notes:
            BaoStock 未直接提供「营业收入同比」，此处 ``revenue_yoy`` 取
            ``query_growth_data`` 的 ``YOYNI``（净利润同比）作为可得近似值并已在此注明。
            ``roe``/``revenue_yoy`` 均 **×100 转为百分比数值**，与 AkShare 口径统一。
        """
        try:
            akapi = self._require_bs()
            self._ensure_login()
            result: dict = {key: None for key in _FUNDAMENTAL_KEYS}
            result["symbol"] = str(symbol)
            result["industry"] = ""
            result["list_date"] = ""
            result["report_period"] = ""
            result["summary"] = ""
            code = _to_bs_code(symbol)

            # 1) 基本信息：名称 / 上市日期
            try:
                basic = _result_to_df(akapi.query_stock_basic(code=code))
                if not basic.empty:
                    row = basic.iloc[0].to_dict()
                    result["name"] = row.get("code_name") or result["name"]
                    result["list_date"] = row.get("ipoDate") or result["list_date"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("query_stock_basic 失败：%s", exc)

            # 2) 行业
            try:
                industry = _result_to_df(akapi.query_stock_industry(code=code))
                if not industry.empty and "industry" in industry.columns:
                    result["industry"] = industry.iloc[0].get("industry") or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("query_stock_industry 失败：%s", exc)

            # 3) ROE（dupont，小数比率 → ×100 百分比）—— 由近及远回退查找最近可用季度
            try:
                row, period = self._latest_financial(akapi, "query_dupont_data", code)
                if row:
                    result["roe"] = _to_pct(row.get("dupontROE"))
                    result["report_period"] = period or result["report_period"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("query_dupont_data 失败：%s", exc)

            # 4) 同比（growth；BaoStock 无营收同比取 YOYNI 近似；小数比率 → ×100 百分比）
            try:
                row, period = self._latest_financial(akapi, "query_growth_data", code)
                if row:
                    yoy = row.get("YOYNI") or row.get("YOYPNI")
                    result["revenue_yoy"] = _to_pct(yoy)
                    result["report_period"] = period or result["report_period"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("query_growth_data 失败：%s", exc)

            result["summary"] = self._build_summary(result)
            return result
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("baostock.get_fundamental(%s) 失败", symbol)
            raise DataSourceError(self.name, exc) from exc

    @staticmethod
    def _latest_financial(
        akapi: Any, method_name: str, code: str
    ) -> tuple[dict, str]:
        """由近及远查找最近一个可用季度的财务行。

        Returns:
            ``(row_dict, statDate)``；全部季度均无数据时返回 ``({}, "")``。
        """
        method = getattr(akapi, method_name, None)
        if method is None:
            return {}, ""
        for year, quarter in _iter_quarters():
            result = method(code=code, year=year, quarter=quarter)
            if str(getattr(result, "error_code", "1")) != "0":
                continue
            frame = _result_to_df(result)
            if not frame.empty:
                row = frame.iloc[0].to_dict()
                return row, str(row.get("statDate", ""))
        return {}, ""

    @staticmethod
    def _build_summary(result: dict) -> str:
        """用已获取字段拼一段可读摘要。"""
        parts: list[str] = []
        if result.get("name"):
            parts.append(str(result["name"]))
        if result.get("industry"):
            parts.append(f"行业：{result['industry']}")
        if result.get("list_date"):
            parts.append(f"上市：{result['list_date']}")
        if result.get("roe") is not None:
            parts.append(f"ROE：{result['roe']}%")
        if result.get("revenue_yoy") is not None:
            parts.append(f"净利同比：{result['revenue_yoy']}%")
        return "；".join(parts)


@register_source("baostock")
def _create_baostock_source() -> BaoStockSource:
    """BaoStock 数据源工厂（供注册表使用）。"""
    return BaoStockSource()
