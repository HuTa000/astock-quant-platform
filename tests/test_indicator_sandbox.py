"""P3-S2 AST 白名单沙箱测试。

聚焦两类：
    * **逃逸封死**：``__globals__``/``__class__``/``__dict__``/``__subclasses__``/``__mro__``/
      ``__import__``、``Import`` 语句、``Lambda`` 等一律 ok=False，且 ``error_type`` 正确；
    * **合法表达式仍可用**：数据列运算、``ta`` 指标、内置指标、``pd``/``np``、比较 /
      条件表达式、上下文 ``symbol``/``period``/``n`` 等照常返回。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engine.indicator import IndicatorEngine, IndicatorResult, get_engine

N = 60


def make_bars(n: int = N) -> pd.DataFrame:
    """构造确定性 Bar 契约 DataFrame（close = 1..n）。"""
    close = np.arange(1.0, n + 1.0)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(100.0, 100.0 + n),
            "amount": close * np.arange(100.0, 100.0 + n),
        }
    )


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture(scope="module")
def engine() -> IndicatorEngine:
    return get_engine()


# --------------------------------------------------------------------------- #
# 逃逸封死（核心）：均 ok=False，且不抛异常
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("expr", "error_type"),
    [
        ("MA.__globals__['__builtins__']['eval']('1+1')", "Security"),
        ("close.__class__.__init__.__globals__['__builtins__']['eval']('1+1')", "Security"),
        ("ta.__dict__", "Security"),
        ("bars.__class__.__mro__[1].__subclasses__()", "Security"),
        ("__import__('os')", "Security"),
        ("import os", "Syntax"),
        ("open('x')", "Type"),
        ("(lambda: 1)()", "Security"),
        ("eval('1+1')", "Security"),
        ("getattr(close, '__class__')", "Security"),
    ],
)
def test_escape_blocked(engine, bars, expr, error_type):
    result = engine.calc(expr, bars)
    assert isinstance(result, IndicatorResult)
    assert result.ok is False
    assert result.error
    assert result.error_type == error_type


def test_escape_never_raises(engine, bars):
    """逃逸类病态输入逐一调用，引擎层均返回 Result，不裸抛。"""
    exprs = [
        "MA.__globals__['__builtins__']['eval']('1+1')",
        "close.__class__.__init__.__globals__['__builtins__']['eval']('1+1')",
        "ta.__dict__",
        "bars.__class__.__mro__[1].__subclasses__()",
        "__import__('os')",
        "import os",
        "open('x')",
        "(lambda: 1)()",
    ]
    for expr in exprs:
        result = engine.calc(expr, bars)
        assert isinstance(result, IndicatorResult)


# --------------------------------------------------------------------------- #
# 名字校验
# --------------------------------------------------------------------------- #
def test_unknown_name_returns_name_error(engine, bars):
    result = engine.calc("foo.rolling(5)", bars)
    assert result.ok is False
    assert result.error_type == "Name"
    assert result.error


def test_forbidden_builtin_name_blocked(engine, bars):
    # open 被开盘价 Series 遮蔽（合法名），但 eval/exec/globals 等危险名被显式拒绝。
    result = engine.calc("eval('1+1')", bars)
    assert result.ok is False
    assert result.error_type == "Security"


# --------------------------------------------------------------------------- #
# 语法错误 → Syntax
# --------------------------------------------------------------------------- #
def test_syntax_error_type(engine, bars):
    result = engine.calc("close.rolling(", bars)
    assert result.ok is False
    assert result.error_type == "Syntax"


# --------------------------------------------------------------------------- #
# 除零 / 运行时错误
# --------------------------------------------------------------------------- #
def test_divide_by_zero_runtime_type(engine, bars):
    result = engine.calc("close/0", bars)
    assert result.ok is False
    assert result.error_type == "Runtime"
    assert result.error


# --------------------------------------------------------------------------- #
# 合法表达式仍可用
# --------------------------------------------------------------------------- #
def test_legit_rolling_mean(engine, bars):
    result = engine.calc("close.rolling(20).mean()", bars)
    assert result.ok is True, result.error
    pd.testing.assert_series_equal(
        result.series, bars["close"].rolling(20).mean(), check_names=False
    )


def test_legit_ta_macd_column(engine, bars):
    result = engine.calc("ta.macd(close)['MACD_12_26_9']", bars)
    assert result.ok is True, result.error
    assert result.series is not None
    assert len(result.series) == len(bars)
    assert result.series.index.equals(bars.index)


def test_legit_ta_sma(engine, bars):
    result = engine.calc("ta.sma(close, 20)", bars)
    assert result.ok is True, result.error
    assert len(result.series) == len(bars)
    assert result.series.iloc[:19].isna().all()


def test_legit_builtin_ma(engine, bars):
    result = engine.calc("MA(bars, 20)", bars)
    assert result.ok is True, result.error
    assert len(result.series) == len(bars)
    assert result.series.iloc[:19].isna().all()


def test_legit_complex_pct_change(engine, bars):
    result = engine.calc("(close - close.shift(1)) / close.shift(1) * 100", bars)
    assert result.ok is True, result.error
    expected = (bars["close"] - bars["close"].shift(1)) / bars["close"].shift(1) * 100
    pd.testing.assert_series_equal(result.series, expected, check_names=False)


def test_legit_compare_returns_bool_series(engine, bars):
    result = engine.calc("close > close.shift(1)", bars)
    assert result.ok is True, result.error
    assert result.series is not None
    # 首元素为 NaN 参与比较 → False；次元素 2.0 > 1.0 → True。
    assert bool(result.series.iloc[0]) is False
    assert bool(result.series.iloc[1]) is True


def test_legit_ifexp(engine, bars):
    result = engine.calc("(close if n > 10 else open)", bars)
    assert result.ok is True, result.error
    pd.testing.assert_series_equal(result.series, bars["close"], check_names=False)


def test_legit_np_and_pd(engine, bars):
    np_result = engine.calc("np.log(close)", bars)
    assert np_result.ok is True, np_result.error
    pd.testing.assert_series_equal(
        np_result.series, np.log(bars["close"]), check_names=False
    )

    pd_result = engine.calc("pd.Series(close)", bars)
    assert pd_result.ok is True, pd_result.error
    assert len(pd_result.series) == len(bars)


def test_legit_binop_arithmetic(engine, bars):
    result = engine.calc("(close + open) / 2", bars)
    assert result.ok is True, result.error
    pd.testing.assert_series_equal(
        result.series, (bars["close"] + bars["open"]) / 2, check_names=False
    )


# --------------------------------------------------------------------------- #
# np/pd 白名单收窄：I/O 与反序列化被封死，纯计算仍可用
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr",
    [
        "np.load('x')",
        "pd.read_pickle('x')",
        "pd.read_csv('x')",
        "np.save('x', close)",
        "np.loadtxt('x')",
    ],
)
def test_narrowed_io_blocked(engine, bars, expr):
    # 这些 I/O 方法不在 np/pd 白名单代理上 → 属性访问 AttributeError → Runtime。
    # 注：``pd.to_pickle`` 现已在 AST 属性黑名单层先命中 Security，见
    # test_pandas_dangerous_methods_blocked。
    result = engine.calc(expr, bars)
    assert result.ok is False
    assert result.error
    assert result.error_type == "Runtime"


@pytest.mark.parametrize(
    "expr",
    [
        "np.log(close)",
        "np.sqrt(close)",
        "np.where(close>30, 1, 0)",
        "np.cumsum(close)",
        "pd.Series(close)",
        "pd.to_numeric(close)",
    ],
)
def test_narrowed_pure_functions_ok(engine, bars, expr):
    result = engine.calc(expr, bars)
    assert result.ok is True, result.error


def test_narrowed_constants_ok(engine, bars):
    assert engine.calc("np.pi", bars).ok is True
    assert engine.calc("np.e", bars).ok is True
    assert engine.calc("np.euler_gamma", bars).ok is True


# --------------------------------------------------------------------------- #
# 上下文注入 symbol / period / n（在表达式中可用）
# --------------------------------------------------------------------------- #
def test_ctx_n_default(engine, bars):
    # n 默认 20，可作为滚动窗口参数使用。
    result = engine.calc("close.rolling(n).mean()", bars)
    assert result.ok is True, result.error
    pd.testing.assert_series_equal(
        result.series, bars["close"].rolling(20).mean(), check_names=False
    )


def test_ctx_symbol_and_period(engine, bars):
    ctx = {"symbol": "000001", "period": "w", "n": 30}
    assert engine.calc("symbol == '000001'", bars, ctx=ctx).data is True
    assert engine.calc("period == 'w'", bars, ctx=ctx).data is True
    # 自定义 n 生效。
    result = engine.calc("close.rolling(n).mean()", bars, ctx=ctx)
    assert result.ok is True, result.error
    pd.testing.assert_series_equal(
        result.series, bars["close"].rolling(30).mean(), check_names=False
    )


# --------------------------------------------------------------------------- #
# ta 白名单收窄：内部模块属性（core / _indicator_loader / 类对象）被封死
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr",
    [
        "ta.core",
        "ta._indicator_loader",
        "ta.core.pd.read_pickle('x')",
        "ta.core.np.load('x')",
        "ta._indicator_loader.importlib.import_module('os').system('echo PWNED')",
        "ta.Category",
        "ta.version",
        "ta.Strategy",
    ],
)
def test_ta_internals_blocked(engine, bars, expr):
    result = engine.calc(expr, bars)
    assert result.ok is False
    assert result.error
    assert result.error_type == "Runtime"


def test_ta_dunder_dict_still_security(engine, bars):
    # ta.__dict__ 属 dunder 属性，仍在 AST 层被拒（Security），不受代理收窄影响。
    result = engine.calc("ta.__dict__['core']", bars)
    assert result.ok is False
    assert result.error_type == "Security"


def test_build_safe_ta_proxy_excludes_internals():
    """直接校验代理对象：危险内部属性缺省、指标函数齐全。"""
    from app.engine.indicator.safe_libs import build_safe_ta

    proxy = build_safe_ta()
    for forbidden in ("core", "_indicator_loader", "Category", "version", "Strategy"):
        assert not hasattr(proxy, forbidden)
    for allowed in ("sma", "ema", "macd", "rsi", "bbands", "kdj"):
        assert hasattr(proxy, allowed)


@pytest.mark.parametrize(
    "expr",
    [
        "ta.sma(close, 20)",
        "ta.ema(close, 20)",
        "ta.rsi(close, 14)",
        "ta.macd(close)['MACD_12_26_9']",
        "ta.bbands(close, 20)['BBM_20_2.0']",
        "ta.kdj(high=high, low=low, close=close, length=9, signal=3)['K_9_3']",
    ],
)
def test_ta_indicators_ok(engine, bars, expr):
    result = engine.calc(expr, bars)
    assert result.ok is True, result.error
    assert result.series is not None
    assert len(result.series) == len(bars)
    assert result.series.index.equals(bars.index)


# --------------------------------------------------------------------------- #
# bars / Series 危险方法名黑名单：字符串重入、写盘、传 callable、信息泄露
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr",
    [
        "bars.eval('close > 0')",
        "bars.eval('close > 0', engine='python')",
        "bars.query('close > 0')",
        "close.to_frame().eval('close > 0')",
        "bars.to_csv('x')",
        "bars.to_pickle('x')",
        "pd.to_pickle('x')",
        "bars.to_dict()",
        "bars.pipe(str)",
        "close.apply(lambda x: x)",
        "'{0}'.format(close)",
    ],
)
def test_pandas_dangerous_methods_blocked(engine, bars, expr):
    result = engine.calc(expr, bars)
    assert result.ok is False
    assert result.error
    assert result.error_type == "Security"


@pytest.mark.parametrize(
    "expr",
    [
        "close.rolling(20).mean()",
        "close.shift(1)",
        "ta.macd(close)['MACD_12_26_9']",
        "MA(bars, 20)",
        "np.log(close)",
        "pd.Series(close)",
        "close > close.shift(1)",
    ],
)
def test_pandas_safe_methods_still_ok(engine, bars, expr):
    result = engine.calc(expr, bars)
    assert result.ok is True, result.error


# --------------------------------------------------------------------------- #
# matplotlib 绘图链写盘：plot/hist/boxplot → figure → savefig/canvas/print_*
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr",
    [
        "bars.plot().figure.savefig('x.png')",
        "bars.plot().figure.savefig('x.png') or close",
        "not bars.plot().figure.savefig('x.png')",
        "bars.boxplot().figure.savefig('x.png')",
        "bars.hist().figure.savefig('x.png')",
        "close.plot().figure.savefig('x.png')",
        "bars.plot()",
        "bars.plot().figure",
        "bars.plot().figure.canvas.print_png('x.png')",
        "bars.boxplot()",
    ],
)
def test_matplotlib_plot_chain_blocked(engine, bars, expr):
    result = engine.calc(expr, bars)
    assert result.ok is False
    assert result.error
    assert result.error_type == "Security"


# --------------------------------------------------------------------------- #
# 原地变异 / numpy 写盘 / info 纵深防御（P3-S2 收尾）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expr",
    [
        "bars.drop(columns=['close'], inplace=True)",
        "bars.insert(0, 'x', close)",
        "bars.pop('close')",
        "close.values.tofile('x')",
        "close.values.dump('x')",
        "bars.info()",
    ],
)
def test_inplace_and_numpy_dump_blocked(engine, bars, expr):
    result = engine.calc(expr, bars)
    assert result.ok is False
    assert result.error
    assert result.error_type == "Security"


def test_build_namespace_injects_deep_copies(engine, bars):
    """改动 1：注入的 bars 与列为深拷贝副本，原地修改副本不影响调用方。"""
    ns = engine._build_namespace(bars)
    assert ns["bars"] is not bars
    assert ns["close"] is not bars["close"]
    # 就地删除副本列，调用方 df 结构不变。
    ns["bars"].drop(columns=["close"], inplace=True)
    assert "close" in bars.columns


def test_copy_isolation_protects_caller_df(engine, bars):
    """端到端：表达式内尝试 inplace 删列后，调用方 df 仍含 close 列。

    注：``drop`` 已在 AST 属性黑名单层被拦（Security），表达式不会真正执行；
    此处一并断言调用方 df 结构未被破坏，与 deep-copy 单元测试形成双重保障。
    """
    result = engine.calc("bars.drop(columns=['close'], inplace=True) or close", bars)
    assert result.ok is False
    assert result.error_type == "Security"
    assert "close" in bars.columns
