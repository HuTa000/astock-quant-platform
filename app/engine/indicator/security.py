"""P3-S2 AST 白名单沙箱：在编译 / 执行前从语法层面封死任意代码执行逃逸。

背景：P3-S1 的 ``calc`` 用受限 ``eval`` 占位，已知
``close.__class__.__init__.__globals__['__builtins__']['eval'](...)`` 可逃逸任意代码执行。
本模块在 AST 层拒绝危险节点 / 危险属性 / 危险名字：

    * **节点白名单**：只允许无副作用的纯表达式节点（算术、比较、调用、下标、切片、属性、
      列表、元组、条件表达式）。``Import/ImportFrom/FunctionDef/ClassDef/Lambda``、各种语句、
      推导式、生成器、``Yield/Await/NamedExpr`` 等**凡不在白名单内一律拒绝**；
    * **属性名**：不得以 ``__`` 开头或结尾（拦 ``__class__/__globals__/__dict__/__mro__/
      __subclasses__/__bases__/__import__/__init__`` 等），并对显式黑名单做双保险；
      显式黑名单除 dunder 属性外，还含**普通危险方法名黑名单**（``eval``/``query``/
      ``to_*``/``pipe``/``apply``、以及 ``plot/hist/boxplot/figure/savefig/canvas/
      print_*``（matplotlib 绘图链写盘）等，用于封死 pandas 字符串重入、写盘与信息泄露）；
    * **名字**：仅 ``Load`` 上下文，必须在调用方给定的白名单名字集合内；未知名抛
      :class:`NameError`（对应 error_type='Name'），``__`` 前后缀名及显式危险名抛
      :class:`IndicatorSecurityError`（对应 error_type='Security'）。
"""
from __future__ import annotations

import ast
from typing import Iterable, Set

__all__ = ["IndicatorSecurityError", "SecurityValidator", "validate_tree"]


class IndicatorSecurityError(Exception):
    """安全违规专用异常（对应 error_type='Security'）。"""


# 允许的 AST 节点类型白名单。凡不在其中的节点一律拒绝。
_ALLOWED_NODE_TYPES: tuple[type, ...] = (
    # 结构节点
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.keyword,
    # 表达式节点
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.IfExp,
    ast.List,
    ast.Tuple,
    # 二元 / 位运算符
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.MatMult,
    # 一元运算符
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    # 布尔 / 比较运算符
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)

# 显式属性黑名单（双保险：即便 ``__`` 规则被绕过也再拦一次）。
# 除 dunder 属性外，还收录 pandas Series/DataFrame 的**普通危险方法名**——它们不是
# dunder、能绕过 ``__`` 检测，但会重入字符串求值 / 写盘 / 传 callable / 信息泄露。
_FORBIDDEN_ATTRS: Set[str] = {
    # dunder 属性（反射 / 逃逸链路）
    "__class__",
    "__mro__",
    "__subclasses__",
    "__bases__",
    "__globals__",
    "__dict__",
    "__builtins__",
    "__import__",
    "__init__",
    "__getattribute__",
    "__getattr__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__call__",
    "__code__",
    "__func__",
    "__self__",
    "__closure__",
    "__defaults__",
    "__kwdefaults__",
    "__module__",
    "__name__",
    "__doc__",
    "__reduce__",
    "__reduce_ex__",
    # 字符串重入（RCE 根治：pandas .eval/.query 内部 AST 求值器绕过外层沙箱）
    "eval",
    "query",
    # 写盘 / 导出副作用（to_* 系列，避免任意文件写入）
    "to_csv",
    "to_pickle",
    "to_excel",
    "to_hdf",
    "to_parquet",
    "to_json",
    "to_html",
    "to_markdown",
    "to_latex",
    "to_sql",
    "to_clipboard",
    "to_dict",
    "to_numpy",
    "to_list",
    "to_frame",
    "to_string",
    "to_timestamp",
    "to_period",
    "to_records",
    "to_stata",
    "to_feather",
    "to_gbq",
    "to_orc",
    "to_xml",
    "to_sparse",
    "to_pytorch",
    # 可传 callable 的通道（即便 lambda 已被节点白名单拦，也从属性层再堵一道）
    "pipe",
    "apply",
    "applymap",
    "map",
    "agg",
    "aggregate",
    "transform",
    "groupby",
    "resample",
    # 信息泄露（str.format 字段访问会做任意 getattr，绕过 ``__`` 检测）
    "format",
    "format_map",
    # matplotlib 绘图链写盘：pandas 绘图入口（返回 matplotlib 对象）
    "plot",
    "hist",
    "boxplot",
    "density",
    "kde",
    "kdeplot",
    "area",
    "bar",
    "barh",
    "pie",
    "scatter",
    "hexbin",
    "box",
    "line",
    # matplotlib Figure / Axes 写盘与画布方法（写文件的真正出口）
    "figure",
    "get_figure",
    "savefig",
    "canvas",
    "print_figure",
    "print_png",
    "print_svg",
    "print_svgz",
    "print_pdf",
    "print_ps",
    "print_eps",
    "print_pgf",
    "print_raw",
    "print_rgba",
    "print_webp",
    "print_tif",
    "print_tiff",
    "print_jpg",
    "print_jpeg",
    "flush_events",
    # numpy 写盘（ndarray 上直接落盘，纵深防御：即便未来注入文件对象/恢复 builtins 也拦死）
    "tofile",
    "dump",
    "dumps",
    "tobytes",
    "tostring",
    # pandas 原地变异（破坏 DataFrame 列结构；指标计算本身不需要增删列）
    "drop",
    "insert",
    "pop",
    "set_axis",
    "rename",
    # 其它潜在写 / 副作用
    "info",
}

# 显式危险名字黑名单（Python 内建 / 反射 / 执行类函数，即便未注入也明确标 Security）。
_FORBIDDEN_NAMES: Set[str] = {
    "__import__",
    "__builtins__",
    "__loader__",
    "__spec__",
    "__package__",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "input",
}


class SecurityValidator(ast.NodeVisitor):
    """AST 白名单校验器。

    用法::

        SecurityValidator(allowed_names).visit(tree)
        # 或
        validate_tree(tree, allowed_names)
    """

    def __init__(self, allowed_names: Iterable[str]) -> None:
        self.allowed_names: Set[str] = set(allowed_names)
        super().__init__()

    def visit(self, node: ast.AST) -> ast.AST:
        """节点级白名单校验：非白名单节点直接拒绝（覆盖 Import/Lambda/推导式等）。"""
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise IndicatorSecurityError(f"禁止的语法节点：{type(node).__name__}")
        return super().visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """名字校验：仅 ``Load`` 上下文 + ``__`` 前后缀 / 显式黑名单 + 白名单。"""
        if not isinstance(node.ctx, ast.Load):
            raise IndicatorSecurityError(f"禁止对名字赋值或删除：{node.id!r}")
        name = node.id
        if name.startswith("__") or name.endswith("__") or name in _FORBIDDEN_NAMES:
            raise IndicatorSecurityError(f"禁止访问名字：{name!r}")
        if name not in self.allowed_names:
            raise NameError(f"未知名字：{name!r}")
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """属性校验：拒绝 ``__`` 前后缀属性 + 显式黑名单（双保险）。"""
        attr = node.attr
        if attr.startswith("__") or attr.endswith("__") or attr in _FORBIDDEN_ATTRS:
            raise IndicatorSecurityError(f"禁止访问属性：{attr!r}")
        self.generic_visit(node)
        return node


def validate_tree(tree: ast.AST, allowed_names: Iterable[str]) -> None:
    """对已解析的 AST 做白名单校验。

    Args:
        tree: ``ast.parse(..., mode='eval')`` 得到的 :class:`ast.AST`。
        allowed_names: 允许引用的名字集合（即表达式命名空间的键）。

    Raises:
        IndicatorSecurityError: 命中禁止节点 / 属性 / 名字。
        NameError: 引用了白名单之外的名字。
    """
    SecurityValidator(allowed_names).visit(tree)
