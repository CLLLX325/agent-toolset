# -*- coding: utf-8 -*-
"""四个工具的定义与实现。

工具是 Agent 唯一的"手"。每个工具做三件事：
1. 用一句人话描述自己（模型靠这句话决定什么时候用）；
2. 声明参数结构（模型靠它生成正确的调用）；
3. 返回结构化结果（成功有 data，失败有 error，模型靠 error 自我纠正）。
"""
import ast
import datetime as dt
import json

import jieba
from rank_bm25 import BM25Okapi

import config
import db


# ---------------------------------------------------------------- 1. SQL 查询
def tool_query_orders(sql: str) -> dict:
    """在订单库上执行只读 SQL。表名 orders，字段见 system prompt。"""
    return db.run_select(sql)


# ---------------------------------------------------------------- 2. 政策检索
_POLICY_CACHE = None


def _load_policy_sections():
    """把政策文档按 ## 小节切开，返回 [(标题, 正文)]。只读一次，缓存起来。"""
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    sections = []
    if config.POLICY_DIR.exists():
        for path in sorted(config.POLICY_DIR.glob("*.md")):
            doc_title = path.stem
            current = None
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("# "):
                    doc_title = line[2:].strip()
                elif line.startswith("## "):
                    current = [f"{doc_title}·{line[3:].strip()}", []]
                    sections.append(current)
                elif current is not None and line.strip():
                    current[1].append(line.strip())
    _POLICY_CACHE = [(t, "\n".join(ls)) for t, ls in sections]
    return _POLICY_CACHE


_POLICY_INDEX = None


def _build_policy_index():
    """建一次 BM25 索引（按小节），之后复用。"""
    global _POLICY_INDEX
    if _POLICY_INDEX is None:
        sections = _load_policy_sections()
        if not sections:
            return None, []
        corpus = [[t for t in jieba.lcut(title + body) if len(t.strip()) > 1]
                  for title, body in sections]
        _POLICY_INDEX = (BM25Okapi(corpus), sections)
    return _POLICY_INDEX


def tool_search_policy(query: str, top_k: int = 2) -> dict:
    """在售后政策文档里做 BM25 关键词检索，返回最相关的几个小节。

    这里没有用向量检索：政策文本里"退休""运费险""价保"这类词是硬约束，
    关键词匹配足够，而且结果可解释（能说清"因为命中了哪些词"）。
    """
    index, sections = _build_policy_index()
    if index is None:
        return {"ok": False, "error": "政策库为空，检查 POLICY_DIR 是否存在"}

    tokens = [t for t in jieba.lcut(query) if len(t.strip()) > 1]
    scores = index.get_scores(tokens)
    order = sorted(range(len(sections)), key=lambda i: -scores[i])[:top_k]
    hits = [i for i in order if scores[i] > 0]
    if not hits:
        return {"ok": False, "error": "政策库里没有找到相关内容，换关键词试试"}
    return {
        "ok": True,
        "results": [{"source": sections[i][0], "text": sections[i][1][:300]} for i in hits],
    }


# ---------------------------------------------------------------- 3. 计算器
_ALLOWED_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub)
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant) + _ALLOWED_OPS


def tool_calculate(expression: str) -> dict:
    """安全地计算一个四则运算表达式（支持括号、乘方、取模）。"""
    text = (expression or "").strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return {"ok": False, "error": f"表达式无法解析：{text}"}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return {"ok": False, "error": f"表达式里有不允许的写法：{type(node).__name__}"}
    try:
        value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return {"ok": False, "error": "除数为零"}
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"ok": True, "result": value}


# ---------------------------------------------------------------- 4. 日期差
def tool_date_diff(start_date: str, end_date: str) -> dict:
    """计算两个日期相差多少天（YYYY-MM-DD）。end 早于 start 时返回负数。"""
    try:
        a = dt.date.fromisoformat(start_date.strip())
        b = dt.date.fromisoformat(end_date.strip())
    except ValueError:
        return {"ok": False, "error": "日期格式必须是 YYYY-MM-DD"}
    return {"ok": True, "days": (b - a).days}


# ---------------------------------------------------------------- 注册表
TOOLS = {
    "query_orders": {
        "func": tool_query_orders,
        "desc": "在订单表上执行只读 SQL，用来查订单、统计金额、计数。",
        "params": {"sql": "SELECT 语句，表名 orders"},
    },
    "search_policy": {
        "func": tool_search_policy,
        "desc": "检索售后政策文档，用来回答规则类问题（退货运费、价保期限等）。",
        "params": {"query": "检索关键词", "top_k": "返回条数，默认 2（可省略）"},
    },
    "calculate": {
        "func": tool_calculate,
        "desc": "计算数学表达式，涉及加减乘除时使用，不要自己心算。",
        "params": {"expression": "如 (399+289)*0.1"},
    },
    "date_diff": {
        "func": tool_date_diff,
        "desc": "计算两个日期相差的天数，用来判断是否超过售后期限。",
        "params": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"},
    },
}


def call_tool(name: str, args: dict) -> dict:
    """统一的工具入口：名字不存在、参数缺失、工具内部报错，都返回 ok=False。"""
    if name not in TOOLS:
        return {"ok": False, "error": f"没有这个工具：{name}，可用工具：{list(TOOLS)}"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "args 必须是对象"}
    func = TOOLS[name]["func"]
    try:
        return func(**args)
    except TypeError as exc:
        return {"ok": False, "error": f"参数不对：{exc}"}
    except Exception as exc:  # 任何意外都不能让 Agent 崩掉
        return {"ok": False, "error": f"工具内部错误：{exc}"}


def tools_prompt() -> str:
    """把工具清单渲染成给模型看的文本。"""
    lines = []
    for name, meta in TOOLS.items():
        params = "，".join(f"{k}（{v}）" for k, v in meta["params"].items())
        lines.append(f"- {name}：{meta['desc']}\n  参数：{params}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 手工冒烟：直接调用四个工具看返回
    print(json.dumps(tool_query_orders("SELECT status, COUNT(*) c FROM orders GROUP BY status"),
                     ensure_ascii=False, indent=2))
    print(json.dumps(tool_search_policy("退货运费谁承担"), ensure_ascii=False, indent=2))
    print(json.dumps(tool_calculate("(399+289)*0.1"), ensure_ascii=False))
    print(json.dumps(tool_date_diff("2026-10-05", "2026-11-04"), ensure_ascii=False))
