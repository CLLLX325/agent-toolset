# -*- coding: utf-8 -*-
"""Agent 项目的单元测试：不调用大模型，只测"确定性"的部分。

这样即使 Ollama 没启动、模型没下载，测试也能跑通——
Agent 项目最容易失控的就是把"模型行为"和"程序逻辑"混在一起测。
"""
import agent
import db
import tools


# ---------------------------------------------------------------- 解析
def test_parse_action_plain_json():
    data = agent.parse_action('{"action": "calculate", "args": {"expression": "1+1"}}')
    assert data["action"] == "calculate"


def test_parse_action_with_fence_and_noise():
    """模型经常把 JSON 包在 ```json 里，或前后加一句解释。"""
    text = '好的，我来算一下：\n```json\n{"final": "答案是 2"}\n```\n完成。'
    assert agent.parse_action(text)["final"] == "答案是 2"


def test_parse_action_rejects_garbage():
    for bad in ["我觉得应该查一下订单", "{不是 json}", ""]:
        try:
            agent.parse_action(bad)
        except Exception:
            continue
        raise AssertionError(f"应当解析失败：{bad!r}")


# ---------------------------------------------------------------- 工具
def test_sql_blocks_writes():
    """只读白名单：任何写操作都必须被拦下来。"""
    for sql in ["DELETE FROM orders", "UPDATE orders SET amount=0",
                "DROP TABLE orders", "SELECT 1; DROP TABLE orders"]:
        assert db.run_select(sql)["ok"] is False, sql


def test_sql_query_works():
    out = db.run_select("SELECT COUNT(*) c FROM orders")
    assert out["ok"] and out["rows"][0]["c"] == 30


def test_calculate_is_sandboxed():
    assert tools.tool_calculate("(399+289)*0.1")["result"] == 68.8
    # 不能执行任意代码
    assert tools.tool_calculate("__import__('os').system('echo hi')")["ok"] is False


def test_date_diff():
    assert tools.tool_date_diff("2026-10-05", "2026-11-04")["days"] == 30
    assert tools.tool_date_diff("2026-11-04", "2026-10-05")["days"] == -30
    assert tools.tool_date_diff("2026/10/05", "2026-11-04")["ok"] is False


def test_unknown_tool_and_bad_args():
    assert tools.call_tool("no_such_tool", {})["ok"] is False
    assert tools.call_tool("calculate", {"wrong": 1})["ok"] is False


def test_policy_search_finds_shipping_fee():
    out = tools.tool_search_policy("退货运费谁承担")
    assert out["ok"] and any("运费" in r["source"] for r in out["results"])
