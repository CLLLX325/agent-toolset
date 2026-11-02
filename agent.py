# -*- coding: utf-8 -*-
"""ReAct 风格的 Agent 主循环。

为什么不用 LangGraph：
这个任务本质上就是"想一步 → 做一步 → 看结果 → 再想"，一个 while 循环足够。
框架能省掉的东西（状态管理、分支、持久化）这里一个都用不上，
而它带来的抽象成本（看不懂执行顺序、报错堆栈深）反而会拖慢调试。

模型的输出格式固定为一行 JSON：
    调用工具：{"thought": "...", "action": "工具名", "args": {...}}
    给出答案：{"thought": "...", "final": "回答内容"}

解析失败不抛异常，而是把错误当作"观测"塞回对话，让模型自己改——
这是让 7B 这种小模型也能稳定跑完的关键一步。
"""
import json
from dataclasses import dataclass, field

import httpx

import config
import tools as tool_lib

SYSTEM_PROMPT = """你是一个电商售后客服助手，可以调用工具查订单和查政策。

可用工具：
{tools}

工作方式：
1. 先用工具查事实，不要凭记忆回答。
2. 每一步只输出一个 JSON 对象，不要输出其他文字。
3. 调用工具时输出：{{"thought": "为什么要调这个工具", "action": "工具名", "args": {{...}}}}
4. 信息足够时输出：{{"thought": "结论依据", "final": "给用户的回答"}}
5. 最多 {max_steps} 步，超时会被强制结束，所以要高效。

订单表的字段：order_id, customer, product, amount, status, order_date, delivered_date, city。
status 只有：已签收、运输中、退货中、已退款、已取消。

注意：
- 涉及算术时用 calculate 工具，不要心算。
- 涉及日期间隔时用 date_diff 工具。
- 回答里带上关键数字（金额、天数、订单号），不要模糊表述。"""


@dataclass
class Step:
    """一步执行记录，用于 trace 和错误归因。"""
    index: int
    thought: str = ""
    action: str = ""
    args: dict = field(default_factory=dict)
    observation: dict = field(default_factory=dict)
    parse_error: str = ""


@dataclass
class RunResult:
    """一次完整运行的结果。"""
    answer: str
    steps: list
    finished: bool
    parse_failures: int = 0
    tool_calls: list = field(default_factory=list)


def _chat(messages, temperature=config.TEMPERATURE):
    """调用 Ollama 的对话接口，强制 JSON 输出。"""
    payload = {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": config.NUM_PREDICT},
    }
    resp = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=config.TIMEOUT)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def parse_action(text: str) -> dict:
    """把模型输出解析成 dict。容忍 ```json 包裹、前后多余文字。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("输出里找不到 JSON 对象")
    return json.loads(text[start:end + 1])


def run(question: str, max_steps: int = config.MAX_STEPS, verbose: bool = False,
        system_prompt: str = None) -> RunResult:
    """跑一个问题，返回答案和完整轨迹。system_prompt 可覆盖，用于对比不同提示词版本。"""
    system = system_prompt or SYSTEM_PROMPT.format(tools=tool_lib.tools_prompt(),
                                                  max_steps=max_steps)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]

    steps, parse_failures, tool_calls = [], 0, []

    for i in range(max_steps):
        step = Step(index=i)
        raw = ""
        try:
            raw = _chat(messages)
            data = parse_action(raw)
        except Exception as exc:  # 解析失败也要记一步，方便统计解析失败率
            parse_failures += 1
            step.parse_error = str(exc)
            steps.append(step)
            if raw:
                messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                             f"上一步输出无法解析（{exc}）。请只输出一个 JSON 对象，不要有任何其他文字。"})
            if verbose:
                print(f"[step {i}] 解析失败：{exc}")
            continue

        step.thought = str(data.get("thought", ""))[:200]

        # 1) 给出最终答案
        if "final" in data:
            answer = str(data["final"])
            steps.append(step)
            if verbose:
                print(f"[step {i}] 最终答案：{answer}")
            return RunResult(answer=answer, steps=steps, finished=True,
                             parse_failures=parse_failures, tool_calls=tool_calls)

        # 2) 调用工具
        action = str(data.get("action", ""))
        args = data.get("args", {})
        step.action, step.args = action, args if isinstance(args, dict) else {}
        obs = tool_lib.call_tool(action, step.args)
        step.observation = obs
        tool_calls.append(action)
        steps.append(step)

        if verbose:
            print(f"[step {i}] {action}({step.args}) -> {'ok' if obs.get('ok') else obs.get('error')}")

        messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
        obs_text = json.dumps(obs, ensure_ascii=False)[:1200]
        if obs.get("ok"):
            messages.append({"role": "user", "content": f"工具返回：{obs_text}"})
        else:
            messages.append({"role": "user", "content":
                             f"工具报错：{obs.get('error')}。请根据错误调整参数或换一个工具。"})

    return RunResult(answer="（超过最大步数，未完成）", steps=steps, finished=False,
                     parse_failures=parse_failures, tool_calls=tool_calls)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "A2012 这个订单买了什么，多少钱？"
    result = run(q, verbose=True)
    print("\n答案：", result.answer)
    print("工具调用：", result.tool_calls, "| 解析失败：", result.parse_failures)
