# -*- coding: utf-8 -*-
"""在 30 条任务上跑评测：确定性判分，不看模型自评。

判分三项：
1. 工具对不对：期望的工具序列是否是实际序列的子序列（少调一次就错，多调一次不罚）
2. 工具序列是否完全一致（更严格，用来区分"够用"和"精确"）
3. 答案对不对：答案里必须出现所有标准答案片段（去空格和逗号后比较）

用法：
    python run_eval.py --variant v2 --limit 5
    python run_eval.py --variant v1 --tag v1-full
"""
import argparse
import json
import re
import time

import agent
import config
import prompts


def normalize(text: str) -> str:
    """统一格式再比对，避免同义写法被判错：
    - 去掉空白、千分位逗号（"16,109" == "16109"）
    - 把"2026年10月5日"补零转成"2026-10-05"
    """
    text = text or ""
    text = re.sub(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
                  lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", text)
    return re.sub(r"[\s,，]", "", text)


def score(task: dict, result) -> dict:
    """返回单条任务的判分明细。"""
    answer = normalize(result.answer)
    missing = [m for m in task["must_contain"] if normalize(m) not in answer]
    answer_ok = not missing

    expected, actual = task["expected_tools"], result.tool_calls
    i = 0
    for name in actual:                       # 期望序列是否为实际序列的子序列
        if i < len(expected) and name == expected[i]:
            i += 1
    tool_ok = i == len(expected)
    tool_exact = actual == expected

    return {
        "answer_ok": answer_ok,
        "missing": missing,
        "tool_ok": tool_ok,
        "tool_exact": tool_exact,
        "success": answer_ok and tool_ok,
        "steps": len(result.steps),
        "tool_calls": actual,
        "parse_failures": result.parse_failures,
        "answer": result.answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v3",
                    help="提示词版本 v1/v2/v3/v4，默认 v3（实验里最好的一版）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全跑")
    ap.add_argument("--tag", default="", help="结果文件名后缀")
    args = ap.parse_args()

    tasks = [json.loads(line) for line in open(config.TASKS_PATH, encoding="utf-8") if line.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    system = prompts.get(args.variant, config.MAX_STEPS)
    rows, started = [], time.time()

    for idx, task in enumerate(tasks, 1):
        t0 = time.time()
        try:
            result = agent.run(task["question"], system_prompt=system)
            detail = score(task, result)
        except Exception as exc:              # 网络或模型异常不能中断整轮评测
            detail = {"answer_ok": False, "missing": [], "tool_ok": False,
                      "tool_exact": False, "success": False, "steps": 0,
                      "tool_calls": [], "parse_failures": 0, "answer": f"ERROR: {exc}"}
        detail.update(id=task["id"], question=task["question"],
                      expected=task["expected_tools"], latency=round(time.time() - t0, 1))
        rows.append(detail)
        flag = "PASS" if detail["success"] else "FAIL"
        print(f"[{idx}/{len(tasks)}] {task['id']} {flag} "
              f"tools={'ok' if detail['tool_ok'] else 'bad'} "
              f"steps={detail['steps']} {detail['latency']}s "
              f"{'| 缺:' + ','.join(detail['missing']) if detail['missing'] else ''}", flush=True)

    n = len(rows)
    success = sum(r["success"] for r in rows)
    tool_ok = sum(r["tool_ok"] for r in rows)
    exact = sum(r["tool_exact"] for r in rows)
    answer_ok = sum(r["answer_ok"] for r in rows)
    parse_fail = sum(r["parse_failures"] for r in rows)
    steps = sum(r["steps"] for r in rows) / n
    elapsed = time.time() - started

    summary = {
        "variant": args.variant, "n": n,
        "success_rate": round(success / n, 3),
        "tool_coverage": round(tool_ok / n, 3),
        "tool_exact": round(exact / n, 3),
        "answer_accuracy": round(answer_ok / n, 3),
        "avg_steps": round(steps, 2),
        "parse_failures": parse_fail,
        "elapsed_s": round(elapsed, 1),
        "model": config.CHAT_MODEL,
    }
    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    tag = args.tag or f"{args.variant}-limit{args.limit or 'all'}"
    out = config.ROOT / f"results-{tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"明细已写入 {out.name}")


if __name__ == "__main__":
    main()
