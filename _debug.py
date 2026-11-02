# -*- coding: utf-8 -*-
"""调试脚本：跑单条任务，把每一步的思考 / 动作 / 参数 / 观测都打出来。

用途：t07（"上海的客户下过哪几个订单号"）在评测里答成"上海没有订单"，
需要看模型到底写了什么 SQL，才能判断是模型错还是工具错。
"""

import sys

import agent

QUESTION = "上海的客户下过哪几个订单号？"

if len(sys.argv) > 1:
    QUESTION = sys.argv[1]

result = agent.run(QUESTION)

print("=" * 60)
for i, step in enumerate(result.steps, 1):
    print(f"第 {i} 步")
    print("  思考:", step.thought)
    print("  动作:", step.action, step.args)
    print("  观测:", str(step.observation)[:500])
print("=" * 60)
print("最终答案:", result.answer)
