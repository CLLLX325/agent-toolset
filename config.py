# -*- coding: utf-8 -*-
"""全局配置。所有可调参数集中在这里，方便对照实验。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "orders.db"
ORDERS_CSV = DATA_DIR / "orders.csv"
TASKS_PATH = ROOT / "tasks.jsonl"
TRACE_DIR = ROOT / "traces"

# 政策语料放在项目自己的 corpus/ 下。
# 早期版本是读隔壁 01-rag-kb/corpus 的，拆成独立仓库后那样会跑不起来，
# 所以复制了一份进来（内容相同，仍是 24 篇）。
POLICY_DIR = ROOT / "corpus"

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:7b"

MAX_STEPS = 8          # 最多几步，防止模型死循环
TEMPERATURE = 0.0
NUM_PREDICT = 512
TIMEOUT = 120

# SQL 工具的安全边界
SQL_MAX_ROWS = 50
SQL_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
                 "attach", "pragma", "replace", "vacuum")
