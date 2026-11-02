# -*- coding: utf-8 -*-
"""把 CSV 订单转成 SQLite 数据库，并提供只读查询。

为什么用 SQLite 而不是让模型直接读 CSV：
1. 模型不需要把 30 行数据全读进上下文，省 token；
2. 聚合问题（求和、计数、最大值）交给数据库算，比让模型心算可靠；
3. 只放行 SELECT，天然限制了破坏性操作。
"""
import csv
import sqlite3

import config


def build_db(force: bool = False) -> None:
    """从 CSV 建库。已存在且不强制重建时直接返回。"""
    if config.DB_PATH.exists() and not force:
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        """
        CREATE TABLE orders (
            order_id       TEXT PRIMARY KEY,
            customer       TEXT NOT NULL,
            product        TEXT NOT NULL,
            amount         REAL NOT NULL,
            status         TEXT NOT NULL,
            order_date     TEXT NOT NULL,
            delivered_date TEXT,
            city           TEXT NOT NULL
        )
        """
    )
    with open(config.ORDERS_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    conn.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)",
        [tuple(r[k] for k in ("order_id", "customer", "product", "amount",
                              "status", "order_date", "delivered_date", "city"))
         for r in rows],
    )
    conn.commit()
    conn.close()


def connect() -> sqlite3.Connection:
    """返回一个把结果当字典取的连接。"""
    build_db()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_select(sql: str, max_rows: int = config.SQL_MAX_ROWS) -> dict:
    """执行一条只读 SQL，返回 {ok, columns, rows, row_count} 或 {ok:False, error}。

    这里做的是"白名单式"防御：只允许以 SELECT 开头的单条语句。
    真实系统还应该用只读账号连库，不能只靠字符串检查。
    """
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        return {"ok": False, "error": "SQL 为空"}
    if not text.lower().startswith("select"):
        return {"ok": False, "error": "只允许 SELECT 查询"}
    if ";" in text:
        return {"ok": False, "error": "不允许一次执行多条语句"}
    low = text.lower()
    for word in config.SQL_FORBIDDEN:
        if word in low:
            return {"ok": False, "error": f"包含被禁止的关键字：{word}"}

    conn = connect()
    try:
        cur = conn.execute(text)
        rows = cur.fetchmany(max_rows)
        columns = [d[0] for d in cur.description] if cur.description else []
        data = [dict(zip(columns, r)) for r in rows]
        return {"ok": True, "columns": columns, "rows": data, "row_count": len(data)}
    except sqlite3.Error as exc:
        return {"ok": False, "error": f"SQL 执行失败：{exc}"}
    finally:
        conn.close()
