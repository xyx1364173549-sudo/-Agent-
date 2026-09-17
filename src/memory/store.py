"""SQLite 存储层：打开连接 + 建表。

分层记忆的三种记忆都存在**同一个 SQLite 文件**里，各占一张表：

    working_memory     工作记忆 —— 当前会话的临时上下文，像手边的草稿纸
    episodic_memory    情景记忆 —— 发生过的事件，像一本日记
    semantic_memory    语义记忆 —— 沉淀下来的稳定事实，像档案卡片

本文件只干两件事：打开连接、建表。具体怎么记、怎么查，是后面几个文件的事。
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import get_settings

# 表结构版本号。
# 以后要改表结构（加字段、改类型）时，把它 +1，并在这里补上对应的升级语句，
# 这样老数据库也能顺利升上来，而不用删库重建。
SCHEMA_VERSION = 1

# 建表语句。用 CREATE TABLE IF NOT EXISTS，重复执行不会报错，
# 所以每次打开连接时顺手调一次是安全的。
SCHEMA_SQL = """
-- 工作记忆：当前会话聊过什么
CREATE TABLE IF NOT EXISTS working_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,              -- user / assistant
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL               -- ISO8601 文本时间
);

-- 按会话取最近几条，是最常用的查询，所以建索引
CREATE INDEX IF NOT EXISTS idx_working_session
    ON working_memory(session_id, id);

-- 情景记忆：发生过的事件
CREATE TABLE IF NOT EXISTS episodic_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    event_type  TEXT NOT NULL,              -- 例如 answered_question / finished_task
    content     TEXT NOT NULL,
    importance  REAL NOT NULL DEFAULT 0.5,  -- 重要程度 0~1，用于遗忘策略
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodic_session
    ON episodic_memory(session_id, created_at);

-- 语义记忆：稳定的事实
-- UNIQUE(user_id, category, key) 是特意加的：同一个人、同一类目下，
-- 同一个 key 只该有一条记录。写入时用 INSERT ... ON CONFLICT 就能实现
-- 「有则更新、无则插入」，不会攒出一堆重复的旧事实。
CREATE TABLE IF NOT EXISTS semantic_memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    category    TEXT NOT NULL,              -- 例如 profile / mastery / preference
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,  -- 置信度 0~1
    updated_at  TEXT NOT NULL,
    UNIQUE(user_id, category, key)
);
"""


def now_iso() -> str:
    """当前时间，ISO8601 文本，例如 ``2026-09-17T17:05:58``。

    时间统一用**文本**存，而不是时间戳数字。原因是 SQLite 里文本时间可以直接
    用 ``>`` ``<`` ``ORDER BY`` 比较，写 SQL 时不用做任何转换，肉眼看着也直观。
    """
    return datetime.now().isoformat(timespec="seconds")


def init_db(conn: sqlite3.Connection) -> None:
    """建表。表已存在就跳过，所以重复调用是安全的。"""
    conn.executescript(SCHEMA_SQL)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """打开数据库连接，并确保表已经建好。

    参数
    ----
    db_path:
        数据库文件路径。不传时用 ``.env`` 里的 ``MEMORY_DB_PATH``
        （默认 ``./data/memory.db``）。

    返回
    ----
    ``sqlite3.Connection``。注意设了 ``row_factory``，所以查询结果可以
    按列名取值：``row["content"]``，比 ``row[2]`` 好读得多。
    """
    path = Path(db_path) if db_path else get_settings().memory_db_path

    # 目录不存在就先建出来。SQLite 只会创建数据库文件，不会帮你建父目录，
    # 少了这一步在全新环境下会直接报 "unable to open database file"。
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn
