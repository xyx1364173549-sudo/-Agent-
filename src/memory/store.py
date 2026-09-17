"""SQLite 存储层：打开连接 + 建表 + 升级老库。

分层记忆的三种记忆都存在**同一个 SQLite 文件**里，各占一张表：

    working_memory     工作记忆 —— 当前会话的临时上下文，像手边的草稿纸
    episodic_memory    情景记忆 —— 发生过的事件，像一本日记
    semantic_memory    语义记忆 —— 沉淀下来的稳定事实，像档案卡片

本文件干三件事：打开连接、建表、**把老库升级到新结构**。
具体怎么记、怎么查，是后面几个文件的事。
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import get_settings

# 表结构版本号。
#
# 改表结构时把它 +1，并在 _migrate() 里补上对应的升级语句，
# 这样已经在用的老库能顺利升上来，不用删库重建、也不会丢数据。
#
#   1  最初版本
#   2  情景记忆加 archived 字段（支持「遗忘」时归档而不是删除）
#   3  新增学习者画像两张表（掌握度 / 目标与偏好）
SCHEMA_VERSION = 3

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
    created_at  TEXT NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0  -- 1 表示已被遗忘（归档），默认查询不再返回
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

-- 学习者画像之一：每个知识点掌握到什么程度
--
-- 为什么不用语义记忆存？因为语义记忆里存的是**文本事实**
-- （「递归 = 待加强」），给人看、给模型看都合适，但没法拿来做计算——
-- 路径规划需要比较「递归 0.3」和「二分查找 0.7」谁更该先学。
-- 所以掌握度单独用**数值**存一张表，两者是同一个人的两种粒度。
CREATE TABLE IF NOT EXISTS learner_mastery (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    topic       TEXT NOT NULL,              -- 知识点，例如「递归」
    mastery     REAL NOT NULL DEFAULT 0.0,  -- 掌握度 0~1
    attempts    INTEGER NOT NULL DEFAULT 0, -- 练过多少题
    correct     INTEGER NOT NULL DEFAULT 0, -- 答对多少题
    updated_at  TEXT NOT NULL,
    UNIQUE(user_id, topic)
);

-- 学习者画像之二：用户级别的设置（学习目标、学习偏好）
--
-- 这类东西没有「多个知识点」的横向结构，就是一个名字对一个值，
-- 所以用 key-value 存，不必为「目标」和「偏好」各建一张表。
CREATE TABLE IF NOT EXISTS learner_meta (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,              -- goal / preference:xxx
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(user_id, key)
);
"""


def now_iso() -> str:
    """当前时间，ISO8601 文本，例如 ``2026-09-17T17:05:58``。

    时间统一用**文本**存，而不是时间戳数字。原因是 SQLite 里文本时间可以直接
    用 ``>`` ``<`` ``ORDER BY`` 比较，写 SQL 时不用做任何转换，肉眼看着也直观。
    """
    return datetime.now().isoformat(timespec="seconds")


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """把老版本的库升级到当前版本。新库不用升（建表时已是新结构）。"""
    if from_version < 2:
        # v1 -> v2：情景记忆加 archived 字段。
        # 新库在 SCHEMA_SQL 里已经带上了这个字段，所以要先查一下再决定加不加——
        # ALTER TABLE ADD COLUMN 对已存在的列会报错。
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(episodic_memory)")}
        if "archived" not in columns:
            conn.execute("ALTER TABLE episodic_memory ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")

    if from_version < 3:
        # v2 -> v3：新增学习者画像两张表。
        # 这次不用写 ALTER：ALTER 只能给已有的表加列，而这里是**加表**，
        # 建表语句里的 CREATE TABLE IF NOT EXISTS 已经覆盖了——它先于本函数执行。
        # 保留这个分支是为了把升级路径写清楚：以后有人看到版本号变了，
        # 能一眼确认「这次升级到底做了什么」。
        pass


def init_db(conn: sqlite3.Connection) -> None:
    """建表并升级老库。表已存在就跳过，所以重复调用是安全的。"""
    current = conn.execute("PRAGMA user_version").fetchone()[0]

    conn.executescript(SCHEMA_SQL)

    if current < SCHEMA_VERSION:
        _migrate(conn, current)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    conn.commit()


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """打开数据库连接，并确保表已经建好、结构是最新的。

    参数
    ----
    db_path:
        数据库文件路径。不传时用 ``.env`` 里的 ``MEMORY_DB_PATH``
        （默认 ``./data/memory.db``）。

    返回
    ----
    ``sqlite3.Connection``。注意设了 ``row_factory``，所以查询结果可以
    按列名取值：``row["content"]``，比 ``row[2]`` 好读得多。

    另外还关掉了 ``check_same_thread`` 检查，原因见下。
    """
    path = Path(db_path) if db_path else get_settings().memory_db_path

    # 目录不存在就先建出来。SQLite 只会创建数据库文件，不会帮你建父目录，
    # 少了这一步在全新环境下会直接报 "unable to open database file"。
    path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False：SQLite 默认禁止连接跨线程使用，而 Web 服务里
    # 一个会话的连接会在多个请求之间复用，FastAPI 的同步端点又跑在线程池上——
    # 不同请求落到不同线程是常态，照默认设置会直接报
    # "SQLite objects created in a thread can only be used in that same thread"。
    #
    # 关掉这个检查安全吗？安全。Python 3.11 起 sqlite3 模块编译为 serialized
    # 模式（sqlite3.threadsafety == 3），底层自带互斥锁，多线程共享同一个连接
    # 有 SQLite 自己兜着。这也意味着并发写会串行执行——对单机应用完全够用，
    # 真上高并发该换 PostgreSQL，而不是在这里硬撑。
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn
