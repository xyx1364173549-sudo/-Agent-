"""SQLite 存储层测试。

重点盯三件事：

1. **建表幂等** —— 每次打开连接都会建表，重复执行绝不能报错；
2. **约束真的生效** —— 语义记忆靠 UNIQUE 约束实现「有则更新」，
   这条约束要是没建上，记忆里就会攒出一堆互相矛盾的旧事实；
3. **目录自动创建** —— 全新环境下 data/ 目录不存在，少这一步直接连不上库。

每个用例都用临时目录，绝不碰项目真实的 data/memory.db。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from src.memory.store import SCHEMA_VERSION, get_connection, init_db, now_iso

EXPECTED_TABLES = {"working_memory", "episodic_memory", "semantic_memory"}


@pytest.fixture
def conn(tmp_path: Path):
    """一个临时数据库连接，用完自动关闭。"""
    connection = get_connection(tmp_path / "test.db")
    yield connection
    connection.close()


# --------------------------------------------------------------------------
# 建表
# --------------------------------------------------------------------------


def test_creates_three_tables(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    names = {row["name"] for row in rows}
    assert EXPECTED_TABLES <= names, f"缺少表：{EXPECTED_TABLES - names}"


def test_creates_indexes(conn: sqlite3.Connection) -> None:
    """按会话查最近记录是最常用的查询，索引必须在。"""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    names = {row["name"] for row in rows}
    assert {"idx_working_session", "idx_episodic_session"} <= names


def test_init_is_idempotent(conn: sqlite3.Connection) -> None:
    """重复建表不能报错——第二次打开同一个库是常态。"""
    init_db(conn)
    init_db(conn)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert len({row["name"] for row in rows} & EXPECTED_TABLES) == 3


def test_schema_version_recorded(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_reopening_existing_db_works(tmp_path: Path) -> None:
    """关掉再打开同一个文件，数据还在，且不会因重复建表而失败。"""
    db = tmp_path / "reopen.db"

    first = get_connection(db)
    first.execute(
        "INSERT INTO working_memory (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        ("s1", "user", "第一条消息", now_iso()),
    )
    first.commit()
    first.close()

    second = get_connection(db)
    count = second.execute("SELECT COUNT(*) AS c FROM working_memory").fetchone()["c"]
    second.close()
    assert count == 1


# --------------------------------------------------------------------------
# 连接行为
# --------------------------------------------------------------------------


def test_creates_parent_directory(tmp_path: Path) -> None:
    """父目录不存在时要自动创建，否则报 unable to open database file。"""
    db = tmp_path / "nested" / "deeper" / "memory.db"
    assert not db.parent.exists()

    connection = get_connection(db)
    connection.close()

    assert db.exists()


def test_rows_support_column_access(conn: sqlite3.Connection) -> None:
    """row_factory 要生效：按列名取值，比 row[2] 好读也好改。"""
    conn.execute(
        "INSERT INTO working_memory (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        ("s1", "user", "你好", now_iso()),
    )
    conn.commit()

    row = conn.execute("SELECT role, content FROM working_memory").fetchone()
    assert row["role"] == "user"
    assert row["content"] == "你好"


# --------------------------------------------------------------------------
# 表结构约束
# --------------------------------------------------------------------------


def test_episodic_importance_defaults(conn: sqlite3.Connection) -> None:
    """importance 不传时应有默认值，不能让调用方每次都填。"""
    conn.execute(
        "INSERT INTO episodic_memory (session_id, event_type, content, created_at) VALUES (?, ?, ?, ?)",
        ("s1", "answered_question", "答对了第 3 题", now_iso()),
    )
    conn.commit()

    row = conn.execute("SELECT importance FROM episodic_memory").fetchone()
    assert row["importance"] == 0.5


def test_semantic_confidence_defaults(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO semantic_memory (user_id, category, key, value, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("u1", "mastery", "递归", "0.8", now_iso()),
    )
    conn.commit()

    row = conn.execute("SELECT confidence FROM semantic_memory").fetchone()
    assert row["confidence"] == 0.5


def test_semantic_unique_constraint_blocks_duplicates(conn: sqlite3.Connection) -> None:
    """同一用户 + 同一类目 + 同一 key，只允许一条记录。

    这条约束是「去重合并」的基础：写入时走 ON CONFLICT 更新，
    就不会攒出一堆自相矛盾的旧事实。
    """
    insert = "INSERT INTO semantic_memory (user_id, category, key, value, updated_at) VALUES (?, ?, ?, ?, ?)"
    conn.execute(insert, ("u1", "mastery", "递归", "0.5", now_iso()))
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, ("u1", "mastery", "递归", "0.9", now_iso()))


def test_semantic_allows_same_key_for_different_users(conn: sqlite3.Connection) -> None:
    """换个用户就该能存同样的 key，约束不能过严。"""
    conn.execute(
        "INSERT INTO semantic_memory (user_id, category, key, value, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("u1", "mastery", "递归", "0.5", now_iso()),
    )
    conn.execute(
        "INSERT INTO semantic_memory (user_id, category, key, value, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("u2", "mastery", "递归", "0.9", now_iso()),
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS c FROM semantic_memory").fetchone()["c"]
    assert count == 2


def test_upsert_pattern_works(conn: sqlite3.Connection) -> None:
    """验证「有则更新、无则插入」的写法真的可行——后面语义记忆就靠它。"""
    sql = """
        INSERT INTO semantic_memory (user_id, category, key, value, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, category, key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
    """
    conn.execute(sql, ("u1", "mastery", "递归", "0.5", "2026-09-17T10:00:00"))
    conn.execute(sql, ("u1", "mastery", "递归", "0.9", "2026-09-17T11:00:00"))
    conn.commit()

    rows = conn.execute("SELECT value FROM semantic_memory").fetchall()
    assert len(rows) == 1, "同一条事实只应留一行"
    assert rows[0]["value"] == "0.9", "应被更新为最新值"


# --------------------------------------------------------------------------
# 时间工具
# --------------------------------------------------------------------------


def test_now_iso_is_parseable() -> None:
    """存的文本时间必须能被解析回来，否则以后想按天分组统计都做不了。"""
    text = now_iso()
    parsed = datetime.fromisoformat(text)
    assert parsed.year >= 2026
    assert len(text) == 19  # 形如 2026-09-17T17:05:58


def test_now_iso_is_sortable_as_text() -> None:
    """时间用文本存储的前提：字符串比较顺序 = 时间先后顺序。"""
    earlier = "2026-09-17T09:00:00"
    later = "2026-09-17T17:00:00"
    assert earlier < later
