"""记忆管理器测试。

重点盯三件事：

1. **只抽用户的话** —— 模型自己的回答不是「关于用户的事实」，抽了会污染语义记忆；
2. **会话与用户是两个维度** —— 换会话时工作/情景记忆该清空，但语义记忆必须留着，
   否则「换个会话就不认识这个人了」；
3. **clear 的语义** —— 它按用户清语义记忆，因为语义记忆本来就跨会话共享。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.manager import MemoryManager
from src.memory.semantic import CATEGORY_MASTERY


@pytest.fixture
def mem(tmp_path: Path):
    """一个独立的临时记忆管理器，用完关闭全部连接。"""
    manager = MemoryManager("s1", "u1", db_path=tmp_path / "mem.db")
    yield manager
    manager.close()


# --------------------------------------------------------------------------
# 初始化
# --------------------------------------------------------------------------


def test_creates_three_layers(mem: MemoryManager) -> None:
    assert mem.working is not None
    assert mem.episodic is not None
    assert mem.semantic is not None
    assert mem.stats() == {"messages": 0, "events": 0, "facts": 0}


def test_user_id_falls_back_to_session_id(tmp_path: Path) -> None:
    """不传 user_id 时退化成 session_id，方便临时试跑。"""
    manager = MemoryManager("only-session", db_path=tmp_path / "m.db")
    assert manager.user_id == "only-session"
    manager.close()


# --------------------------------------------------------------------------
# 写入
# --------------------------------------------------------------------------


def test_add_message_writes_working_memory(mem: MemoryManager) -> None:
    mem.add_message("user", "我想学递归")
    mem.add_message("assistant", "好的")
    assert mem.working.count() == 2


def test_add_message_extracts_facts_from_user(mem: MemoryManager) -> None:
    """用户说「我已经掌握了递归」，顺手就该沉淀成一条事实。"""
    learned = mem.add_message("user", "我已经掌握了递归")
    assert len(learned) == 1
    assert mem.semantic.get(CATEGORY_MASTERY, "递归")["value"] == "已掌握"


def test_add_message_skips_extraction_for_assistant(mem: MemoryManager) -> None:
    """模型自己的回答不是关于用户的事实，抽了会污染语义记忆。"""
    learned = mem.add_message("assistant", "我已经掌握了递归")
    assert learned == []
    assert mem.semantic.count() == 0


def test_add_message_can_disable_learning(mem: MemoryManager) -> None:
    learned = mem.add_message("user", "我已经掌握了递归", learn=False)
    assert learned == []
    assert mem.semantic.count() == 0
    assert mem.working.count() == 1, "关掉抽取也不该影响原文入库"


def test_record_event(mem: MemoryManager) -> None:
    event_id = mem.record_event("struggled", "练习递归 5 题错 3 题", importance=0.8)
    assert event_id > 0
    event = mem.episodic.recent()[0]
    assert event["content"] == "练习递归 5 题错 3 题"
    assert event["importance"] == 0.8


def test_learn_writes_facts_without_touching_working(mem: MemoryManager) -> None:
    """learn 只写语义记忆，不动工作记忆——用于从外部材料学事实。"""
    mem.learn("我学会了二分查找")
    assert mem.semantic.count() == 1
    assert mem.working.count() == 0


# --------------------------------------------------------------------------
# 会话与用户是两个维度
# --------------------------------------------------------------------------


def test_working_and_episodic_are_per_session(tmp_path: Path) -> None:
    """换会话，工作记忆和情景记忆都该是空的。"""
    db = tmp_path / "mem.db"

    first = MemoryManager("session-1", "student-1", db_path=db)
    first.add_message("user", "会话一里说的话")
    first.record_event("struggled", "会话一里发生的事")
    first.close()

    second = MemoryManager("session-2", "student-1", db_path=db)
    assert second.working.count() == 0
    assert second.episodic.count() == 0
    second.close()


def test_semantic_memory_survives_across_sessions(tmp_path: Path) -> None:
    """换会话但人没变，学到的事实必须还在——这正是 user_id 存在的意义。"""
    db = tmp_path / "mem.db"

    first = MemoryManager("session-1", "student-1", db_path=db)
    first.add_message("user", "我已经掌握了递归")
    first.close()

    second = MemoryManager("session-2", "student-1", db_path=db)
    fact = second.semantic.get(CATEGORY_MASTERY, "递归")
    assert fact is not None, "换会话也要认得出这个人"
    assert fact["value"] == "已掌握"
    second.close()


def test_different_users_do_not_share_facts(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"

    a = MemoryManager("s1", "student-a", db_path=db)
    b = MemoryManager("s1", "student-b", db_path=db)
    a.learn("我已经掌握了递归")

    assert a.semantic.count() == 1
    assert b.semantic.count() == 0
    a.close()
    b.close()


# --------------------------------------------------------------------------
# 统计、清空、关闭
# --------------------------------------------------------------------------


def test_stats_reflects_all_layers(mem: MemoryManager) -> None:
    mem.add_message("user", "我已经掌握了递归")
    mem.record_event("struggled", "卡住了")
    assert mem.stats() == {"messages": 1, "events": 1, "facts": 1}


def test_clear_empties_everything(mem: MemoryManager) -> None:
    mem.add_message("user", "我已经掌握了递归")
    mem.record_event("struggled", "卡住了")
    mem.clear()
    assert mem.stats() == {"messages": 0, "events": 0, "facts": 0}


def test_close_then_reopen_keeps_data(tmp_path: Path) -> None:
    """关掉再打开，数据还在——说明确实落到了 SQLite 文件里。"""
    db = tmp_path / "mem.db"

    first = MemoryManager("s1", "u1", db_path=db)
    first.add_message("user", "我已经掌握了递归")
    first.close()

    second = MemoryManager("s1", "u1", db_path=db)
    assert second.semantic.count() == 1
    second.close()
