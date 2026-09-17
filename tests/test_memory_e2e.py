"""记忆模块端到端测试。

前面的测试文件都是按模块单独测的，这里刻意换个视角：**从使用者的角度
走一遍完整流程**，看三层记忆能不能协同工作。

走的是这样一条线：

    用户说话 → 记进工作记忆 → 顺手抽出事实 → 记下关键事件
    → 把三层组装成上下文 → 交给模型 → 过一段时间 → 该忘的忘掉

单模块全绿不代表串起来也对——参数顺序、会话与用户的传法、时间字段的格式，
这些问题只有走完整流程才暴露得出来。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.memory.manager import MemoryManager


@pytest.fixture
def mem(tmp_path: Path):
    manager = MemoryManager("session-1", "student-1", db_path=tmp_path / "e2e.db")
    yield manager
    manager.close()


# --------------------------------------------------------------------------
# 完整流程
# --------------------------------------------------------------------------


def test_full_learning_flow(mem: MemoryManager) -> None:
    """一次完整的「聊了 → 记了事 → 沉淀了事实」。"""
    mem.add_message("user", "我最近在学递归")
    mem.add_message("assistant", "好的，我们从最简单的开始")
    mem.add_message("user", "递归我还是不太会")
    mem.record_event("struggled", "练习递归 5 题，答错 3 题", importance=0.8)

    stats = mem.stats()
    assert stats["messages"] == 3, "三条对话都进了工作记忆"
    assert stats["events"] == 1, "事件进了情景记忆"
    assert stats["facts"] == 1, "「递归不太会」被抽成了一条事实"


def test_context_merges_all_three_layers(mem: MemoryManager) -> None:
    """组装出来的上下文，三层的内容都要在。"""
    mem.add_message("user", "递归我还是不太会")
    mem.record_event("struggled", "练习递归 5 题，答错 3 题", importance=0.8)

    context = mem.context()
    assert "【学习者情况】" in context, "语义记忆"
    assert "【最近发生的事】" in context, "情景记忆"
    assert "【最近对话】" in context, "工作记忆"
    assert "递归" in context


def test_context_grows_as_memory_accumulates(mem: MemoryManager) -> None:
    assert mem.context() == "", "一开始什么都没记，上下文是空的"

    mem.add_message("user", "我想学递归")
    assert mem.context() != ""


# --------------------------------------------------------------------------
# 时间推进
# --------------------------------------------------------------------------


def test_fading_over_time(mem: MemoryManager) -> None:
    """时间快进：琐碎的事被忘掉，重要的事还留着。"""
    mem.record_event("chitchat", "聊了句天气", importance=0.3)
    mem.record_event("finished_task", "拿下递归这一关", importance=0.8)

    future = datetime.now() + timedelta(days=10)
    archived = mem.episodic.forget_weak(now=future)

    assert archived == 1, "只该忘掉那条不重要的"
    remaining = [e["content"] for e in mem.episodic.recent()]
    assert remaining == ["拿下递归这一关"]


def test_forgotten_events_leave_context(mem: MemoryManager) -> None:
    """被忘掉的事不该再出现在给模型的上下文里——否则「遗忘」就白做了。"""
    mem.record_event("chitchat", "一句无关紧要的闲聊", importance=0.3)
    before = mem.context()
    assert "无关紧要的闲聊" in before

    mem.episodic.forget_weak(now=datetime.now() + timedelta(days=10))
    after = mem.context()
    assert "无关紧要的闲聊" not in after


def test_reinforcement_slows_forgetting(mem: MemoryManager) -> None:
    """同类问题反复出现时加深印象，它在遗忘曲线上就能撑得更久。"""
    event_id = mem.record_event("struggled", "又卡在终止条件", importance=0.4)

    future = datetime.now() + timedelta(days=14)
    before = mem.episodic.decay_report(now=future)[0]["strength"]

    for _ in range(3):
        mem.episodic.reinforce(event_id)

    after = mem.episodic.decay_report(now=future)[0]["strength"]
    assert after > before, "巩固之后，同样的时间点应该剩下更多"


def test_decay_report_is_ordered_for_cleanup(mem: MemoryManager) -> None:
    """报表按强度升序——最该忘的排最前，方便按顺序清理。"""
    mem.record_event("t", "刚发生的重要事", importance=1.0)
    mem.episodic._conn.execute(
        "INSERT INTO episodic_memory (session_id, event_type, content, importance, created_at) VALUES (?,?,?,?,?)",
        ("session-1", "t", "很久以前的小事", 0.3, (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")),
    )
    mem.episodic._conn.commit()

    report = mem.episodic.decay_report()
    assert report[0]["content"] == "很久以前的小事"


# --------------------------------------------------------------------------
# 隔离
# --------------------------------------------------------------------------


def test_two_students_do_not_mix(mem: MemoryManager, tmp_path: Path) -> None:
    """两个学生共用一个库，但互相看不到对方的记忆。"""
    other = MemoryManager("session-2", "student-2", db_path=tmp_path / "e2e.db")

    mem.add_message("user", "我学会了递归")
    other.add_message("user", "我学会了二分查找")

    assert mem.semantic.count() == 1
    assert other.semantic.count() == 1
    assert mem.semantic.get("mastery", "递归") is not None
    assert other.semantic.get("mastery", "递归") is None

    other.close()


def test_new_session_for_same_student_keeps_profile(tmp_path: Path) -> None:
    """同一个学生换一次会话：旧的对话清空，但「他是什么水平」还认得。"""
    db = tmp_path / "e2e.db"

    first = MemoryManager("session-a", "student-1", db_path=db)
    first.add_message("user", "递归我还是不太会")
    first.add_message("user", "我们下次再聊")
    first.close()

    second = MemoryManager("session-b", "student-1", db_path=db)
    assert second.working.count() == 0, "新会话没有旧对话"
    assert second.semantic.get("mastery", "递归") is not None, "但记得这个学生的薄弱点"

    context = second.context()
    assert "学习者情况" in context, "开新会话时，画像应该立刻可用"
    second.close()
