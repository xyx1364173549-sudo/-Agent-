"""上下文组装测试。

重点盯三件事：

1. **顺序** —— 语义 → 情景 → 工作，稳定的排前面。顺序错了，预算紧张时
   被砍掉的就会是最该留下的那部分；
2. **预算真的生效** —— 给紧预算时结果要变短，但语义记忆必须活下来；
3. **空输入不炸** —— 三层都空时返回空字符串，下游拿到空串也不会出问题。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.context import build_context
from src.memory.episodic import EpisodicMemory
from src.memory.manager import MemoryManager
from src.memory.semantic import SemanticMemory
from src.memory.working import WorkingMemory, count_tokens


@pytest.fixture
def mems(tmp_path: Path):
    """三层的裸实例（不经过 MemoryManager），用完统一关闭。"""
    db = tmp_path / "ctx.db"
    working = WorkingMemory("s1", db_path=db)
    episodic = EpisodicMemory("s1", db_path=db)
    semantic = SemanticMemory("u1", db_path=db)
    yield working, episodic, semantic
    working.close()
    episodic.close()
    semantic.close()


# --------------------------------------------------------------------------
# 空输入与单层
# --------------------------------------------------------------------------


def test_all_empty_returns_empty_string(mems) -> None:
    working, episodic, semantic = mems
    assert build_context(working, episodic, semantic) == ""


def test_only_semantic(mems) -> None:
    working, episodic, semantic = mems
    semantic.remember("mastery", "递归", "待加强")

    text = build_context(working, episodic, semantic)
    assert "【学习者情况】" in text
    assert "- 递归：待加强" in text
    assert "【最近对话】" not in text


def test_only_events(mems) -> None:
    working, episodic, semantic = mems
    episodic.record("struggled", "练习递归 5 题错 3 题")

    text = build_context(working, episodic, semantic)
    assert "【最近发生的事】" in text
    assert "练习递归 5 题错 3 题" in text


def test_only_working(mems) -> None:
    working, episodic, semantic = mems
    working.add("user", "我想学递归")

    text = build_context(working, episodic, semantic)
    assert "【最近对话】" in text
    assert "- user：我想学递归" in text


# --------------------------------------------------------------------------
# 顺序
# --------------------------------------------------------------------------


def test_section_order_is_stable_first(mems) -> None:
    """顺序必须是 语义 → 情景 → 工作。稳定的放前面，预算不够时才砍后面的。"""
    working, episodic, semantic = mems
    semantic.remember("mastery", "递归", "待加强")
    episodic.record("struggled", "卡住了")
    working.add("user", "你好")

    text = build_context(working, episodic, semantic)
    assert text.index("【学习者情况】") < text.index("【最近发生的事】")
    assert text.index("【最近发生的事】") < text.index("【最近对话】")


# --------------------------------------------------------------------------
# 每层限量
# --------------------------------------------------------------------------


def test_max_facts_limits_lines(mems) -> None:
    working, episodic, semantic = mems
    for i in range(8):
        semantic.remember("mastery", f"知识点{i}", "已掌握")

    text = build_context(working, episodic, semantic, max_facts=3)
    assert text.count("- ") == 3


def test_max_events_limits_lines(mems) -> None:
    working, episodic, semantic = mems
    for i in range(8):
        episodic.record("t", f"第{i}件事")

    text = build_context(working, episodic, semantic, max_events=2)
    assert text.count("- ") == 2


# --------------------------------------------------------------------------
# 预算
# --------------------------------------------------------------------------


def test_zero_budget_returns_empty(mems) -> None:
    working, episodic, semantic = mems
    semantic.remember("mastery", "递归", "待加强")
    assert build_context(working, episodic, semantic, max_tokens=0) == ""


def test_negative_budget_returns_empty(mems) -> None:
    working, episodic, semantic = mems
    semantic.remember("mastery", "递归", "待加强")
    assert build_context(working, episodic, semantic, max_tokens=-10) == ""


def test_tight_budget_shortens_result(mems) -> None:
    working, episodic, semantic = mems
    semantic.remember("mastery", "递归", "待加强")
    for i in range(10):
        working.add("user", f"这是第{i}条比较长的对话内容，专门用来把预算占满")

    generous = build_context(working, episodic, semantic, max_tokens=2000)
    tight = build_context(working, episodic, semantic, max_tokens=60)

    assert count_tokens(tight) < count_tokens(generous)


def test_semantic_memory_survives_tight_budget(mems) -> None:
    """预算再紧，也要保住「这个人是什么水平」这个核心判断。"""
    working, episodic, semantic = mems
    semantic.remember("mastery", "递归", "待加强")
    for i in range(10):
        working.add("user", f"很长的一段闲聊内容第 {i} 条，用来挤占预算")

    tight = build_context(working, episodic, semantic, max_tokens=40)
    assert "【学习者情况】" in tight
    assert "递归" in tight


def test_result_never_exceeds_budget(mems) -> None:
    """组装结果不能超预算——超了就等于没裁，模型照样会拒收。"""
    working, episodic, semantic = mems
    for i in range(20):
        semantic.remember("mastery", f"知识点{i}", "已掌握")
        episodic.record("t", f"第{i}件事发生了")
        working.add("user", f"第{i}条对话内容，写长一点好占位置")

    for budget in (30, 80, 200, 500):
        text = build_context(working, episodic, semantic, max_tokens=budget)
        assert count_tokens(text) <= budget, f"预算 {budget} 时超了"


# --------------------------------------------------------------------------
# 与 MemoryManager 的集成
# --------------------------------------------------------------------------


def test_manager_exposes_context(tmp_path: Path) -> None:
    mem = MemoryManager("s1", "u1", db_path=tmp_path / "m.db")
    mem.add_message("user", "我已经掌握了递归")
    mem.record_event("struggled", "还是不太会")

    text = mem.context()
    assert "【学习者情况】" in text
    assert "递归" in text
    assert "【最近发生的事】" in text
    mem.close()


def test_manager_context_respects_budget(tmp_path: Path) -> None:
    mem = MemoryManager("s1", "u1", db_path=tmp_path / "m.db")
    for i in range(10):
        mem.add_message("user", f"第 {i} 条很长的对话内容，用来占预算")

    assert count_tokens(mem.context(max_tokens=40)) <= 40
    mem.close()
