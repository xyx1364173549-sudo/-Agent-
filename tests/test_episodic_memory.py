"""情景记忆测试。

四条主线：

1. **时间线顺序** —— 最新的事件排最前，且同一秒内的多条要靠 id 兜底；
2. **关键词检索** —— 子串能匹配到，不匹配时返回空而不是报错；
3. **importance 校验** —— 越界必须拦下来，否则后面遗忘策略会算乱；
4. **会话隔离** —— 各自的日记本不能串页。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.episodic import EpisodicMemory


@pytest.fixture
def mem(tmp_path: Path):
    """一个独立的临时情景记忆，用完关闭连接。"""
    memory = EpisodicMemory("s1", db_path=tmp_path / "episodic.db")
    yield memory
    memory.close()


# --------------------------------------------------------------------------
# 写入
# --------------------------------------------------------------------------


def test_starts_empty(mem: EpisodicMemory) -> None:
    assert mem.count() == 0
    assert mem.recent() == []


def test_record_increases_count(mem: EpisodicMemory) -> None:
    mem.record("answered_question", "答对了第 3 题")
    mem.record("struggled", "连续答错 2 题")
    assert mem.count() == 2


def test_record_returns_id(mem: EpisodicMemory) -> None:
    """返回新事件的 id，方便调用方后续引用这一条。"""
    first = mem.record("answered_question", "第一件")
    second = mem.record("answered_question", "第二件")
    assert second > first


def test_recorded_fields_are_complete(mem: EpisodicMemory) -> None:
    mem.record("struggled", "卡在终止条件", importance=0.9)

    event = mem.recent()[0]
    assert event["event_type"] == "struggled"
    assert event["content"] == "卡在终止条件"
    assert event["importance"] == 0.9
    assert event["id"] == 1
    assert len(event["created_at"]) == 19  # ISO8601 文本时间


def test_default_importance(mem: EpisodicMemory) -> None:
    mem.record("chitchat", "随便聊了两句")
    assert mem.recent()[0]["importance"] == 0.5


# --------------------------------------------------------------------------
# 时间线
# --------------------------------------------------------------------------


def test_recent_returns_newest_first(tmp_path: Path) -> None:
    """时间线是「最近发生的事排最前」。"""
    memory = EpisodicMemory("s1", db_path=tmp_path / "e.db")
    memory.record("t", "早上做的事")
    memory.record("t", "中午做的事")
    memory.record("t", "晚上做的事")

    contents = [e["content"] for e in memory.recent()]
    assert contents == ["晚上做的事", "中午做的事", "早上做的事"]
    memory.close()


def test_same_second_ordering(tmp_path: Path) -> None:
    """同一秒内记的多条事件时间戳完全相同，必须靠 id 兜底排序。

    这是很容易漏的坑：只按 created_at 排序时，SQLite 对同值的行
    返回顺序是不确定的，可能今天对、明天就乱了。
    """
    memory = EpisodicMemory("s1", db_path=tmp_path / "e.db")
    memory.record("t", "第一条")
    memory.record("t", "第二条")
    memory.record("t", "第三条")

    contents = [e["content"] for e in memory.recent()]
    assert contents == ["第三条", "第二条", "第一条"]
    memory.close()


def test_recent_respects_limit(mem: EpisodicMemory) -> None:
    for i in range(10):
        mem.record("t", f"第{i}件事")

    assert len(mem.recent(3)) == 3
    assert [e["content"] for e in mem.recent(3)] == ["第9件事", "第8件事", "第7件事"]


# --------------------------------------------------------------------------
# 关键词检索
# --------------------------------------------------------------------------


def test_search_matches_substring(mem: EpisodicMemory) -> None:
    mem.record("struggled", "卡在递归的终止条件上")
    mem.record("answered_question", "答对了二分查找的题")
    mem.record("struggled", "又把递归写成了死循环")

    hits = mem.search("递归")
    assert len(hits) == 2
    assert all("递归" in e["content"] for e in hits)


def test_search_returns_empty_when_no_match(mem: EpisodicMemory) -> None:
    """没匹配到就返回空列表，不能报错——下游会遍历它。"""
    mem.record("t", "只有这一条")
    assert mem.search("完全不相干的词") == []


def test_search_can_match_event_type_text(mem: EpisodicMemory) -> None:
    """LIKE 只搜内容，不搜类型——这个边界要明确，免得调用方误用。"""
    mem.record("struggled", "题目太难了")
    assert mem.search("struggled") == []


def test_search_respects_limit(mem: EpisodicMemory) -> None:
    for i in range(5):
        mem.record("t", f"第{i}次提到递归")

    assert len(mem.search("递归", limit=2)) == 2


# --------------------------------------------------------------------------
# 按类型取
# --------------------------------------------------------------------------


def test_by_type_filters(mem: EpisodicMemory) -> None:
    mem.record("struggled", "卡住了")
    mem.record("answered_question", "答对了")
    mem.record("struggled", "又卡住了")

    assert len(mem.by_type("struggled")) == 2
    assert len(mem.by_type("answered_question")) == 1
    assert mem.by_type("finished_task") == []


# --------------------------------------------------------------------------
# importance 校验
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2])
def test_importance_out_of_range_rejected(mem: EpisodicMemory, bad: float) -> None:
    """越界必须拦下，否则后面按 importance 做遗忘排序时会算乱。"""
    with pytest.raises(ValueError, match="importance"):
        mem.record("t", "内容", importance=bad)


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_importance_boundaries_accepted(mem: EpisodicMemory, good: float) -> None:
    """边界值本身合法，校验不能写成开区间。"""
    mem.record("t", "内容", importance=good)
    assert mem.count() == 1


def test_failed_validation_writes_nothing(mem: EpisodicMemory) -> None:
    """校验失败时不能留下半条脏数据。"""
    with pytest.raises(ValueError):
        mem.record("t", "内容", importance=9)
    assert mem.count() == 0


# --------------------------------------------------------------------------
# 隔离与清空
# --------------------------------------------------------------------------


def test_sessions_are_isolated(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    a = EpisodicMemory("a", db_path=db)
    b = EpisodicMemory("b", db_path=db)

    a.record("t", "A 的事件")
    b.record("t", "B 的事件")

    assert [e["content"] for e in a.recent()] == ["A 的事件"]
    assert [e["content"] for e in b.recent()] == ["B 的事件"]
    a.close()
    b.close()


def test_search_does_not_leak_across_sessions(tmp_path: Path) -> None:
    """检索也要按会话隔离，不能搜出别人的日记。"""
    db = tmp_path / "e.db"
    a = EpisodicMemory("a", db_path=db)
    b = EpisodicMemory("b", db_path=db)

    a.record("t", "我在学递归")
    b.record("t", "我也在学递归")

    assert len(a.search("递归")) == 1
    assert len(b.search("递归")) == 1
    a.close()
    b.close()


def test_clear_only_affects_current_session(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    a = EpisodicMemory("a", db_path=db)
    b = EpisodicMemory("b", db_path=db)

    a.record("t", "A 的事件")
    b.record("t", "B 的事件")
    a.clear()

    assert a.count() == 0
    assert b.count() == 1
    a.close()
    b.close()
