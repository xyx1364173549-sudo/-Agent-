"""工作记忆测试。

三条主线：

1. **滑动窗口** —— 超上限时删的是最老的，不是最新的；
2. **token 预算** —— 装不下时同样丢最老的；
3. **顺序** —— 取出来的必须是时间正序。

第 3 条最容易写错：token 预算那部分是「从最新往最老」累加的，
累加完如果忘了翻回正序，模型收到的对话顺序就是反的——用户的话
排在回答后面，模型会读得莫名其妙。所以专门有测试盯着它。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.working import WorkingMemory, count_tokens


@pytest.fixture
def mem(tmp_path: Path):
    """一个独立的临时工作记忆，用完关闭连接。"""
    memory = WorkingMemory("s1", db_path=tmp_path / "wm.db")
    yield memory
    memory.close()


# --------------------------------------------------------------------------
# 基本读写
# --------------------------------------------------------------------------


def test_starts_empty(mem: WorkingMemory) -> None:
    assert mem.count() == 0
    assert mem.messages() == []


def test_add_and_read_back(mem: WorkingMemory) -> None:
    mem.add("user", "我想学递归")
    mem.add("assistant", "好，从最简单的开始")

    assert mem.count() == 2
    assert mem.messages() == [
        {"role": "user", "content": "我想学递归"},
        {"role": "assistant", "content": "好，从最简单的开始"},
    ]


def test_messages_are_plain_dicts(mem: WorkingMemory) -> None:
    """返回普通 dict，可以直接丢给模型，不用再转换一次。"""
    mem.add("user", "hi")
    assert set(mem.messages()[0].keys()) == {"role", "content"}


# --------------------------------------------------------------------------
# 滑动窗口
# --------------------------------------------------------------------------


def test_sliding_window_keeps_newest(tmp_path: Path) -> None:
    """超上限时删掉最老的，留下最新的——新对话比旧对话重要。"""
    memory = WorkingMemory("s1", max_messages=3, db_path=tmp_path / "wm.db")
    for i in range(5):
        memory.add("user", f"第{i}条")

    contents = [m["content"] for m in memory.messages()]
    assert contents == ["第2条", "第3条", "第4条"]
    assert memory.count() == 3
    memory.close()


def test_default_limit_is_twenty(mem: WorkingMemory) -> None:
    for i in range(25):
        mem.add("user", f"第{i}条")
    assert mem.count() == 20


def test_max_messages_one(tmp_path: Path) -> None:
    """边界：上限设为 1 时只剩最新一条。"""
    memory = WorkingMemory("s1", max_messages=1, db_path=tmp_path / "wm.db")
    memory.add("user", "第一条")
    memory.add("user", "第二条")

    assert [m["content"] for m in memory.messages()] == ["第二条"]
    memory.close()


def test_sessions_are_isolated(tmp_path: Path) -> None:
    """两个会话共用一个库，但互相看不到对方的对话。"""
    db = tmp_path / "wm.db"
    a = WorkingMemory("a", db_path=db)
    b = WorkingMemory("b", db_path=db)

    a.add("user", "A 说的话")
    b.add("user", "B 说的话")

    assert [m["content"] for m in a.messages()] == ["A 说的话"]
    assert [m["content"] for m in b.messages()] == ["B 说的话"]
    a.close()
    b.close()


def test_window_applies_per_session(tmp_path: Path) -> None:
    """滑动窗口是按会话各算各的，不能把别的会话的消息挤掉。"""
    db = tmp_path / "wm.db"
    a = WorkingMemory("a", max_messages=2, db_path=db)
    b = WorkingMemory("b", max_messages=2, db_path=db)

    for i in range(4):
        a.add("user", f"a{i}")
    b.add("user", "b0")

    assert a.count() == 2
    assert b.count() == 1
    a.close()
    b.close()


# --------------------------------------------------------------------------
# token 预算
# --------------------------------------------------------------------------


def test_keeps_all_when_within_budget(mem: WorkingMemory) -> None:
    mem.add("user", "短消息")
    mem.add("assistant", "也短")
    assert len(mem.messages()) == 2


def test_token_budget_drops_oldest(tmp_path: Path) -> None:
    """预算不够时从最老的开始丢，且剩下的仍按时间正序排列。

    注意这里**不能用「字数」估算 token**：实测 ``"A"*40`` 是 5 个 token，
    而 ``"B"*40`` 是 10 个——同样 40 个字符，差了一倍。
    所以预算按实际算出来的 token 数来定，不靠假设。
    """
    texts = ["A" * 40, "B" * 40, "C" * 40]
    budget = count_tokens(texts[1]) + count_tokens(texts[2])

    # 预算刚好装下最后两条；再加上最老的 A 就超了
    memory = WorkingMemory("s1", max_tokens=budget, db_path=tmp_path / "wm.db")
    for text in texts:
        memory.add("user", text)

    contents = [m["content"] for m in memory.messages()]
    assert contents == texts[1:], "应丢掉最老的 A，且顺序不能颠倒"
    memory.close()


def test_token_budget_does_not_delete_from_db(tmp_path: Path) -> None:
    """token 预算只影响「取出来多少」，不删除库里的数据。

    这一点很重要：库里留着完整轨迹，后面情景记忆才能从中抽取事件。
    """
    memory = WorkingMemory("s1", max_tokens=1, db_path=tmp_path / "wm.db")
    memory.add("user", "一条比较长的消息内容")

    assert memory.messages() == [], "预算装不下，取出来是空的"
    assert memory.count() == 1, "但库里仍然留着这条记录"
    memory.close()


def test_single_message_over_budget_returns_empty(tmp_path: Path) -> None:
    """边界：最新一条自己就超预算，那就一条都不带。

    这是刻意保留的行为——宁可不带上下文，也好过超限被模型拒收。
    """
    memory = WorkingMemory("s1", max_tokens=1, db_path=tmp_path / "wm.db")
    memory.add("user", "这句话肯定不止一个 token")
    assert memory.messages() == []
    memory.close()


def test_messages_accepts_per_call_budget(mem: WorkingMemory) -> None:
    """可以临时指定预算，不改实例的默认值。"""
    mem.add("user", "一条消息")
    assert len(mem.messages(max_tokens=1000)) == 1
    assert mem.messages(max_tokens=0) == []
    assert len(mem.messages()) == 1, "临时预算不该影响实例默认设置"


# --------------------------------------------------------------------------
# 清空
# --------------------------------------------------------------------------


def test_clear_only_affects_current_session(tmp_path: Path) -> None:
    db = tmp_path / "wm.db"
    a = WorkingMemory("a", db_path=db)
    b = WorkingMemory("b", db_path=db)

    a.add("user", "A 的话")
    b.add("user", "B 的话")
    a.clear()

    assert a.count() == 0
    assert b.count() == 1
    a.close()
    b.close()


# --------------------------------------------------------------------------
# token 计数
# --------------------------------------------------------------------------


def test_count_tokens_basics() -> None:
    assert count_tokens("") == 0
    assert count_tokens("你好") == 2


def test_chinese_costs_more_than_english() -> None:
    """同样字数，中文占的 token 明显多于英文——做预算时不能按字数估算。"""
    assert count_tokens("你好世界") > count_tokens("abcd")


def test_longer_text_costs_more() -> None:
    assert count_tokens("你好世界你好世界") > count_tokens("你好世界")
