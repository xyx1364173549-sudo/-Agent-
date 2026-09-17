"""语义记忆测试。

语义记忆的难点不在「存」而在「遇到矛盾怎么办」，所以测试重点放在
``remember`` 的四种结果上：

    created     第一次见到
    reinforced  内容一样、又被说了一遍 -> 提高置信度
    updated     内容变了、新说法更可信 -> 覆盖
    kept        内容变了、但新说法不太可信 -> 保留旧的

其中 ``kept`` 最容易被忽略，也最要紧：它保证「用户随口一句」不会把
「用户明确肯定过」的结论冲掉。

另外如实记录了两处**规则版抽取的已知缺陷**（见文件末尾），这是基线版本的
真实表现，将来换成大模型抽取时应删掉对应测试。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.semantic import (
    CATEGORY_MASTERY,
    CATEGORY_PREFERENCE,
    SemanticMemory,
    extract_facts,
)


@pytest.fixture
def mem(tmp_path: Path):
    """一个独立的临时语义记忆，用完关闭连接。"""
    memory = SemanticMemory("u1", db_path=tmp_path / "semantic.db")
    yield memory
    memory.close()


# --------------------------------------------------------------------------
# 规则抽取
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["我已经掌握了递归", "我掌握了递归", "我学会了递归", "递归我学会了"],
)
def test_extract_mastered(text: str) -> None:
    facts = extract_facts(text)
    assert len(facts) == 1
    assert facts[0]["category"] == CATEGORY_MASTERY
    assert facts[0]["key"] == "递归"
    assert facts[0]["value"] == "已掌握"


@pytest.mark.parametrize(
    "text",
    ["递归我还是不太会", "递归我不太会", "递归不太会", "递归还是不太会", "二分查找我不太会"],
)
def test_extract_weak(text: str) -> None:
    """各种连接词（我 / 还 / 还是）都不能被吃进知识点名里。"""
    facts = extract_facts(text)
    assert len(facts) == 1
    assert facts[0]["category"] == CATEGORY_MASTERY
    assert facts[0]["value"] == "待加强"
    assert facts[0]["key"] in {"递归", "二分查找"}


@pytest.mark.parametrize("text", ["递归有点难", "递归我感觉有点难"])
def test_extract_hard(text: str) -> None:
    facts = extract_facts(text)
    assert facts[0]["key"] == "递归"
    assert facts[0]["value"] == "待加强"


def test_extract_preference() -> None:
    facts = extract_facts("我喜欢用图来理解")
    assert facts[0]["category"] == CATEGORY_PREFERENCE
    assert facts[0]["value"] == "偏好"


@pytest.mark.parametrize("text", ["递归这个东西吧，我总觉得哪儿没通", "今天天气不错", "", "   "])
def test_extract_returns_empty_when_unrecognized(text: str) -> None:
    """认不出来的句式返回空列表，不能报错——调用方会直接遍历它。"""
    assert extract_facts(text) == []


def test_extract_splits_sentences() -> None:
    """一句话里说了两件事，应当分别抽出来，不能糊成一团。"""
    facts = extract_facts("我学会了递归，但二分查找我不太会")
    assert len(facts) == 2
    assert facts[0]["key"] == "递归"
    assert facts[0]["value"] == "已掌握"


# --------------------------------------------------------------------------
# remember 的四种结果
# --------------------------------------------------------------------------


def test_created_on_first_sight(mem: SemanticMemory) -> None:
    assert mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.8) == "created"
    assert mem.count() == 1


def test_reinforced_when_same_value_again(mem: SemanticMemory) -> None:
    """同一件事被再说一遍，说明更可信——置信度该往上走。"""
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.5)
    assert mem.remember(CATEGORY_MASTERY, "递归", "已掌握") == "reinforced"
    assert mem.get(CATEGORY_MASTERY, "递归")["confidence"] == pytest.approx(0.6)


def test_reinforced_caps_at_one(mem: SemanticMemory) -> None:
    """置信度提满 1.0 就不再涨，不能出现 1.2 这种值。"""
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.95)
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握")
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握")
    assert mem.get(CATEGORY_MASTERY, "递归")["confidence"] == 1.0


def test_updated_when_new_value_more_confident(mem: SemanticMemory) -> None:
    """新说法更可信就覆盖——用户后来明确说「不会了」，得认。"""
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.4)
    assert mem.remember(CATEGORY_MASTERY, "递归", "待加强", confidence=0.9) == "updated"
    assert mem.get(CATEGORY_MASTERY, "递归")["value"] == "待加强"


def test_updated_when_confidence_ties(mem: SemanticMemory) -> None:
    """置信度打平时以最新说法为准（>= 而非 >）。"""
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.5)
    assert mem.remember(CATEGORY_MASTERY, "递归", "待加强", confidence=0.5) == "updated"
    assert mem.get(CATEGORY_MASTERY, "递归")["value"] == "待加强"


def test_kept_when_new_value_less_confident(mem: SemanticMemory) -> None:
    """最要紧的一条：低可信度的新说法不能冲掉高可信度的旧结论。"""
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.9)
    assert mem.remember(CATEGORY_MASTERY, "递归", "待加强", confidence=0.3) == "kept"
    assert mem.get(CATEGORY_MASTERY, "递归")["value"] == "已掌握", "旧结论必须保住"


def test_kept_does_not_change_confidence(mem: SemanticMemory) -> None:
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.9)
    mem.remember(CATEGORY_MASTERY, "递归", "待加强", confidence=0.3)
    assert mem.get(CATEGORY_MASTERY, "递归")["confidence"] == 0.9


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2])
def test_confidence_out_of_range_rejected(mem: SemanticMemory, bad: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=bad)


@pytest.mark.parametrize("good", [0.0, 0.5, 1.0])
def test_confidence_boundaries_accepted(mem: SemanticMemory, good: float) -> None:
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=good)
    assert mem.count() == 1


def test_repeated_remember_keeps_single_row(mem: SemanticMemory) -> None:
    """反复记同一条事实，库里始终只有一行——这是 UNIQUE 约束在起作用。"""
    for _ in range(5):
        mem.remember(CATEGORY_MASTERY, "递归", "已掌握")
    assert mem.count() == 1


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------


def test_get_missing_returns_none(mem: SemanticMemory) -> None:
    assert mem.get(CATEGORY_MASTERY, "不存在的知识点") is None


def test_get_returns_all_fields(mem: SemanticMemory) -> None:
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.7)
    fact = mem.get(CATEGORY_MASTERY, "递归")
    assert fact["category"] == CATEGORY_MASTERY
    assert fact["key"] == "递归"
    assert fact["value"] == "已掌握"
    assert fact["confidence"] == 0.7
    assert len(fact["updated_at"]) == 19


def test_all_facts_returns_everything(mem: SemanticMemory) -> None:
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握")
    mem.remember(CATEGORY_PREFERENCE, "图", "偏好")
    assert len(mem.all_facts()) == 2


def test_all_facts_filters_by_category(mem: SemanticMemory) -> None:
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握")
    mem.remember(CATEGORY_MASTERY, "二分查找", "待加强")
    mem.remember(CATEGORY_PREFERENCE, "图", "偏好")

    assert len(mem.all_facts(CATEGORY_MASTERY)) == 2
    assert len(mem.all_facts(CATEGORY_PREFERENCE)) == 1


def test_all_facts_sorted_by_confidence(mem: SemanticMemory) -> None:
    """置信度高的排前面——上层组装上下文时通常只要最确定的那几条。"""
    mem.remember(CATEGORY_MASTERY, "低", "待加强", confidence=0.2)
    mem.remember(CATEGORY_MASTERY, "高", "已掌握", confidence=0.9)
    mem.remember(CATEGORY_MASTERY, "中", "待加强", confidence=0.5)

    keys = [f["key"] for f in mem.all_facts()]
    assert keys == ["高", "中", "低"]


# --------------------------------------------------------------------------
# 从文字里学事实
# --------------------------------------------------------------------------


def test_learn_from_text_writes_facts(mem: SemanticMemory) -> None:
    learned = mem.learn_from_text("我已经掌握了递归")
    assert len(learned) == 1
    assert learned[0]["action"] == "created"
    assert mem.get(CATEGORY_MASTERY, "递归")["value"] == "已掌握"


def test_learn_from_text_returns_actions(mem: SemanticMemory) -> None:
    """返回实际做了什么，调用方据此知道自己的话有没有被采纳。"""
    mem.learn_from_text("我已经掌握了递归")
    learned = mem.learn_from_text("我已经掌握了递归")
    assert learned[0]["action"] == "reinforced"


def test_learn_from_text_with_nothing_to_learn(mem: SemanticMemory) -> None:
    assert mem.learn_from_text("今天天气不错") == []
    assert mem.count() == 0


def test_learn_from_text_multiple_sentences(mem: SemanticMemory) -> None:
    mem.learn_from_text("我学会了递归，但二分查找我不太会")
    assert mem.count() == 2


# --------------------------------------------------------------------------
# 维护与隔离
# --------------------------------------------------------------------------


def test_forget_existing(mem: SemanticMemory) -> None:
    mem.remember(CATEGORY_MASTERY, "递归", "已掌握")
    assert mem.forget(CATEGORY_MASTERY, "递归") is True
    assert mem.count() == 0


def test_forget_missing_returns_false(mem: SemanticMemory) -> None:
    assert mem.forget(CATEGORY_MASTERY, "从来没记过") is False


def test_users_are_isolated(tmp_path: Path) -> None:
    """两个学生的档案不能串——同一条 key 各记各的。"""
    db = tmp_path / "semantic.db"
    a = SemanticMemory("student-a", db_path=db)
    b = SemanticMemory("student-b", db_path=db)

    a.remember(CATEGORY_MASTERY, "递归", "已掌握", confidence=0.9)
    b.remember(CATEGORY_MASTERY, "递归", "待加强", confidence=0.3)

    assert a.get(CATEGORY_MASTERY, "递归")["value"] == "已掌握"
    assert b.get(CATEGORY_MASTERY, "递归")["value"] == "待加强"
    a.close()
    b.close()


def test_clear_only_affects_current_user(tmp_path: Path) -> None:
    db = tmp_path / "semantic.db"
    a = SemanticMemory("student-a", db_path=db)
    b = SemanticMemory("student-b", db_path=db)

    a.remember(CATEGORY_MASTERY, "递归", "已掌握")
    b.remember(CATEGORY_MASTERY, "递归", "已掌握")
    a.clear()

    assert a.count() == 0
    assert b.count() == 1
    a.close()
    b.close()


# --------------------------------------------------------------------------
# 规则版的已知缺陷（如实记录，换成大模型抽取后应删除）
# --------------------------------------------------------------------------


def test_known_limitation_connector_leaks_into_key() -> None:
    """句首的连接词会被带进知识点名：抽出的是「但二分查找」而不是「二分查找」。

    这条**不是期望的行为**，只是把规则版的真实表现固定下来，
    免得将来无意中改了还以为修好了。要真正解决得靠大模型抽取。
    """
    facts = extract_facts("我学会了递归，但二分查找我不太会")
    assert [f["key"] for f in facts] == ["递归", "但二分查找"]


def test_known_limitation_paraphrase_not_recognized() -> None:
    """换个说法就抽不到——这正是要用大模型改进它的理由。"""
    assert extract_facts("递归这个东西吧，我总觉得哪儿没通") == []
