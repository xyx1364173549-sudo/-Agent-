"""切分策略测试。

三条主线：

1. **不丢内容** —— 块拼起来要能还原原文（用 overlap=0 时才可比）；
2. **参数校验** —— ``overlap >= chunk_size`` 会让切分原地打转，必须拦下来；
3. **语义切分能离线测** —— ``embed_fn`` 由外部传入，测试里塞个假函数就行，
   不必真的下载模型。这个设计正是为了让切分逻辑可测。

每个策略都会有一条「和另外两种行为不同」的测试，确保它们不是同一份代码
换个名字——毕竟论文实验二要拿它们做横向对比。
"""

from __future__ import annotations

import re

import pytest

from src.rag.splitter import (
    split_fixed,
    split_recursive,
    split_semantic,
)


def fake_embed(texts: list[str]) -> list[list[float]]:
    """假向量化：拿「关键词是否出现」当特征。

    真人用真模型，测试用它——完全离线、结果确定，还能精确控制
    「哪两句该被判为不同话题」。
    """
    return [
        [
            1.0 if "递归" in text else 0.0,
            1.0 if "二分查找" in text else 0.0,
            0.1,
        ]
        for text in texts
    ]


# --------------------------------------------------------------------------
# 参数校验（三种策略共用）
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_size", [0, -1, -100])
def test_chunk_size_must_be_positive(bad_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        split_fixed("内容", chunk_size=bad_size)


def test_overlap_must_not_exceed_chunk_size() -> None:
    """overlap 大于等于块长时，每次只前进 0 步或负步，会陷入死循环。"""
    with pytest.raises(ValueError, match="overlap"):
        split_fixed("内容" * 100, chunk_size=50, overlap=50)

    with pytest.raises(ValueError, match="overlap"):
        split_fixed("内容" * 100, chunk_size=50, overlap=80)


def test_negative_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_recursive("内容", overlap=-1)


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t"])
def test_blank_text_returns_empty(text: str) -> None:
    assert split_fixed(text) == []
    assert split_recursive(text) == []
    assert split_semantic(text, embed_fn=fake_embed) == []


# --------------------------------------------------------------------------
# 固定长度
# --------------------------------------------------------------------------


def test_fixed_cuts_at_exact_size() -> None:
    text = "a" * 100
    chunks = split_fixed(text, chunk_size=30, overlap=0)
    assert [len(chunk) for chunk in chunks] == [30, 30, 30, 10]


def test_fixed_keeps_all_content_without_overlap() -> None:
    text = "abcdefghij" * 20
    chunks = split_fixed(text, chunk_size=30, overlap=0)
    assert "".join(chunks) == text


def test_fixed_overlap_repeats_previous_tail() -> None:
    """开启重叠后，每个块的开头会重复上一块的结尾。"""
    text = "0123456789" * 10
    chunks = split_fixed(text, chunk_size=30, overlap=10)
    assert chunks[1].startswith(chunks[0][-10:])


def test_fixed_short_text_not_split() -> None:
    assert split_fixed("短文本", chunk_size=100) == ["短文本"]


def test_fixed_ignores_semantic_boundaries() -> None:
    """固定长度的特点就是「不管语义」——它会切在句子中间。

    这不是 bug，是它的定位：作为实验里最笨的基线，用来量化
    「讲究的切分能带来多少提升」。
    """
    text = "第一句话在这里。第二句话在这里。第三句话在这里。"
    chunks = split_fixed(text, chunk_size=12, overlap=0)
    assert len(chunks) > 1
    # 至少有一块不是以句末标点结尾，说明确实切断了句子
    assert any(not chunk.endswith("。") for chunk in chunks)


# --------------------------------------------------------------------------
# 递归
# --------------------------------------------------------------------------


def test_recursive_prefers_paragraph_boundary() -> None:
    """段落边界优先——两个短段落不该被硬凑在一起、也不该被切开。"""
    text = "第一段的内容。\n\n第二段的内容。\n\n第三段的内容。"
    chunks = split_recursive(text, chunk_size=20, overlap=0)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)


def test_recursive_keeps_all_content_without_overlap() -> None:
    """块拼起来要还原原文。

    允许的差异只有空白：每个块首尾的空白会被清掉（对检索没意义），
    所以比较前先把所有空白字符都去掉。
    """
    text = "第一段的内容写在这里。\n\n第二段的内容写在这里。\n\n第三段的内容写在这里。"
    chunks = split_recursive(text, chunk_size=25, overlap=0)

    def strip_all(value: str) -> str:
        return re.sub(r"\s+", "", value)

    assert strip_all("".join(chunks)) == strip_all(text)


def test_recursive_splits_long_paragraph_further() -> None:
    """一整段超长、没有空行可切时，要能退到用句号切。"""
    text = "".join(f"这是第{i}句话的内容。 " for i in range(30))
    chunks = split_recursive(text, chunk_size=60, overlap=0)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks), "块长不该失控"


def test_recursive_handles_text_without_any_separator() -> None:
    """连一个分隔符都没有时也不能崩，最后要退化成硬切。"""
    text = "无分隔符的长文本" * 30
    chunks = split_recursive(text, chunk_size=50, overlap=0, separators=["\n\n", "\n"])
    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_recursive_overlap_applied() -> None:
    text = "甲" * 100 + "。" + "乙" * 100
    chunks = split_recursive(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[1].startswith(chunks[0][-10:])


def test_recursive_short_text_untouched() -> None:
    assert split_recursive("很短的一句话。", chunk_size=100) == ["很短的一句话。"]


# --------------------------------------------------------------------------
# 语义
# --------------------------------------------------------------------------


def test_semantic_splits_on_topic_change() -> None:
    """话题从「递归」转到「二分查找」时切一刀。"""
    text = "递归是函数自己调用自己。递归需要终止条件。二分查找要求数组有序。二分查找每次折半。"
    chunks = split_semantic(text, embed_fn=fake_embed, chunk_size=200, threshold=0.5)

    assert len(chunks) == 2
    assert "递归" in chunks[0]
    assert "二分查找" in chunks[1]


def test_semantic_keeps_single_topic_together() -> None:
    """通篇一个话题就不该切开。"""
    text = "递归是函数自己调用自己。递归需要终止条件。递归要注意栈溢出。"
    chunks = split_semantic(text, embed_fn=fake_embed, chunk_size=200, threshold=0.5)
    assert len(chunks) == 1


def test_semantic_high_threshold_produces_more_chunks() -> None:
    """阈值调高 = 更敏感 = 切得更碎。"""
    text = "递归是函数自己调用自己。递归需要终止条件。二分查找要求数组有序。"
    loose = split_semantic(text, embed_fn=fake_embed, chunk_size=200, threshold=0.5)
    tight = split_semantic(text, embed_fn=fake_embed, chunk_size=200, threshold=1.1)
    assert len(tight) >= len(loose)


def test_semantic_single_sentence_returns_as_is() -> None:
    assert split_semantic("只有一句话。", embed_fn=fake_embed, chunk_size=100) == ["只有一句话。"]


def test_semantic_rejects_mismatched_embeddings() -> None:
    """向量数量和句子数量对不上，说明 embed_fn 用错了，必须报错而不是凑合。"""
    with pytest.raises(ValueError, match="向量"):
        split_semantic("甲。乙。", embed_fn=lambda texts: [], chunk_size=100)


def test_semantic_does_not_exceed_chunk_size_drastically() -> None:
    """语义切完之后仍要受块长约束，不能因为「话题整块」就无限长。"""
    text = "".join(f"递归的第{i}种用法很有意思。" for i in range(50))
    chunks = split_semantic(text, embed_fn=fake_embed, chunk_size=100, threshold=0.5)
    assert all(len(chunk) <= 100 for chunk in chunks), "超长的话题块也要再切开"


# --------------------------------------------------------------------------
# 三种策略的差异
# --------------------------------------------------------------------------


def test_strategies_differ_from_each_other() -> None:
    """三种策略在同一段文本上应当给出不同结果。

    如果它们结果完全一样，论文实验二就没得比了——这条测试等于守住
    「三种策略确实是三种」。
    """
    text = (
        "递归是函数自己调用自己。这是它的基本定义。\n\n"
        "递归必须有终止条件。否则会无限递归下去。\n\n"
        "二分查找要求数组有序。它每次把范围折半。\n\n"
        "二分查找的时间复杂度是对数级别的。"
    )

    fixed = split_fixed(text, chunk_size=40, overlap=0)
    recursive = split_recursive(text, chunk_size=40, overlap=0)
    semantic = split_semantic(text, embed_fn=fake_embed, chunk_size=40, threshold=0.5)

    assert fixed != recursive or fixed != semantic, "三种策略不该完全一样"
