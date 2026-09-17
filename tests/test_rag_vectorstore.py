"""向量库测试。

重点盯四件事：

1. **幂等** —— 同一份资料重复入库不能堆出重复块，否则检索结果里全是复制品；
2. **语义命中** —— 这是向量检索存在的全部理由：字面不同、意思相同也要能搜到；
3. **得分方向** —— 统一成「越大越相关」。Chroma 原始返回的是距离（越小越相关），
   如果这里忘了翻转，后面融合多路结果时就会把最不相关的排到第一；
4. **持久化** —— 关掉重开数据还在，否则每次启动都要重新入库。

这些测试依赖 chromadb 与它自带的本地向量模型（首次会自动下载，之后离线可用）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.vectorstore import VectorStore, make_chunk_id

SAMPLE_CHUNKS = [
    "递归是函数自己调用自己，必须有终止条件",
    "二分查找要求数组有序，每次把范围折半",
    "Python 的列表推导式可以快速生成列表",
]


@pytest.fixture
def store(tmp_path: Path):
    """一个独立的临时向量库。"""
    return VectorStore(persist_dir=tmp_path / "chroma", collection_name="test_collection")


# --------------------------------------------------------------------------
# 块 id 的生成
# --------------------------------------------------------------------------


def test_chunk_id_is_stable() -> None:
    """同样内容必须算出同样 id，这是「幂等入库」的基础。"""
    assert make_chunk_id("一样的内容", "a.md") == make_chunk_id("一样的内容", "a.md")


def test_chunk_id_differs_by_content() -> None:
    assert make_chunk_id("内容甲", "a.md") != make_chunk_id("内容乙", "a.md")


def test_chunk_id_differs_by_source() -> None:
    """同样一句话来自不同文件，应当算两个块——它们出处不同，引用时要分开。"""
    assert make_chunk_id("同一句话", "a.md") != make_chunk_id("同一句话", "b.md")


def test_chunk_id_is_short_hex() -> None:
    chunk_id = make_chunk_id("内容")
    assert len(chunk_id) == 16
    assert all(char in "0123456789abcdef" for char in chunk_id)


# --------------------------------------------------------------------------
# 写入与幂等
# --------------------------------------------------------------------------


def test_starts_empty(store: VectorStore) -> None:
    assert store.count() == 0


def test_add_returns_written_count(store: VectorStore) -> None:
    assert store.add_chunks(SAMPLE_CHUNKS, source="kb.md") == 3
    assert store.count() == 3


def test_repeated_add_is_idempotent(store: VectorStore) -> None:
    """同样的内容重复入库两次，库里仍然只有一份。

    资料目录反复构建知识库是常态，如果每次入库都堆一份，
    检索结果里就全是复制品，白白挤掉真正不同的内容。
    """
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    assert store.count() == 3


def test_duplicates_within_one_batch_collapsed(store: VectorStore) -> None:
    """同一批里的重复也得去掉——否则 Chroma 会因为 id 重复直接报错。"""
    assert store.add_chunks(["重复内容", "重复内容", "另一条"]) == 2
    assert store.count() == 2


def test_same_text_from_different_sources_are_distinct(store: VectorStore) -> None:
    store.add_chunks(["相同的一段话"], source="a.md")
    store.add_chunks(["相同的一段话"], source="b.md")
    assert store.count() == 2


def test_blank_chunks_are_skipped(store: VectorStore) -> None:
    """空块算出来的向量没有意义，只会污染检索结果。"""
    assert store.add_chunks(["有内容", "", "   ", "\n\n"]) == 1
    assert store.count() == 1


def test_add_empty_list_returns_zero(store: VectorStore) -> None:
    assert store.add_chunks([]) == 0


def test_chunks_are_stripped(store: VectorStore) -> None:
    """首尾空白要清掉，否则「内容」和「内容  」会被当成两个不同的块。"""
    store.add_chunks(["  带空白的块  "])
    store.add_chunks(["带空白的块"])
    assert store.count() == 1


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------


def test_semantic_match_across_different_wording(store: VectorStore) -> None:
    """**这条是向量检索存在的全部理由**。

    问题「循环调用自己是怎么回事」里，没有任何一个块含这些字，
    关键词检索会一条都搜不到（M2.3 情景记忆就是这样），
    但向量检索能靠语义把「递归」那段找出来。
    """
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")

    hits = store.search("循环调用自己是怎么回事", top_k=3)

    assert hits, "应当至少命中一条"
    assert "递归" in hits[0]["text"], "最相关的应该是讲递归的那块"


def test_search_returns_descending_scores(store: VectorStore) -> None:
    """得分必须从高到低，而且越大越相关。

    忘了把 Chroma 的「距离」翻转成「相似度」，就会把最不相关的排第一 ——
    这条测试就是守着这个方向的。
    """
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")

    hits = store.search("递归的终止条件", top_k=3)
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0]["score"] > hits[-1]["score"]


def test_search_respects_top_k(store: VectorStore) -> None:
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    assert len(store.search("递归", top_k=2)) == 2


def test_search_top_k_larger_than_library(store: VectorStore) -> None:
    """要 10 条但库里只有 3 条时，返回 3 条就行，不该报错。"""
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    assert len(store.search("递归", top_k=10)) == 3


def test_search_on_empty_store(store: VectorStore) -> None:
    """空库检索返回空列表——下游会直接遍历它。"""
    assert store.search("随便问点什么") == []


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_search_with_blank_query(store: VectorStore, blank: str) -> None:
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    assert store.search(blank) == []


@pytest.mark.parametrize("bad", [0, -1])
def test_search_rejects_bad_top_k(store: VectorStore, bad: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        store.search("递归", top_k=bad)


def test_search_carries_source_metadata(store: VectorStore) -> None:
    """结果要带来源，能追溯这段话是从哪份资料里找出来的。"""
    store.add_chunks(["递归的终止条件是递归的出口"], source="算法笔记.md")

    hit = store.search("递归什么时候停", top_k=1)[0]
    assert hit["source"] == "算法笔记.md"
    assert hit["id"]


# --------------------------------------------------------------------------
# 持久化与清空
# --------------------------------------------------------------------------


def test_persists_across_reopen(tmp_path: Path) -> None:
    """关掉重开数据还在，否则每次启动都要重新入库。"""
    first = VectorStore(persist_dir=tmp_path / "chroma", collection_name="persist_test")
    first.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    assert first.count() == 3

    second = VectorStore(persist_dir=tmp_path / "chroma", collection_name="persist_test")
    assert second.count() == 3, "重开之后数据应当还在"
    assert second.search("递归", top_k=1)


def test_collections_are_isolated(tmp_path: Path) -> None:
    """同一个目录下的不同集合互不干扰，用来隔离不同知识库。"""
    first = VectorStore(persist_dir=tmp_path / "chroma", collection_name="collection_a")
    second = VectorStore(persist_dir=tmp_path / "chroma", collection_name="collection_b")

    first.add_chunks(["甲库的内容"])
    assert first.count() == 1
    assert second.count() == 0


def test_clear_empties_collection(store: VectorStore) -> None:
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    store.clear()
    assert store.count() == 0
    assert store.search("递归") == []


def test_usable_after_clear(store: VectorStore) -> None:
    """清空之后还要能继续用——clear 是「删掉再建」，不能把集合弄成坏的。"""
    store.add_chunks(SAMPLE_CHUNKS, source="kb.md")
    store.clear()
    store.add_chunks(["清空之后新加的内容"], source="new.md")
    assert store.count() == 1


def test_creates_persist_dir(tmp_path: Path) -> None:
    """目录不存在时要自动创建，否则全新环境下第一次入库就报错。"""
    target = tmp_path / "nested" / "chroma"
    VectorStore(persist_dir=target, collection_name="auto_dir_test")
    assert target.is_dir()
