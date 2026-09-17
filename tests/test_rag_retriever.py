"""检索与重排测试。

重点盯三块：

1. **关键词打分的尺度** —— 用「覆盖率」而非「出现次数」，长文本不该占便宜；
2. **融合会不会重复计数** —— 多路检索命中同一个块时必须合并而不是各算各的；
3. **重排的容错** —— 模型回复可能多嘴、编号越界、说「无」，
   解析器要扛得住，不能把 1 号当 11 号、也不能因为越界就崩。

重排部分用假模型，完全离线；这样测的是**解析与排序逻辑**，不是模型水平。
"""

from __future__ import annotations

import pytest

from src.rag.retriever import (
    Retriever,
    keyword_score,
    keyword_search,
    merge_results,
    ngrams,
    rerank_with_llm,
)


class FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """假模型：invoke 时固定返回预先设定好的内容。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeReply:
        self.prompts.append(prompt)
        return FakeReply(self.content)


# --------------------------------------------------------------------------
# 二元组切分
# --------------------------------------------------------------------------


def test_ngrams_basic() -> None:
    assert ngrams("递归不难") == {"递归", "归不", "不难"}


def test_ngrams_ignores_whitespace() -> None:
    """空白不该影响匹配——「递归 不难」和「递归不难」应当切出一样的结果。"""
    assert ngrams("递归 不难") == ngrams("递归不难")


def test_ngrams_short_text() -> None:
    assert ngrams("递") == {"递"}


def test_ngrams_empty() -> None:
    assert ngrams("") == set()
    assert ngrams("   ") == set()


# --------------------------------------------------------------------------
# 关键词打分
# --------------------------------------------------------------------------


def test_full_match_scores_one() -> None:
    assert keyword_score("递归是函数自己调用自己", "递归") == pytest.approx(1.0)


def test_no_match_scores_zero() -> None:
    assert keyword_score("二分查找要求有序", "递归") == 0.0


def test_partial_match_between_zero_and_one() -> None:
    score = keyword_score("递归需要终止条件", "递归终止")
    assert 0.0 < score < 1.0


def test_long_chunk_does_not_get_advantage() -> None:
    """用覆盖率而不是出现次数，长文本不该因为字数多就得分更高。"""
    short = "递归的终止条件很重要"
    long = short + "。" + "这里是大量与问题无关的填充文字。" * 20
    assert keyword_score(long, "递归的终止条件") <= keyword_score(short, "递归的终止条件")


def test_empty_query_scores_zero() -> None:
    assert keyword_score("任意内容", "") == 0.0


# --------------------------------------------------------------------------
# 关键词检索
# --------------------------------------------------------------------------


def test_keyword_search_orders_by_score() -> None:
    chunks = [
        "二分查找要求数组有序",  # 与问题无关
        "递归需要终止条件，否则会无限递归",  # 高度相关
        "递归是一种函数调用自身的方法",  # 相关
    ]
    results = keyword_search(chunks, "递归终止条件")

    assert results[0]["index"] == 1
    assert results[0]["score"] > results[-1]["score"]


def test_keyword_search_drops_zero_scores() -> None:
    """完全不相关的块要丢掉——宁缺毋滥，塞给模型反而会让它硬编。"""
    chunks = ["完全无关的内容", "另一个无关话题"]
    assert keyword_search(chunks, "递归") == []


def test_keyword_search_respects_top_k() -> None:
    chunks = [f"递归的第{i}种用法" for i in range(10)]
    assert len(keyword_search(chunks, "递归", top_k=3)) == 3


def test_keyword_search_on_empty_input() -> None:
    assert keyword_search([], "递归") == []


def test_keyword_search_returns_original_index() -> None:
    """返回的下标要能指回原始列表，否则没法从向量库里取回对应文档。"""
    chunks = ["无关", "递归相关", "无关二"]
    results = keyword_search(chunks, "递归")
    assert results[0]["index"] == 1
    assert results[0]["text"] == "递归相关"


# --------------------------------------------------------------------------
# 结果融合
# --------------------------------------------------------------------------


def test_merge_deduplicates() -> None:
    """两路命中同一个块，只能算一条，并且要记下「两路都命中了它」。"""
    group_a = [{"index": 0, "text": "甲", "source_type": "vector"}]
    group_b = [{"index": 0, "text": "甲", "source_type": "keyword"}]

    merged = merge_results(group_a, group_b)
    assert len(merged) == 1
    assert merged[0]["sources"] == ["keyword", "vector"]


def test_merge_rewards_agreement() -> None:
    """两路都排在前面的块，总得分要高于只被一路命中的块。"""
    group_a = [{"index": 1, "text": "两路都命中"}, {"index": 2, "text": "只有向量命中"}]
    group_b = [{"index": 1, "text": "两路都命中"}]

    merged = merge_results(group_a, group_b)
    assert merged[0]["index"] == 1
    assert merged[0]["score"] > merged[1]["score"]


def test_merge_respects_top_k() -> None:
    group = [{"index": i, "text": f"块{i}"} for i in range(10)]
    assert len(merge_results(group, top_k=3)) == 3


def test_merge_with_no_input() -> None:
    assert merge_results() == []


@pytest.mark.parametrize("bad", [0, -1])
def test_merge_rejects_bad_top_k(bad: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        merge_results([{"index": 0, "text": "甲"}], top_k=bad)


def test_merge_result_is_sorted() -> None:
    group_a = [{"index": 0, "text": "甲"}, {"index": 1, "text": "乙"}]
    group_b = [{"index": 1, "text": "乙"}]

    merged = merge_results(group_a, group_b)
    scores = [item["score"] for item in merged]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# 重排
# --------------------------------------------------------------------------


def test_rerank_picks_in_model_order() -> None:
    model = FakeModel("2,1")
    candidates = ["第一段资料", "第二段资料"]

    result = rerank_with_llm("问题", candidates, top_k=2, model=model)

    assert [item["index"] for item in result] == [1, 0]
    assert result[0]["rank"] == 1
    assert result[0]["text"] == "第二段资料"


def test_rerank_ignores_out_of_range_indexes() -> None:
    """模型给越界编号时必须丢掉，不能把 9 号当有效索引去取列表。"""
    model = FakeModel("1,9,99")
    candidates = ["甲", "乙"]

    result = rerank_with_llm("问题", candidates, top_k=3, model=model)
    assert [item["index"] for item in result] == [0]


def test_rerank_handles_duplicate_indexes() -> None:
    model = FakeModel("1,1,2")
    candidates = ["甲", "乙"]

    result = rerank_with_llm("问题", candidates, top_k=3, model=model)
    assert [item["index"] for item in result] == [0, 1]


def test_rerank_returns_empty_when_model_says_none() -> None:
    """模型说「都不相关」时返回空——允许检索为空，好过硬凑不相关的内容。"""
    model = FakeModel("无")
    assert rerank_with_llm("问题", ["甲", "乙"], model=model) == []


def test_rerank_handles_empty_candidates() -> None:
    """没有候选就不该去调模型，直接返回空。"""
    model = FakeModel("1")
    assert rerank_with_llm("问题", [], model=model) == []
    assert model.prompts == [], "不该浪费一次调用"


def test_rerank_prompt_lists_candidates_with_numbers() -> None:
    model = FakeModel("1")
    rerank_with_llm("递归怎么学", ["第一段", "第二段"], top_k=1, model=model)

    prompt = model.prompts[0]
    assert "[1] 第一段" in prompt
    assert "[2] 第二段" in prompt
    assert "递归怎么学" in prompt


def test_rerank_respects_top_k() -> None:
    model = FakeModel("4,3,2,1")
    candidates = ["甲", "乙", "丙", "丁"]

    result = rerank_with_llm("问题", candidates, top_k=2, model=model)
    assert len(result) == 2


@pytest.mark.parametrize("bad", [0, -1])
def test_rerank_rejects_bad_top_k(bad: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        rerank_with_llm("问题", ["甲"], top_k=bad, model=FakeModel("1"))


def test_rerank_tolerates_chatty_reply() -> None:
    """模型可能多说两句，要从一堆文字里把编号捞出来。"""
    model = FakeModel("我认为最相关的是第 2 段，其次是第 1 段。")
    candidates = ["甲", "乙"]

    result = rerank_with_llm("问题", candidates, top_k=2, model=model)
    assert [item["index"] for item in result] == [1, 0]


# --------------------------------------------------------------------------
# Retriever 编排
# --------------------------------------------------------------------------


class FakeStore:
    """假向量库：直接返回预设结果，绕开 chromadb 与本地模型。

    这样测的是 Retriever 的**编排逻辑**（模式分派、下标对齐、结果补全），
    而不是向量库本身——那部分在 test_rag_vectorstore.py 里单独测。
    """

    def __init__(self, chunks: list[dict], hits: list[dict] | None = None) -> None:
        self._chunks = chunks
        self._hits = hits if hits is not None else []

    def all_chunks(self) -> list[dict]:
        return list(self._chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[dict]:
        return list(self._hits[:top_k])


CHUNKS = [
    {"id": "c0", "text": "二分查找要求数组有序，每次把范围折半", "source": "算法.md"},
    {"id": "c1", "text": "递归是函数自己调用自己，必须有终止条件", "source": "算法.md"},
    {"id": "c2", "text": "Python 列表推导式可以一行生成列表", "source": "语法.md"},
]

# 假装向量检索只看语义，认为 c1 最相关
VECTOR_HITS = [
    {"id": "c1", "text": CHUNKS[1]["text"], "score": 0.93, "source": "算法.md"},
    {"id": "c0", "text": CHUNKS[0]["text"], "score": 0.61, "source": "算法.md"},
]


def test_keyword_mode_uses_literal_match() -> None:
    retriever = Retriever(FakeStore(CHUNKS))

    results = retriever.search("递归终止条件", mode="keyword", top_k=3)

    assert results
    assert results[0]["index"] == 1
    assert results[0]["source_type"] == "keyword"


def test_vector_mode_uses_store_hits() -> None:
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))

    results = retriever.search("循环调用自己", mode="vector", top_k=2)

    assert [item["index"] for item in results] == [1, 0]
    assert results[0]["source_type"] == "vector"


def test_vector_mode_aligns_to_chunk_index() -> None:
    """向量库给的是 id，要换成全库下标才能和关键词结果对齐去重。"""
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))

    results = retriever.search("循环调用自己", mode="vector", top_k=2)

    assert results[0]["index"] == 1
    assert results[0]["source"] == "算法.md"


def test_hybrid_mode_merges_both_paths() -> None:
    """混合模式下两路都要有贡献。

    「递归」这个词在 c1 里出现过，所以关键词那路也会命中它；
    向量那路本来就把它排第一。两路重叠的块应当被标出来。
    """
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))

    results = retriever.search("递归", mode="hybrid", top_k=3)

    assert results
    hit_indexes = {item["index"] for item in results}
    assert 1 in hit_indexes, "讲递归的那块必须在结果里"

    merged_hit = next(item for item in results if item["index"] == 1)
    assert merged_hit["sources"] == ["keyword", "vector"], "应当记录两路都命中了它"


def test_hybrid_marks_sources() -> None:
    """每个结果都要标明它被哪几路命中，方便事后分析哪路更有用。"""
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))

    results = retriever.search("递归", mode="hybrid", top_k=3)
    assert all("sources" in item for item in results)


def test_hybrid_respects_top_k() -> None:
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))
    assert len(retriever.search("递归", mode="hybrid", top_k=1)) == 1


def test_invalid_mode_rejected() -> None:
    retriever = Retriever(FakeStore(CHUNKS))
    with pytest.raises(ValueError, match="检索模式"):
        retriever.search("递归", mode="随便")


@pytest.mark.parametrize("bad", [0, -1])
def test_retriever_rejects_bad_top_k(bad: int) -> None:
    retriever = Retriever(FakeStore(CHUNKS))
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("递归", top_k=bad)


def test_empty_store_returns_empty() -> None:
    assert Retriever(FakeStore([])).search("递归") == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_query_returns_empty(blank: str) -> None:
    assert Retriever(FakeStore(CHUNKS, VECTOR_HITS)).search(blank) == []


def test_rerank_disabled_by_default() -> None:
    """重排要多花一次模型调用，默认不能悄悄开。"""
    model = FakeModel("2,1")
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))

    results = retriever.search("递归", mode="vector", top_k=2, model=model)

    assert model.prompts == [], "没开 rerank 就不该调模型"
    assert results


def test_rerank_reorders_results() -> None:
    """开启重排后，顺序应当以模型的判断为准。"""
    model = FakeModel("2,1")
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))

    results = retriever.search("递归", mode="vector", top_k=2, rerank=True, model=model)

    assert len(model.prompts) == 1, "应当只调一次模型"
    assert results[0]["rank"] == 1
    assert results[0]["index"] == 0, "模型说第 2 段最相关"


def test_rerank_keeps_original_metadata() -> None:
    """重排之后，下标和来源这些信息不能丢。"""
    model = FakeModel("1")
    retriever = Retriever(FakeStore(CHUNKS, VECTOR_HITS))

    results = retriever.search("递归", mode="vector", top_k=2, rerank=True, model=model)

    assert "source" in results[0]
    assert "index" in results[0]
