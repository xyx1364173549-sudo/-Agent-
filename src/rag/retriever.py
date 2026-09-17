"""检索与重排：从一堆资料块里，找出跟问题最相关的几块。

三种检索方式，脾气各不相同：

================= ==============================================================
**向量检索**      按**语义**找。搜「循环调用自己」也能命中讲递归的段落，
                  因为它是拿向量算距离，不看字面。需要 embedding（见 vectorstore）
**关键词检索**    按**字面**找。搜什么词就得出现什么词。快、稳、不依赖模型，
                  M2.3 情景记忆里的 ``search()`` 就是这个路子
**混合检索**      两个都跑，再把结果融合。实践里通常比单用一种强
================= ==============================================================

外加一个可选步骤：

**重排（rerank）** —— 先粗筛一批候选，再用更贵的方法精排，取前几个。
RAG 里的常见套路：**召回求全，精排求准**。向量检索负责别漏，
重排负责别乱。

为什么中文要按「两个字一组」来切？因为中文词之间没有空格，而装个分词器
（jieba 之类）又要多一份依赖、多一层解释成本。用字符二元组是个务实的折中：
「递归」切成「递归」，「终止条件」切成「终止/止条/条件」，匹配效果够用，
而且零依赖、结果可复现。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.llm import create_chat_model

if TYPE_CHECKING:  # 只为类型提示，运行时不必真的去加载 chromadb
    from src.rag.vectorstore import VectorStore

# 关键词检索默认返回几条
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------- 关键词检索


def _normalize(text: str) -> str:
    """去掉空白，方便按字符切二元组。"""
    return re.sub(r"\s+", "", text)


def ngrams(text: str, size: int = 2) -> set[str]:
    """把文本切成字符 n 元组的集合。

    ``"递归不难"`` -> ``{"递归", "归不", "不难"}``（size=2）。

    注意返回的是**集合**：同一个二元组出现多次只算一次。因为我们关心的是
    「这个词组出现过没有」，而不是「出现了几次」——重复出现往往只是啰嗦，
    不代表更相关。
    """
    cleaned = _normalize(text)
    if len(cleaned) <= size:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + size] for i in range(len(cleaned) - size + 1)}


def keyword_score(chunk: str, query: str, *, size: int = 2) -> float:
    """关键词匹配度，取值 ``0.0 ~ 1.0``。

    算法很直白：**query 切出来的词组，有多大比例在 chunk 里出现过**。

    用「覆盖率」而不是「出现次数」，是为了不被长文本占便宜——一个啰嗦十倍的
    块不该因为字数多就排前面。
    """
    query_grams = ngrams(query, size)
    if not query_grams:
        return 0.0

    chunk_grams = ngrams(chunk, size)
    hits = len(query_grams & chunk_grams)
    return hits / len(query_grams)


def keyword_search(
    chunks: Sequence[str],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """按关键词排序，返回前 ``top_k`` 个。

    返回 ``[{"index": 原始下标, "text": 内容, "score": 得分}, ...]``，
    得分从高到低。得分为 0 的块会被丢掉——**宁缺毋滥**：把完全不相关的块
    塞给模型，比不给还糟，它会拿着不相关的资料硬编。

    它最大的好处是**不依赖任何模型**：没有网络、没有 API Key 也能跑，
    所以它是向量检索的天然对照组（论文实验二用得上）。
    """
    scored = [
        {"index": index, "text": chunk, "score": keyword_score(chunk, query)}
        for index, chunk in enumerate(chunks)
    ]
    scored = [item for item in scored if item["score"] > 0]
    scored.sort(key=lambda item: (-item["score"], item["index"]))
    return scored[:top_k]


# ---------------------------------------------------------------- 重排


def _build_rerank_prompt(query: str, candidates: Sequence[str], top_k: int) -> str:
    listed = "\n".join(f"[{index + 1}] {text}" for index, text in enumerate(candidates))
    return (
        f"下面有 {len(candidates)} 段候选资料，请挑出与问题最相关的 {top_k} 段。\n\n"
        f"问题：{query}\n\n"
        f"候选资料：\n{listed}\n\n"
        f"只输出最相关的 {top_k} 个编号，用逗号分隔，不要任何解释。"
        f"例如：2,5,1\n"
        f"如果都不相关，就输出：无"
    )


def _parse_indexes(reply: str, total: int, top_k: int) -> list[int]:
    """从模型回复里抠出编号。容错处理：模型可能多嘴、可能越界、可能重复。"""
    numbers = re.findall(r"\d+", reply)
    picked: list[int] = []
    for raw in numbers:
        index = int(raw) - 1  # 模型用的是从 1 开始的编号
        if 0 <= index < total and index not in picked:
            picked.append(index)
        if len(picked) >= top_k:
            break
    return picked


def rerank_with_llm(
    query: str,
    candidates: Sequence[str],
    *,
    top_k: int = 3,
    model=None,
) -> list[dict]:
    """用大模型给候选块精排，返回最相关的 ``top_k`` 个。

    流程：把候选块编号后交给模型，让它挑出最相关的几个。典型用法是
    「向量检索先召回 10 个，再精排取 3 个」——先求别漏，再求别乱。

    为什么值得多花一次调用？因为向量检索算的是「整体语义像不像」，
    而模型读得懂「这段资料到底能不能回答这个问题」。前者快而糙，后者慢而准。

    返回 ``[{"index": 原始下标, "text": 内容, "rank": 排第几}, ...]``。

    如果模型说「都不相关」，返回空列表——**允许检索结果为空**，
    这比硬凑几段不相关的资料要好。
    """
    if not candidates:
        return []
    if top_k <= 0:
        raise ValueError(f"top_k 需为正整数，实际为 {top_k}")

    chat = model or create_chat_model(temperature=0.0)
    prompt = _build_rerank_prompt(query, candidates, min(top_k, len(candidates)))
    reply = chat.invoke(prompt).content

    if isinstance(reply, list):  # 有些模型会返回内容块列表
        reply = "".join(str(part) for part in reply)

    picked = _parse_indexes(str(reply), len(candidates), min(top_k, len(candidates)))
    return [
        {"index": index, "text": candidates[index], "rank": rank + 1}
        for rank, index in enumerate(picked)
    ]


# ---------------------------------------------------------------- 结果融合


def merge_results(
    *result_groups: Sequence[dict],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """把多路检索的结果合并去重。

    混合检索的做法：向量检索给一批、关键词检索给一批，两边可能命中同一个块。
    合并规则是**同名次相加**——第 1 名得 ``1/(1+1)``，第 2 名得 ``1/(2+1)``，
    两路都排前面的块，总分会明显高出来。

    这种融合方式叫倒数排名融合（RRF），好处是**不需要把两边的分数调到同一量纲**：
    向量相似度是 0~1 的余弦值，关键词覆盖率也是 0~1，但两者含义完全不同，
    直接加权是没道理的。按名次算就绕开了这个问题。
    """
    if top_k <= 0:
        raise ValueError(f"top_k 需为正整数，实际为 {top_k}")

    scores: dict[int, float] = {}
    texts: dict[int, str] = {}
    sources: dict[int, set[str]] = {}

    for group in result_groups:
        for rank, item in enumerate(group, start=1):
            index = item["index"]
            scores[index] = scores.get(index, 0.0) + 1.0 / (rank + 1)
            texts[index] = item["text"]
            # 记下这一块是被哪一路命中的（vector / keyword）。
            # 事后能拿它分析「哪路检索更有用」——论文实验二要的正是这个。
            sources.setdefault(index, set()).add(str(item.get("source_type", "unknown")))

    merged = [
        {
            "index": index,
            "text": texts[index],
            "score": round(score, 6),
            "sources": sorted(sources[index]),
        }
        for index, score in scores.items()
    ]
    merged.sort(key=lambda item: (-item["score"], item["index"]))
    return merged[:top_k]


class Retriever:
    """统一的检索入口：把三种检索方式和重排串成一条线。

    用法::

        retriever = Retriever(VectorStore())
        retriever.search("循环调用自己是怎么回事", mode="hybrid", rerank=True)

    模式说明：

    ================ ==========================================================
    ``vector``       只走向量检索。语义强，但可能漏掉字面精确匹配的内容
    ``keyword``      只走关键词检索。字面准，但换个说法就搜不到
    ``hybrid``       两路都跑再融合。实践里通常最好，也是默认值
    ================ ==========================================================

    注意上面这句话「混合通常更好」是**要靠实验验证的**，不是结论——
    论文实验二正好拿这三种模式做横向对比，用数据说话。
    """

    def __init__(self, store: "VectorStore") -> None:
        self.store = store

    def _vector_group(self, query: str, top_k: int) -> list[dict]:
        """向量检索，并把结果对齐到「全库下标」。

        为什么要对齐下标？因为融合两路结果时需要统一的标识来判重——
        向量库给的是 id，关键词检索给的是列表下标，不统一就没法比较。
        """
        chunks = self.store.all_chunks()
        index_of = {chunk["id"]: position for position, chunk in enumerate(chunks)}

        group: list[dict] = []
        for hit in self.store.search(query, top_k=top_k):
            position = index_of.get(hit["id"])
            if position is None:
                continue
            group.append(
                {
                    "index": position,
                    "text": hit["text"],
                    "score": hit["score"],
                    "source": hit["source"],
                    "source_type": "vector",
                }
            )
        return group

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        model=None,
    ) -> list[dict]:
        """检索。

        ``rerank=True`` 会在粗筛之后再用大模型精排一次。默认关着，因为它
        **要多花一次模型调用**——典型用法是「混合检索召回 10 条 → 重排取 3 条」，
        只在结果质量比响应速度更重要时才开。
        """
        if mode not in {"vector", "keyword", "hybrid"}:
            raise ValueError(f"不支持的检索模式 {mode!r}，可选：vector / keyword / hybrid")
        if top_k <= 0:
            raise ValueError(f"top_k 需为正整数，实际为 {top_k}")

        chunks = self.store.all_chunks()
        if not chunks or not query.strip():
            return []
        texts = [chunk["text"] for chunk in chunks]

        if mode == "keyword":
            results = keyword_search(texts, query, top_k=top_k)
            for item in results:
                item["source_type"] = "keyword"
                item["source"] = chunks[item["index"]]["source"]

        elif mode == "vector":
            results = self._vector_group(query, top_k)

        else:  # hybrid：两路各多取一些，融合后再截断
            pool = top_k * 2
            keyword_group = keyword_search(texts, query, top_k=pool)
            for item in keyword_group:
                item["source_type"] = "keyword"

            results = merge_results(
                self._vector_group(query, pool),
                keyword_group,
                top_k=top_k,
            )
            for item in results:
                item["source"] = chunks[item["index"]]["source"]

        if rerank and results:
            picked = rerank_with_llm(
                query,
                [item["text"] for item in results],
                top_k=top_k,
                model=model,
            )
            # 重排只返回「挑中的几条」，这里把原来的下标、来源补回去
            by_text = {item["text"]: item for item in results}
            results = [
                {**by_text[item["text"]], "rank": item["rank"]}
                for item in picked
                if item["text"] in by_text
            ]

        return results


def to_json(data: object) -> str:
    """调试用：把结果打成 JSON 字符串，避免直接 print 长文本。"""
    return json.dumps(data, ensure_ascii=False, indent=2)
