"""检索增强生成四件套（对应计划 M3、论文第 6 章）。

流程::

    加载（loader）→ 切分（splitter）→ 向量化与索引（vectorstore）→ 检索与重排（retriever）

- ``loader``      把 md/txt/pdf 读成干净文本
- ``splitter``    三种切分策略：固定长度 / 递归 / 语义
- ``vectorstore`` 向量化并存入 Chroma（本地 ONNX 模型，无需额外 API）
- ``retriever``   向量 / 关键词 / 混合检索 + 大模型重排

最短用法::

    from src.rag import VectorStore, Retriever, load_directory, split_recursive

    chunks = []
    for doc in load_directory("data/knowledge"):
        chunks += split_recursive(doc["text"])

    store = VectorStore()
    store.add_chunks(chunks, source="知识库")
    for hit in Retriever(store).search("循环调用自己是怎么回事"):
        print(hit["score"], hit["text"])
"""

from src.rag.loader import clean_text, load_directory, load_text
from src.rag.retriever import (
    Retriever,
    keyword_score,
    keyword_search,
    merge_results,
    rerank_with_llm,
)
from src.rag.splitter import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    split_fixed,
    split_recursive,
    split_semantic,
)
from src.rag.vectorstore import VectorStore, make_chunk_id

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "Retriever",
    "VectorStore",
    "clean_text",
    "keyword_score",
    "keyword_search",
    "load_directory",
    "load_text",
    "make_chunk_id",
    "merge_results",
    "rerank_with_llm",
    "split_fixed",
    "split_recursive",
    "split_semantic",
]
