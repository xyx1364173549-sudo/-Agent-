"""向量化与 Chroma 索引。

把切好的文本块变成向量、存进 Chroma，之后就能按**语义**检索——
搜「循环调用自己」也能命中讲递归的段落，这是关键词检索做不到的。

为什么用本地模型而不是调 API（答辩很可能被问到这个）

    因为 **DeepSeek 不提供 embedding 接口**。查过它的模型列表，只有
    ``deepseek-flash`` 和 ``deepseek-v4-pro`` 两个对话模型，没有向量化服务。
    所以这里用 Chroma 自带的本地 ONNX 模型：

    ================ ====================================================
    好处             完全本地、免费、不需要额外申请 Key、不受网络波动影响
    代价             模型偏小，中文效果不如专门的中文向量模型；
                     但对本项目的检索需求与对比实验来说够用
    ================ ====================================================

    首次使用会自动下载模型文件（约 80MB），之后离线可用、可复现。

**实测的能力边界**（别把「语义检索」想得太神）

    拿这个模型做过对照测试，结论是：它能处理「词形相近」的替换，
    但处理不了「字面完全不同、纯靠语义」的对应关系。

    ======================================= ========================================
    查询「函数什么时候该停下来」              命中 ✓（原文写的是「必须给出停止的时刻」）
    查询「自我引用是什么意思」                命中 ✓（原文含「自我引用」）
    查询「循环调用自己是怎么回事」            未命中 ✗（原文写的是「自我引用」）
    ======================================= ========================================

    原因很直白：``all-MiniLM-L6-v2`` 是**英文模型**，中文只能算「勉强能用」。
    这恰恰说明**向量模型的质量直接决定检索效果的上限**——换成中文专用模型
    （BGE、text2vec 之类）会明显更好。这是论文里可以写的一条改进方向，
    也已用 ``xfail`` 测试把这条边界固定下来（见 test_rag_build_kb.py）。

增量更新怎么做的

    每个块的 id 由「来源 + 内容」的哈希生成，所以**同一份资料重复入库
    不会产生重复块**：内容没变，id 就没变，Chroma 会覆盖而不是新增。
    资料改了，哈希变了，就自然变成一条新记录——不需要自己比对差异。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import chromadb

from src.config import get_settings

# collection 名字。注意 chromadb 要求 3~512 个字符、且只能用
# 字母数字和 _ . -（"kb" 这种两个字符的名字会被直接拒绝）
DEFAULT_COLLECTION = "knowledge_base"

# 默认返回几条
DEFAULT_TOP_K = 5


def make_chunk_id(text: str, source: str = "") -> str:
    """给一个块算 id：来源 + 内容的 MD5 前 16 位。

    用哈希而不是自增序号，是为了**幂等**——同一份资料入库两次，
    两次算出的 id 完全一样，Chroma 就会覆盖而不是堆重复。
    这也是「增量更新」能这么简单的全部秘密。
    """
    digest = hashlib.md5(f"{source}\n{text}".encode("utf-8")).hexdigest()
    return digest[:16]


class VectorStore:
    """Chroma 向量库的封装，只管存和按语义取。

    用法::

        store = VectorStore()
        store.add_chunks(["递归是函数自己调用自己", "二分查找要求有序"])
        for hit in store.search("循环调用是怎么回事"):
            print(hit["score"], hit["text"])
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        参数
        ----
        persist_dir:
            向量库落盘目录。不传时用 ``.env`` 里的 ``CHROMA_PERSIST_DIR``
            （默认 ``./chroma_db``）。目录会自动创建。
        collection_name:
            集合名。同一个目录下可以有多个集合，用来隔离不同的知识库。
        """
        path = Path(persist_dir) if persist_dir else get_settings().chroma_persist_dir
        path.mkdir(parents=True, exist_ok=True)

        self.persist_dir = path
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    # ---------- 写入 ----------

    def add_chunks(
        self,
        chunks: Sequence[str],
        *,
        source: str = "unknown",
    ) -> int:
        """把一批文本块写进库里，返回**实际写入的条数**。

        ``source`` 用来标记这批块来自哪个文件，检索结果里会带回来，
        方便追溯「这段话是从哪份资料里找出来的」。
        空的块会被跳过——空文本算出来的向量没有意义，只会污染检索结果。
        """
        texts = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        if not texts:
            return 0

        # 同一批里如果有重复内容，先去重，否则 Chroma 会因为 id 重复报错
        unique: dict[str, str] = {make_chunk_id(text, source): text for text in texts}

        self._collection.upsert(
            ids=list(unique.keys()),
            documents=list(unique.values()),
            metadatas=[{"source": source} for _ in unique],
        )
        return len(unique)

    # ---------- 读取 ----------

    def search(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """按语义检索，返回最相关的若干块。

        返回 ``[{"id":..., "text":..., "score":..., "source":...}, ...]``，
        按相关度从高到低。

        ``score`` 是**相似度**（``1 - 距离``），越接近 1 越相关。
        Chroma 原始返回的是距离（越小越相关），这里转成相似度，
        是为了和关键词检索的得分方向一致——都是「越大越相关」，
        融合时不容易搞反。
        """
        if top_k <= 0:
            raise ValueError(f"top_k 需为正整数，实际为 {top_k}")
        if not query.strip() or self.count() == 0:
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict] = []
        for index, text in enumerate(documents):
            distance = distances[index] if index < len(distances) else 0.0
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            hits.append(
                {
                    "id": ids[index] if index < len(ids) else "",
                    "text": text,
                    "score": round(1.0 - float(distance), 6),
                    "source": metadata.get("source", "unknown"),
                }
            )
        return hits

    def count(self) -> int:
        """库里现在有多少个块。"""
        return self._collection.count()

    def all_chunks(self) -> list[dict]:
        """取出库里全部块。

        关键词检索必须遍历全库才能算覆盖率——它不像向量检索那样有近似索引可用。
        当前规模（几百到几千块）完全没问题；资料量真的很大时，
        这里该换成 SQLite FTS 之类的全文索引。
        """
        if self.count() == 0:
            return []

        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        chunks: list[dict] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            chunks.append(
                {
                    "id": ids[index] if index < len(ids) else "",
                    "text": text,
                    "source": metadata.get("source", "unknown"),
                }
            )
        return chunks

    # ---------- 维护 ----------

    def clear(self) -> None:
        """清空这个集合。

        做法是「删掉再建」而不是逐条删——Chroma 没有清空集合的原子操作，
        重建是最省事也最不容易出错的方式。
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)
