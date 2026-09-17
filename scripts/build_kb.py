"""知识库构建脚本：把资料目录灌进向量库。

用法::

    python scripts/build_kb.py data/knowledge                      # 默认递归切分
    python scripts/build_kb.py data/knowledge --strategy fixed --chunk-size 300
    python scripts/build_kb.py data/knowledge --strategy semantic
    python scripts/build_kb.py data/knowledge --reset              # 先清空再入库

为什么要脚本化，而不是每次现写几行

    因为论文实验二要跑「三种切分 × 三种检索」的交叉对比，同一批资料
    要用不同参数反复入库。脚本化的好处是**可复现**——同样的命令跑两次，
    结果完全一样，实验数据才站得住。

构建逻辑被抽成 ``build()`` 函数而不是全塞在 ``main()`` 里，是为了让它
**能被测试调用**；不然这个脚本只能靠人肉跑一遍来验证，改了也不知道有没有坏。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import (  # noqa: E402
    VectorStore,
    load_directory,
    split_fixed,
    split_recursive,
    split_semantic,
)

# Windows 控制台默认 GBK，打印中文可能报编码错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 可选的切分策略。语义切分需要向量模型，所以单独处理
SIMPLE_STRATEGIES = ("fixed", "recursive")
ALL_STRATEGIES = SIMPLE_STRATEGIES + ("semantic",)

# 语义切分要调向量模型，实例化一次就够，反复创建会反复加载模型
_embedder = None


def _embed(texts: list[str]) -> list[list[float]]:
    """给语义切分用的向量化函数，复用 Chroma 自带的本地模型。

    这样切分和检索用的是**同一个**向量模型，语义判断的标准一致。
    """
    global _embedder
    if _embedder is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _embedder = DefaultEmbeddingFunction()
    return [list(vector) for vector in _embedder(list(texts))]


def build(
    source: str | Path,
    *,
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
    reset: bool = False,
    store: VectorStore | None = None,
) -> dict[str, int | str]:
    """读目录 → 切分 → 入库，返回这次构建的统计。

    按文件逐个入库，并把**文件名作为来源标记**——检索结果里能看到
    「这段话出自哪份资料」，写论文引用时用得上。
    """
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"未知切分策略 {strategy!r}，可选：{' / '.join(ALL_STRATEGIES)}")
    if strategy in SIMPLE_STRATEGIES and overlap >= chunk_size:
        # 提前拦下来给人话，不然会从切分器里抛出一个技术味很重的错误
        raise ValueError(
            f"overlap（{overlap}）必须小于 chunk_size（{chunk_size}），否则切分原地打转。"
            f"把 --overlap 调小，或把 --chunk-size 调大"
        )

    documents = load_directory(source)
    stats: dict[str, int | str] = {
        "documents": len(documents),
        "chunks": 0,
        "written": 0,
        "strategy": strategy,
    }
    if not documents:
        return stats

    vector_store = store or VectorStore()
    if reset:
        vector_store.clear()

    for document in documents:
        text = document["text"]
        if strategy == "fixed":
            chunks = split_fixed(text, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "recursive":
            chunks = split_recursive(text, chunk_size=chunk_size, overlap=overlap)
        else:
            chunks = split_semantic(text, embed_fn=_embed, chunk_size=chunk_size)

        stats["chunks"] += len(chunks)
        stats["written"] += vector_store.add_chunks(chunks, source=document["path"])

    stats["total_in_store"] = vector_store.count()
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把资料目录灌进向量库")
    parser.add_argument("source", help="资料目录（会递归子目录）")
    parser.add_argument(
        "--strategy",
        choices=ALL_STRATEGIES,
        default="recursive",
        help="切分策略，默认 recursive",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="块大小（字符数）")
    parser.add_argument("--overlap", type=int, default=50, help="块间重叠（字符数）")
    parser.add_argument("--reset", action="store_true", help="入库前先清空向量库")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_dir():
        print(f"[FAIL] 资料目录不存在：{source}")
        return 1

    print(f"资料目录：{source}")
    print(f"切分策略：{args.strategy}（块大小 {args.chunk_size}，重叠 {args.overlap}）")
    if args.reset:
        print("模式：先清空再入库")

    stats = build(
        source,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        reset=args.reset,
    )

    print()
    if stats["documents"] == 0:
        print("[WARN] 目录里没有可用的文档（支持 md / txt / pdf）")
        return 0

    print(f"读到文档：{stats['documents']} 份")
    print(f"切成块数：{stats['chunks']} 块")
    print(f"实际写入：{stats['written']} 块（重复内容会被幂等合并，所以可能少于切分出的块数）")
    print(f"库中总量：{stats['total_in_store']} 块")
    print()
    print("[OK] 完成。可以这样试检索：")
    print('    python -c "from src.rag import VectorStore, Retriever; print(Retriever(VectorStore()).search(\'你的问题\'))"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
