"""检索增强生成四件套（对应计划 M3、论文第 6 章）。

流程：

    加载（loader）→ 切分（splitter）→ 向量化与索引（vectorstore）→ 检索与重排（retriever）

- ``loader``      把 md/txt/pdf 读成干净文本
- ``splitter``    三种切分策略：固定长度 / 递归 / 语义
- ``vectorstore`` 向量化并存入 Chroma（待实现，M3.3）
- ``retriever``   向量 / 关键词 / 混合检索 + 重排（待实现，M3.4）
"""

from src.rag.loader import clean_text, load_directory, load_text
from src.rag.splitter import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    split_fixed,
    split_recursive,
    split_semantic,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "clean_text",
    "load_directory",
    "load_text",
    "split_fixed",
    "split_recursive",
    "split_semantic",
]
