"""文本切分：三种策略。

为什么非要切？因为向量检索的单位是「块」（chunk），不是整篇文档。

    块太大：语义被稀释——一段话里混了十来个话题，算出来的向量是个四不像，
            搜什么都匹配不准；
    块太小：上下文丢失——检索到「终止条件」四个字，却不知道说的是什么的终止条件。

所以切分策略直接影响检索质量。这也正是论文实验二要横向对比的东西：
不同的切分方式，检索命中率差多少。

三种策略：

=============== =========================================================
``fixed``       每 N 个字符切一刀。最笨，但**作为基线必须留着**——
                不跟笨办法比，就说不清「讲究的切分到底带来多少提升」
``recursive``   按分隔符优先级（段落 → 换行 → 句末标点 → 空格）逐级找切点，
                尽量不切断句子。实践中最常用
``semantic``    用向量判断相邻句子的相似度，在「话题变了」的地方下刀。
                需要外部提供 embedding 函数（M3.3 的向量化负责）
=============== =========================================================

重叠（overlap）：相邻块之间留一段重复内容，避免刚好把一句话切在两个块的交界处，
导致两边都读不完整。``fixed`` 和 ``recursive`` 用得上；``semantic`` 不用——
它本来就切在话题边界上，再加重叠反而把两个话题又揉到一起了。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

# 默认块大小（字符数）。500 字上下是中文资料里比较通用的取值：
# 大约是一段到两段的量，既能装下一个完整论点，又不至于太泛。
DEFAULT_CHUNK_SIZE = 500

# 默认重叠长度（字符数）
DEFAULT_OVERLAP = 50

# 递归切分的分隔符优先级：从「最像段落边界」到「最不像」。
# 空字符串放最后，意思是「实在切不动了就逐字硬切」。
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ". ",
    "! ",
    "? ",
    "; ",
    " ",
    "",
)

# embedding 函数的形状：给一批文本，返回一批等长向量
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


# ---------------------------------------------------------------- 公共校验


def _check_params(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 需为正整数，实际为 {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负数，实际为 {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap（{overlap}）必须小于 chunk_size（{chunk_size}），否则会原地打转")


# ---------------------------------------------------------------- 工具


def _split_sentences(text: str) -> list[str]:
    """把文本切成句子（保留标点）。中文标点和英文标点都认。"""
    parts = re.split(r"(?<=[。！？；!?;])", text)
    return [part.strip() for part in parts if part.strip()]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度：两个向量方向越接近越接近 1。

    只看方向、不看长度，所以「递归的定义」和「递归的定义是什么」这种
    长短不同但意思接近的句子，也能算出高相似度。
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """给每个块前面贴上上一块的结尾，减少「句子被切在两块交界处」的损失。

    注意：贴完之后单块长度会略超 ``chunk_size``。这是刻意的取舍——
    多出几十个字符换回上下文完整，对检索更有利。
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for index in range(1, len(chunks)):
        result.append(chunks[index - 1][-overlap:] + chunks[index])
    return result


# ---------------------------------------------------------------- 策略一：固定长度


def split_fixed(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """每 ``chunk_size`` 个字符切一刀，完全不管语义边界。

    会切在句子中间、词中间，切出来的块读起来是断的。它是**基线**：
    实验里拿它的检索效果去比另外两种，才知道「讲究切分」值不值。
    """
    _check_params(chunk_size, overlap)
    if not text.strip():
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
    return chunks


# ---------------------------------------------------------------- 策略二：递归


def _split_by_separator(text: str, separator: str) -> list[str]:
    """按分隔符切开，并把分隔符粘回前一段的末尾（不丢标点）。"""
    if separator == "":
        return list(text)

    parts = text.split(separator)
    if len(parts) == 1:
        return [text]
    return [part + separator for part in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # 所有分隔符都试过了还切不开，只能逐字硬切
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *rest = separators
    pieces = _split_by_separator(text, separator)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if len(piece) > chunk_size:
            # 这一片自己就超长，先把缓冲区冲出去，再对它换更细的分隔符继续切
            if buffer.strip():
                chunks.append(buffer)
            buffer = ""
            chunks.extend(_recursive_split(piece, chunk_size, rest))
        elif len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            if buffer.strip():
                chunks.append(buffer)
            buffer = piece

    if buffer.strip():
        chunks.append(buffer)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def split_recursive(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """按分隔符优先级逐级往下切，尽量不切断句子。

    思路是「先试粗的边界，不行再试细的」：先按空行（段落）切，
    某一段还是太长就按换行切，再长就按句号切，最后实在不行才逐字切。

    这是实践中最常用的策略——它在「块大小可控」和「语义尽量完整」之间
    取得了比较好的平衡。
    """
    _check_params(chunk_size, overlap)
    if not text.strip():
        return []

    chunks = _recursive_split(text, chunk_size, list(separators))
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return _apply_overlap(chunks, overlap)


# ---------------------------------------------------------------- 策略三：语义


def split_semantic(
    text: str,
    *,
    embed_fn: EmbedFn,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    threshold: float = 0.6,
) -> list[str]:
    """在「话题变了」的地方下刀。

    做法分四步：

    1. 先把文本切成句子；
    2. 用 ``embed_fn`` 把每个句子变成向量；
    3. 算相邻两句的余弦相似度，**相似度低于阈值就认为换话题了，在这里切一刀**；
    4. 把切出来的片段按 ``chunk_size`` 合并或再切，保证块大小可控。

    参数
    ----
    embed_fn:
        形如 ``f(文本列表) -> 向量列表`` 的函数。M3.3 会把 Chroma 的
        向量化能力包成这个形状传进来。之所以要求外部传入而不是内部直接调，
        是为了让切分逻辑本身可以被**离线测试**——测试里传个假函数即可，
        不必真的去下模型。
    threshold:
        相似度阈值。调高 → 切得碎（更敏感）；调低 → 切得整。

    取一个折中值 0.6：太低会把不同话题粘在一起，太高会把同一个话题切碎。
    """
    _check_params(chunk_size, 0)
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()]

    vectors = embed_fn(sentences)
    if len(vectors) != len(sentences):
        raise ValueError(f"embed_fn 返回了 {len(vectors)} 个向量，但输入是 {len(sentences)} 个句子")

    # 第一步：按「话题是否变了」分组
    groups: list[str] = []
    current: list[str] = [sentences[0]]
    for index in range(1, len(sentences)):
        similarity = _cosine_similarity(vectors[index - 1], vectors[index])
        if similarity < threshold:
            groups.append("".join(current))
            current = []
        current.append(sentences[index])
    if current:
        groups.append("".join(current))

    # 第二步：只保证不超上限，**不把小组揉回一起**。
    #
    # 语义切出来的每一组就是一个话题，再合并回去等于白切——这也正是它和
    # 「递归切分」最大的行为差异。代价是块大小可能不均（短话题会切出小块），
    # 这是语义切分的固有特点，也是实验里要观察的指标之一。
    chunks: list[str] = []
    for group in groups:
        if len(group) <= chunk_size:
            chunks.append(group)
        else:
            chunks.extend(group[i : i + chunk_size] for i in range(0, len(group), chunk_size))

    return [chunk.strip() for chunk in chunks if chunk.strip()]
