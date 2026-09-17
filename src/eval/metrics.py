"""实验指标：全是纯函数，输入输出都是数字。

之所以把指标单独抽出来、还配单元测试，是因为**论文里的数字一旦算错，
整章结论都站不住**。指标函数是最不该"凭感觉写"的一类代码：
它跑起来不报错，只是悄悄给你一个偏高的分数。

三个实验各用一组指标：

    实验一（记忆消融）   要点覆盖率 —— 回答里命中了几个关键信息
    实验二（检索策略）   Hit@k / MRR / 召回率 —— 检索领域的标准指标
    实验三（动态规划）   达到目标所需轮数、无效练习比例
"""

from collections.abc import Collection, Sequence


# --------------------------------------------------------------------------
# 检索指标（实验二）
# --------------------------------------------------------------------------


def hit_at_k(sources: Sequence[str], relevant: Collection[str], k: int) -> float:
    """前 k 条结果里有没有命中相关文档。命中算 1，没命中算 0。

    参数
    ----
    sources:
        检索结果按名次排好的来源列表（第一条最相关）。
    relevant:
        标注为「对这个查询相关」的来源集合。
    k:
        只看前 k 条。

    这个指标回答的是「用户翻前几条时，能不能看到有用的东西」。
    它比准确率更贴近实际体验：排在第 1 和第 5 名，用户体验差很多。
    """
    if k <= 0:
        raise ValueError(f"k 需为正整数，实际为 {k}")
    if not relevant:
        raise ValueError("相关标注不能为空——没有标准答案就算不出指标")

    return 1.0 if any(source in relevant for source in sources[:k]) else 0.0


def reciprocal_rank(sources: Sequence[str], relevant: Collection[str]) -> float:
    """第一个命中结果的名次的倒数。第 1 名命中得 1.0，第 2 名得 0.5，第 5 名得 0.2。

    为什么用倒数而不是直接用名次：名次是「越小越好」，倒数是「越大越好」，
    这样多个查询求平均时不会出现「一个极差的查询把整体拉爆」的情况——
    排到第 50 名和第 5 名的差距被压缩了，更符合直觉。

    一条都没命中返回 0.0。
    """
    if not relevant:
        raise ValueError("相关标注不能为空")

    for rank, source in enumerate(sources, start=1):
        if source in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(sources: Sequence[str], relevant: Collection[str], k: int) -> float:
    """前 k 条里覆盖了相关文档的几成。

    Hit@k 只管「有没有命中」，这个管「命中了多少」——两者要一起看：
    一个策略可能每次都命中一条，但相关文档一共 3 篇却只找回来 1 篇。
    """
    if k <= 0:
        raise ValueError(f"k 需为正整数，实际为 {k}")
    if not relevant:
        raise ValueError("相关标注不能为空")

    found = {source for source in sources[:k] if source in relevant}
    return len(found) / len(relevant)


# --------------------------------------------------------------------------
# 要点覆盖率（实验一）
# --------------------------------------------------------------------------


def coverage(marks: Sequence[bool]) -> float:
    """要点覆盖率 = 命中几个 / 一共几个。

    只做数学，**不判断「算不算命中」**。判定方式有两种，由调用方选：

        关键词匹配（``match_keywords``）—— 零成本、完全可复现，适合当基线
        大模型判定                     —— 准，但要花钱

    分开的好处是：换判定方式不用动指标代码，两种方式的分数也能放在同一张表里比。
    """
    return round(sum(marks) / len(marks), 4) if marks else 0.0


# 要点句里常见的虚词，匹配前先剔掉
_STOPWORDS = (
    "提到", "说明", "指出", "给出", "包含", "应该", "需要", "能够", "可以",
    "至少", "一个", "具体", "明确", "以及", "并且",
    "的", "了", "是", "在", "和", "与", "或", "把", "被", "对", "为",
)


def match_keywords(answer: str, points: Sequence[str]) -> list[bool]:
    """关键词版判定：要点里的实词在回答中出现，就算这一条被覆盖。

    为什么不能拿整句要点去匹配：要点写着「提到终止条件」，而回答里
    永远不会原样出现「提到」两个字，整句匹配必然全不命中。

    所以先剔虚词、再取实词，并且**只取开头的 4 个字**做匹配——
    要点通常是「终止条件要显式写出来」这种句子，代表它的就是前几个字。
    这个做法很土，但零成本、结果稳定，正好当实验一的基线。
    """
    text = answer
    marks: list[bool] = []

    for point in points:
        cleaned = point
        for word in _STOPWORDS:
            cleaned = cleaned.replace(word, " ")

        keywords = [
            piece.strip("，。、：；（）()[]【】 \"'")[:4]
            for piece in cleaned.split()
        ]
        keywords = [word for word in keywords if len(word) >= 2]

        marks.append(any(word in text for word in keywords))

    return marks


# --------------------------------------------------------------------------
# 查询分类（实验二）
# --------------------------------------------------------------------------


def bigrams(text: str) -> set[str]:
    """把一段文字切成相邻两字的片段集合（二元组）。

    为什么是二元组：中文没有空格，单字太短（「条」几乎什么都命中），
    整句又太长（换个说法就不一样）。两个字是既能表达含义、
    又不至于太苛刻的最小单位。

    标点和空白先去掉——它们不承载语义，留着只会干扰比对。
    """
    cleaned = [char for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff"]
    joined = "".join(cleaned)
    return {joined[i : i + 2] for i in range(len(joined) - 1)}


def literal_overlap(query: str, document: str) -> float:
    """查询里有多大比例的字面片段，能在文档中原样找到。

    返回 0 表示「完全没有字面重叠」——这类查询只有理解意思才能命中，
    是向量检索真正该发挥作用的场景。返回 0.8 表示这段话几乎照抄了文档，
    关键词检索闭着眼睛都能命中。

    **为什么需要这个函数**：实验二要把查询分成「字面型」和「语义型」两类
    分别统计。靠人眼判断属于拍脑袋——写查询的时候自己知道答案，
    很容易觉得「这个肯定算语义型」，实际却和文档有好几个字重合。
    用这个函数客观算一遍，分类才有依据。
    """
    query_grams = bigrams(query)
    if not query_grams:
        return 0.0

    document_grams = bigrams(document)
    return round(len(query_grams & document_grams) / len(query_grams), 4)


# --------------------------------------------------------------------------
# 学习过程指标（实验三）
# --------------------------------------------------------------------------


def efficiency(rounds: int, baseline_rounds: int) -> float:
    """学习效率：相比基线省下了几成的轮数。

    基线通常是「静态路径」——固定顺序、不管掌握度。返回 0.3 表示
    比基线少花 30% 的轮数。
    """
    if baseline_rounds <= 0:
        raise ValueError("基线轮数需为正数")
    return round((baseline_rounds - rounds) / baseline_rounds, 4)


def idle_ratio(attempts: list[dict]) -> float:
    """无效练习比例：练了但本来就会的比例。

    定义「本来就会」用的是**练习前**的掌握度——这个区分很关键：
    练完之后掌握度当然高，拿练之后的值去判断，这个指标永远是 0。

    这个指标回答的是「路径规划有没有把时间浪费在已经会的东西上」，
    是「个性化」最直接的一个证据。
    """
    if not attempts:
        return 0.0
    idle = sum(1 for item in attempts if item.get("mastery_before", 0.0) >= item.get("threshold", 0.7))
    return round(idle / len(attempts), 4)


# --------------------------------------------------------------------------
# 统计工具
# --------------------------------------------------------------------------


def mean(values: Sequence[float]) -> float:
    """平均值。空列表返回 0.0，不让它除零崩掉。"""
    return round(sum(values) / len(values), 4) if values else 0.0


def std(values: Sequence[float]) -> float:
    """样本标准差（除以 n-1）。

    仿真实验要报误差，光给平均值撑不住结论——两组均值差 0.2，
    但一组波动 0.01、另一组波动 0.5，可信度完全不同。
    """
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return round(variance**0.5, 4)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """一次给出均值、标准差、最小值和最大值。"""
    return {
        "mean": mean(values),
        "std": std(values),
        "min": round(min(values), 4) if values else 0.0,
        "max": round(max(values), 4) if values else 0.0,
        "n": len(values),
    }
