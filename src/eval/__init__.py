"""实验与评估（对应计划 M7、论文第 6 章的三组实验）。

    dataset.py   三个实验的输入：资料、查询标注、学生轨迹、知识点图
    metrics.py   纯函数的指标计算
    judge.py     要点覆盖的模型判定（零成本的关键词判定在 metrics 里）
    report.py    结果汇总与图表生成

三个实验分别回答：

    实验一  分层记忆到底有没有用？           消融对比（无记忆 / 仅工作 / 全分层）
    实验二  切分和检索策略怎么选？           3 种切分 × 3 种检索的交叉对比
    实验三  动态规划和静态路径差在哪？       仿真对比学习轮数与无效练习比例

设计上刻意做的一处区分：**实验三用仿真学生而不是真实模型**。
要对比的是规划算法，模型随机性只会把结论搅浑；而且仿真零成本、
可复现、能跑多次做统计。
"""

from src.eval.dataset import (
    KNOWLEDGE_GRAPH,
    MATERIALS,
    MEMORY_CONDITIONS,
    PROFILE_QUESTIONS,
    RETRIEVAL_QUERIES,
    SIM_STUDENT,
    STUDENT_SESSION,
    TARGET_TOPICS,
)
from src.eval.metrics import (
    bigrams,
    coverage,
    efficiency,
    hit_at_k,
    idle_ratio,
    literal_overlap,
    match_keywords,
    mean,
    recall_at_k,
    reciprocal_rank,
    std,
    summarize,
)

__all__ = [
    "KNOWLEDGE_GRAPH",
    "MATERIALS",
    "MEMORY_CONDITIONS",
    "PROFILE_QUESTIONS",
    "RETRIEVAL_QUERIES",
    "SIM_STUDENT",
    "STUDENT_SESSION",
    "TARGET_TOPICS",
    "bigrams",
    "coverage",
    "efficiency",
    "hit_at_k",
    "idle_ratio",
    "literal_overlap",
    "match_keywords",
    "mean",
    "recall_at_k",
    "reciprocal_rank",
    "std",
    "summarize",
]
