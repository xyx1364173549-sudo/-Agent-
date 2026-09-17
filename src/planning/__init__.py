"""学习者画像与动态任务规划（对应计划 M4、论文第 5 章）。

流水线是四步，前三步在本模块、第四步负责把它们串起来：

    profile     学习者画像   —— 这个人会什么、想学什么、喜欢怎么学
    decomposer  目标拆解器   —— 把「掌握动态规划」拆成具体知识点（用大模型）
    planner     路径规划器   —— 排出学习顺序（用算法，不用大模型）
    graph       LangGraph 编排 —— 学 → 练 → 评 → 调，循环推进

其中最值得说的分工是：**大模型负责理解和生成，算法负责保证正确性**。
「该先学什么」有确定答案，交给拓扑排序；「该拆成哪几块」没有确定答案，
才交给大模型。
"""

from src.planning.decomposer import DEFAULT_MAX_TOPICS, decompose
from src.planning.planner import (
    STATUS_FOCUS,
    STATUS_PRACTICE,
    STATUS_REVIEW,
    classify,
    next_step,
    plan,
    progress,
    render,
    topo_layers,
)
from src.planning.profile import (
    MASTERED_THRESHOLD,
    WEAK_THRESHOLD,
    LearnerProfile,
)

__all__ = [
    "DEFAULT_MAX_TOPICS",
    "MASTERED_THRESHOLD",
    "STATUS_FOCUS",
    "STATUS_PRACTICE",
    "STATUS_REVIEW",
    "WEAK_THRESHOLD",
    "LearnerProfile",
    "classify",
    "decompose",
    "next_step",
    "plan",
    "progress",
    "render",
    "topo_layers",
]
