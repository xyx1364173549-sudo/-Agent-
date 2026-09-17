"""功能 Agent 集合（对应计划 M4）。

导师 Agent（讲解）、出题 Agent（生成练习）、评估 Agent（批改与诊断）。
每个 Agent 只负责一件事，由 ``src/planning/graph.py`` 统一编排。

三者共用一个很薄的基类 ``Agent``——它们长得一样：拿一份提示词去问模型、
把回答取成文字。差别只在提示词写什么。做成工具调用、自主循环那些复杂形态
不在这一层，交给编排层。
"""

from src.agents.base import Agent, TokenStream
from src.agents.grader import GraderAgent
from src.agents.quiz import QuizAgent
from src.agents.tutor import TutorAgent

__all__ = ["Agent", "GraderAgent", "QuizAgent", "TokenStream", "TutorAgent"]
