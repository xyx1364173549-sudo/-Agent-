"""LLM 接入层（对应计划 M1）。

对外只暴露 ``create_chat_model()``。

业务代码不要直接 ``from langchain_openai import ChatOpenAI``——那样以后
换模型、加日志、改超时，就得满项目去找散落的构造代码了。
"""

from src.llm.factory import DEFAULT_MODEL, create_chat_model

__all__ = ["DEFAULT_MODEL", "create_chat_model"]
