"""LLM 接入层（对应计划 M1）。

对外只暴露 ``create_chat_model()``。

业务代码不要自己写 ``ChatDeepSeek(...)``，统一从这里拿：
以后换模型、调温度、加日志，都只改 ``factory.py`` 一个地方。
"""

from src.llm.factory import DEFAULT_MODEL, create_chat_model

__all__ = ["DEFAULT_MODEL", "create_chat_model"]
