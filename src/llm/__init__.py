"""多模型统一接入层（对应计划 M1）。

对外只暴露一个 ``create_chat_model(provider)`` 入口，屏蔽 DeepSeek 与小米 MiMo
在鉴权、接口地址、模型名上的差异。

命名说明：用 ``create_`` 而非 ``get_``，是因为每次调用都会构造**新的**模型实例，
不存在缓存或单例语义；``get_`` 前缀容易让人误以为拿到的是同一个对象，
一旦哪天需要按会话隔离模型参数，这种误解会带来难查的 bug。
"""

from src.llm.factory import (
    DEFAULT_MODELS,
    LLMFactoryError,
    available_providers,
    create_chat_model,
    default_model_for,
)

__all__ = [
    "DEFAULT_MODELS",
    "LLMFactoryError",
    "available_providers",
    "create_chat_model",
    "default_model_for",
]
