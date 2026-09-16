"""DeepSeek 聊天模型工厂。

全项目只通过 ``create_chat_model()`` 拿模型，不在业务代码里散落
``ChatOpenAI(...)``。好处是：以后想换模型、调超时、加日志，只改这一个文件。

**只支持 DeepSeek 一个提供方。** 曾设计过一套「多提供方映射表」，但在只有
一个厂商的场景下，那层抽象只增加阅读成本、不产生任何价值，已移除。
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 默认模型。换模型时改这一行，或在调用时显式传 model= 参数。
DEFAULT_MODEL = "deepseek-flash"


def create_chat_model(
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = 0.3,
    timeout: float = 60.0,
    api_key: str | None = None,
) -> BaseChatModel:
    """创建一个 DeepSeek 聊天模型对象。

    参数
    ----
    model:
        模型名，默认 ``deepseek-flash``。
    temperature:
        采样温度。0 最稳定、1 最发散。默认 0.3——学习路径规划要的是可复现，
        不是天马行空。
    timeout:
        单次请求超时秒数（等模型回复的最长时间），默认 60 秒。
    api_key:
        一般不用传，默认从 ``.env`` 读取；写测试时可以显式传入假密钥。

    返回
    ----
    LangChain 的 ``BaseChatModel``，可以直接 ``invoke(...)`` 或 ``stream(...)``。

    说明
    ----
    本函数**不发起网络请求**，只是把参数装配成一个对象。真正的调用发生在
    ``llm.invoke(...)`` 那一刻。

    若 ``.env`` 里没配 ``DEEPSEEK_API_KEY``，会抛 ``ConfigError``，
    并提示该改哪个文件、填哪个变量。
    """
    settings = get_settings()
    key = api_key or settings.require_api_key()

    logger.info("创建聊天模型 | model=%s | base_url=%s", model, settings.deepseek_base_url)

    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        timeout=timeout,
    )
