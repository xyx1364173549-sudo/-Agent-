"""统一聊天模型工厂。

四条设计原则：

1. **双通道同构**：DeepSeek 与小米 MiMo 都走 OpenAI 兼容协议，因此共用
   ``ChatOpenAI`` 实现，差异只收敛到「密钥 / 接口地址 / 默认模型」三处映射表。
   新增一个兼容 OpenAI 协议的提供方，只需往映射表加一行。
2. **返回统一抽象**：对外只暴露 ``BaseChatModel``，让后续的 chain、LangGraph
   编排、流式输出无需感知具体提供方。
3. **构造期零网络**：工厂只做参数装配与校验，不发起任何请求，真正的网络调用
   发生在 ``invoke`` / ``stream``。这使得单元测试可以完全离线运行。
4. **密钥只进不出**：日志与异常信息中出现提供方与模型名，绝不出现明文密钥。
"""

from __future__ import annotations

from typing import Any, Final

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.utils.logger import get_logger

__all__ = [
    "DEFAULT_MODELS",
    "LLMFactoryError",
    "available_providers",
    "create_chat_model",
    "default_model_for",
]

logger = get_logger(__name__)

# 提供方 -> 默认模型名
DEFAULT_MODELS: Final[dict[str, str]] = {
    "deepseek": "deepseek-chat",
    "mimo": "mimo-v2.5",
}

# 提供方 -> Settings 中的接口地址字段名
_BASE_URL_FIELD: Final[dict[str, str]] = {
    "deepseek": "deepseek_base_url",
    "mimo": "mimo_base_url",
}

DEFAULT_TEMPERATURE: Final[float] = 0.3
DEFAULT_TIMEOUT: Final[float] = 60.0
DEFAULT_MAX_RETRIES: Final[int] = 2


class LLMFactoryError(Exception):
    """工厂参数非法时抛出（提供方未知、模型为空、超参越界等）。

    注意：**缺少 API Key 不抛本异常**，而是由 ``Settings.require_api_key()``
    抛出 ``ConfigError``——前者是「参数写错了」，后者是「配置没准备好」，
    语义不同，调用方需要分开处理。
    """


# ---------------------------------------------------------------- 校验辅助


def available_providers() -> list[str]:
    """返回当前支持的提供方名称（按字母序）。"""
    return sorted(DEFAULT_MODELS)


def _check_provider(provider: str) -> str:
    """校验并规范化提供方名称（去空格、转小写）。"""
    if not isinstance(provider, str) or not provider.strip():
        raise LLMFactoryError(f"provider 必须是非空字符串，实际收到 {provider!r}")

    name = provider.strip().lower()
    if name not in DEFAULT_MODELS:
        supported = "、".join(available_providers())
        raise LLMFactoryError(f"未支持的提供方 {provider!r}，当前支持：{supported}")
    return name


def _non_empty(value: object, field: str) -> str:
    """要求取值是非空字符串，返回去除首尾空白的结果。"""
    if not isinstance(value, str) or not value.strip():
        raise LLMFactoryError(f"{field} 必须是非空字符串，实际收到 {value!r}")
    return value.strip()


def _check_temperature(value: float) -> float:
    """温度需落在 [0.0, 2.0]；非数值与 NaN 一并拦截。"""
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMFactoryError(f"temperature 需为数值，实际收到 {value!r}") from exc

    # 注意 NaN 的比较恒为 False，因此下面这个写法能顺带把 NaN 挡掉
    if not 0.0 <= temperature <= 2.0:
        raise LLMFactoryError(f"temperature 需在 [0.0, 2.0] 区间内，实际为 {temperature}")
    return temperature


def _check_timeout(value: float) -> float:
    """超时时间必须为正数（秒）；NaN 同样拦截。"""
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMFactoryError(f"timeout 需为数值（秒），实际收到 {value!r}") from exc

    if not timeout > 0:
        raise LLMFactoryError(f"timeout 需为正数，实际为 {timeout}")
    return timeout


def _check_max_retries(value: int) -> int:
    """重试次数必须是非负整数。

    显式排除 ``bool``：Python 中 ``True`` 也是 ``int``，若放行会让
    ``max_retries=True`` 意外通过校验。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMFactoryError(f"max_retries 需为非负整数，实际收到 {value!r}")
    if value < 0:
        raise LLMFactoryError(f"max_retries 不能为负数，实际为 {value}")
    return value


# ---------------------------------------------------------------- 主入口


def default_model_for(provider: str, settings: Settings | None = None) -> str:
    """取提供方的默认模型名。

    MiMo 的模型名支持通过 ``.env`` 的 ``MIMO_MODEL`` 覆盖；DeepSeek 暂用固定
    默认值。若后续需要在多个 DeepSeek 模型间切换，再把它提到配置层即可。
    """
    name = _check_provider(provider)
    if name == "mimo":
        config = settings or get_settings()
        return config.mimo_model or DEFAULT_MODELS[name]
    return DEFAULT_MODELS[name]


def create_chat_model(
    provider: str = "deepseek",
    *,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: str | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """按提供方创建一个聊天模型实例。

    参数
    ----
    provider:
        提供方名称，支持 ``"deepseek"`` 与 ``"mimo"``，大小写不敏感。
    model:
        模型名；传 ``None`` 时取该提供方的默认模型。
    temperature:
        采样温度，取值区间 ``[0.0, 2.0]``。默认 ``0.3``，偏向稳定输出——
        学习路径规划场景要的是可复现，而不是天马行空。
    timeout:
        单次请求超时秒数，默认 ``60``。
    max_retries:
        SDK 层自动重试次数，默认 ``2``。
    api_key:
        显式密钥，主要用于测试；传 ``None`` 时从 ``.env`` 读取。
    settings:
        配置对象，传 ``None`` 时取全局单例。测试时应显式注入，避免真实 ``.env`` 干扰。
    **kwargs:
        透传给 ``ChatOpenAI`` 的其他参数，例如 ``max_tokens``、``top_p``。

    异常
    ----
    LLMFactoryError:
        提供方未知、模型名为空、超参越界。
    ConfigError:
        对应的 API Key 未配置。
    """
    config = settings or get_settings()
    name = _check_provider(provider)

    resolved_model = _non_empty(model, "model") if model is not None else default_model_for(name, config)
    resolved_temperature = _check_temperature(temperature)
    resolved_timeout = _check_timeout(timeout)
    resolved_retries = _check_max_retries(max_retries)
    resolved_key = _non_empty(api_key, "api_key") if api_key is not None else config.require_api_key(name)
    base_url = getattr(config, _BASE_URL_FIELD[name])

    chat_model = ChatOpenAI(
        model=resolved_model,
        api_key=resolved_key,
        base_url=base_url,
        temperature=resolved_temperature,
        timeout=resolved_timeout,
        max_retries=resolved_retries,
        **kwargs,
    )

    logger.info(
        "已创建聊天模型 | provider=%s | model=%s | base_url=%s | temperature=%s | timeout=%.1fs | max_retries=%d",
        name,
        resolved_model,
        base_url,
        resolved_temperature,
        resolved_timeout,
        resolved_retries,
    )
    return chat_model
