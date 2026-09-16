"""聊天模型工厂测试。

重点盯三类容易被「看起来能跑」掩盖的问题：

1. **串号**：把 MiMo 的密钥配到 DeepSeek 的接口地址上——这类错误不会报错，
   只会让请求静默失败或账单打错地方；
2. **越界超参**：``temperature=3``、``max_retries=True`` 这种参数，SDK 未必拦得住；
3. **密钥泄漏**：密钥一旦进日志，就会随日志文件与 CI 输出扩散出去。

全部用例均不联网——工厂在构造期不发起任何请求，这是它可以离线测试的前提。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from src.config import ConfigError, Settings
from src.llm import (
    DEFAULT_MODELS,
    LLMFactoryError,
    available_providers,
    create_chat_model,
    default_model_for,
)

DEEPSEEK_KEY = "sk-deepseek-fake-key-for-test"
MIMO_KEY = "sk-mimo-fake-key-for-test"


def build(env_file: Path, **env: str) -> Settings:
    """构造完全受控的配置。

    显式传 ``environ`` 会完全替代系统环境变量，因此本机真实的 ``.env``
    不会泄进断言，避免「换台机器结果就变了」的假绿。
    """
    return Settings.from_env(environ=env, env_file=env_file)


@pytest.fixture
def settings(missing_env_file: Path) -> Settings:
    """两套密钥齐备的配置对象。"""
    return build(missing_env_file, DEEPSEEK_API_KEY=DEEPSEEK_KEY, MIMO_API_KEY=MIMO_KEY)


# --------------------------------------------------------------------------
# 基本构造
# --------------------------------------------------------------------------


def test_returns_langchain_chat_model(settings: Settings) -> None:
    """对外暴露的必须是 LangChain 统一抽象，后续 chain 与 LangGraph 才接得上。"""
    llm = create_chat_model(settings=settings)
    assert isinstance(llm, BaseChatModel)


def test_default_provider_is_deepseek(settings: Settings) -> None:
    llm = create_chat_model(settings=settings)
    assert llm.model_name == DEFAULT_MODELS["deepseek"]
    assert llm.openai_api_base == settings.deepseek_base_url


def test_available_providers_are_sorted() -> None:
    assert available_providers() == ["deepseek", "mimo"]


def test_default_hyperparams(settings: Settings) -> None:
    """默认超参即契约：温度偏低求稳定，超时与重试有兜底。"""
    llm = create_chat_model(settings=settings)
    assert llm.temperature == 0.3
    assert llm.request_timeout == 60.0
    assert llm.max_retries == 2


# --------------------------------------------------------------------------
# 提供方映射（串号防护）
# --------------------------------------------------------------------------


def test_mimo_uses_its_own_base_url_and_key(settings: Settings) -> None:
    """MiMo 必须走自己的地址与密钥，不能借道 DeepSeek 的配置。"""
    llm = create_chat_model("mimo", settings=settings)
    assert llm.openai_api_base == settings.mimo_base_url
    assert llm.openai_api_key.get_secret_value() == MIMO_KEY


def test_deepseek_key_never_reaches_mimo(settings: Settings) -> None:
    llm = create_chat_model("mimo", settings=settings)
    assert llm.openai_api_key.get_secret_value() != DEEPSEEK_KEY


def test_providers_do_not_share_base_url(settings: Settings) -> None:
    """两个提供方的地址必须不同——若哪天在 .env 里配成同一个，这条会报警。"""
    deepseek = create_chat_model("deepseek", settings=settings)
    mimo = create_chat_model("mimo", settings=settings)
    assert deepseek.openai_api_base != mimo.openai_api_base


@pytest.mark.parametrize("raw", ["MIMO", " Mimo ", "mimo"])
def test_provider_name_case_and_space_insensitive(settings: Settings, raw: str) -> None:
    llm = create_chat_model(raw, settings=settings)
    assert llm.model_name == settings.mimo_model


# --------------------------------------------------------------------------
# 模型名解析
# --------------------------------------------------------------------------


def test_mimo_model_overridable_via_env(missing_env_file: Path) -> None:
    """MiMo 模型名允许从 .env 覆盖，换型号时不用改代码。"""
    s = build(missing_env_file, MIMO_API_KEY=MIMO_KEY, MIMO_MODEL="mimo-v9")
    assert default_model_for("mimo", s) == "mimo-v9"


def test_explicit_model_overrides_default(settings: Settings) -> None:
    llm = create_chat_model("deepseek", model="deepseek-reasoner", settings=settings)
    assert llm.model_name == "deepseek-reasoner"


def test_explicit_api_key_overrides_settings(settings: Settings) -> None:
    llm = create_chat_model(settings=settings, api_key="sk-explicit-override")
    assert llm.openai_api_key.get_secret_value() == "sk-explicit-override"


def test_injected_settings_win_over_real_env(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """注入的配置优先级最高，测试结果不受本机环境变量影响。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-real-env-should-be-ignored")
    llm = create_chat_model(settings=settings)
    assert llm.openai_api_key.get_secret_value() == DEEPSEEK_KEY


def test_extra_kwargs_passed_through(settings: Settings) -> None:
    """额外参数（如 max_tokens）应能透传到 SDK，否则未来逐个加参数会没完没了。"""
    llm = create_chat_model(settings=settings, max_tokens=256)
    assert llm.max_tokens == 256


# --------------------------------------------------------------------------
# 非法参数
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["gemini", "gpt-4", "deepseek-chat"])
def test_unknown_provider_rejected(settings: Settings, bad: str) -> None:
    """把模型名当提供方传进来是高频笔误，必须明确报错而不是静默走默认值。"""
    with pytest.raises(LLMFactoryError, match="未支持的提供方"):
        create_chat_model(bad, settings=settings)


@pytest.mark.parametrize("bad", ["", "   "])
def test_blank_provider_rejected(settings: Settings, bad: str) -> None:
    with pytest.raises(LLMFactoryError, match="provider"):
        create_chat_model(bad, settings=settings)


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_model_rejected(settings: Settings, bad: str) -> None:
    with pytest.raises(LLMFactoryError, match="model"):
        create_chat_model(settings=settings, model=bad)


@pytest.mark.parametrize("bad", [-0.1, 2.1, 3, "hot"])
def test_temperature_out_of_range_rejected(settings: Settings, bad: object) -> None:
    with pytest.raises(LLMFactoryError, match="temperature"):
        create_chat_model(settings=settings, temperature=bad)  # type: ignore[arg-type]


def test_temperature_rejects_nan(settings: Settings) -> None:
    """NaN 的比较恒为 False，最容易被区间校验漏放行。"""
    with pytest.raises(LLMFactoryError, match="temperature"):
        create_chat_model(settings=settings, temperature=float("nan"))


@pytest.mark.parametrize("good", [0.0, 2.0])
def test_temperature_boundaries_accepted(settings: Settings, good: float) -> None:
    """边界值本身是合法的，校验不能写成开区间。"""
    llm = create_chat_model(settings=settings, temperature=good)
    assert llm.temperature == good


@pytest.mark.parametrize("bad", [0, -1, "soon"])
def test_timeout_must_be_positive(settings: Settings, bad: object) -> None:
    with pytest.raises(LLMFactoryError, match="timeout"):
        create_chat_model(settings=settings, timeout=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-1, 1.5, "2"])
def test_max_retries_must_be_non_negative_int(settings: Settings, bad: object) -> None:
    with pytest.raises(LLMFactoryError, match="max_retries"):
        create_chat_model(settings=settings, max_retries=bad)  # type: ignore[arg-type]


def test_max_retries_rejects_bool(settings: Settings) -> None:
    """``True`` 在 Python 里也是 ``int``，不显式挡掉就会静默通过。"""
    with pytest.raises(LLMFactoryError, match="max_retries"):
        create_chat_model(settings=settings, max_retries=True)


def test_zero_max_retries_accepted(settings: Settings) -> None:
    llm = create_chat_model(settings=settings, max_retries=0)
    assert llm.max_retries == 0


# --------------------------------------------------------------------------
# 密钥缺失
# --------------------------------------------------------------------------


def test_missing_deepseek_key_raises_config_error(missing_env_file: Path) -> None:
    """缺密钥抛 ConfigError（配置没准备好），而非 LLMFactoryError（参数写错了）。"""
    s = build(missing_env_file)
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        create_chat_model("deepseek", settings=s)


def test_missing_mimo_key_raises_config_error(missing_env_file: Path) -> None:
    """DeepSeek 配了但 MiMo 没配时，只报缺的那一个。"""
    s = build(missing_env_file, DEEPSEEK_API_KEY=DEEPSEEK_KEY)
    with pytest.raises(ConfigError, match="MIMO_API_KEY"):
        create_chat_model("mimo", settings=s)


def test_blank_key_treated_as_missing(missing_env_file: Path) -> None:
    """空白串等同没配，不能拿着空密钥去发请求。"""
    s = build(missing_env_file, DEEPSEEK_API_KEY="   ")
    with pytest.raises(ConfigError):
        create_chat_model("deepseek", settings=s)


def test_parameter_error_precedes_key_check(missing_env_file: Path) -> None:
    """没有密钥时，参数错误应当先报——否则会误以为是配置问题去翻 .env。"""
    s = build(missing_env_file)
    with pytest.raises(LLMFactoryError, match="temperature"):
        create_chat_model("deepseek", settings=s, temperature=9)


# --------------------------------------------------------------------------
# 零网络与密钥安全
# --------------------------------------------------------------------------


def test_construction_makes_no_network_call(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """构造期必须只装配、不发请求，否则离线开发与 CI 全都跑不起来。"""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("构造聊天模型时不应发起网络请求")

    monkeypatch.setattr("httpx.Client.send", _boom)
    monkeypatch.setattr("httpx.AsyncClient.send", _boom)

    llm = create_chat_model(settings=settings)
    assert isinstance(llm, BaseChatModel)


def test_unreachable_base_url_still_constructs(missing_env_file: Path) -> None:
    """指向一个必然连不上的地址也能构造成功，反证构造期无网络活动。"""
    s = build(
        missing_env_file,
        DEEPSEEK_API_KEY=DEEPSEEK_KEY,
        DEEPSEEK_BASE_URL="http://127.0.0.1:1",
    )
    llm = create_chat_model(settings=s)
    assert llm.openai_api_base == "http://127.0.0.1:1"


def test_api_key_never_appears_in_logs(caplog: pytest.LogCaptureFixture, settings: Settings) -> None:
    """日志可以记提供方与模型名，但绝不能记密钥。"""
    with caplog.at_level(logging.INFO):
        create_chat_model("deepseek", settings=settings)
        create_chat_model("mimo", settings=settings)

    assert DEEPSEEK_KEY not in caplog.text
    assert MIMO_KEY not in caplog.text
    # 同时确认日志确实写了内容，否则上面的断言会因「什么都没记」而假通过
    assert "已创建聊天模型" in caplog.text
