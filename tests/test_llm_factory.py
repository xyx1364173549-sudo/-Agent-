"""DeepSeek 聊天模型工厂测试。

只盯两件真正要紧的事：

1. **不联网** —— 工厂只装配参数，构造时不发请求，所以测试能离线跑；
2. **不泄漏密钥** —— 日志里只能有模型名，不能有明文 API Key。

超参是否越界交给 SDK 自己管，这里不重复校验，省得两处规则打架。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from src.config import ConfigError, Settings
from src.llm import DEFAULT_MODEL, create_chat_model

FAKE_KEY = "sk-fake-key-for-test"


def use_settings(monkeypatch: pytest.MonkeyPatch, env_file: Path, **env: str) -> Settings:
    """把工厂读到的配置换成受控配置。

    必须做这一步：否则本机真实的 ``.env`` 会泄进测试，
    导致「换台机器结果就变了」的假绿。
    """
    settings = Settings.from_env(environ=env, env_file=env_file)
    monkeypatch.setattr("src.llm.factory.get_settings", lambda: settings)
    return settings


# --------------------------------------------------------------------------
# 正常构造
# --------------------------------------------------------------------------


def test_returns_langchain_chat_model(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    """必须返回 LangChain 统一抽象，后续 chain 与 LangGraph 才接得上。"""
    use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY=FAKE_KEY)
    assert isinstance(create_chat_model(), BaseChatModel)


def test_default_model_is_deepseek_chat(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY=FAKE_KEY)
    assert create_chat_model().model_name == DEFAULT_MODEL


def test_uses_configured_base_url(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    settings = use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY=FAKE_KEY)
    assert create_chat_model().openai_api_base == settings.deepseek_base_url


def test_custom_model_name(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY=FAKE_KEY)
    assert create_chat_model("deepseek-reasoner").model_name == "deepseek-reasoner"


@pytest.mark.parametrize(("temperature", "timeout"), [(0.0, 30.0), (0.9, 15.0), (1.5, 120.0)])
def test_hyperparams_passed_through(
    monkeypatch: pytest.MonkeyPatch, missing_env_file: Path, temperature: float, timeout: float
) -> None:
    """传进去的超参必须原样落到模型对象上，不能被默默改掉。"""
    use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY=FAKE_KEY)
    llm = create_chat_model(temperature=temperature, timeout=timeout)
    assert llm.temperature == temperature
    assert llm.request_timeout == timeout


# --------------------------------------------------------------------------
# 密钥
# --------------------------------------------------------------------------


def test_missing_api_key_gives_actionable_error(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    """报错必须说清「改哪个文件、填哪个变量」，否则用户只能猜。"""
    use_settings(monkeypatch, missing_env_file)
    with pytest.raises(ConfigError) as exc:
        create_chat_model()

    assert "DEEPSEEK_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)


def test_blank_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    """空白串等同没配，不能拿着空密钥去发请求。"""
    use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY="   ")
    with pytest.raises(ConfigError):
        create_chat_model()


def test_explicit_api_key_wins(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    """显式传入的密钥优先于配置文件——测试与临时切换账号都靠它。"""
    use_settings(monkeypatch, missing_env_file)
    assert create_chat_model(api_key="sk-explicit").openai_api_key.get_secret_value() == "sk-explicit"

    # 同时确认注入的配置真的生效了，而不是碰巧读了本机 .env
    assert not (missing_env_file.exists())


# --------------------------------------------------------------------------
# 离线保证
# --------------------------------------------------------------------------


def test_construction_makes_no_network_call(monkeypatch: pytest.MonkeyPatch, missing_env_file: Path) -> None:
    """构造期若发请求，就会在这里炸——这是整个测试套件能离线跑的前提。"""
    use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY=FAKE_KEY)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("构造聊天模型时不应发起网络请求")

    monkeypatch.setattr("httpx.Client.send", _boom)
    monkeypatch.setattr("httpx.AsyncClient.send", _boom)

    assert isinstance(create_chat_model(), BaseChatModel)


def test_api_key_never_appears_in_logs(
    monkeypatch: pytest.MonkeyPatch, missing_env_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """密钥一旦进日志，就会跟着日志文件和 CI 输出扩散出去。"""
    use_settings(monkeypatch, missing_env_file, DEEPSEEK_API_KEY=FAKE_KEY)

    with caplog.at_level(logging.INFO):
        create_chat_model()

    assert FAKE_KEY not in caplog.text
    # 反过来确认日志真的写了内容，避免「什么都没记」让上面那句假通过
    assert "创建聊天模型" in caplog.text
