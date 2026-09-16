"""DeepSeek 模型接入测试。

只盯两件真正要紧的事：

1. **不联网** —— 构造模型只是拼参数，不发请求，所以测试能离线跑；
2. **读得到配置** —— .env 里的密钥与地址确实被读进来了。

超参是否越界交给 SDK 管，这里不重复校验，省得两处规则打架。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_deepseek import ChatDeepSeek

from src.llm import DEFAULT_MODEL, create_chat_model, factory

HAS_ENV_FILE = (factory.PROJECT_ROOT / ".env").exists()


# --------------------------------------------------------------------------
# 正常构造
# --------------------------------------------------------------------------


def test_returns_chat_deepseek() -> None:
    """返回的必须是 LangChain 的聊天模型对象，后续 chain 与 LangGraph 才接得上。"""
    assert isinstance(create_chat_model(), ChatDeepSeek)


def test_default_model_name() -> None:
    assert DEFAULT_MODEL == "deepseek-v4-flash"


def test_uses_default_model_when_not_specified() -> None:
    assert create_chat_model().model_name == DEFAULT_MODEL


def test_custom_model_name() -> None:
    assert create_chat_model("deepseek-chat").model_name == "deepseek-chat"


@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0, 2.0])
def test_temperature_passed_through(temperature: float) -> None:
    """传进去的温度要原样落到模型对象上，不能被默默改掉。"""
    assert create_chat_model(temperature=temperature).temperature == temperature


def test_base_url_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "DEEPSEEK_BASE_URL", "https://example.test/v1")
    assert create_chat_model().api_base == "https://example.test/v1"


# --------------------------------------------------------------------------
# 缺密钥
# --------------------------------------------------------------------------


def test_missing_api_key_gives_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """报错必须说清「改哪个文件、填哪个变量」，否则用户只能猜。"""
    monkeypatch.setattr(factory, "DEEPSEEK_API_KEY", None)

    with pytest.raises(RuntimeError) as exc:
        create_chat_model()

    message = str(exc.value)
    assert "DEEPSEEK_API_KEY" in message
    assert ".env" in message
    # 用户上次就是把密钥填进了模板文件，提示里要专门点出来
    assert ".env.example" in message


def test_empty_api_key_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """写成 ``DEEPSEEK_API_KEY=``（空值）等同没配，不能拿着空密钥去发请求。"""
    monkeypatch.setattr(factory, "DEEPSEEK_API_KEY", "")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        create_chat_model()


# --------------------------------------------------------------------------
# 离线保证
# --------------------------------------------------------------------------


def test_construction_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造期若发请求，就会在这里炸——这是整个测试套件能离线跑的前提。"""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("构造模型时不应发起网络请求")

    monkeypatch.setattr("httpx.Client.send", _boom)
    monkeypatch.setattr("httpx.AsyncClient.send", _boom)
    monkeypatch.setattr(factory, "DEEPSEEK_API_KEY", "sk-fake")

    assert isinstance(create_chat_model(), ChatDeepSeek)


# --------------------------------------------------------------------------
# .env 读取（本机环境自检）
# --------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ENV_FILE, reason="本机没有 .env 文件，跳过环境自检")
def test_project_env_file_is_actually_loaded() -> None:
    """模块被 import 时确实读了项目根目录的 .env。

    这条测试专门守着上次踩的坑：密钥填错文件（填进了 .env.example），
    程序却读 .env，结果「明明配了却读不到」。
    """
    env_file = factory.PROJECT_ROOT / ".env"
    assert env_file.exists()
    assert factory.DEEPSEEK_API_KEY, "应能从 .env 读到 DEEPSEEK_API_KEY"
    assert factory.DEEPSEEK_API_KEY.startswith("sk-")
    assert factory.DEEPSEEK_BASE_URL.startswith("http")


@pytest.mark.skipif(not HAS_ENV_FILE, reason="本机没有 .env 文件，跳过环境自检")
def test_env_file_is_gitignored() -> None:
    """守死一条底线：.env 绝不能被 git 跟踪。"""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", ".env"],
        cwd=factory.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, ".env 未被 .gitignore 排除，密钥有泄漏风险"
