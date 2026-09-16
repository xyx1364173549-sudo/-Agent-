"""配置中心测试。

重点不在「能不能读到值」，而在**异常路径**——缺键、类型错、越界、
地址非法、密钥打码、目录幂等创建。这些才是上线后会咬人的地方。
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, ConfigError, Settings, get_settings


def build(env_file: Path, **env: str) -> Settings:
    """构造一份完全受控的配置。

    显式传入 ``environ`` 会**完全替代**系统环境变量，因此本机真实的
    ``.env`` 与已有的同名环境变量都不会干扰断言。
    """
    return Settings.from_env(environ=env, env_file=env_file)


# --------------------------------------------------------------------------
# 默认值与路径解析
# --------------------------------------------------------------------------


def test_defaults_when_nothing_configured(missing_env_file: Path) -> None:
    """零配置也要能构造出可用的 Settings（密钥为空，但不报错）。"""
    s = build(missing_env_file)
    assert s.deepseek_api_key is None
    assert s.mimo_api_key is None
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8000
    assert s.mimo_model == "mimo-v2.5"


def test_relative_path_anchored_to_project_root(missing_env_file: Path) -> None:
    """相对路径必须挂到项目根目录，否则换个工作目录跑就找不到库文件。"""
    s = build(missing_env_file)
    assert s.memory_db_path == PROJECT_ROOT / "data" / "memory.db"
    assert s.chroma_persist_dir == PROJECT_ROOT / "chroma_db"


def test_absolute_path_kept_as_is(missing_env_file: Path, tmp_path: Path) -> None:
    target = tmp_path / "custom" / "memory.db"
    s = build(missing_env_file, MEMORY_DB_PATH=str(target))
    assert s.memory_db_path == target


def test_blank_value_falls_back_to_default(missing_env_file: Path) -> None:
    """空白值等同于「没配置」，不能把端口变成空字符串再炸在 int() 上。"""
    s = build(missing_env_file, API_PORT="   ", MIMO_MODEL="")
    assert s.api_port == 8000
    assert s.mimo_model == "mimo-v2.5"


# --------------------------------------------------------------------------
# 非法取值
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["abc", "80.5", "1e3", "--"])
def test_port_must_be_integer(missing_env_file: Path, raw: str) -> None:
    with pytest.raises(ConfigError, match="需要整数"):
        build(missing_env_file, API_PORT=raw)


@pytest.mark.parametrize("raw", ["0", "65536", "-1"])
def test_port_must_be_in_valid_range(missing_env_file: Path, raw: str) -> None:
    with pytest.raises(ConfigError, match="区间"):
        build(missing_env_file, API_PORT=raw)


def test_base_url_must_look_like_url(missing_env_file: Path) -> None:
    with pytest.raises(ConfigError, match="http"):
        build(missing_env_file, MIMO_BASE_URL="api.xiaomi.com")


def test_base_url_trailing_slash_stripped(missing_env_file: Path) -> None:
    s = build(missing_env_file, DEEPSEEK_BASE_URL="https://api.deepseek.com/")
    assert s.deepseek_base_url == "https://api.deepseek.com"


# --------------------------------------------------------------------------
# 密钥校验：延迟到使用时
# --------------------------------------------------------------------------


def test_api_key_not_required_at_construction(missing_env_file: Path) -> None:
    """构造期不该因为缺密钥就失败——离线建索引、跑测试都不需要密钥。"""
    s = build(missing_env_file)
    assert s.require_api_key.__self__ is s  # 仅确认方法可访问，未触发校验


def test_require_api_key_raises_with_actionable_message(missing_env_file: Path) -> None:
    s = build(missing_env_file)
    with pytest.raises(ConfigError) as exc:
        s.require_api_key("deepseek")
    # 报错要说清「改哪个文件、填哪个变量」，否则用户只能猜
    assert "DEEPSEEK_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)


def test_require_api_key_returns_value(missing_env_file: Path) -> None:
    s = build(missing_env_file, MIMO_API_KEY="sk-mimo-abcdefgh")
    assert s.require_api_key("mimo") == "sk-mimo-abcdefgh"


def test_require_api_key_rejects_unknown_provider(missing_env_file: Path) -> None:
    s = build(missing_env_file)
    with pytest.raises(ConfigError, match="未知的模型提供方"):
        s.require_api_key("openai")


def test_strict_mode_rejects_missing_keys(missing_env_file: Path) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(environ={}, env_file=missing_env_file, strict=True)


def test_strict_mode_passes_when_all_keys_present(missing_env_file: Path) -> None:
    s = Settings.from_env(
        environ={"DEEPSEEK_API_KEY": "sk-ds", "MIMO_API_KEY": "sk-mimo"},
        env_file=missing_env_file,
        strict=True,
    )
    assert s.deepseek_api_key == "sk-ds"


# --------------------------------------------------------------------------
# 密钥打码
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("sk-abcdefghijklmn", "sk-a***mn"), ("short", "***"), ("12345678", "***")],
)
def test_as_dict_masks_secrets(missing_env_file: Path, raw: str, expected: str) -> None:
    """密钥绝不能整串落进日志或接口响应，打码逻辑要卡住边界长度。"""
    s = build(missing_env_file, DEEPSEEK_API_KEY=raw)
    assert s.as_dict()["deepseek_api_key"] == expected


def test_as_dict_can_reveal_when_explicitly_asked(missing_env_file: Path) -> None:
    s = build(missing_env_file, DEEPSEEK_API_KEY="sk-abcdefghijklmn")
    assert s.as_dict(mask_secrets=False)["deepseek_api_key"] == "sk-abcdefghijklmn"


def test_as_dict_leaves_none_untouched(missing_env_file: Path) -> None:
    assert build(missing_env_file).as_dict()["mimo_api_key"] is None


# --------------------------------------------------------------------------
# .env 文件加载与优先级
# --------------------------------------------------------------------------


def test_env_file_is_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("API_PORT=9000\nMIMO_MODEL=custom-model\n", encoding="utf-8")
    s = Settings.from_env(environ={}, env_file=env_file)
    assert s.api_port == 9000
    assert s.mimo_model == "custom-model"


def test_environ_overrides_env_file(tmp_path: Path) -> None:
    """真实环境变量优先级高于 .env 文件——这是部署时的覆盖手段。"""
    env_file = tmp_path / ".env"
    env_file.write_text("API_PORT=9000\n", encoding="utf-8")
    s = Settings.from_env(environ={"API_PORT": "9100"}, env_file=env_file)
    assert s.api_port == 9100


def test_env_file_absent_is_not_an_error(missing_env_file: Path) -> None:
    assert Settings.from_env(environ={}, env_file=missing_env_file).api_port == 8000


def test_env_file_does_not_pollute_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """读取 .env 只解析、不写回 os.environ，否则测试之间会互相污染。"""
    monkeypatch.delenv("MIMO_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("MIMO_MODEL=leaked-model\n", encoding="utf-8")
    Settings.from_env(environ=None, env_file=env_file)
    assert "MIMO_MODEL" not in os.environ


# --------------------------------------------------------------------------
# 目录创建
# --------------------------------------------------------------------------


def test_ensure_dirs_creates_missing_only(missing_env_file: Path, tmp_path: Path) -> None:
    s = build(
        missing_env_file,
        MEMORY_DB_PATH=str(tmp_path / "data" / "memory.db"),
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
        LOG_DIR=str(tmp_path / "logs"),
    )
    created = s.ensure_dirs()
    assert len(created) == 3
    # 第二次调用不该重复创建（幂等）
    assert s.ensure_dirs() == []
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "chroma").is_dir()


def test_ensure_dirs_reuses_existing_parent(missing_env_file: Path, tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    s = build(
        missing_env_file,
        MEMORY_DB_PATH=str(tmp_path / "data" / "memory.db"),
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
        LOG_DIR=str(tmp_path / "logs"),
    )
    assert (tmp_path / "data") not in s.ensure_dirs()


# --------------------------------------------------------------------------
# 单例
# --------------------------------------------------------------------------


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_get_settings_reload_returns_new_object() -> None:
    first = get_settings()
    second = get_settings(reload=True)
    assert second is not first
    assert second.progress_safe_equal(first) if hasattr(second, "progress_safe_equal") else True


def test_settings_is_immutable(missing_env_file: Path) -> None:
    """frozen dataclass：防止某处代码偷偷改全局配置引发蝴蝶效应。"""
    s = build(missing_env_file)
    with pytest.raises(Exception):
        s.api_port = 9999  # type: ignore[misc]
