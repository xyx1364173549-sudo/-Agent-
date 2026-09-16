"""配置中心测试（路径与端口部分）。

重点不在「能不能读到值」，而在**异常路径**——类型错、越界、路径解析、
目录幂等创建。这些才是上线后会咬人的地方。

注意：大模型的密钥不在本模块测试范围内，它归 ``src/llm/factory.py`` 管，
对应的测试在 ``tests/test_llm_factory.py``。
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
    """零配置也要能构造出可用的 Settings。"""
    s = build(missing_env_file)
    assert s.api_host == "127.0.0.1"
    assert s.api_port == 8000
    assert s.embedding_model == "text-embedding-3-small"


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
    s = build(missing_env_file, API_PORT="   ", EMBEDDING_MODEL="")
    assert s.api_port == 8000
    assert s.embedding_model == "text-embedding-3-small"


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


# --------------------------------------------------------------------------
# .env 文件加载与优先级
# --------------------------------------------------------------------------


def test_env_file_is_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("API_PORT=9000\nEMBEDDING_MODEL=custom-embedding\n", encoding="utf-8")
    s = Settings.from_env(environ={}, env_file=env_file)
    assert s.api_port == 9000
    assert s.embedding_model == "custom-embedding"


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
    monkeypatch.delenv("LOG_DIR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LOG_DIR=./leaked-logs\n", encoding="utf-8")
    Settings.from_env(environ=None, env_file=env_file)
    assert "LOG_DIR" not in os.environ


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


def test_settings_is_immutable(missing_env_file: Path) -> None:
    """frozen dataclass：防止某处代码偷偷改全局配置引发蝴蝶效应。"""
    s = build(missing_env_file)
    with pytest.raises(FrozenInstanceError):
        s.api_port = 9999  # type: ignore[misc]
