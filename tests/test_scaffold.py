"""工程骨架冒烟测试。

保证目录结构、包可导入性与关键约定不被误改。
这类测试很「浅」，但能在重构时第一时间报出问题，性价比很高。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

EXPECTED_PACKAGES = [
    "src",
    "src.config",
    "src.utils",
    "src.utils.logger",
    "src.llm",
    "src.memory",
    "src.rag",
    "src.planning",
    "src.agents",
    "src.api",
    "scripts",
    "scripts.gen_progress",
]

EXPECTED_DIRS = ["src", "tests", "docs", "data", "scripts"]

EXPECTED_FILES = ["README.md", "PLAN.md", "AGENTS.md", "pytest.ini", ".env.example", ".gitignore"]


@pytest.mark.parametrize("name", EXPECTED_PACKAGES)
def test_package_importable(name: str) -> None:
    assert importlib.import_module(name) is not None


@pytest.mark.parametrize("name", EXPECTED_DIRS)
def test_directory_exists(project_root: Path, name: str) -> None:
    assert (project_root / name).is_dir(), f"缺少目录：{name}"


@pytest.mark.parametrize("name", EXPECTED_FILES)
def test_key_file_exists(project_root: Path, name: str) -> None:
    assert (project_root / name).is_file(), f"缺少文件：{name}"


def test_version_is_declared() -> None:
    import src

    assert isinstance(src.__version__, str)
    assert src.__version__.count(".") == 2


def test_gitignore_protects_secrets(project_root: Path) -> None:
    """密钥与数据库绝不能进仓库——这是不可逆事故，值得一条测试守着。"""
    content = (project_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in [".env", ".venv/", "__pycache__/", "*.db", "chroma_db/"]:
        assert pattern in content, f".gitignore 缺少排除项：{pattern}"


def test_env_example_documents_required_vars(project_root: Path) -> None:
    """模板即文档：换台机器时能否跑起来，取决于模板是否写全。"""
    content = (project_root / ".env.example").read_text(encoding="utf-8")
    for var in ["DEEPSEEK_API_KEY", "MIMO_API_KEY", "MIMO_BASE_URL", "MEMORY_DB_PATH", "API_PORT"]:
        assert var in content, f".env.example 未说明变量：{var}"
