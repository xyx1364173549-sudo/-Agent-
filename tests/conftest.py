"""pytest 全局夹具与路径准备。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录。"""
    return PROJECT_ROOT


@pytest.fixture
def missing_env_file(tmp_path: Path) -> Path:
    """指向一个不存在的 .env 文件。

    测试配置中心时必须用它，否则本机真实的 ``.env`` 会泄进测试，
    造成「换个环境结果就变了」的假绿。
    """
    return tmp_path / "absent.env"
