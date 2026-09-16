"""日志模块测试。

核心关注点是**幂等**：``setup_logging()`` 被调用多次时，
不能出现同一条日志打印两遍的情况（Python logging 的经典坑）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.utils.logger import _HANDLER_FLAG, get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_logging(tmp_path: Path):
    """每个用例用独立目录重建日志配置，避免用例之间互相影响。

    teardown 必须同时把模块级 ``_configured`` 复位、并摘掉全部 handler，
    否则下一个用例会遇到「已配置但无 handler」的中间态，日志静默丢失。
    """
    from src.utils import logger as logger_module

    setup_logging(log_dir=tmp_path, force=True)
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    logger_module._configured = False


def _handlers(kind: str) -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, _HANDLER_FLAG, None) == kind]


def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    """重复调用不能重复挂 handler。"""
    setup_logging(log_dir=tmp_path)
    setup_logging(log_dir=tmp_path)
    setup_logging(log_dir=tmp_path)
    assert len(_handlers("file")) == 1
    assert len(_handlers("console")) == 1


def test_log_is_written_to_file(tmp_path: Path) -> None:
    setup_logging(log_dir=tmp_path, force=True)
    get_logger("test.channel").info("hello-from-test")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "hello-from-test" in content
    assert "test.channel" in content


def test_log_file_returns_path(tmp_path: Path) -> None:
    path = setup_logging(log_dir=tmp_path, force=True)
    assert path == tmp_path / "app.log"


def test_get_logger_auto_initializes() -> None:
    """即使忘了调 setup_logging，也要能拿到可用 logger，而不是丢日志。"""
    from src.utils import logger as logger_module

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    logger_module._configured = False  # 模拟「全新进程，尚未初始化」

    logger = get_logger("test.auto")
    assert isinstance(logger, logging.Logger)
    assert root.handlers, "首次取 logger 应自动完成初始化"


def test_log_dir_created_automatically(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "logs"
    setup_logging(log_dir=target, force=True)
    assert target.is_dir()


def test_unwritable_log_dir_degrades_gracefully(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """日志目录不可写时，应降级为仅控制台输出，而不是让程序直接崩。"""
    from src.utils import logger as logger_module

    monkeypatch.setattr(logger_module, "_build_file_handler", lambda *a, **k: None)
    result = setup_logging(log_dir=tmp_path, force=True)
    assert result is None
    assert len(_handlers("console")) == 1
