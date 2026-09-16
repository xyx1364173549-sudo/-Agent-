"""统一日志模块。

目标是「一次配置，处处可用」：
- ``setup_logging()`` 只在程序入口调用一次，挂载控制台与文件两个 handler；
- 业务代码只调 ``get_logger(__name__)``，不自己配置 handler。

幂等的关键：重复调用 ``setup_logging()`` 不会导致日志重复输出
（这是 Python logging 最常见的坑——同一个 logger 被反复 ``addHandler``）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 标记 handler 归属，避免重复挂载
_HANDLER_FLAG = "_learning_agent_handler"

_configured: bool = False


def _already_has(logger: logging.Logger, kind: str) -> bool:
    """判断该 logger 是否已挂载指定类型的 handler。"""
    return any(getattr(h, _HANDLER_FLAG, None) == kind for h in logger.handlers)


def _build_console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    setattr(handler, _HANDLER_FLAG, "console")
    return handler


def _build_file_handler(path: Path, level: int) -> logging.Handler | None:
    """创建文件 handler；目录不可写时返回 None，让日志降级为仅控制台输出。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:  # pragma: no cover - 依赖文件系统权限
        return None

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    setattr(handler, _HANDLER_FLAG, "file")
    return handler


def setup_logging(
    level: int = logging.INFO,
    *,
    log_dir: str | Path | None = None,
    log_file: str = "app.log",
    force: bool = False,
) -> Path | None:
    """初始化根 logger。

    返回日志文件路径；若文件 handler 创建失败（如目录无写权限）则返回 ``None``。
    ``force=True`` 会清空已有的本项目 handler 后重新挂载。
    """
    global _configured

    root = logging.getLogger()
    root.setLevel(level)

    if force:
        for handler in list(root.handlers):
            if getattr(handler, _HANDLER_FLAG, None):
                root.removeHandler(handler)
        _configured = False

    if _configured:
        return next(
            (Path(h.baseFilename) for h in root.handlers if getattr(h, _HANDLER_FLAG, None) == "file"),
            None,
        )

    if not _already_has(root, "console"):
        root.addHandler(_build_console_handler(level))

    target_dir = Path(log_dir) if log_dir else get_settings().log_dir

    file_path: Path | None = None
    if not _already_has(root, "file"):
        handler = _build_file_handler(target_dir / log_file, level)
        if handler is not None:
            root.addHandler(handler)
            file_path = Path(handler.baseFilename)

    _configured = True
    return file_path


def get_logger(name: str | None = None) -> logging.Logger:
    """获取业务 logger。首次调用会自动完成一次默认初始化，保证不漏日志。"""
    if not _configured:
        setup_logging()
    return logging.getLogger(name or PROJECT_ROOT.name)
