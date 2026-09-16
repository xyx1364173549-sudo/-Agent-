"""配置中心：路径与端口。

管的是「文件放哪、服务监听哪」这类项目配置——数据库、向量库、日志目录、
API 监听地址。这些配置的共同点是：**和调用大模型无关**。

大模型的密钥与接口地址**不在这里**，而在 ``src/llm/factory.py``。
两部分刻意分开：模型接入层自己负责自己的连接配置，
换模型、换密钥时不必翻到这个文件里来。

设计原则：

1. **单一入口**：全项目只通过 ``get_settings()`` 读取，不散落 ``os.getenv``。
2. **可注入**：``from_env(environ=...)`` 允许测试完全替换环境变量来源，
   避免本机真实 ``.env`` 干扰测试结果。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# 项目根目录（本文件位于 src/ 下，上一级即根目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """配置缺失或取值非法时抛出。"""


def _to_int(raw: str, key: str, *, minimum: int, maximum: int) -> int:
    """把字符串转成整数并做区间校验。"""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"配置项 {key} 需要整数，实际收到 {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"配置项 {key} 需在 [{minimum}, {maximum}] 区间内，实际为 {value}")
    return value


def _to_path(raw: str, key: str) -> Path:
    """相对路径统一挂到项目根目录下，保证换目录运行结果一致。"""
    path = Path(str(raw).strip()).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def read_env_file(env_file: Path) -> dict[str, str]:
    """读取 ``.env`` 文件；文件不存在时返回空字典（视为全部走默认值）。"""
    if not env_file.exists():
        return {}

    try:
        from dotenv import dotenv_values
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确指引
        raise ConfigError("缺少 python-dotenv 依赖，请先执行 pip install -r requirements.txt") from exc

    # 用 dotenv_values 而非 load_dotenv：只解析不污染 os.environ，便于测试
    return {k: v for k, v in dotenv_values(env_file).items() if v is not None}


@dataclass(frozen=True)
class Settings:
    """项目配置对象。字段全部为不可变类型，创建后不允许就地修改。"""

    embedding_model: str
    api_host: str
    api_port: int
    memory_db_path: Path
    chroma_persist_dir: Path
    log_dir: Path
    env_file: Path

    # ---------- 构造 ----------

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        env_file: str | Path | None = None,
    ) -> "Settings":
        """从环境变量构造配置。

        参数
        ----
        environ:
            显式传入的环境变量映射。**传入时会完全替代系统环境变量**，
            以保证测试结果不受本机已有变量干扰；传 ``None`` 则读取 ``os.environ``。
        env_file:
            ``.env`` 文件路径，默认取项目根目录下的 ``.env``。
        """
        path = _to_path(str(env_file), "env_file") if env_file else (PROJECT_ROOT / ".env")

        # 优先级：.env 文件 < 环境变量（后者覆盖前者）
        merged: dict[str, str] = {**read_env_file(path), **(dict(os.environ) if environ is None else dict(environ))}

        def get(key: str, default: str | None = None) -> str | None:
            value = merged.get(key)
            if value is None or str(value).strip() == "":
                return default
            return str(value).strip()

        return cls(
            embedding_model=get("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
            api_host=get("API_HOST", "127.0.0.1") or "127.0.0.1",
            api_port=_to_int(get("API_PORT", "8000") or "8000", "API_PORT", minimum=1, maximum=65535),
            memory_db_path=_to_path(get("MEMORY_DB_PATH", "./data/memory.db") or "", "MEMORY_DB_PATH"),
            chroma_persist_dir=_to_path(get("CHROMA_PERSIST_DIR", "./chroma_db") or "", "CHROMA_PERSIST_DIR"),
            log_dir=_to_path(get("LOG_DIR", "./logs") or "", "LOG_DIR"),
            env_file=path,
        )

    # ---------- 便捷方法 ----------

    def ensure_dirs(self) -> list[Path]:
        """创建运行期需要的目录（数据目录、向量库目录、日志目录）。

        返回实际新建的目录列表；已存在的目录不重复创建。目录已在
        ``.gitignore`` 中排除，因此不会产生无意义的提交内容。
        """
        created: list[Path] = []
        for target in (self.memory_db_path.parent, self.chroma_persist_dir, self.log_dir):
            if not target.exists():
                target.mkdir(parents=True, exist_ok=True)
                created.append(target)
        return created

    def as_dict(self) -> dict[str, object]:
        """导出为字典，便于日志打印与接口调试。"""
        return {
            "embedding_model": self.embedding_model,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "memory_db_path": str(self.memory_db_path),
            "chroma_persist_dir": str(self.chroma_persist_dir),
            "log_dir": str(self.log_dir),
            "env_file": str(self.env_file),
        }


# 进程内单例缓存
_settings: Settings | None = None


def get_settings(*, reload: bool = False) -> Settings:
    """获取全局配置单例。

    ``reload=True`` 会重新读取环境变量，主要供测试与热更新场景使用。
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings.from_env()
    return _settings
