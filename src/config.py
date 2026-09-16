"""配置中心。

三条设计原则：

1. **单一入口**：全项目只通过 ``get_settings()`` 读取配置，不散落 ``os.getenv``。
2. **缺密钥不炸导入**：读取配置时只做类型转换，不在导入期校验密钥。
   只有真正要调外部服务时，才用 ``Settings.require_api_key()`` 强校验。
   这样单元测试和离线开发不会因为缺 ``.env`` 而整体崩掉。
3. **可注入**：``from_env(environ=...)`` 允许测试完全替换环境变量来源，避免真实
   环境干扰测试结果。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# 项目根目录（本文件位于 src/ 下，上一级即根目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 支持的模型提供方 -> 对应的密钥字段名
PROVIDER_KEY_FIELD: dict[str, str] = {
    "deepseek": "deepseek_api_key",
    "mimo": "mimo_api_key",
}


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


def _to_url(raw: str, key: str) -> str:
    """接口地址必须以 http(s):// 开头，否则八成是填错了。"""
    value = str(raw).strip().rstrip("/")
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"配置项 {key} 需为 http(s):// 开头的地址，实际为 {value!r}")
    return value


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
    """全项目配置对象。字段全部为不可变类型，创建后不允许就地修改。"""

    deepseek_api_key: str | None
    deepseek_base_url: str
    mimo_api_key: str | None
    mimo_base_url: str
    mimo_model: str
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
        strict: bool = False,
    ) -> "Settings":
        """从环境变量构造配置。

        参数
        ----
        environ:
            显式传入的环境变量映射。**传入时会完全替代系统环境变量**，
            以保证测试结果不受本机已有变量干扰；传 ``None`` 则读取 ``os.environ``。
        env_file:
            ``.env`` 文件路径，默认取项目根目录下的 ``.env``。
        strict:
            为 ``True`` 时，缺少任一模型 API Key 直接抛 ``ConfigError``；
            为 ``False``（默认）时允许密钥为空，推迟到调用前再校验。
        """
        path = _to_path(str(env_file), "env_file") if env_file else (PROJECT_ROOT / ".env")

        # 优先级：.env 文件 < 环境变量（后者覆盖前者）
        merged: dict[str, str] = {**read_env_file(path), **(dict(os.environ) if environ is None else dict(environ))}

        def get(key: str, default: str | None = None) -> str | None:
            value = merged.get(key)
            if value is None or str(value).strip() == "":
                return default
            return str(value).strip()

        settings = cls(
            deepseek_api_key=get("DEEPSEEK_API_KEY"),
            deepseek_base_url=_to_url(get("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "", "DEEPSEEK_BASE_URL"),
            mimo_api_key=get("MIMO_API_KEY"),
            mimo_base_url=_to_url(get("MIMO_BASE_URL", "https://api.xiaomi.com/v1") or "", "MIMO_BASE_URL"),
            mimo_model=get("MIMO_MODEL", "mimo-v2.5") or "mimo-v2.5",
            embedding_model=get("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
            api_host=get("API_HOST", "127.0.0.1") or "127.0.0.1",
            api_port=_to_int(get("API_PORT", "8000") or "8000", "API_PORT", minimum=1, maximum=65535),
            memory_db_path=_to_path(get("MEMORY_DB_PATH", "./data/memory.db") or "", "MEMORY_DB_PATH"),
            chroma_persist_dir=_to_path(get("CHROMA_PERSIST_DIR", "./chroma_db") or "", "CHROMA_PERSIST_DIR"),
            log_dir=_to_path(get("LOG_DIR", "./logs") or "", "LOG_DIR"),
            env_file=path,
        )

        if strict:
            for provider in PROVIDER_KEY_FIELD:
                settings.require_api_key(provider)

        return settings

    # ---------- 使用期校验 ----------

    def require_api_key(self, provider: str) -> str:
        """取出指定提供方的 API Key；未配置时抛出可读错误。

        密钥的校验刻意推迟到这一步，是为了让「不调用模型」的代码路径
        （如单元测试、离线索引构建）无需配置任何密钥即可运行。
        """
        field_name = PROVIDER_KEY_FIELD.get(provider)
        if field_name is None:
            supported = "、".join(sorted(PROVIDER_KEY_FIELD))
            raise ConfigError(f"未知的模型提供方 {provider!r}，当前支持：{supported}")

        value = getattr(self, field_name)
        if not value:
            env_name = provider.upper() + "_API_KEY"
            raise ConfigError(
                f"未配置 {provider} 的 API Key。请在 {self.env_file} 中设置 {env_name}，"
                f"或参考 .env.example 生成配置文件。"
            )
        return value

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

    def as_dict(self, *, mask_secrets: bool = True) -> dict[str, object]:
        """导出为字典，便于日志打印与接口调试。默认对密钥打码。"""

        def mask(value: str | None) -> str | None:
            if value is None:
                return None
            if not mask_secrets:
                return value
            return value[:4] + "***" + value[-2:] if len(value) > 8 else "***"

        return {
            "deepseek_api_key": mask(self.deepseek_api_key),
            "deepseek_base_url": self.deepseek_base_url,
            "mimo_api_key": mask(self.mimo_api_key),
            "mimo_base_url": self.mimo_base_url,
            "mimo_model": self.mimo_model,
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
