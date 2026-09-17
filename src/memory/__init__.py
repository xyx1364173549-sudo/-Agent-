"""分层记忆三件套（对应计划 M2、论文第 4 章）。

- 工作记忆 WorkingMemory：当前会话的短期上下文
- 情景记忆 EpisodicMemory：发生过的事件与交互轨迹
- 语义记忆 SemanticMemory：沉淀下来的稳定事实与结论
- 记忆管理器 MemoryManager：把上面三种收拢成一个入口
- 上下文组装 build_context：三层合并成一段可直接喂给模型的文字
- 遗忘与巩固 forgetting.strength：按半衰期计算记忆强度

三种记忆统一存在**同一个 SQLite 文件**里，各占一张表，底层连接、建表与
版本升级由 ``store.py`` 负责。

日常使用推荐直接从 ``MemoryManager`` 入手：``mem.add_message(...)`` 写入，
``mem.context()`` 取出组装好的上下文。需要细粒度操作时，再到它下面
三个属性上去取。
"""

from src.memory.context import DEFAULT_MAX_TOKENS, build_context
from src.memory.episodic import EpisodicMemory
from src.memory.forgetting import (
    DEFAULT_FORGET_THRESHOLD,
    DEFAULT_HALF_LIFE_DAYS,
    days_since,
    strength,
)
from src.memory.manager import MemoryManager
from src.memory.semantic import SemanticMemory, extract_facts
from src.memory.store import SCHEMA_VERSION, get_connection, init_db, now_iso
from src.memory.working import WorkingMemory, count_tokens

__all__ = [
    "DEFAULT_FORGET_THRESHOLD",
    "DEFAULT_HALF_LIFE_DAYS",
    "DEFAULT_MAX_TOKENS",
    "SCHEMA_VERSION",
    "EpisodicMemory",
    "MemoryManager",
    "SemanticMemory",
    "WorkingMemory",
    "build_context",
    "count_tokens",
    "days_since",
    "extract_facts",
    "get_connection",
    "init_db",
    "now_iso",
    "strength",
]
