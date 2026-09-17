"""会话缓存：把「学到哪了」留在进程里。

LangGraph 的 checkpointer 已经把图的状态存下来了，但那个存档是按
``thread_id`` 找的；Web 层还需要 ``LearningSession`` 对象本身——它握着数据库
连接、学习者画像和记忆管理器。所以这里再放一层索引，把
``session_id → LearningSession`` 记下来。

**为什么放内存而不是数据库**：这部分状态天然属于进程。服务器重启后丢掉的只有
「这一轮做到第几题」，而画像、记忆、学习路径全都在 SQLite 里躺着，重新开一场
会话接着学就行。要多实例部署时把它换成 Redis——接口就下面这几个方法，
替换成本很低。
"""

import threading
from typing import Any

from src.planning.graph import LearningSession
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SessionNotFound(KeyError):
    """会话不存在（没建过，或者服务重启后缓存没了）。"""


class SessionStore:
    """``session_id`` 到 ``LearningSession`` 的映射。

    加了锁：FastAPI 的同步端点跑在线程池里，两个请求可能同时进来，
    而不带锁的字典在并发写入时行为不确定。
    """

    def __init__(self, *, db_path: Any = None, model: Any = None) -> None:
        """参数
        ----
        db_path:
            数据库路径，传给每个新会话。不传用 ``.env`` 里的配置。
        model:
            聊天模型。不传时每个会话各自创建默认的 DeepSeek 模型；
            **测试时塞一个假模型进来**，整个 API 就能离线跑通。
        """
        self._db_path = db_path
        self._model = model
        self._items: dict[str, LearningSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        goal: str,
        user_id: str,
        session_id: str,
        max_topics: int = 8,
    ) -> LearningSession:
        """建一场会话（不跑图，等前端来拉才开始）。

        同一个 ``session_id`` 已经存在时抛 ``ValueError``，而不是悄悄复用——
        复用会让「我以为开了新课，结果接着上次的进度」这种困惑很难排查。
        """
        with self._lock:
            if session_id in self._items:
                raise ValueError(f"会话 {session_id} 已存在，换个 id 或先删掉它")

            session = LearningSession(
                goal=goal,
                user_id=user_id,
                session_id=session_id,
                model=self._model,
                db_path=self._db_path,
                max_topics=max_topics,
            )
            self._items[session_id] = session

        logger.info("会话已创建 | session=%s | user=%s | goal=%s", session_id, user_id, goal)
        return session

    def get(self, session_id: str) -> LearningSession | None:
        """取一个会话，没有则返回 ``None``。"""
        with self._lock:
            return self._items.get(session_id)

    def require(self, session_id: str) -> LearningSession:
        """取一个会话，没有就抛 ``SessionNotFound``。

        和 ``get`` 分开，是为了让调用方自己决定「找不到」算不算异常——
        查询类接口适合返回 404，而内部流程适合直接抛。
        """
        session = self.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        return session

    def drop(self, session_id: str) -> bool:
        """删掉一场会话并关掉它的数据库连接。返回是否真的删到了。"""
        with self._lock:
            session = self._items.pop(session_id, None)

        if session is None:
            return False

        session.close()
        logger.info("会话已删除 | session=%s", session_id)
        return True

    def count(self) -> int:
        """当前缓存了几场会话。"""
        with self._lock:
            return len(self._items)

    def close_all(self) -> None:
        """关掉全部会话的连接。服务停止时调用，避免 SQLite 连接悬着。"""
        with self._lock:
            sessions = list(self._items.values())
            self._items.clear()

        for session in sessions:
            session.close()
        logger.info("已关闭 %d 场会话", len(sessions))
