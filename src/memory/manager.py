"""记忆管理器：三种记忆的统一入口。

前面三个文件分别实现了工作记忆、情景记忆、语义记忆，但它们目前还是三个孤岛——
上层 Agent 要用的话，得分别 new 三个对象、各自传一遍参数；而且「记一句话」这件事
往往要同时动两三处：既要留原文，又要抽事实，有时还要记个事件。

这个文件把它们收拢成一个入口::

    mem = MemoryManager(session_id="s1", user_id="u1")
    mem.add_message("user", "我最近在学递归，感觉有点难")
    mem.record_event("struggled", "练习 5 题错 3 题", importance=0.8)
    print(mem.stats())

三种记忆的分工不变：

    工作记忆   这次会话聊过什么 —— 会随窗口滑走
    情景记忆   发生过什么事件   —— 长期留
    语义记忆   这个人是什么水平 —— 跨会话，最稳定

**会话（session）和用户（user）是两个维度**：同一个人可能开很多次会话，
工作记忆和情景记忆按会话分，语义记忆按用户分——这样换个会话，学到的
事实仍然认得出是同一个人。
"""

from pathlib import Path

from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.working import WorkingMemory


class MemoryManager:
    """一个会话 + 一个用户的三种记忆。

    三层记忆作为属性直接可用，需要细粒度操作时不必绕路::

        mem.working.messages()          # 取当前窗口的对话
        mem.episodic.by_type("struggled")  # 取某类事件
        mem.semantic.all_facts()        # 取全部沉淀事实
    """

    def __init__(
        self,
        session_id: str,
        user_id: str | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        """初始化三种记忆，共用同一个数据库文件。

        参数
        ----
        session_id:
            会话 id，工作记忆与情景记忆按它划分。
        user_id:
            用户 id，语义记忆按它划分。**不传时退化成跟 session_id 一样**，
            相当于「一个人只开一次会话」——方便临时试跑，但正式使用时
            应该显式传，否则换会话就认不出是同一个人了。
        db_path:
            数据库文件路径，不传时用 ``.env`` 里的 ``MEMORY_DB_PATH``。
        """
        self.session_id = session_id
        self.user_id = user_id or session_id

        self.working = WorkingMemory(session_id, db_path=db_path)
        self.episodic = EpisodicMemory(session_id, db_path=db_path)
        self.semantic = SemanticMemory(self.user_id, db_path=db_path)

    # ---------- 写入 ----------

    def add_message(self, role: str, content: str, *, learn: bool = True) -> list[dict[str, str]]:
        """记一条对话消息。

        做两件事：把原文写进工作记忆；如果是用户说的话，顺带**抽一遍事实**
        写进语义记忆（抽不到就什么也不做）。

        返回这次抽到并处理了哪些事实，调 ``learn=False`` 可关掉自动抽取。

        为什么只抽用户说的话？因为模型自己的回答不是「关于用户的事实」，
        抽了只会污染语义记忆。
        """
        self.working.add(role, content)

        if not learn or role != "user":
            return []
        return self.semantic.learn_from_text(content)

    def record_event(self, event_type: str, content: str, importance: float = 0.5) -> int:
        """记一件发生的事，返回事件 id。"""
        return self.episodic.record(event_type, content, importance)

    def learn(self, text: str, confidence: float = 0.5) -> list[dict[str, str]]:
        """从一段文字里抽事实并记下来（不写工作记忆）。

        适合「用户的话已经记过了，但想手动补一次抽取」或「从外部材料学事实」。
        """
        return self.semantic.learn_from_text(text, confidence)

    # ---------- 状态 ----------

    def stats(self) -> dict[str, int]:
        """看一眼三层记忆各存了多少东西，调试时很省事。"""
        return {
            "messages": self.working.count(),
            "events": self.episodic.count(),
            "facts": self.semantic.count(),
        }

    # ---------- 维护 ----------

    def clear(self) -> None:
        """清空这个会话的记忆。

        注意：语义记忆是按**用户**清的——因为它本来就跨会话共享，
        只清当前会话的重开一次还会读到，反而让人困惑。
        """
        self.working.clear()
        self.episodic.clear()
        self.semantic.clear()

    def close(self) -> None:
        """关闭三个数据库连接。"""
        for memory in (self.working, self.episodic, self.semantic):
            memory.close()
