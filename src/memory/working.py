"""工作记忆：当前会话的短期上下文。

它就像**手边的一张草稿纸**：写满了就得把最旧的擦掉，只留最近的内容。

为什么不能把所有对话都留下？两个原因：

1. **模型有上下文长度上限**，塞太多会被拒绝；
2. **token 是要花钱的**，历史越长每轮调用越贵。

所以工作记忆只管两件事：

    滑动窗口   一个会话最多留 N 条消息，超了就删最老的
    token 预算 取出来喂给模型时，总量不超过预算，超了同样从最老的开始丢

为什么丢老的而不是丢新的？因为**最新的对话最重要**——用户刚说的话，
肯定比十轮之前那句更相关。
"""

import sqlite3
from pathlib import Path

import tiktoken

from src.memory.store import get_connection, now_iso

# 分词器。tiktoken 用来数 token。
#
# 注意：DeepSeek 用的是它自己的分词器，这里数出来的数量会有少量偏差。
# 但做「预算裁剪」这种估算完全够用，而且不需要真的调一次 API 才能知道长度。
_ENCODER = tiktoken.get_encoding("cl100k_base")

# 默认最多留 20 条消息（一问一答算 2 条，也就是约 10 轮对话）
DEFAULT_MAX_MESSAGES = 20

# 默认单次取给模型的 token 上限
DEFAULT_MAX_TOKENS = 2000


def count_tokens(text: str) -> int:
    """数一段文字大概有多少 token。

    实测参考：``"你好"`` 是 2 个 token，``"a" * 100`` 是 13 个 token。
    中文比英文「贵」得多——一个汉字约 2 个 token，而一串英文单词常被合并成一个。
    """
    return len(_ENCODER.encode(text))


class WorkingMemory:
    """某个会话的工作记忆。

    用法::

        mem = WorkingMemory("session-1")
        mem.add("user", "我想学递归")
        mem.add("assistant", "好，我们从最简单的开始")
        print(mem.messages())     # 取出来喂给模型
    """

    def __init__(
        self,
        session_id: str,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        db_path: str | Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self._conn: sqlite3.Connection = get_connection(db_path)

    # ---------- 写入 ----------

    def add(self, role: str, content: str) -> None:
        """写一条消息，并把超出条数上限的最老消息删掉。

        role 是 ``"user"``（用户说的）或 ``"assistant"``（模型答的）。
        """
        self._conn.execute(
            "INSERT INTO working_memory (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (self.session_id, role, content, now_iso()),
        )

        # 滑动窗口：删掉「不属于最近 max_messages 条」的消息。
        # 这句读作：删除本会话中，id 不在【最近 N 条的 id 集合】里的所有行。
        self._conn.execute(
            """
            DELETE FROM working_memory
            WHERE session_id = ?
              AND id NOT IN (
                  SELECT id FROM working_memory
                  WHERE session_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (self.session_id, self.session_id, self.max_messages),
        )
        self._conn.commit()

    # ---------- 读取 ----------

    def messages(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """取出当前窗口的消息，按时间**正序**排列（最早的在前）。

        如果总量超过 token 预算，就从最老的开始丢，直到装得下为止。
        模型要求消息从早到晚排列，所以最后会翻回正序。

        ``max_tokens`` 不传时用实例的 ``self.max_tokens``。
        """
        budget = self.max_tokens if max_tokens is None else max_tokens

        rows = self._conn.execute(
            "SELECT role, content FROM working_memory WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,),
        ).fetchall()

        kept: list[dict[str, str]] = []
        used = 0
        # 从最新的往前累加：装不下就停手，等于自动丢掉了更老的那些
        for row in reversed(rows):
            cost = count_tokens(row["content"])
            if used + cost > budget:
                break
            used += cost
            kept.append({"role": row["role"], "content": row["content"]})

        kept.reverse()  # 翻回时间正序
        return kept

    def count(self) -> int:
        """当前存了多少条消息。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM working_memory WHERE session_id = ?",
            (self.session_id,),
        ).fetchone()
        return row["c"]

    # ---------- 维护 ----------

    def clear(self) -> None:
        """清空这个会话的工作记忆（其它会话不受影响）。"""
        self._conn.execute("DELETE FROM working_memory WHERE session_id = ?", (self.session_id,))
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()
