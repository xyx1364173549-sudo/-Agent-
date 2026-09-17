"""情景记忆：发生过的事件，像一本日记。

工作记忆是「手边的草稿纸」，用完就撕；情景记忆是「要长期留存的本子」，
记的是**发生过什么**：

    2026-09-17 10:30  用户问了递归的定义
    2026-09-17 10:35  用户答对了第 3 题
    2026-09-17 10:40  用户连续答错 3 题，卡在递归的终止条件上

为什么单独存一层？因为「发生了什么」和「聊了什么」是两回事：

- 聊天的**原文**又长又碎，塞给模型很费 token；
- 而**事件**是提炼过的短句，还能标上重要程度，检索起来又快又准。

所以工作记忆负责「这轮带多少原文进上下文」，情景记忆负责「从历史里
沉淀出事件、按时间和关键词翻出来」。两者共用同一份原始轨迹。
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from src.memory.forgetting import DEFAULT_FORGET_THRESHOLD, DEFAULT_HALF_LIFE_DAYS, strength
from src.memory.store import get_connection, now_iso


class EpisodicMemory:
    """某个会话的情景记忆。

    用法::

        mem = EpisodicMemory("session-1")
        mem.record("answered_question", "答对了第 3 题：递归的终止条件", importance=0.8)
        mem.record("struggled", "连续答错 3 题，卡在递归的终止条件上", importance=0.9)

        for event in mem.recent(10):          # 时间线：最近发生了什么
            print(event["created_at"], event["content"])

        mem.search("递归")                    # 关键词检索
    """

    def __init__(self, session_id: str, db_path: str | Path | None = None) -> None:
        self.session_id = session_id
        self._conn: sqlite3.Connection = get_connection(db_path)

    # ---------- 写入 ----------

    def record(self, event_type: str, content: str, importance: float = 0.5) -> int:
        """记下一件发生的事，返回这条事件的 id。

        参数
        ----
        event_type:
            事件类型，自己定一套命名，例如 ``answered_question``（答了题）、
            ``struggled``（卡住了）、``finished_task``（完成一个任务）。
        content:
            事件内容，一句话说清发生了什么。
        importance:
            重要程度，取值 ``0.0 ~ 1.0``，默认 0.5。越重要的事越不容易被忘掉
            （遗忘策略在 M2.7 实现）。
        """
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance 需在 [0.0, 1.0] 区间内，实际为 {importance}")

        cursor = self._conn.execute(
            """
            INSERT INTO episodic_memory (session_id, event_type, content, importance, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self.session_id, event_type, content, importance, now_iso()),
        )
        self._conn.commit()
        return cursor.lastrowid

    # ---------- 读取 ----------

    def recent(self, limit: int = 10) -> list[dict]:
        """按**时间线**取最近的事件，最新的排在最前面。

        排序用了 ``created_at DESC, id DESC`` 两个字段：时间精确到秒，
        同一秒内记的多条事件时间戳完全相同，这时靠 id 兜底才能保证
        「后记的排在前面」。
        """
        rows = self._conn.execute(
            """
            SELECT id, event_type, content, importance, created_at
            FROM episodic_memory
            WHERE session_id = ? AND archived = 0
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (self.session_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def search(self, keyword: str, limit: int = 10) -> list[dict]:
        """按**关键词**搜事件内容。

        这里做的是最朴素的文字匹配（SQL 的 ``LIKE``），不算聪明——
        搜「递归」能匹配到「递归的终止条件」，但搜「循环调用自己」就匹配不到，
        因为字面上不一样。

        真正按**语义**找相关记忆是 M3 的任务（用向量检索）。这一层先用简单
        办法跑通，也方便后面做对比实验：关键词检索 vs 向量检索，正好是
        论文实验二能用的素材。
        """
        rows = self._conn.execute(
            """
            SELECT id, event_type, content, importance, created_at
            FROM episodic_memory
            WHERE session_id = ? AND archived = 0 AND content LIKE ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (self.session_id, f"%{keyword}%", limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def by_type(self, event_type: str, limit: int = 10) -> list[dict]:
        """按事件类型取，例如「把用户所有卡住的事件翻出来」。"""
        rows = self._conn.execute(
            """
            SELECT id, event_type, content, importance, created_at
            FROM episodic_memory
            WHERE session_id = ? AND archived = 0 AND event_type = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (self.session_id, event_type, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        """当前**有效**的事件条数（不含已被遗忘归档的）。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM episodic_memory WHERE session_id = ? AND archived = 0",
            (self.session_id,),
        ).fetchone()
        return row["c"]

    # ---------- 遗忘与巩固 ----------

    def decay_report(
        self,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> list[dict]:
        """列出每条事件的当前强度，**最该忘的排最前面**。

        用来调试，也用来给论文画遗忘曲线。``now`` 可以显式指定——
        这样不用真等一周，就能验证「一周后强度减半」。
        """
        rows = self._conn.execute(
            """
            SELECT id, event_type, content, importance, created_at
            FROM episodic_memory
            WHERE session_id = ? AND archived = 0
            """,
            (self.session_id,),
        ).fetchall()

        report = [
            {
                **dict(row),
                "strength": strength(
                    row["importance"],
                    row["created_at"],
                    now=now,
                    half_life_days=half_life_days,
                ),
            }
            for row in rows
        ]
        report.sort(key=lambda item: item["strength"])
        return report

    def forget_weak(
        self,
        threshold: float = DEFAULT_FORGET_THRESHOLD,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> int:
        """把强度低于阈值的事件**归档**，返回归档了几条。

        为什么归档而不是删除？因为删了就真的没了。归档之后默认查询看不到它，
        但数据还在库里——万一以后要回溯（比如论文里统计「遗忘曲线长什么样」），
        原始记录不会丢。
        """
        weak_ids = [
            item["id"]
            for item in self.decay_report(half_life_days=half_life_days, now=now)
            if item["strength"] < threshold
        ]
        if not weak_ids:
            return 0

        self._conn.executemany(
            "UPDATE episodic_memory SET archived = 1 WHERE id = ?",
            [(event_id,) for event_id in weak_ids],
        )
        self._conn.commit()
        return len(weak_ids)

    def reinforce(self, event_id: int, boost: float = 0.1) -> float:
        """给某条事件「加深印象」，返回提升后的 importance。

        什么时候用？同类的事又发生了一次，说明这不是偶然——把重要性提上去，
        它在遗忘曲线上就能撑得更久。这就是「巩固」：**重复出现延缓遗忘**。
        """
        row = self._conn.execute(
            "SELECT importance FROM episodic_memory WHERE id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"没有 id 为 {event_id} 的事件")

        new_importance = min(1.0, row["importance"] + boost)
        self._conn.execute(
            "UPDATE episodic_memory SET importance = ? WHERE id = ?",
            (new_importance, event_id),
        )
        self._conn.commit()
        return new_importance

    def archived_count(self) -> int:
        """已经被遗忘（归档）的事件条数。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM episodic_memory WHERE session_id = ? AND archived = 1",
            (self.session_id,),
        ).fetchone()
        return row["c"]

    # ---------- 维护 ----------

    def clear(self) -> None:
        """清空这个会话的情景记忆（其它会话不受影响）。"""
        self._conn.execute("DELETE FROM episodic_memory WHERE session_id = ?", (self.session_id,))
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()
