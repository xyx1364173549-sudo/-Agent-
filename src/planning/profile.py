"""学习者画像：这个人会什么、想学什么、喜欢怎么学。

对应论文第 5 章的第一块。画像回答三个问题：

    会什么      每个知识点掌握到什么程度 —— **数值**，用来做排序和计算
    想学什么    学习目标
    喜欢怎么学  学习偏好

## 为什么掌握度要用数值单独存一张表

第 4 章的语义记忆里已经存了「递归 = 待加强」这种文本事实，为什么还要再来一份？

因为两者**用途不同**：文本事实是给人看、给大模型看的（「这个人递归不太行」），
而路径规划需要**比较**——「递归 0.3」和「二分查找 0.7」，谁该排在前面？
文本说法没法比大小，数值可以。所以这是同一个人的两种粒度，不是重复存储。

## 掌握度怎么更新

用**指数移动平均**：新掌握度 = 旧值 × (1 - α) + 本次得分 × α，α 取 0.3。

直白讲就是：答对一次，掌握度往 1.0 靠三成，而不是直接跳到 1.0。
理由很实在——蒙对一题不代表真会。反过来，偶尔失手一次也不会把之前的
积累全抹掉。这个「近期表现权重更高、但不推翻历史」的性质，正好对上
学习者水平的真实变化方式。

典型数值：连续答对 0.00 → 0.30 → 0.51 → 0.66 → 0.76 → 0.83 → 0.88…
"""

import math
import sqlite3
from pathlib import Path

from src.memory.store import get_connection, now_iso

# 指数移动平均的更新幅度。越大越「看重最近一次表现」。
LEARNING_RATE = 0.3

# 掌握度到这个值就算「已掌握」。0.7 大致相当于连续答对 4 题。
MASTERED_THRESHOLD = 0.7

# 掌握度低于这个值算「薄弱」，规划路径时会优先安排。
WEAK_THRESHOLD = 0.4

# learner_meta 表里存学习目标的固定 key。
GOAL_KEY = "goal"

# 学习偏好的 key 前缀，后面拼序号，例如 preference:1
PREFERENCE_PREFIX = "preference:"


class LearnerProfile:
    """一个用户的学习者画像。

    用法::

        profile = LearnerProfile("student-1")
        profile.set_goal("三个月内掌握动态规划")
        profile.add_preference("喜欢先看图示再看代码")

        profile.record_attempt("递归", correct=True)   # 练了一题，答对
        profile.record_attempt("递归", correct=False)  # 又练一题，答错

        print(profile.snapshot())
    """

    def __init__(self, user_id: str, db_path: str | Path | None = None) -> None:
        """绑定一个用户。

        参数
        ----
        user_id:
            用户 id。画像跟着人走，不跟会话走——换个会话重开，水平还是那个水平。
        db_path:
            数据库路径，不传时用 ``.env`` 里的 ``MEMORY_DB_PATH``。
        """
        self.user_id = user_id
        self._conn: sqlite3.Connection = get_connection(db_path)

    # ---------- 掌握度 ----------

    def set_mastery(self, topic: str, mastery: float) -> float:
        """直接把某个知识点的掌握度设成指定值（用于导入初始数据或测试）。

        日常学习过程中应该用 ``record_attempt``，让它按答题结果逐步调整；
        这个方法相当于「直接改答案」，绕过累积过程。
        """
        _check_mastery(mastery)
        topic = topic.strip()
        if not topic:
            raise ValueError("知识点名不能为空")

        self._conn.execute(
            """
            INSERT INTO learner_mastery (user_id, topic, mastery, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, topic) DO UPDATE SET
                mastery     = excluded.mastery,
                updated_at  = excluded.updated_at
            """,
            (self.user_id, topic, mastery, now_iso()),
        )
        self._conn.commit()
        return mastery

    def mastery_of(self, topic: str) -> float:
        """查某个知识点的掌握度。

        **没记录过的一律算 0.0**（还没学过），而不是报错——
        路径规划要能对着一个全新用户算出「先学什么」，不能因为查不到就崩。
        """
        row = self._conn.execute(
            "SELECT mastery FROM learner_mastery WHERE user_id = ? AND topic = ?",
            (self.user_id, topic),
        ).fetchone()
        return row["mastery"] if row else 0.0

    def record_attempt(self, topic: str, correct: bool) -> float:
        """记一次练习结果，返回更新后的掌握度。

        这是画像最核心的入口：评估 Agent 批改完一道题就调一次，
        画像随学习过程逐步变化，路径规划据此动态调整。
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("知识点名不能为空")

        old = self.mastery_of(topic)
        score = 1.0 if correct else 0.0
        new = old * (1 - LEARNING_RATE) + score * LEARNING_RATE

        # 浮点数累加会产生 0.30000000000000004 这种尾巴，四舍五入到 4 位，
        # 存进库里看着干净，比较时也不会因为 1e-17 的差判成不等。
        new = round(new, 4)
        _check_mastery(new)

        self._conn.execute(
            """
            INSERT INTO learner_mastery (user_id, topic, mastery, attempts, correct, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(user_id, topic) DO UPDATE SET
                mastery     = ?,
                attempts    = attempts + 1,
                correct     = correct + ?,
                updated_at  = excluded.updated_at
            """,
            (self.user_id, topic, new, int(correct), now_iso(), new, int(correct)),
        )
        self._conn.commit()
        return new

    def all_topics(self) -> list[dict]:
        """全部有记录的知识点，按掌握度**从低到高**排（最该补的排前面）。"""
        rows = self._conn.execute(
            """
            SELECT topic, mastery, attempts, correct, updated_at
            FROM learner_mastery
            WHERE user_id = ?
            ORDER BY mastery ASC, attempts DESC, topic ASC
            """,
            (self.user_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def weak_topics(self, limit: int | None = None) -> list[dict]:
        """薄弱的那些（掌握度低于 ``WEAK_THRESHOLD``），最弱的排前面。"""
        weak = [item for item in self.all_topics() if item["mastery"] < WEAK_THRESHOLD]
        return weak[:limit] if limit else weak

    def mastered_topics(self, threshold: float = MASTERED_THRESHOLD) -> list[str]:
        """已经掌握的知识点名字。"""
        _check_mastery(threshold)
        return [item["topic"] for item in self.all_topics() if item["mastery"] >= threshold]

    def accuracy(self, topic: str) -> float:
        """某个知识点的历史正确率。没练过返回 0.0。"""
        row = self._conn.execute(
            "SELECT attempts, correct FROM learner_mastery WHERE user_id = ? AND topic = ?",
            (self.user_id, topic),
        ).fetchone()
        if not row or row["attempts"] == 0:
            return 0.0
        return round(row["correct"] / row["attempts"], 4)

    def forget_topic(self, topic: str) -> bool:
        """删掉某个知识点的记录，返回是否真的删到了。

        掌握度过期作废，或用户说「这个我早会了，别老安排」时用。
        """
        cursor = self._conn.execute(
            "DELETE FROM learner_mastery WHERE user_id = ? AND topic = ?",
            (self.user_id, topic),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ---------- 目标与偏好 ----------

    def set_goal(self, goal: str) -> None:
        """设置学习目标，例如「三个月内掌握动态规划」。"""
        self._set_meta(GOAL_KEY, goal.strip())

    def goal(self) -> str:
        """取学习目标。没设置过返回空字符串。"""
        return self._get_meta(GOAL_KEY) or ""

    def add_preference(self, text: str) -> None:
        """加一条学习偏好，例如「喜欢先看图示再看代码」。

        同一条偏好重复添加不会存两份——已经有了就跳过。
        """
        text = text.strip()
        if not text:
            raise ValueError("偏好内容不能为空")
        if text in self.preferences():
            return

        # 按序号往后排，保证顺序稳定（输出的画像简报每次长得一样）
        index = len(self.preferences()) + 1
        self._set_meta(f"{PREFERENCE_PREFIX}{index}", text)

    def preferences(self) -> list[str]:
        """全部学习偏好，按添加顺序。"""
        rows = self._conn.execute(
            """
            SELECT value FROM learner_meta
            WHERE user_id = ? AND key LIKE ?
            ORDER BY key ASC
            """,
            (self.user_id, f"{PREFERENCE_PREFIX}%"),
        ).fetchall()
        return [row["value"] for row in rows]

    def clear_preferences(self) -> None:
        """清掉全部偏好。"""
        self._conn.execute(
            "DELETE FROM learner_meta WHERE user_id = ? AND key LIKE ?",
            (self.user_id, f"{PREFERENCE_PREFIX}%"),
        )
        self._conn.commit()

    # ---------- 输出 ----------

    def snapshot(self) -> str:
        """把画像拼成一段可以直接塞进提示词的文字。

        注意这里**只讲掌握情况**，不含「最近发生了什么」和「刚才聊了什么」——
        那是记忆负责的部分（见 ``src/memory/context.py``）。画像讲的是
        「这个人的稳定状态」，两者拼在一起才是完整的上下文。
        """
        lines: list[str] = []

        goal = self.goal()
        if goal:
            lines.append(f"- 学习目标：{goal}")

        topics = self.all_topics()
        if topics:
            mastered = [item["topic"] for item in topics if item["mastery"] >= MASTERED_THRESHOLD]
            weak = [item for item in topics if item["mastery"] < WEAK_THRESHOLD]

            if mastered:
                lines.append(f"- 已掌握：{'、'.join(mastered)}")
            for item in weak:
                lines.append(
                    f"- 待加强：{item['topic']}"
                    f"（掌握度 {item['mastery']:.2f}，"
                    f"练过 {item['attempts']} 题对 {item['correct']} 题）"
                )

            # 既不算掌握也不算薄弱的，属于「学过但没吃透」，单独列出来——
            # 漏掉它们会让画像看起来像「要么会要么不会」，不真实。
            middle = [
                item["topic"]
                for item in topics
                if WEAK_THRESHOLD <= item["mastery"] < MASTERED_THRESHOLD
            ]
            if middle:
                lines.append(f"- 学习中：{'、'.join(middle)}")
        else:
            lines.append("- 还没有学习记录")

        prefs = self.preferences()
        if prefs:
            lines.append(f"- 学习偏好：{'；'.join(prefs)}")

        return "\n".join(lines)

    # ---------- 维护 ----------

    def count(self) -> int:
        """有记录的知识点数量。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM learner_mastery WHERE user_id = ?",
            (self.user_id,),
        ).fetchone()
        return row["c"]

    def clear(self) -> None:
        """清空这个用户的画像（掌握度 + 目标 + 偏好）。"""
        self._conn.execute("DELETE FROM learner_mastery WHERE user_id = ?", (self.user_id,))
        self._conn.execute("DELETE FROM learner_meta WHERE user_id = ?", (self.user_id,))
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()

    # ---------- 内部 ----------

    def _set_meta(self, key: str, value: str) -> None:
        """写一条用户级设置（有则更新、无则插入）。"""
        self._conn.execute(
            """
            INSERT INTO learner_meta (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value      = excluded.value,
                updated_at = excluded.updated_at
            """,
            (self.user_id, key, value, now_iso()),
        )
        self._conn.commit()

    def _get_meta(self, key: str) -> str | None:
        """读一条用户级设置。"""
        row = self._conn.execute(
            "SELECT value FROM learner_meta WHERE user_id = ? AND key = ?",
            (self.user_id, key),
        ).fetchone()
        return row["value"] if row else None


def _check_mastery(value: float) -> None:
    """掌握度必须在 [0, 1] 且不是 NaN。

    必须显式挡 NaN：``NaN < 0`` 和 ``NaN > 1`` 都是 False，光写区间判断
    是拦不住它的，而 NaN 一旦进了库，排序结果会变得毫无规律。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"掌握度必须是数字，实际收到 {type(value).__name__}")
    if math.isnan(value):
        raise ValueError("掌握度不能是 NaN")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"掌握度需在 [0.0, 1.0] 区间内，实际为 {value}")
