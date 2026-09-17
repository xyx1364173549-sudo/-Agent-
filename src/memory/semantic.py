"""语义记忆：沉淀下来的稳定事实，像一张张档案卡片。

前两种记忆都只是「存和取」，语义记忆要难一些，因为它要处理**矛盾**：

    9 月 10 日  用户说「我已经掌握了递归」   -> 记下：递归 = 已掌握
    9 月 17 日  用户说「递归我还是不太会」   -> 怎么办？

如果两条都留着，记忆里就自相矛盾了；如果随便覆盖，又可能把准确的信息丢掉。
所以这里定了一套明确的规则（见 ``remember``），让每次写入都有据可依。

为什么需要这一层？因为聊天原文又长又乱，而**事实**是提炼过的短句。
上层 Agent 每次都要知道「这个学生目前是什么水平」，总不能每次重新读一遍
全部聊天记录——成本高、还容易读出互相矛盾的说法。
"""

import re
import sqlite3
from pathlib import Path

from src.memory.store import get_connection, now_iso

# 同一条事实被反复确认时，每次把置信度往上提这么多（上限 1.0）。
# 道理很直白：一个人说了三遍「我学会了递归」，总比只说一遍更可信。
CONFIDENCE_BOOST = 0.1

# 置信度上限，提满就不再加了
MAX_CONFIDENCE = 1.0

# 事实类目。定成常量，免得各处手写字符串写错字。
CATEGORY_MASTERY = "mastery"        # 知识点掌握情况
CATEGORY_PREFERENCE = "preference"  # 学习偏好

# 断句用的分隔符：中英文标点与空白
_SENTENCE_SPLIT = re.compile(r"[，。,.！!？?\s]+")

# 「规则版」的事实抽取模式。
#
# 每条是 (正则, 类目, 取值)，正则里的**第一个捕获组**就是要记的对象。
# 规则版只认下面这几种固定句式，其它说法一律抽不出来——这正是它的局限，
# 也是后面改用大模型抽取的动机（两版对比可以放进论文实验）。
_EXTRACT_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # 知识点在前：「递归我学会了」。
    # 注意捕获组要把「我」排除掉，否则「我学会了递归」会被这两条先命中，
    # 抽出个知识点叫「我」——这就是规则版必须逐条试出来的坑。
    (re.compile(r"([^\s，。,.！!？?我]+)我(?:已经)?掌握了?"), CATEGORY_MASTERY, "已掌握"),
    (re.compile(r"([^\s，。,.！!？?我]+)我学会了?"), CATEGORY_MASTERY, "已掌握"),
    # 知识点在后：「我学会了递归」
    # 这里的 ``?+`` 是「占有量词」：匹配上就不再回头。用普通的 ``?`` 时，
    # 正则为了整体匹配成功会把已经吃掉的「了」吐出来，结果抽出个知识点叫「了」。
    (re.compile(r"我?(?:已经)?掌握了?+([^\s，。,.！!？?]+)"), CATEGORY_MASTERY, "已掌握"),
    (re.compile(r"我?学会了?+([^\s，。,.！!？?]+)"), CATEGORY_MASTERY, "已掌握"),
    (re.compile(r"([^\s，。,.！!？?我是还]+)(?:我)?(?:还)?(?:是)?不太?会"), CATEGORY_MASTERY, "待加强"),
    (re.compile(r"([^\s，。,.！!？?我]+)(?:我)?(?:感觉)?有点难"), CATEGORY_MASTERY, "待加强"),
    (re.compile(r"我?(?:更|比较)?喜欢(?:用)?([^\s，。,.！!？?]+)"), CATEGORY_PREFERENCE, "偏好"),
]


def extract_facts(text: str) -> list[dict[str, str]]:
    """用**规则**从一段文字里抽事实，抽不出来就返回空列表。

    只认固定句式，例如::

        "我已经掌握了递归"   -> [{"category": "mastery", "key": "递归", "value": "已掌握"}]
        "递归我还是不太会"   -> [{"category": "mastery", "key": "递归", "value": "待加强"}]
        "我喜欢用图来理解"   -> [{"category": "preference", "key": "图来理解", "value": "偏好"}]

    局限有两处，都很明确：

    1. **换个说法就抽不到**——「递归这个东西吧，我总觉得哪儿没通」完全识别不了；
    2. **捕获的边界不精确**——上面第三条抽出的 key 是「图来理解」而不是「图」，
       因为中文没有空格，正则不知道该在哪儿断词。

    这是规则版的固有代价，换来的是**免费、快、结果可复现**。这两处局限正好就是
    后面改用大模型抽取的动机，也是论文里「规则 vs 大模型」对比实验的素材。
    """
    facts: list[dict[str, str]] = []

    # 先按标点把话切成小句，每句单独去匹配。
    # 不这么做的话，「我学会了递归，但二分查找我不太会」后半句会把连接词
    # 「但」也一起带进知识点名里。
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        for pattern, category, value in _EXTRACT_RULES:
            match = pattern.search(sentence)
            if match is None:
                continue

            key = match.group(1).strip()
            if key:
                facts.append({"category": category, "key": key, "value": value})
    return facts


class SemanticMemory:
    """某个用户的语义记忆。

    用法::

        mem = SemanticMemory("student-1")
        mem.remember("mastery", "递归", "已掌握", confidence=0.8)
        mem.remember("mastery", "递归", "已掌握")        # 又被确认一次 -> 置信度提高
        print(mem.get("mastery", "递归"))
    """

    def __init__(self, user_id: str, db_path: str | Path | None = None) -> None:
        self.user_id = user_id
        self._conn: sqlite3.Connection = get_connection(db_path)

    # ---------- 写入 ----------

    def remember(self, category: str, key: str, value: str, confidence: float = 0.5) -> str:
        """记一条事实，返回这次实际做了什么。

        四种结果：

        ============ ==========================================================
        ``created``    第一次见到这条事实，直接插入
        ``reinforced`` 内容一样、又被说了一遍 —— 说明更可信，把置信度调高
        ``updated``    内容变了，且新说法置信度不低于旧的 —— 以新的为准
        ``kept``       内容变了，但新说法置信度更低 —— 保留旧的，不动
        ============ ==========================================================

        为什么要分四种而不是一律覆盖？因为「用户随口一句」和「用户明确肯定」
        的可信度不一样。低可信度的新说法不该把高可信度的旧结论冲掉。
        """
        if not 0.0 <= confidence <= MAX_CONFIDENCE:
            raise ValueError(f"confidence 需在 [0.0, 1.0] 区间内，实际为 {confidence}")

        existing = self.get(category, key)

        if existing is None:
            self._conn.execute(
                """
                INSERT INTO semantic_memory (user_id, category, key, value, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.user_id, category, key, value, confidence, now_iso()),
            )
            self._conn.commit()
            return "created"

        if existing["value"] == value:
            boosted = min(MAX_CONFIDENCE, existing["confidence"] + CONFIDENCE_BOOST)
            self._update_confidence(category, key, boosted)
            return "reinforced"

        if confidence >= existing["confidence"]:
            self._conn.execute(
                """
                UPDATE semantic_memory
                SET value = ?, confidence = ?, updated_at = ?
                WHERE user_id = ? AND category = ? AND key = ?
                """,
                (value, confidence, now_iso(), self.user_id, category, key),
            )
            self._conn.commit()
            return "updated"

        return "kept"

    def learn_from_text(self, text: str, confidence: float = 0.5) -> list[dict[str, str]]:
        """从一段文字里抽事实并记下来。

        返回这次抽到并处理了哪些事实，每项带一个 ``action`` 说明实际做了什么
        （取值含义见 ``remember``）。抽不到东西时返回空列表。
        """
        learned: list[dict[str, str]] = []
        for fact in extract_facts(text):
            action = self.remember(fact["category"], fact["key"], fact["value"], confidence)
            learned.append({**fact, "action": action})
        return learned

    def _update_confidence(self, category: str, key: str, confidence: float) -> None:
        self._conn.execute(
            """
            UPDATE semantic_memory
            SET confidence = ?, updated_at = ?
            WHERE user_id = ? AND category = ? AND key = ?
            """,
            (confidence, now_iso(), self.user_id, category, key),
        )
        self._conn.commit()

    # ---------- 读取 ----------

    def get(self, category: str, key: str) -> dict | None:
        """取一条事实；没有就返回 ``None``。"""
        row = self._conn.execute(
            """
            SELECT id, category, key, value, confidence, updated_at
            FROM semantic_memory
            WHERE user_id = ? AND category = ? AND key = ?
            """,
            (self.user_id, category, key),
        ).fetchone()
        return dict(row) if row is not None else None

    def all_facts(self, category: str | None = None) -> list[dict]:
        """取全部事实；传 ``category`` 则只取某一类。

        置信度高的排前面——上层组装上下文时，往往只想要最确定的那几条。
        """
        if category is None:
            rows = self._conn.execute(
                """
                SELECT id, category, key, value, confidence, updated_at
                FROM semantic_memory
                WHERE user_id = ?
                ORDER BY confidence DESC, updated_at DESC
                """,
                (self.user_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, category, key, value, confidence, updated_at
                FROM semantic_memory
                WHERE user_id = ? AND category = ?
                ORDER BY confidence DESC, updated_at DESC
                """,
                (self.user_id, category),
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        """当前记了多少条事实。"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM semantic_memory WHERE user_id = ?",
            (self.user_id,),
        ).fetchone()
        return row["c"]

    # ---------- 维护 ----------

    def forget(self, category: str, key: str) -> bool:
        """删掉一条事实，返回是否真的删掉了（原本不存在则返回 ``False``）。"""
        cursor = self._conn.execute(
            "DELETE FROM semantic_memory WHERE user_id = ? AND category = ? AND key = ?",
            (self.user_id, category, key),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        """清空这个用户的语义记忆（其它用户不受影响）。"""
        self._conn.execute("DELETE FROM semantic_memory WHERE user_id = ?", (self.user_id,))
        self._conn.commit()

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()
