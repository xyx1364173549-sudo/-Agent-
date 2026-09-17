"""学习者画像测试。

重点盯三件事：

1. **掌握度的更新算法** —— 指数移动平均是否真的「往目标靠三成」，
   且一次答对不会直接跳到满分；
2. **区间与非法值拦截** —— NaN 特别要紧，因为 ``NaN < 0`` 是 False，
   光写区间判断拦不住它，而它一旦进库会让排序彻底乱掉；
3. **用户隔离** —— 两个学生的画像绝不能互相看见。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.memory.store import SCHEMA_VERSION, get_connection
from src.planning.profile import (
    LEARNING_RATE,
    MASTERED_THRESHOLD,
    WEAK_THRESHOLD,
    LearnerProfile,
)


@pytest.fixture
def profile(tmp_path: Path):
    """一个独立的学习者画像，用完关闭连接。"""
    item = LearnerProfile("student-1", db_path=tmp_path / "profile.db")
    yield item
    item.close()


# --------------------------------------------------------------------------
# 掌握度的读写
# --------------------------------------------------------------------------


def test_unknown_topic_starts_at_zero(profile: LearnerProfile) -> None:
    """没记录过的知识点算 0 分，而不是报错。

    全新用户第一次来，路径规划要能算出「先学什么」，不能因为查不到就崩。
    """
    assert profile.mastery_of("量子力学") == 0.0


def test_set_mastery_then_read_back(profile: LearnerProfile) -> None:
    profile.set_mastery("递归", 0.65)
    assert profile.mastery_of("递归") == 0.65


def test_set_mastery_is_idempotent(profile: LearnerProfile) -> None:
    """重复设置同一个知识点是更新，不是插第二条。"""
    profile.set_mastery("递归", 0.2)
    profile.set_mastery("递归", 0.8)

    assert profile.count() == 1
    assert profile.mastery_of("递归") == 0.8


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0, -1.0])
def test_set_mastery_rejects_out_of_range(profile: LearnerProfile, bad: float) -> None:
    with pytest.raises(ValueError, match="区间"):
        profile.set_mastery("递归", bad)


@pytest.mark.parametrize("bad", [0.0, 0.5, 1.0])
def test_set_mastery_accepts_boundaries(profile: LearnerProfile, bad: float) -> None:
    """边界值本身合法，校验不能写成开区间。"""
    profile.set_mastery("递归", bad)
    assert profile.mastery_of("递归") == bad


def test_set_mastery_rejects_nan(profile: LearnerProfile) -> None:
    """NaN 必须显式拦截。

    ``NaN < 0`` 和 ``NaN > 1`` 都是 False，所以「不在 [0,1] 区间」这个判断
    对 NaN 不成立。不专门挡掉，NaN 就会写进库，之后任何排序都不再有意义。
    """
    with pytest.raises(ValueError, match="NaN"):
        profile.set_mastery("递归", float("nan"))


def test_set_mastery_rejects_bool(profile: LearnerProfile) -> None:
    """True 在 Python 里也是数字，但不能当掌握度用。"""
    with pytest.raises(ValueError, match="数字"):
        profile.set_mastery("递归", True)  # type: ignore[arg-type]


def test_set_mastery_rejects_blank_topic(profile: LearnerProfile) -> None:
    with pytest.raises(ValueError, match="知识点名"):
        profile.set_mastery("   ", 0.5)


# --------------------------------------------------------------------------
# 答题结果更新掌握度
# --------------------------------------------------------------------------


def test_record_attempt_moves_toward_score(profile: LearnerProfile) -> None:
    """答对一次，掌握度往 1.0 靠 LEARNING_RATE 那么多。"""
    new = profile.record_attempt("递归", correct=True)
    assert new == pytest.approx(LEARNING_RATE, abs=1e-9)

    # 再答对一次：在 0.3 的基础上再往里靠三成 -> 0.51
    new = profile.record_attempt("递归", correct=True)
    assert new == pytest.approx(0.51, abs=1e-9)


def test_record_attempt_never_oversteps(profile: LearnerProfile) -> None:
    """一次答对不会直接跳到满分——蒙对一题不代表真会。"""
    profile.set_mastery("递归", 0.9)
    new = profile.record_attempt("递归", correct=True)

    assert new < 1.0
    assert new > 0.9


def test_wrong_answer_lowers_mastery(profile: LearnerProfile) -> None:
    """答错往下走，但也不会一次归零——偶尔失手不该抹掉历史积累。"""
    profile.set_mastery("递归", 0.8)
    new = profile.record_attempt("递归", correct=False)

    assert new == pytest.approx(0.8 * (1 - LEARNING_RATE), abs=1e-9)
    assert 0.0 < new < 0.8


def test_record_attempt_counts_attempts_and_correct(profile: LearnerProfile) -> None:
    profile.record_attempt("递归", correct=True)
    profile.record_attempt("递归", correct=False)
    profile.record_attempt("递归", correct=True)

    item = profile.all_topics()[0]
    assert item["attempts"] == 3
    assert item["correct"] == 2


def test_mastery_stays_in_range_after_many_attempts(profile: LearnerProfile) -> None:
    """反复答对不能冲破上限，反复答错不能跌破下限。"""
    for _ in range(30):
        profile.record_attempt("递归", correct=True)
    assert 0.0 <= profile.mastery_of("递归") <= 1.0

    for _ in range(60):
        profile.record_attempt("二分", correct=False)
    assert profile.mastery_of("二分") == 0.0


def test_accuracy(profile: LearnerProfile) -> None:
    profile.record_attempt("递归", correct=True)
    profile.record_attempt("递归", correct=True)
    profile.record_attempt("递归", correct=False)

    assert profile.accuracy("递归") == pytest.approx(2 / 3, abs=1e-4)


def test_accuracy_of_unpracticed_topic_is_zero(profile: LearnerProfile) -> None:
    """没练过就是 0，不能因为「除以 0」直接崩。"""
    assert profile.accuracy("没练过的知识点") == 0.0


def test_record_attempt_rejects_blank_topic(profile: LearnerProfile) -> None:
    with pytest.raises(ValueError, match="知识点名"):
        profile.record_attempt("  ", correct=True)


# --------------------------------------------------------------------------
# 分类：薄弱 / 已掌握
# --------------------------------------------------------------------------


@pytest.fixture
def graded(profile: LearnerProfile) -> LearnerProfile:
    """一个有三档水平的学生：会 / 半会 / 不会。"""
    profile.set_mastery("递归", 0.9)
    profile.set_mastery("二分查找", 0.55)
    profile.set_mastery("动态规划", 0.15)
    return profile


def test_all_topics_sorted_by_mastery_ascending(graded: LearnerProfile) -> None:
    """最该补的排最前，方便直接取前几个安排学习。"""
    assert [item["topic"] for item in graded.all_topics()] == ["动态规划", "二分查找", "递归"]


def test_weak_topics_filter(graded: LearnerProfile) -> None:
    weak = graded.weak_topics()
    assert [item["topic"] for item in weak] == ["动态规划"]
    assert all(item["mastery"] < WEAK_THRESHOLD for item in weak)


def test_weak_topics_limit(graded: LearnerProfile) -> None:
    """只学得动一个的时候，取最弱的那一个。"""
    graded.set_mastery("贪心", 0.05)
    assert len(graded.weak_topics(limit=1)) == 1
    assert graded.weak_topics(limit=1)[0]["topic"] == "贪心"


def test_mastered_topics(graded: LearnerProfile) -> None:
    assert graded.mastered_topics() == ["递归"]
    assert "递归" not in graded.mastered_topics(threshold=0.95)


def test_weak_and_mastered_are_disjoint(graded: LearnerProfile) -> None:
    """一个人不可能同时被算作「薄弱」和「已掌握」——阈值范围不能重叠。"""
    assert WEAK_THRESHOLD < MASTERED_THRESHOLD

    weak = {item["topic"] for item in graded.weak_topics()}
    mastered = set(graded.mastered_topics())
    assert weak & mastered == set()


def test_forget_topic(graded: LearnerProfile) -> None:
    assert graded.forget_topic("动态规划") is True
    assert graded.mastery_of("动态规划") == 0.0
    # 第二次删就删不到了，返回值如实反映
    assert graded.forget_topic("动态规划") is False


# --------------------------------------------------------------------------
# 目标与偏好
# --------------------------------------------------------------------------


def test_goal_roundtrip(profile: LearnerProfile) -> None:
    assert profile.goal() == ""  # 没设置过返回空串，不是 None
    profile.set_goal("三个月内掌握动态规划")
    assert profile.goal() == "三个月内掌握动态规划"


def test_goal_update_does_not_duplicate(profile: LearnerProfile) -> None:
    profile.set_goal("先学递归")
    profile.set_goal("改主意了，先学二分查找")
    assert profile.goal() == "改主意了，先学二分查找"


def test_preferences_keep_order(profile: LearnerProfile) -> None:
    """偏好按添加顺序取回，输出才会稳定。"""
    profile.add_preference("喜欢先看图示再看代码")
    profile.add_preference("习惯晚上学习")
    assert profile.preferences() == ["喜欢先看图示再看代码", "习惯晚上学习"]


def test_duplicate_preference_is_skipped(profile: LearnerProfile) -> None:
    """同一句话重复加，不该存两份。"""
    profile.add_preference("喜欢用图理解")
    profile.add_preference("喜欢用图理解")
    assert profile.preferences() == ["喜欢用图理解"]


def test_clear_preferences(profile: LearnerProfile) -> None:
    profile.add_preference("喜欢用图理解")
    profile.clear_preferences()
    assert profile.preferences() == []


def test_add_blank_preference_rejected(profile: LearnerProfile) -> None:
    with pytest.raises(ValueError, match="偏好"):
        profile.add_preference("   ")


# --------------------------------------------------------------------------
# 画像简报
# --------------------------------------------------------------------------


def test_snapshot_of_empty_profile(profile: LearnerProfile) -> None:
    """全新用户也要有话说，不能输出一片空白。"""
    assert "还没有学习记录" in profile.snapshot()


def test_snapshot_contains_all_three_levels(graded: LearnerProfile) -> None:
    """会 / 半会 / 不会三档都要出现。

    漏掉「学习中」那一档，画像看起来就像「要么会要么不会」，不真实。
    """
    text = graded.snapshot()

    assert "已掌握：递归" in text
    assert "学习中：二分查找" in text
    assert "待加强：动态规划" in text


def test_snapshot_shows_attempt_detail(graded: LearnerProfile) -> None:
    """薄弱项要带上「练过几题对几题」——导师 Agent 制定方案时要看这个。"""
    graded.record_attempt("动态规划", correct=False)
    text = graded.snapshot()

    assert "练过 1 题对 0 题" in text


def test_snapshot_includes_goal_and_preference(graded: LearnerProfile) -> None:
    graded.set_goal("三个月内掌握动态规划")
    graded.add_preference("喜欢先看图示")

    text = graded.snapshot()
    assert "学习目标：三个月内掌握动态规划" in text
    assert "学习偏好：喜欢先看图示" in text


# --------------------------------------------------------------------------
# 隔离与持久化
# --------------------------------------------------------------------------


def test_users_are_isolated(tmp_path: Path) -> None:
    """两个学生的画像必须互不可见。"""
    db = tmp_path / "shared.db"
    alice = LearnerProfile("alice", db_path=db)
    bob = LearnerProfile("bob", db_path=db)

    alice.set_mastery("递归", 0.9)
    alice.set_goal("打牢基础")
    bob.set_mastery("递归", 0.1)

    assert alice.mastery_of("递归") == 0.9
    assert bob.mastery_of("递归") == 0.1
    assert bob.goal() == ""
    assert "打牢基础" not in bob.snapshot()

    alice.close()
    bob.close()


def test_data_survives_reopen(tmp_path: Path) -> None:
    """关掉再打开，画像还在——这是「跨会话记得你」的前提。"""
    db = tmp_path / "profile.db"

    first = LearnerProfile("student-1", db_path=db)
    first.record_attempt("递归", correct=True)
    first.set_goal("三个月内掌握动态规划")
    first.add_preference("喜欢用图理解")
    first.close()

    second = LearnerProfile("student-1", db_path=db)
    assert second.mastery_of("递归") == pytest.approx(LEARNING_RATE, abs=1e-9)
    assert second.goal() == "三个月内掌握动态规划"
    assert second.preferences() == ["喜欢用图理解"]
    second.close()


def test_clear_removes_everything(profile: LearnerProfile) -> None:
    profile.set_mastery("递归", 0.9)
    profile.set_goal("打牢基础")
    profile.add_preference("喜欢用图理解")

    profile.clear()

    assert profile.count() == 0
    assert profile.goal() == ""
    assert profile.preferences() == []


def test_old_database_upgrades_to_new_schema(tmp_path: Path) -> None:
    """v2 的老库打开时要能自动补上画像两张表。

    这条要是失守，就是「用户升级一次、画像功能直接报错」。
    """
    db = tmp_path / "old.db"

    # 手工造一个 v2 的库：只有记忆要用的表，没有画像表
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            event_type TEXT NOT NULL, content TEXT NOT NULL,
            importance REAL NOT NULL DEFAULT 0.5, created_at TEXT NOT NULL,
            archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE semantic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            category TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5, updated_at TEXT NOT NULL,
            UNIQUE(user_id, category, key)
        );
        PRAGMA user_version = 2;
        """
    )
    conn.commit()
    conn.close()

    # 用新版本代码打开它
    upgraded = LearnerProfile("student-1", db_path=db)
    upgraded.set_mastery("递归", 0.5)

    assert upgraded.mastery_of("递归") == 0.5
    upgraded.close()

    # 版本号也要跟着升上去，否则每次打开都会重跑一遍迁移
    conn = get_connection(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()
