"""遗忘与巩固测试。

分三块：

1. **强度计算**（纯函数）—— 半衰期是否真的「过一期减半」。测试里显式传 ``now``，
   不用真等一周才能验证，结果也完全可复现；
2. **归档** —— 低于阈值的被归档而不是删除，归档后默认查询看不到、但数据还在；
3. **老库升级** —— v1 的库（没有 archived 字段）打开时能自动升级，且老数据不丢。
   这条最容易被忽略，但一旦出错就是「用户升级后数据全没了」。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.memory.episodic import EpisodicMemory
from src.memory.forgetting import (
    DEFAULT_FORGET_THRESHOLD,
    days_since,
    strength,
)
from src.memory.store import SCHEMA_VERSION, get_connection


@pytest.fixture
def mem(tmp_path: Path):
    """一个独立的临时情景记忆，用完关闭连接。"""
    memory = EpisodicMemory("s1", db_path=tmp_path / "episodic.db")
    yield memory
    memory.close()


# --------------------------------------------------------------------------
# 强度计算（纯函数）
# --------------------------------------------------------------------------

CREATED = "2026-09-01T00:00:00"


def test_fresh_event_keeps_full_strength() -> None:
    """刚写下的事件强度就等于它的重要性。"""
    now = datetime(2026, 9, 1, 0, 0, 0)
    assert strength(0.8, CREATED, now=now) == pytest.approx(0.8)


def test_one_half_life_halves_strength() -> None:
    """过一个半衰期，强度减半——这就是「半衰期」的定义。"""
    now = datetime(2026, 9, 8, 0, 0, 0)  # 正好 7 天后
    assert strength(1.0, CREATED, now=now, half_life_days=7.0) == pytest.approx(0.5)


def test_two_half_lives_quarter_strength() -> None:
    now = datetime(2026, 9, 15, 0, 0, 0)  # 14 天后
    assert strength(1.0, CREATED, now=now, half_life_days=7.0) == pytest.approx(0.25)


def test_more_important_decays_slower_in_absolute_terms() -> None:
    """同样过 7 天，重要的事剩下的强度仍然更高——这正是我们想要的。"""
    now = datetime(2026, 9, 8, 0, 0, 0)
    important = strength(1.0, CREATED, now=now)
    trivial = strength(0.4, CREATED, now=now)
    assert important > trivial


def test_shorter_half_life_forgets_faster() -> None:
    now = datetime(2026, 9, 8, 0, 0, 0)
    assert strength(1.0, CREATED, now=now, half_life_days=3.0) < strength(
        1.0, CREATED, now=now, half_life_days=14.0
    )


@pytest.mark.parametrize("bad", [0, -1, -7.5])
def test_non_positive_half_life_rejected(bad: float) -> None:
    """半衰期必须为正；为 0 时 0.5 ** inf 会直接除爆。"""
    with pytest.raises(ValueError, match="half_life_days"):
        strength(1.0, CREATED, now=datetime(2026, 9, 8), half_life_days=bad)


def test_future_timestamp_treated_as_zero_age() -> None:
    """时钟不准导致 created_at 在未来时，年龄按 0 算，不能让强度反而变大。"""
    now = datetime(2026, 9, 1, 0, 0, 0)
    future = "2026-09-05T00:00:00"
    assert days_since(future, now) == 0.0
    assert strength(0.6, future, now=now) == pytest.approx(0.6)


def test_days_since_counts_fractional_days() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0)
    assert days_since(CREATED, now) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 强度报表
# --------------------------------------------------------------------------


def test_decay_report_sorted_weakest_first(mem: EpisodicMemory) -> None:
    """报表按强度升序——最该忘的排最前面，一眼看出该清理谁。"""
    old = (datetime.now() - timedelta(days=30)).isoformat(timespec="seconds")
    fresh = datetime.now().isoformat(timespec="seconds")

    mem._conn.execute(
        "INSERT INTO episodic_memory (session_id, event_type, content, importance, created_at) VALUES (?,?,?,?,?)",
        ("s1", "t", "很久以前的小事", 0.3, old),
    )
    mem._conn.execute(
        "INSERT INTO episodic_memory (session_id, event_type, content, importance, created_at) VALUES (?,?,?,?,?)",
        ("s1", "t", "刚刚发生的重要事", 0.9, fresh),
    )
    mem._conn.commit()

    report = mem.decay_report()
    assert [item["content"] for item in report] == ["很久以前的小事", "刚刚发生的重要事"]
    assert report[0]["strength"] < report[1]["strength"]


def test_decay_report_excludes_archived(mem: EpisodicMemory) -> None:
    """已经归档的不该再出现在报表里。"""
    mem.record("t", "会被忘记的事", importance=0.1)
    assert len(mem.decay_report()) == 1

    # 用一个未来的时间来「快进」，让它自然衰到阈值以下
    future = datetime.now() + timedelta(days=365)
    mem.forget_weak(now=future)
    assert mem.decay_report() == []


# --------------------------------------------------------------------------
# 归档
# --------------------------------------------------------------------------


def test_forget_weak_archives_low_strength(mem: EpisodicMemory) -> None:
    mem.record("struggled", "很久以前的一件小事", importance=0.1)
    mem.record("finished_task", "刚完成的重要任务", importance=1.0)

    # 10 天后：0.1 那条衰到 0.04（该忘），1.0 那条还有 0.37（该留）
    future = datetime.now() + timedelta(days=10)
    archived = mem.forget_weak(now=future)

    assert archived == 1
    assert [e["content"] for e in mem.recent()] == ["刚完成的重要任务"]


def test_forget_weak_keeps_important_events(mem: EpisodicMemory) -> None:
    """重要的事就算过很久也该留着——这正是「重要的事记得久」。"""
    mem.record("finished_task", "关键突破", importance=1.0)

    future = datetime.now() + timedelta(days=14)  # 两个半衰期，剩 0.25
    assert mem.forget_weak(threshold=0.2, now=future) == 0
    assert mem.count() == 1


def test_forget_weak_returns_zero_when_nothing_to_forget(mem: EpisodicMemory) -> None:
    mem.record("t", "刚发生的事", importance=0.9)
    assert mem.forget_weak() == 0


def test_archived_data_is_not_deleted(mem: EpisodicMemory) -> None:
    """归档不等于删除：默认查询看不到，但数据还在库里。"""
    mem.record("t", "会被忘记的事", importance=0.05)
    future = datetime.now() + timedelta(days=365)
    mem.forget_weak(now=future)

    assert mem.count() == 0, "默认查询看不到"
    assert mem.archived_count() == 1, "但归档计数里在"
    row = mem._conn.execute("SELECT content FROM episodic_memory WHERE archived = 1").fetchone()
    assert row["content"] == "会被忘记的事", "原始数据仍在"


def test_archived_excluded_from_search_and_by_type(mem: EpisodicMemory) -> None:
    mem.record("struggled", "关于递归的旧困惑", importance=0.05)
    future = datetime.now() + timedelta(days=365)
    mem.forget_weak(now=future)

    assert mem.search("递归") == []
    assert mem.by_type("struggled") == []


def test_default_threshold_is_reasonable() -> None:
    assert 0 < DEFAULT_FORGET_THRESHOLD < 1


# --------------------------------------------------------------------------
# 巩固
# --------------------------------------------------------------------------


def test_reinforce_raises_importance(mem: EpisodicMemory) -> None:
    """同类的事又发生一次，就该给旧的那条加深印象。"""
    event_id = mem.record("struggled", "又卡在终止条件", importance=0.5)
    assert mem.reinforce(event_id) == pytest.approx(0.6)
    assert mem.recent()[0]["importance"] == pytest.approx(0.6)


def test_reinforce_caps_at_one(mem: EpisodicMemory) -> None:
    event_id = mem.record("t", "内容", importance=0.95)
    mem.reinforce(event_id)
    assert mem.reinforce(event_id) == 1.0


def test_reinforce_extends_lifetime(mem: EpisodicMemory) -> None:
    """巩固的意义：加深印象后，同样的时间点它就不该被忘掉。"""
    event_id = mem.record("struggled", "反复出现的困难", importance=0.3)
    future = datetime.now() + timedelta(days=14)

    before = mem.decay_report(now=future)[0]["strength"]
    for _ in range(5):
        mem.reinforce(event_id)
    after = mem.decay_report(now=future)[0]["strength"]

    assert after > before


def test_reinforce_missing_event_raises(mem: EpisodicMemory) -> None:
    with pytest.raises(ValueError, match="没有 id"):
        mem.reinforce(99999)


# --------------------------------------------------------------------------
# 老库升级
# --------------------------------------------------------------------------


def test_migrates_v1_database_without_losing_data(tmp_path: Path) -> None:
    """v1 的老库（没有 archived 字段）打开时自动升级，且老数据不丢。

    这条要是失守，用户升级一次数据就没了——所以专门测。
    """
    db = tmp_path / "legacy.db"

    # 手工造一个 v1 结构的老库：没有 archived 列，版本号是 1
    legacy = sqlite3.connect(db)
    legacy.execute(
        """
        CREATE TABLE episodic_memory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            content     TEXT NOT NULL,
            importance  REAL NOT NULL DEFAULT 0.5,
            created_at  TEXT NOT NULL
        )
        """
    )
    legacy.execute(
        "INSERT INTO episodic_memory (session_id, event_type, content, created_at) VALUES (?, ?, ?, ?)",
        ("s1", "t", "升级前就存在的老数据", "2026-09-01T10:00:00"),
    )
    legacy.execute("PRAGMA user_version = 1")
    legacy.commit()
    legacy.close()

    # 用正常途径打开，应当自动升级
    upgraded = get_connection(db)

    columns = {row["name"] for row in upgraded.execute("PRAGMA table_info(episodic_memory)")}
    assert "archived" in columns, "老库应当被补上 archived 字段"
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    row = upgraded.execute("SELECT content, archived FROM episodic_memory").fetchone()
    assert row["content"] == "升级前就存在的老数据", "老数据必须还在"
    assert row["archived"] == 0, "老数据默认是未归档状态"
    upgraded.close()


def test_reopening_current_database_is_noop(tmp_path: Path) -> None:
    """已经是最新结构的库再打开一次，不该重复执行升级、也不该报错。"""
    db = tmp_path / "current.db"
    first = get_connection(db)
    first.close()

    second = get_connection(db)
    assert second.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    second.close()
