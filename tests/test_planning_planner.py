"""路径规划器测试。

这一步全是确定性算法（拓扑排序 + 排序规则），所以测试可以写得很硬：
不是「大致按顺序」而是「第 3 个必须是 X」。

核心要守住的是那条**顺序铁律**：任何一个知识点的前置，都必须排在它前面。
这条要是破了，学生会遇到「还没学递归就开始练尾递归」这种荒唐路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.planning.planner import (
    STATUS_FOCUS,
    STATUS_PRACTICE,
    STATUS_REVIEW,
    classify,
    next_step,
    plan,
    progress,
    render,
    topo_layers,
)
from src.planning.profile import LearnerProfile


@pytest.fixture
def profile(tmp_path: Path):
    """一个空白画像，用完关连接。"""
    item = LearnerProfile("student-1", db_path=tmp_path / "profile.db")
    yield item
    item.close()


def topics(*specs: tuple[str, tuple[str, ...]]) -> list[dict]:
    """便捷构造：topics(("递归", ()), ("尾递归", ("递归",)))"""
    return [{"topic": name, "depends_on": list(deps), "reason": ""} for name, deps in specs]


# --------------------------------------------------------------------------
# 状态分类
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mastery", "expected"),
    [
        (0.0, STATUS_FOCUS),
        (0.39, STATUS_FOCUS),
        (0.4, STATUS_PRACTICE),  # 边界属于「学习中」
        (0.69, STATUS_PRACTICE),
        (0.7, STATUS_REVIEW),  # 边界属于「已掌握」
        (1.0, STATUS_REVIEW),
    ],
)
def test_classify_boundaries(mastery: float, expected: str) -> None:
    """阈值边界要卡准，否则「刚好 0.7」会被判成还要重点攻。"""
    assert classify(mastery) == expected


# --------------------------------------------------------------------------
# 拓扑分层
# --------------------------------------------------------------------------


def test_no_dependencies_is_one_layer() -> None:
    """互相不依赖的知识点全在第 0 层，可以任意顺序学。"""
    layers = topo_layers(topics(("递归", ()), ("二分查找", ()), ("排序", ())))

    assert len(layers) == 1
    assert sorted(layers[0]) == ["二分查找", "排序", "递归"]


def test_chain_produces_one_node_per_layer() -> None:
    """一条链 A→B→C 就该分三层。"""
    layers = topo_layers(topics(("C", ("B",)), ("B", ("A",)), ("A", ())))

    assert layers == [["A"], ["B"], ["C"]]


def test_diamond_shape() -> None:
    """菱形依赖：A → (B, C) → D。中间两个可以并行。"""
    layers = topo_layers(
        topics(("A", ()), ("B", ("A",)), ("C", ("A",)), ("D", ("B", "C")))
    )

    assert layers[0] == ["A"]
    assert sorted(layers[1]) == ["B", "C"]
    assert layers[2] == ["D"]


def test_dependencies_never_precede_their_prerequisite() -> None:
    """核心不变式：前置一定排在后面。

    这条用一组比较绕的数据验证——只要拓扑排序写对了就必然成立。
    """
    data = topics(
        ("A", ()),
        ("B", ("A",)),
        ("C", ("A",)),
        ("D", ("B", "C")),
        ("E", ("D",)),
        ("F", ()),
    )
    layers = topo_layers(data)

    position: dict[str, int] = {}
    for index, layer in enumerate(layers):
        for name in layer:
            position[name] = index

    for item in data:
        for dep in item["depends_on"]:
            assert position[dep] < position[item["topic"]], f"{item['topic']} 排在了它的前置 {dep} 前面"


def test_dangling_dependency_ignored() -> None:
    """依赖了不存在的知识点时忽略这条边，而不是崩掉。

    正常流程里拆解器已经清洗过了，这里是第二道防线——万一有人直接调 planner。
    """
    layers = topo_layers(topics(("递归", ("函数调用栈",))))

    assert layers == [["递归"]]


def test_cycle_raises() -> None:
    """有环要报错，不能死循环。"""
    data = topics(("A", ("B",)), ("B", ("A",)))

    with pytest.raises(ValueError, match="环"):
        topo_layers(data)


def test_empty_topics_returns_no_layers() -> None:
    assert topo_layers([]) == []


# --------------------------------------------------------------------------
# 层内排序
# --------------------------------------------------------------------------


def test_within_layer_sorted_by_mastery(profile: LearnerProfile) -> None:
    """同一层里，掌握度低的排前面。"""
    profile.set_mastery("递归", 0.8)
    profile.set_mastery("排序", 0.2)
    profile.set_mastery("查找", 0.5)

    result = plan(topics(("递归", ()), ("排序", ()), ("查找", ())), profile)

    assert [step["topic"] for step in result["steps"]] == ["排序", "查找", "递归"]


def test_weak_topics_come_before_practiced_ones(profile: LearnerProfile) -> None:
    """薄弱的排在「学了一半」的前面——先补短板。"""
    profile.set_mastery("A", 0.5)  # practice
    profile.set_mastery("B", 0.1)  # focus

    result = plan(topics(("A", ()), ("B", ())), profile)

    assert [step["topic"] for step in result["steps"]] == ["B", "A"]


def test_review_topics_go_last(profile: LearnerProfile) -> None:
    """已掌握的排最后——不用专门花时间，总复习时带一遍就行。"""
    profile.set_mastery("A", 0.9)  # review
    profile.set_mastery("B", 0.1)  # focus
    profile.set_mastery("C", 0.5)  # practice

    result = plan(topics(("A", ()), ("B", ()), ("C", ())), profile)

    assert [step["topic"] for step in result["steps"]] == ["B", "C", "A"]


def test_sorting_is_deterministic() -> None:
    """掌握度全一样时，排序结果必须稳定。

    不靠名字兜底的话，同分项的顺序取决于字典遍历顺序，
    同一份数据可能排出两条不同的路径——测试会变成随机飘绿，也说不清原因。
    """
    data = topics(("丙", ()), ("甲", ()), ("乙", ()))
    first = [step["topic"] for step in plan(data, None)["steps"]]
    second = [step["topic"] for step in plan(list(reversed(data)), None)["steps"]]

    assert first == second == sorted(["甲", "乙", "丙"])


def test_new_learner_gets_zero_mastery() -> None:
    """不传画像（全新用户）时掌握度一律按 0 算。"""
    result = plan(topics(("递归", ())), None)

    assert result["steps"][0]["mastery"] == 0.0
    assert result["steps"][0]["status"] == STATUS_FOCUS


# --------------------------------------------------------------------------
# 路径结果
# --------------------------------------------------------------------------


def test_plan_step_fields(profile: LearnerProfile) -> None:
    profile.set_mastery("递归", 0.25)
    result = plan(
        [{"topic": "递归", "depends_on": [], "reason": "一切的基础"}],
        profile,
        goal="掌握递归",
    )

    step = result["steps"][0]
    assert step["order"] == 1
    assert step["layer"] == 0
    assert step["topic"] == "递归"
    assert step["mastery"] == 0.25
    assert step["status"] == STATUS_FOCUS
    assert step["depends_on"] == []
    assert step["reason"] == "一切的基础"


def test_order_is_continuous(profile: LearnerProfile) -> None:
    """order 必须是 1、2、3…… 连续编号，不能跳号。"""
    data = topics(("A", ()), ("B", ("A",)), ("C", ("B",)), ("D", ("A",)))
    result = plan(data, profile)

    assert [step["order"] for step in result["steps"]] == [1, 2, 3, 4]


def test_plan_records_goal_and_layers(profile: LearnerProfile) -> None:
    result = plan(topics(("A", ()), ("B", ("A",))), profile, goal="掌握 A")

    assert result["goal"] == "掌握 A"
    assert result["layers"] == [["A"], ["B"]]


def test_empty_topics_produces_empty_plan() -> None:
    result = plan([], None, goal="没有目标")

    assert result["steps"] == []
    assert result["layers"] == []
    assert result["goal"] == "没有目标"


# --------------------------------------------------------------------------
# 取下一步
# --------------------------------------------------------------------------


def test_next_step_returns_first_pending(profile: LearnerProfile) -> None:
    profile.set_mastery("A", 0.1)
    profile.set_mastery("B", 0.5)
    result = plan(topics(("A", ()), ("B", ())), profile)

    assert next_step(result)["topic"] == "A"


def test_untouched_topic_ranks_before_practiced_one(profile: LearnerProfile) -> None:
    """完全没碰过的（0.0）排在「练过但很弱」（0.1）的前面。

    这是一条**有意为之**的规则：排序只看掌握度高低，不区分「没学过」和
    「学过但很差」。理由是两者都属于「不会」，而先把一个科目整体过一遍，
    比反复磨一个已经投入过时间的点更划算——后者可以等第二轮再补。
    """
    profile.set_mastery("练过的", 0.1)
    result = plan(topics(("练过的", ()), ("没碰过的", ())), profile)

    assert [step["topic"] for step in result["steps"]] == ["没碰过的", "练过的"]


def test_next_step_skips_mastered(profile: LearnerProfile) -> None:
    """已掌握的不占时间，直接跳到下一个。"""
    profile.set_mastery("A", 0.95)
    profile.set_mastery("B", 0.2)
    result = plan(topics(("A", ()), ("B", ())), profile)

    assert next_step(result)["topic"] == "B"


def test_next_step_returns_none_when_all_mastered(profile: LearnerProfile) -> None:
    """全都会了就没有「下一步」，返回 None 而不是硬凑一个。"""
    profile.set_mastery("A", 0.9)
    profile.set_mastery("B", 0.9)
    result = plan(topics(("A", ()), ("B", ())), profile)

    assert next_step(result) is None


def test_next_step_on_empty_plan() -> None:
    assert next_step(plan([], None)) is None


# --------------------------------------------------------------------------
# 进度与渲染
# --------------------------------------------------------------------------


def test_progress_counts(profile: LearnerProfile) -> None:
    profile.set_mastery("A", 0.9)
    profile.set_mastery("B", 0.5)
    result = plan(topics(("A", ()), ("B", ())), profile)

    assert progress(result) == {"total": 2, "done": 1, "ratio": 0.5}


def test_progress_of_empty_plan() -> None:
    """空路径不能因为除以 0 崩掉。"""
    assert progress({"steps": []})["ratio"] == 0.0


def test_render_mentions_layers_and_status(profile: LearnerProfile) -> None:
    profile.set_mastery("B", 0.9)
    result = plan(topics(("A", ()), ("B", ("A",))), profile, goal="掌握 A")

    text = render(result)
    assert "掌握 A" in text
    assert "第 1 批" in text
    assert "第 2 批" in text
    assert "重点攻" in text
    assert "快速过" in text


def test_render_empty_plan() -> None:
    assert "空" in render({"steps": [], "goal": ""})


# --------------------------------------------------------------------------
# 重规划：画像变了，路径就得变
# --------------------------------------------------------------------------


def test_replan_after_mastery_change(profile: LearnerProfile) -> None:
    """这是「动态重规划」的核心：不存路径，每次按最新画像重算。

    学生把 A 练会了之后，A 自动从「重点攻」变成「快速过」，
    排序也随之让位给还没掌握的 B。
    """
    data = topics(("A", ()), ("B", ()))

    before = plan(data, profile)
    assert [step["topic"] for step in before["steps"]] == ["A", "B"]
    assert before["steps"][0]["status"] == STATUS_FOCUS

    for _ in range(6):  # 连对 6 题，A 掌握度升到 0.9 以上
        profile.record_attempt("A", correct=True)

    after = plan(data, profile)
    assert after["steps"][0]["topic"] == "B", "A 已掌握，该轮到 B 了"
    assert after["steps"][-1]["topic"] == "A"
    assert after["steps"][-1]["status"] == STATUS_REVIEW
