"""目标拆解器测试。

这个模块的价值有一半在**清洗**上，所以测试重心也在这里：
模型会输出悬空依赖、自依赖、环、重复项、裹着解释文字的 JSON，
这些都得能被收拾干净，而不是让问题漏到后面的路径规划里去炸。

全部用假模型，不联网、不花钱、结果可复现。
"""

from __future__ import annotations

import json

import pytest

from src.planning.decomposer import decompose


class FakeReply:
    """假模型返回的消息对象，只要有个 content 属性就够。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """假模型：invoke 时固定返回预先设定好的内容。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> FakeReply:
        self.prompts.append(prompt)
        return FakeReply(self.content)


def make(topics: list[dict]) -> FakeModel:
    """按知识点列表造一个假模型（自动序列化成 JSON）。"""
    return FakeModel(json.dumps(topics, ensure_ascii=False))


# --------------------------------------------------------------------------
# 正常路径
# --------------------------------------------------------------------------


def test_decompose_basic() -> None:
    model = make(
        [
            {"topic": "递归的含义", "depends_on": [], "reason": "一切的基础"},
            {"topic": "递归的终止条件", "depends_on": ["递归的含义"], "reason": "不写会死循环"},
        ]
    )

    topics = decompose("掌握递归", model=model)

    assert [item["topic"] for item in topics] == ["递归的含义", "递归的终止条件"]
    assert topics[1]["depends_on"] == ["递归的含义"]
    assert topics[0]["reason"] == "一切的基础"


def test_prompt_contains_goal() -> None:
    """目标要真的进了提示词，不能被模板吞掉。"""
    model = make([{"topic": "递归", "depends_on": []}])
    decompose("掌握动态规划", model=model)

    assert "掌握动态规划" in model.prompts[0]


def test_missing_reason_defaults_to_empty() -> None:
    """reason 是补充说明，模型不写也不该报错。"""
    model = make([{"topic": "递归", "depends_on": []}])
    topics = decompose("掌握递归", model=model)

    assert topics[0]["reason"] == ""


# --------------------------------------------------------------------------
# 解析容错：模型不听话的各种情况
# --------------------------------------------------------------------------


def test_parses_json_wrapped_in_fence() -> None:
    """模型爱用 Markdown 代码围栏包 JSON。"""
    payload = [{"topic": "递归", "depends_on": []}]
    model = FakeModel(f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```")

    assert len(decompose("掌握递归", model=model)) == 1


def test_parses_json_with_surrounding_text() -> None:
    """围栏外面还夹着客套话。"""
    payload = [{"topic": "递归", "depends_on": []}]
    model = FakeModel(f"好的，这是拆解结果：{json.dumps(payload, ensure_ascii=False)} 希望对你有帮助。")

    assert len(decompose("掌握递归", model=model)) == 1


def test_parses_json_object_wrapper() -> None:
    """模型把数组包在对象里，例如 {"topics": [...]}。"""
    model = FakeModel(json.dumps({"topics": [{"topic": "递归", "depends_on": []}]}))

    assert len(decompose("掌握递归", model=model)) == 1


def test_unparsable_output_raises() -> None:
    """完全解析不出来时要报错。

    这里不能「返回空列表」了事——上层会以为「这个目标不需要学任何东西」，
    比直接报错更难排查。
    """
    model = FakeModel("抱歉，我不太理解你的问题。")

    with pytest.raises(ValueError, match="解析"):
        decompose("掌握递归", model=model)


def test_blank_goal_rejected() -> None:
    with pytest.raises(ValueError, match="学习目标"):
        decompose("   ", model=make([{"topic": "递归"}]))


def test_non_dict_items_are_skipped() -> None:
    """数组里混进了字符串等杂物，跳过即可。"""
    model = FakeModel(json.dumps(["递归", {"topic": "二分查找", "depends_on": []}]))

    topics = decompose("掌握算法", model=model)
    assert [item["topic"] for item in topics] == ["二分查找"]


# --------------------------------------------------------------------------
# 清洗：重复与空值
# --------------------------------------------------------------------------


def test_duplicate_topics_deduped() -> None:
    """同一个知识点被输出两次，只留第一次。"""
    model = make(
        [
            {"topic": "递归", "depends_on": []},
            {"topic": "递归", "depends_on": [], "reason": "重复的"},
        ]
    )

    topics = decompose("掌握递归", model=model)
    assert len(topics) == 1


def test_blank_topic_dropped() -> None:
    model = make([{"topic": "   ", "depends_on": []}, {"topic": "递归", "depends_on": []}])

    topics = decompose("掌握递归", model=model)
    assert [item["topic"] for item in topics] == ["递归"]


def test_all_invalid_raises() -> None:
    """全是垃圾条目时不能返回空列表。"""
    model = make([{"topic": ""}, {"depends_on": ["递归"]}])

    with pytest.raises(ValueError, match="不合法"):
        decompose("掌握递归", model=model)


# --------------------------------------------------------------------------
# 清洗：依赖边
# --------------------------------------------------------------------------


def test_dangling_dependency_dropped() -> None:
    """指向没被拆出来的知识点——丢掉这条边，但保留知识点本身。"""
    model = make(
        [
            {"topic": "递归", "depends_on": ["函数调用栈"], "reason": "外部概念"},
        ]
    )

    topics = decompose("掌握递归", model=model)
    assert topics[0]["topic"] == "递归"
    assert topics[0]["depends_on"] == []


def test_self_dependency_dropped() -> None:
    """自己依赖自己。"""
    model = make([{"topic": "递归", "depends_on": ["递归"]}])

    assert decompose("掌握递归", model=model)[0]["depends_on"] == []


def test_dependency_deduped() -> None:
    model = make(
        [
            {"topic": "递归", "depends_on": []},
            {"topic": "尾递归", "depends_on": ["递归", "递归"]},
        ]
    )

    topics = decompose("掌握递归", model=model)
    assert topics[1]["depends_on"] == ["递归"]


def test_non_string_dependency_dropped() -> None:
    """依赖列表里混进了数字或嵌套对象。"""
    model = make(
        [
            {"topic": "递归", "depends_on": []},
            {"topic": "尾递归", "depends_on": ["递归", 42, {"a": 1}, ""]},
        ]
    )

    topics = decompose("掌握递归", model=model)
    assert topics[1]["depends_on"] == ["递归"]


def test_non_list_dependency_dropped() -> None:
    """depends_on 写成了字符串而不是列表。"""
    model = make([{"topic": "递归", "depends_on": "无"}])

    assert decompose("掌握递归", model=model)[0]["depends_on"] == []


# --------------------------------------------------------------------------
# 清洗：环
# --------------------------------------------------------------------------


def test_two_node_cycle_broken() -> None:
    """A 依赖 B、B 依赖 A，必须断掉一条边，否则拓扑排序会死锁。"""
    model = make(
        [
            {"topic": "A", "depends_on": ["B"]},
            {"topic": "B", "depends_on": ["A"]},
        ]
    )

    topics = decompose("随便", model=model)
    edges = [(item["topic"], dep) for item in topics for dep in item["depends_on"]]

    assert len(edges) == 1, "两条边里应该只剩一条"


def test_three_node_cycle_broken() -> None:
    model = make(
        [
            {"topic": "A", "depends_on": ["C"]},
            {"topic": "B", "depends_on": ["A"]},
            {"topic": "C", "depends_on": ["B"]},
        ]
    )

    topics = decompose("随便", model=model)
    assert _is_acyclic(topics)


def test_node_outside_cycle_keeps_its_edges() -> None:
    """环外的节点不该被殃及。

    这条是特意加的：早期实现用「摘掉没有前置的点、剩下都算环上节点」的做法，
    会把「依赖了环上节点」的环外节点也一起删边。改成正经的 DFS 找回边后
    才精确——只有真正绕回去的那条边会被剪。
    """
    model = make(
        [
            {"topic": "基础", "depends_on": []},
            {"topic": "A", "depends_on": ["B"]},
            {"topic": "B", "depends_on": ["A"]},
            {"topic": "进阶", "depends_on": ["基础", "B"]},
        ]
    )

    topics = decompose("随便", model=model)
    by_name = {item["topic"]: item for item in topics}

    assert _is_acyclic(topics)
    assert by_name["基础"]["depends_on"] == []
    # 「进阶」依赖「基础」这条边跟环无关，必须留着
    assert "基础" in by_name["进阶"]["depends_on"]


def test_self_loop_is_also_a_cycle() -> None:
    """自依赖在算法眼里也是环，前面已单独清洗，这里确认两条路都通。"""
    model = make([{"topic": "A", "depends_on": ["A"]}, {"topic": "B", "depends_on": ["A"]}])

    topics = decompose("随便", model=model)
    assert _is_acyclic(topics)


def _is_acyclic(topics: list[dict]) -> bool:
    """独立实现的环检测，专门用来验证被清洗过的结果。

    刻意不调用被测代码里的任何函数——用同一套实现去验证自己，
    等于没验证。
    """
    deps = {item["topic"]: list(item["depends_on"]) for item in topics}
    done: set[str] = set()

    for _ in range(len(deps) + 1):
        ready = [name for name, lst in deps.items() if name not in done and not lst]
        if not ready:
            break
        for name in ready:
            done.add(name)
        for lst in deps.values():
            lst[:] = [dep for dep in lst if dep not in done]

    return len(done) == len(deps)


# --------------------------------------------------------------------------
# 数量上限
# --------------------------------------------------------------------------


def test_max_topics_truncates() -> None:
    """超出上限的部分丢掉。"""
    model = make([{"topic": f"知识点{i}", "depends_on": []} for i in range(20)])

    assert len(decompose("随便", model=model, max_topics=5)) == 5


def test_truncation_removes_dangling_edges() -> None:
    """被截掉的知识点，不能留下指向它的悬空依赖。

    这条盯着一个顺序问题：必须先截断、再清理依赖。
    反过来的话，「知识点3」被截掉了，但「知识点4」还依赖着它，
    结果就是一条谁也不认识的悬空边。
    """
    model = make(
        [
            {"topic": "知识点0", "depends_on": []},
            {"topic": "知识点1", "depends_on": []},
            {"topic": "知识点2", "depends_on": ["知识点1"]},
            {"topic": "知识点3", "depends_on": ["知识点2"]},
        ]
    )

    topics = decompose("随便", model=model, max_topics=2)
    names = {item["topic"] for item in topics}

    for item in topics:
        assert set(item["depends_on"]) <= names, "依赖里出现了没被保留的知识点"


def test_max_topics_respected_in_prompt() -> None:
    """上限要写进提示词，让模型自己少写几个，而不是全靠事后截断。"""
    model = make([{"topic": "递归", "depends_on": []}])
    decompose("掌握递归", model=model, max_topics=3)

    assert "3" in model.prompts[0]
