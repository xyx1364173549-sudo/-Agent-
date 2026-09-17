"""路径规划器：把知识点和依赖关系排成一条能照着走的学习路径。

## 这一步刻意**不用**大模型

拆解目标（``decomposer``）是开放式的，需要常识，适合大模型。
但「谁该排在谁前面」是个**有确定答案**的问题：前置知识必须先学。
这种活交给算法，好处有三条：

1. **正确**：拓扑排序保证不会出现「还没学递归就开始练尾递归」这种荒唐顺序；
   交给大模型排，它可能排错，而且错得没有规律、难以复现；
2. **免费**：不花一次 API 调用；
3. **可测**：同样的输入永远得到同样的输出，测试能盯死。

所以这里的分工是：**大模型负责理解和生成，算法负责保证正确性**。
这条在论文里可以直接写成方法论。

## 排序规则

先用拓扑排序把知识点分成若干**层**：同一层内的知识点互不依赖，可以任意顺序学，
但必须等上一层的全部学完。然后层内按掌握度排序：

    待加强（< 0.4）      先攻——这是真正的短板
    学习中（0.4 ~ 0.7）  再巩固
    已掌握（≥ 0.7）      最后快速复习

## 与「动态重规划」的关系

本模块是**纯函数**：不写数据库，也不记「学到哪了」。

「学到哪了」这件事由学习者画像回答——每练一题，掌握度就更新一次，
下次调用 ``plan()`` 自然算出不同的路径。这样就不存在「路径状态」和
「掌握度状态」两份数据打架的可能。知识点总共才十几个，重算一遍的开销
可以忽略，**简单比聪明更重要**。
"""

from typing import Any

from src.planning.profile import MASTERED_THRESHOLD, WEAK_THRESHOLD, LearnerProfile

# 路径里每一步的状态，同时决定了它的学习方式。
STATUS_FOCUS = "focus"  # 薄弱，重点攻
STATUS_PRACTICE = "practice"  # 学过但没吃透，需要练
STATUS_REVIEW = "review"  # 已掌握，快速过一遍即可

# 层内排序时的先后：薄弱的排最前
_STATUS_ORDER = {STATUS_FOCUS: 0, STATUS_PRACTICE: 1, STATUS_REVIEW: 2}


def classify(mastery: float) -> str:
    """按掌握度判断这一步该怎么学。"""
    if mastery < WEAK_THRESHOLD:
        return STATUS_FOCUS
    if mastery < MASTERED_THRESHOLD:
        return STATUS_PRACTICE
    return STATUS_REVIEW


def topo_layers(topics: list[dict[str, Any]]) -> list[list[str]]:
    """把知识点按依赖关系分层。

    第 0 层是「没有任何前置」的，第 1 层是「前置全在第 0 层」的，依此类推。

    参数
    ----
    topics:
        每项形如 ``{"topic": "尾递归", "depends_on": ["递归"]}``。

    返回
    ----
    分层的知识点名，例如 ``[["递归"], ["尾递归", "记忆化搜索"]]``——
    外层顺序是必须遵守的学习顺序，内层可以并行。

    抛出
    ----
    ValueError:
        依赖关系成环时。正常流程里拆解器已经把环剪掉了，这里是兜底——
        与其让循环转不出来，不如大声报错。
    """
    names = [item["topic"] for item in topics]
    known = set(names)
    deps: dict[str, set[str]] = {
        item["topic"]: {dep for dep in item.get("depends_on", []) if dep in known}
        for item in topics
    }

    layers: list[list[str]] = []
    remaining = dict(deps)

    while remaining:
        # 保持输入顺序，这样同层内的初始次序是稳定的、可复现的
        ready = [name for name in names if name in remaining and not remaining[name]]
        if not ready:
            stuck = "、".join(name for name in names if name in remaining)
            raise ValueError(f"依赖关系存在环，无法排序。涉及：{stuck}")

        layers.append(ready)
        for name in ready:
            del remaining[name]
        for pending in remaining.values():
            pending -= set(ready)

    return layers


def plan(
    topics: list[dict[str, Any]],
    profile: LearnerProfile | None = None,
    *,
    goal: str = "",
) -> dict[str, Any]:
    """排出一条学习路径。

    参数
    ----
    topics:
        拆解器给出的知识点列表（含 ``depends_on``）。
    profile:
        学习者画像，用来决定**层内**谁先谁后、每步该怎么学。
        传 ``None`` 表示全新用户（所有掌握度都是 0）。
    goal:
        学习目标，原样带进结果里，方便上层展示「为了什么而学」。

    返回
    ----
    ``{"goal": ..., "steps": [...], "layers": [...]}``，其中 ``steps`` 已按
    学习顺序排好，每项包含::

        {"order": 1, "layer": 0, "topic": "递归", "mastery": 0.0,
         "status": "focus", "depends_on": [], "reason": "..."}
    """
    if not topics:
        return {"goal": goal, "steps": [], "layers": []}

    layers = topo_layers(topics)
    info = {item["topic"]: item for item in topics}

    steps: list[dict[str, Any]] = []
    order = 0

    for layer_index, layer in enumerate(layers):
        for name in sorted(layer, key=lambda item: _sort_key(item, profile)):
            order += 1
            mastery = profile.mastery_of(name) if profile else 0.0
            steps.append(
                {
                    "order": order,
                    "layer": layer_index,
                    "topic": name,
                    "mastery": mastery,
                    "status": classify(mastery),
                    "depends_on": list(info[name].get("depends_on", [])),
                    "reason": info[name].get("reason", ""),
                }
            )

    return {"goal": goal, "steps": steps, "layers": layers}


def next_step(plan_result: dict[str, Any]) -> dict[str, Any] | None:
    """从路径里挑出「现在该学的那一个」。

    跳过已掌握的（``review``）——那些不用专门花时间，只需要在总复习时带一遍。
    全部都掌握时返回 ``None``，表示这条路径已经走完了。
    """
    for step in plan_result.get("steps", []):
        if step["status"] != STATUS_REVIEW:
            return step
    return None


def progress(plan_result: dict[str, Any]) -> dict[str, Any]:
    """算一下这条路径走到哪了。

    返回 ``{"total": 5, "done": 2, "ratio": 0.4}``。
    「已掌握」的算走过，其余算没走。
    """
    steps = plan_result.get("steps", [])
    total = len(steps)
    done = sum(1 for step in steps if step["status"] == STATUS_REVIEW)
    return {
        "total": total,
        "done": done,
        "ratio": round(done / total, 4) if total else 0.0,
    }


def render(plan_result: dict[str, Any]) -> str:
    """把路径渲染成给人看的文字，用于命令行输出和提示词。"""
    steps = plan_result.get("steps", [])
    if not steps:
        return "（这条学习路径还是空的）"

    goal = plan_result.get("goal") or "（未指定目标）"
    lines = [f"学习目标：{goal}", ""]

    current_layer = -1
    for step in steps:
        if step["layer"] != current_layer:
            current_layer = step["layer"]
            prefix = "第 1 批（可以任意顺序学）" if current_layer == 0 else f"第 {current_layer + 1} 批（需先完成前一批）"
            lines.append(f"【{prefix}】")

        mark = {"focus": "重点攻", "practice": "多练习", "review": "快速过"}[step["status"]]
        lines.append(f"  {step['order']}. {step['topic']}　[{mark}]　掌握度 {step['mastery']:.2f}")
        if step["depends_on"]:
            lines.append(f"     前置：{'、'.join(step['depends_on'])}")

    return "\n".join(lines)


def _sort_key(name: str, profile: LearnerProfile | None) -> tuple:
    """同层内的排序依据：先按状态（薄弱优先），再按掌握度（越低越前）。

    第三项用知识点名字兜底，保证排序结果稳定——否则掌握度相同的两项
    谁在前谁在后取决于字典遍历顺序，同一份数据可能排出两条路径。
    """
    mastery = profile.mastery_of(name) if profile else 0.0
    return (_STATUS_ORDER[classify(mastery)], mastery, name)
