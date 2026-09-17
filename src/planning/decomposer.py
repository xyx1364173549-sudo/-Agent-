"""目标拆解器：把「我想学会动态规划」这种大目标，拆成能动手练的知识点。

## 谁来做这件事

交给大模型。理由：拆解是**开放式生成**——同一个目标可以拆成 5 个点也可以拆成 10 个，
规则写不出来；而且拆得对不对，看的是它有没有常识（「动态规划」该先学「记忆化搜索」
还是先学「状态转移方程」），这正好是大模型擅长的。

## 但大模型给的依赖关系不能直接用

模型输出的是**自由文本**，会出各种问题：

    指向根本不存在的前置    「递归」依赖「函数调用栈」——后者没被拆出来
    自己依赖自己            「递归」依赖「递归」
    成环                   A 依赖 B，B 又依赖 A —— 拓扑排序直接死锁
    重复输出                同一个知识点出现两次
    话多                    JSON 外面裹着解释文字和 ```json 围栏

所以本文件有一半篇幅在做**清洗**。这不是「多此一举的防御性编程」——
模型每次输出都可能不同，不校验的话，问题会在几百行之外的路径规划里炸掉，
排查起来极其痛苦。

清洗后保证：无重复、无自依赖、无悬空依赖、**无环**。
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm import create_chat_model
from src.utils.json_parse import extract_json_array
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 拆出来的知识点数量上限。太少学不透，太多一次学不完。
DEFAULT_MAX_TOPICS = 8

# 提示词。写清楚「不要说话，只输出 JSON」，能省掉一半的解析麻烦。
PROMPT_TEMPLATE = """你是一位课程设计专家。请把下面的学习目标拆解成若干知识点，并标明它们之间的先后依赖关系。

学习目标：{goal}

要求：
1. 知识点要**具体可练**，例如「递归的终止条件」而不是「递归思想」；
2. 按「先学什么、后学什么」的顺序排列，被依赖的排前面；
3. `depends_on` 只填**本次拆解出的知识点名**，不要引入外部概念；
4. 依赖关系不能成环（A 依赖 B，B 就不能再依赖 A）；
5. 拆分粒度控制在 {max_topics} 个以内；
6. 只输出 JSON 数组，不要任何解释文字、不要 Markdown 围栏。

输出格式：
[
  {{"topic": "知识点名", "depends_on": ["前置知识点名"], "reason": "为什么需要学它"}}
]"""


def decompose(
    goal: str,
    *,
    model: BaseChatModel | None = None,
    max_topics: int = DEFAULT_MAX_TOPICS,
) -> list[dict[str, Any]]:
    """把一个学习目标拆成带依赖关系的知识点列表。

    参数
    ----
    goal:
        学习目标，越具体越好。例如「掌握动态规划」比「学好算法」强得多。
    model:
        聊天模型。不传时用 ``src.llm`` 里的默认 DeepSeek——
        显式传入是为了让测试能塞一个假模型，不必真的联网。
    max_topics:
        知识点数量上限。

    返回
    ----
    列表，每项形如::

        {"topic": "递归的终止条件", "depends_on": ["递归"], "reason": "..."}

    已经过清洗：无重复、无自依赖、无悬空依赖、无环。依赖方向是
    「本知识点需要先学的前提」，即 ``depends_on`` 里的都该排在前面。

    抛出
    ----
    ValueError:
        模型输出完全没法解析成知识点列表时。这种情况硬凑一个空路径
        比报错更糟——上层会以为「这个目标不需要学什么」。
    """
    goal = goal.strip()
    if not goal:
        raise ValueError("学习目标不能为空")

    model = model or create_chat_model()
    prompt = PROMPT_TEMPLATE.format(goal=goal, max_topics=max_topics)

    reply = model.invoke(prompt)
    text = reply.content if hasattr(reply, "content") else str(reply)

    raw = extract_json_array(text)
    if not raw:
        raise ValueError(f"没能从模型输出里解析出知识点列表。原始输出：{text[:200]}")

    topics = _clean(raw, max_topics=max_topics)
    if not topics:
        raise ValueError(f"模型输出的知识点全部不合法，已全部丢弃。原始输出：{text[:200]}")

    logger.info("目标拆解完成 | goal=%s | 知识点 %d 个", goal, len(topics))
    return topics


# --------------------------------------------------------------------------
# 清洗
# --------------------------------------------------------------------------


def _clean(raw: list[dict], *, max_topics: int) -> list[dict[str, Any]]:
    """把模型输出的原始条目洗成可用的知识点列表。

    顺序很讲究：**先按数量截断，再清理依赖**。反过来的话，被截掉的知识点
    会留下指向它的边，变成悬空依赖。
    """
    # 第一步：抽出合法条目，同名的只留第一次出现的
    topics: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in raw:
        name = str(item.get("topic", "")).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        topics.append(
            {
                "topic": name,
                # 原样带过来，等截断之后再清洗：那时才知道哪些名字还在
                "depends_on": item.get("depends_on", []),
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    if not topics:
        return []

    # 第二步：截断到上限。超出部分直接丢——它们通常是模型「顺手多写了几个」，
    # 而且是排在后面的、更细枝末节的内容。
    if len(topics) > max_topics:
        dropped = [item["topic"] for item in topics[max_topics:]]
        logger.info("知识点超过上限 %d，丢弃：%s", max_topics, "、".join(dropped))
        topics = topics[:max_topics]

    # 第三步：清理依赖边。此时能引用的名字就是剩下的这些。
    valid_names = {item["topic"] for item in topics}
    for item in topics:
        item["depends_on"] = _clean_deps(item, valid_names)

    # 第四步：打破环。必须放在最后——前面几步的删边操作可能改变环的情况。
    _break_cycles(topics)

    return topics


def _clean_deps(item: dict[str, Any], valid_names: set[str]) -> list[str]:
    """清洗单个知识点的依赖列表。

    去掉：非列表、非字符串、空串、自己、不存在的名字、重复项。
    """
    raw = item.get("depends_on", [])
    if not isinstance(raw, list):
        return []

    cleaned: list[str] = []
    for name in raw:
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name == item["topic"] or name not in valid_names:
            continue
        if name in cleaned:
            continue
        cleaned.append(name)
    return cleaned


def _break_cycles(topics: list[dict[str, Any]]) -> None:
    """打破知识点之间的循环依赖，**原地修改**。

    做法是深度优先遍历，找**回边**：

        从某个知识点往下走它的前置，一路上把走过的点标成「正在走」。
        如果走到的下一个点已经在「正在走」的名单里，说明这条路绕回了自己，
        这条边就是回边——剪掉它，环就断了。

    为什么必须处理？拓扑排序遇到环会死锁（永远找不到「没有前置」的节点）。
    与其让上层崩掉，不如在这里丢掉出问题的那条边，并把日志写清楚——
    这类问题几乎总是模型输出质量问题，少一条边不影响整体路径可用。
    """
    deps = {item["topic"]: item["depends_on"] for item in topics}

    # 0 未访问 / 1 正在走（在当前这条路径上）/ 2 已走完
    color: dict[str, int] = {}
    removed: list[tuple[str, str]] = []

    def visit(name: str) -> None:
        color[name] = 1
        kept: list[str] = []
        for dep in deps[name]:
            state = color.get(dep, 0)
            if state == 1:
                # 回边：dep 正在当前路径上，再加这条边就绕成环
                removed.append((name, dep))
                continue
            if state == 0:
                visit(dep)
            kept.append(dep)
        deps[name] = kept
        color[name] = 2

    # 从每个点都试着走一遍：图可能是不连通的，只从一个点出发会漏掉整片子图
    for item in topics:
        if color.get(item["topic"], 0) == 0:
            visit(item["topic"])

    # 把剪过的边写回原对象
    for item in topics:
        item["depends_on"] = deps[item["topic"]]

    for name, dep in removed:
        logger.warning("依赖关系成环，已丢掉「%s 依赖 %s」这条边", name, dep)
