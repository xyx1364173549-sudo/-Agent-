"""记忆上下文组装：把三层记忆拼成一段能直接喂给模型的话。

为什么要单独一个文件？因为**这里才是分层记忆真正交付价值的地方**——
前面几个文件都在「存」，这个文件负责把存下来的东西用起来。

三条规则：

1. **按稳定性排序**：语义记忆（这个人是什么水平）→ 情景记忆（最近发生了什么）
   → 工作记忆（刚才聊了什么）。越稳定、越浓缩的放前面。
2. **按预算裁剪**：总长度不超过 ``max_tokens``。前面几层用剩的额度才轮到后面。
   万一预算被压得很短，砍掉的也是又长又碎的原文，留下的是核心判断。
3. **每层限量**：语义记忆最多 10 条、情景记忆最多 5 条，免得某一层把额度吃光。

组装出来的样子::

    【学习者情况】
    - 递归：待加强
    - 二分查找：已掌握
    【最近发生的事】
    - 17:29 练习递归 5 题，答错 3 题
    【最近对话】
    - user：我最近在学递归，感觉有点难
    - assistant：递归确实不容易，关键是想清楚终止条件
"""

from __future__ import annotations

from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.working import WorkingMemory, count_tokens

# 默认总预算。取值偏保守，因为后面还要给系统提示词和用户当前问题留位置。
DEFAULT_MAX_TOKENS = 800

# 每层最多取多少条
DEFAULT_MAX_FACTS = 10
DEFAULT_MAX_EVENTS = 5


def build_context(
    working: WorkingMemory,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_facts: int = DEFAULT_MAX_FACTS,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> str:
    """把三层记忆拼成一段文字。三层都是空的时候返回空字符串。

    参数
    ----
    max_tokens:
        组装结果的总 token 上限。语义记忆和情景记忆先按限量取，
        它们用完剩下的额度才给工作记忆。
    max_facts / max_events:
        每层最多取几条。

    实现上的一个要点
    ----------------
    判断「加这一行会不会超预算」时，**每次都把整段重新数一遍**，
    而不是把每行的 token 数相加。因为 BPE 分词在行与行的边界处切法不同，
    「各行之和」并不等于「整段的实际长度」——实测按相加估算会低估约 5%，
    结果就是「以为没超、其实超了」。行数很少，多算几次完全划算。
    """
    if max_tokens <= 0:
        return ""

    sections: list[str] = []
    remaining = max_tokens

    def fits(block: list[str]) -> bool:
        """这一整段（含标题行）放得进剩余额度吗？"""
        return count_tokens("\n".join(block)) <= remaining

    # ---- 第 1 层：语义记忆（最稳定、最浓缩）----
    facts = semantic.all_facts()[:max_facts]
    if facts:
        block = ["【学习者情况】"]
        for fact in facts:
            block.append(f"- {fact['key']}：{fact['value']}")
            if not fits(block):
                block.pop()
                break
        if len(block) > 1:
            text = "\n".join(block)
            sections.append(text)
            remaining -= count_tokens(text)

    # ---- 第 2 层：情景记忆（补时间线）----
    events = episodic.recent(max_events)
    if events:
        block = ["【最近发生的事】"]
        for event in events:
            block.append(f"- {event['created_at'][11:16]} {event['content']}")
            if not fits(block):
                block.pop()
                break
        if len(block) > 1:
            text = "\n".join(block)
            sections.append(text)
            remaining -= count_tokens(text)

    # ---- 第 3 层：工作记忆（最细，也最占地方）----
    # 先把剩余额度交给它自己裁（它内部就是按 token 预算裁的），
    # 再在这里收敛一次——它的估算是逐条相加的，同样会偏乐观。
    header = "【最近对话】"
    header_cost = count_tokens(header)
    if remaining > header_cost:
        messages = working.messages(max_tokens=remaining - header_cost)
        if messages:
            block = [header, *[f"- {m['role']}：{m['content']}" for m in messages]]
            while len(block) > 1 and not fits(block):
                block.pop()
            if len(block) > 1:
                sections.append("\n".join(block))

    # ---- 最后再校一次总长 ----
    # 段落之间拼接也有分词边界效应，可能比「各段之和」多几个 token。
    # 真超了就整段砍掉最后一段（工作记忆，最不值钱的那个）。
    result = "\n".join(sections)
    while sections and count_tokens(result) > max_tokens:
        sections.pop()
        result = "\n".join(sections)

    return result
