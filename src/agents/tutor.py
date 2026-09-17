"""导师 Agent：针对具体学生讲清一个知识点。

## 它和「直接问大模型」差在哪

直接问大模型：「什么是递归？」
——它会给你一段标准答案，可能很好，但它不知道问的人是谁。

导师 Agent 收到的是**这个学生的档案**：他递归掌握度 0.2、练过 5 题错 3 题、
喜欢先看图再写代码。于是讲解会变成「你上次卡在终止条件，我们从这儿说起」。

这中间的差别，就是第 4 章分层记忆存在的意义。所以 ``explain()`` 的第二个
参数是一段现成的上下文文字——由 ``MemoryManager.context()`` 和
``LearnerProfile.snapshot()`` 拼好传进来。

**为什么是「传进来」而不是「自己去查」**：这个 Agent 不需要知道记忆存在
SQLite 里、分三层、怎么检索。它只管拿着上下文讲课。查记忆是上层编排的事，
职责分开，两边都能单独测试。
"""

from src.agents.base import Agent

PROMPT_TEMPLATE = """你是一位耐心的一对一编程导师，正在给一位大学生讲知识点。

需要讲解的知识点：{topic}

以下是这位学生的历史情况：
{context}

讲解要求：
1. 先用一个**生活化的类比**开场，让他先有直觉，再讲原理；
2. 讲清楚「为什么是这样」，不要只给结论；
3. 给一段**最短**的示例代码（能说明问题即可，不要写完整程序）；
4. 如果上面提到了他的薄弱点，就针对那个点重点讲，别泛泛而谈；
5. 全文控制在 350 字以内，用大白话，不要堆术语。

直接输出讲解内容，不要写「好的」「下面是」之类的开场白。"""

# 没有历史记录时用的替代文字。写清楚「按初学者处理」，
# 免得不带上下文时模型自己瞎编一个学生画像。
NO_CONTEXT = "（暂无这位学生的学习记录，请按零基础处理，从最基本的概念讲起。）"


class TutorAgent(Agent):
    """讲解知识点。"""

    name = "tutor"

    def explain(self, topic: str, *, context: str = "") -> str:
        """讲解一个知识点。

        参数
        ----
        topic:
            要讲的知识点，例如「递归的终止条件」。
        context:
            学生情况简报。通常传 ``MemoryManager.context()`` 与
            ``LearnerProfile.snapshot()`` 拼起来的文字；不传就按零基础讲。

        返回
        ----
        讲解正文（纯文本）。
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("知识点不能为空")

        prompt = PROMPT_TEMPLATE.format(topic=topic, context=context.strip() or NO_CONTEXT)
        return self._ask(prompt)
