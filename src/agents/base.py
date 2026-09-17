"""功能 Agent 的共同部分。

三个 Agent（导师 / 出题 / 评估）长得其实一样：

    拿着一份提示词 → 问模型 → 把回答取出来

差别只在提示词写什么、输出是文字还是结构化的题目。所以把「持有模型」
和「取回答」这两件事抽出来一次，另外三个文件就能专心写各自的提示词。

这里刻意**做得很薄**：没有工具调用、没有自主循环、没有中间件。
那些要等 LangGraph 编排层来做（``src/planning/graph.py``）——
每个 Agent 只负责一件事，怎么把它们串起来是编排层的事。
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm import create_chat_model


class Agent:
    """功能 Agent 的基类。

    只做两件事：存住模型对象，以及把模型回答统一取成一段文字。
    """

    #: Agent 名字，出现在日志里，方便看清是哪一步出的问题
    name = "agent"

    def __init__(self, model: BaseChatModel | None = None) -> None:
        """参数
        ----
        model:
            聊天模型。不传时用 ``src.llm`` 的默认 DeepSeek。
            **显式传入主要是为了测试**——塞个假模型就能离线跑完整个流程，
            不花 API 费用、结果也稳定可复现。
        """
        self.model = model or create_chat_model()

    def _ask(self, prompt: str) -> str:
        """问模型一句，拿回纯文本。

        不同模型的返回类型略有差异（有的是消息对象，有的直接是字符串），
        这里统一处理，上层就不用关心了。
        """
        reply: Any = self.model.invoke(prompt)
        text = reply.content if hasattr(reply, "content") else str(reply)
        return text.strip()
