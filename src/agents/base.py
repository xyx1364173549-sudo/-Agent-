"""功能 Agent 的共同部分。

三个 Agent（导师 / 出题 / 评估）长得其实一样：

    拿着一份提示词 → 问模型 → 把回答取出来

差别只在提示词写什么、输出是文字还是结构化的题目。所以把「持有模型」
和「取回答」这两件事抽出来一次，另外三个文件就能专心写各自的提示词。

这里刻意**做得很薄**：没有工具调用、没有自主循环、没有中间件。
那些要等 LangGraph 编排层来做（``src/planning/graph.py``）——
每个 Agent 只负责一件事，怎么把它们串起来是编排层的事。
"""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm import create_chat_model


class TokenStream:
    """讲解片段的出口：一处设置回调，多处往里送文字。

    ## 为什么需要一个对象，而不是直接传个函数

    节点需要知道「**到底有没有人在听**」，才好决定用流式还是一次性生成：

        有人在听（浏览器等着逐字渲染）→ 流式，边生成边推
        没人听（命令行脚本、单元测试）→ 一次性生成，省一趟来回

    但如果直接传函数，外面裹一层包装函数之后就**永远不是 None** 了，
    节点无从判断，于是脚本和测试也会被拖进流式路径——最后报一句
    「这个假模型没有 stream 方法」，排查半天才发现是判断条件写错了。

    所以这里给「出口」一个显式的身份：``active`` 为假就说明没人在听。

    回调可以随时替换（``stream.callback = ...``），所以 Web 层可以先建好会话，
    等 SSE 连接就绪了再把回调挂上，不必重建整张图。
    """

    def __init__(self, callback: Callable[[str], None] | None = None) -> None:
        self.callback = callback

    @property
    def active(self) -> bool:
        """有没有人在听。"""
        return self.callback is not None

    def __call__(self, text: str) -> None:
        if self.callback is not None:
            self.callback(text)

    def as_agent_arg(self) -> Callable[[str], None] | None:
        """给 Agent 用的形式：没人听就返回 ``None``，让 Agent 走非流式路径。"""
        return self if self.active else None


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

    def _ask_stream(self, prompt: str, on_token: Callable[[str], None]) -> str:
        """边生成边回调，最后把完整文本一起返回。

        为什么两件事一起做：**前端要的是「一个字一个字冒出来」的过程**，
        而业务要的是完整的讲解文本（要写进记忆、要存进状态、要能回看）。
        所以每拿到一小段就回调一次给前端，同时把片段攒起来，最后一次给出全文。

        参数
        ----
        on_token:
            每生成一小段就调一次。Web 层在它里面把文字推进 SSE 队列。
        """
        pieces: list[str] = []

        for chunk in self.model.stream(prompt):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not text:
                continue
            pieces.append(text)
            on_token(text)

        return "".join(pieces).strip()
