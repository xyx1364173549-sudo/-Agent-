"""SSE（Server-Sent Events）报文的格式化与事件名约定。

## SSE 是什么

普通的 HTTP 请求是「问一次、等一次答、连接关闭」。SSE 是一条**不关的管道**：
服务器可以持续往里写字，浏览器收到一段就渲染一段。做「文字一个个冒出来」
的流式对话，用的就是它。

一条报文长这样::

    event: token
    data: {"text": "递归"}

    event: token
    data: {"text": "就是"}

注意尾巴上那个**空行**——空行才是「这条消息结束了」的标志，少一个空行，
浏览器会一直等这条消息的剩余部分，表现就是「卡在那儿没反应」。

## 为什么会话要用 SSE

讲解有 350 字，一次性等它生成完要好几秒，屏幕上什么都没有。
逐字推送之后，学生第一秒就能开始读。
"""

import json
from typing import Any

# 事件名。集中在这里定义，前后端照着对，避免各写各的字符串。
EVENT_STATUS = "status"  # 正在做什么（拆解中、讲解中……）
EVENT_TOKEN = "token"  # 讲解的一小段文字
EVENT_STATE = "state"  # 一轮的完整状态（路径、题目、记忆）
EVENT_ERROR = "error"  # 出错了
EVENT_DONE = "done"  # 这一轮结束，可以关连接了


def sse_event(event: str, data: Any) -> str:
    """把一条消息格式化成 SSE 报文。

    ``data`` 会被转成 JSON。这一步不只是为了整齐——**它顺手解决了换行问题**：
    SSE 以换行分隔字段，如果直接把含换行的讲解文字塞进去，整个报文会被撕裂。
    JSON 把换行转义成 ``\\n``，正好躲开这个坑。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_headers() -> dict[str, str]:
    """SSE 响应该带的头。

    ``X-Accel-Buffering: no`` 是给反向代理看的：不加的话 Nginx 会攒够一批
    再转发，流式效果就没了——本地调试一切正常，一上服务器就变成「等半天
    然后一次性全出来」，很难查。
    """
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
