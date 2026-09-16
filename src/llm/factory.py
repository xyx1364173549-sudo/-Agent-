"""DeepSeek 模型接入。

整个项目要调用大模型，都从这里拿。核心就三步：

    1. load_dotenv()      读项目根目录的 .env
    2. os.getenv(...)     取出密钥和接口地址
    3. ChatDeepSeek(...)  拼成一个模型对象

为什么外面要包一层 create_chat_model()，而不像下面这样直接写在模块里？

    model = ChatDeepSeek(api_key=..., api_base=..., model_name=...)

因为模块级代码在 **import 的那一刻就会执行**。万一 .env 还没配好，
整个项目连 import 都会崩，后面所有模块都用不了。包成函数后，
只有真正要调用模型时才构造它。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 项目根目录（本文件在 src/llm/ 下，往上三层就是根目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 读取 .env。这里写死路径，而不是用 load_dotenv() 的默认行为，
# 是为了保证「不管从哪个目录运行程序」都能读到项目根目录下那个 .env
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 默认模型。想换模型就改这一行，或者调用时传 model_name= 参数
DEFAULT_MODEL = "deepseek-v4-flash"


def create_chat_model(
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> ChatDeepSeek:
    """创建一个 DeepSeek 聊天模型。

    用法::

        model = create_chat_model()
        print(model.invoke("你好"))

    参数
    ----
    model_name:
        模型名，默认 ``deepseek-v4-flash``。
    temperature:
        采样温度。0 最稳定、1 最发散。默认 0.3——学习路径规划要的是
        可复现，不是天马行空。

    注意
    ----
    这一步**不会联网**，只是把参数拼成一个对象。真正发请求是在
    ``model.invoke(...)`` 的时候。

    ``.env`` 里没配 ``DEEPSEEK_API_KEY`` 时会抛 ``RuntimeError``。
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "没读到 DEEPSEEK_API_KEY。请检查项目根目录下是否存在 .env 文件，"
            "且里面有 DEEPSEEK_API_KEY=sk-xxxx 这一行。"
            "（注意：要填在 .env 里，不是 .env.example）"
        )

    return ChatDeepSeek(
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        model_name=model_name,
        temperature=temperature,
    )
