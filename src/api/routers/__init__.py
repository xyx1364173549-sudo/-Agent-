"""路由模块。

按「谁在操作什么」分两组，而不是按技术分层：

    session.py   会话与对话——学生在学
    inspect.py   记忆、画像、知识库——看系统知道什么

分组依据是**使用场景**，将来加接口时不用纠结放哪：跟学习过程有关进 session，
跟查看状态有关进 inspect。
"""

from src.api.routers.inspect import router as inspect_router
from src.api.routers.session import router as session_router

__all__ = ["inspect_router", "session_router"]
