"""分层记忆驱动的学习路径规划 Agent —— 核心源码包。

模块划分：
- ``llm``      DeepSeek 聊天模型接入（统一工厂）
- ``memory``   分层记忆三件套（工作 / 情景 / 语义）
- ``rag``      检索增强生成四件套（切分 / 向量化 / 检索 / 重排）
- ``planning`` 学习者画像、动态任务规划与 LangGraph 编排
- ``agents``   导师 / 出题 / 评估等功能 Agent
- ``api``      FastAPI 服务层
- ``utils``    通用工具（日志等）
"""

__version__ = "0.1.0"
