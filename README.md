# 分层记忆驱动的学习路径规划 Agent 设计与实现

本科毕业论文项目。构建一个具备**分层记忆**与**动态任务规划**能力的个性化学习伴侣 Agent 系统。

## 核心研究点

| 章节 | 内容 | 关键技术 |
| --- | --- | --- |
| 第 4 章 | 分层记忆机制 | 工作记忆 / 情景记忆 / 语义记忆，SQLite 持久化 |
| 第 5 章 | 动态任务规划 | LangGraph 多 Agent 编排（导师 / 出题 / 评估 / 规划） |
| 第 6 章 | 知识检索与验证 | RAG 四件套 + 向量库 + 前端 + 3 组对比实验 |

## 技术栈

- **语言**：Python
- **Agent 框架**：LangChain（`init_chat_model` / PromptTemplate / @tool）、LangGraph
- **RAG**：文本切分 → 向量化 → Chroma 检索 → 重排
- **后端**：FastAPI + SSE 流式输出
- **持久化**：SQLite（分层记忆）、JSON（早期原型）
- **模型接入**：DeepSeek API（OpenAI 兼容协议）
- **工程化**：Docker 容器化、MCP 协议

## 目录规划

```
.
├── src/              # 核心源码
│   ├── memory/       # 分层记忆三件套
│   ├── agents/       # 多 Agent 实现
│   ├── planning/     # 动态任务规划
│   ├── rag/          # 检索增强生成
│   └── api/          # FastAPI 服务层
├── tests/            # 单元测试与边界测试
├── docs/             # 设计文档、实验记录
├── data/             # 数据文件（不入库）
├── .env.example      # 环境变量模板
└── README.md
```

## 快速开始

```bash
# 1. 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
copy .env.example .env        # 然后填入自己的 API Key

# 4. 运行
python -m src.main
```

## 环境变量

| 变量名 | 说明 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek 平台密钥（**必填**） |
| `DEEPSEEK_BASE_URL` | 接口地址，默认 `https://api.deepseek.com` |
| `MEMORY_DB_PATH` | 分层记忆 SQLite 文件位置 |
| `CHROMA_PERSIST_DIR` | 向量库持久化目录 |
| `API_HOST` / `API_PORT` | FastAPI 监听地址与端口 |
| `LOG_DIR` | 日志文件目录 |

> `.env` 已在 `.gitignore` 中排除，请勿将真实密钥提交到仓库。

## 开发约定

- 提交信息格式：`<类型>: <简述>`，类型取 `feat` / `fix` / `docs` / `refactor` / `test` / `chore`
- 分支策略：`main` 保持可用，功能开发走 `feat/xxx` 分支
- 每完成一个功能模块提交一次，保持提交粒度清晰

## 作者

肖宇轩 · 广州商学院 · 计算机科学与技术（2023–2027）
