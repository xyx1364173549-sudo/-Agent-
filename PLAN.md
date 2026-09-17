# 项目实施计划表

> 本文件是本项目进度的**唯一权威来源**。所有工作按模块（M0–M8）顺序推进，一次只做一条子任务。
>
> **状态标记**：`[ ]` 待办 · `[~]` 进行中 · `[x]` 已完成 · `[!]` 阻塞
>
> **看板生成**：`python scripts/gen_progress.py` —— 解析本文件，产出 `docs/progress.html` 可视化看板
> **一键校验**：`python scripts/check.py` —— 跑全部测试 + 重新生成看板
>
> **执行约定**（来自 `AGENTS.md`）：每完成一个模块，必须补测试并全绿，再创建对应 git commit。

---

## 总览

| 模块 | 名称 | 对应章节 | 子任务数 |
| --- | --- | --- | --- |
| M0 | 工程地基 | 全局前置 | 7 |
| M1 | LLM 接入层 | 第 4/5/6 章支撑 | 6 |
| M2 | 分层记忆三件套 | **第 4 章（创新点 1）** | 8 |
| M3 | RAG 四件套 | **第 6 章（创新点 3）** | 6 |
| M4 | 画像 · 动态规划 · 多 Agent | **第 5 章（创新点 2）** | 7 |
| M5 | FastAPI 服务层 | 第 6 章 | 4 |
| M6 | 前端交互界面 | 第 6 章 | 3 |
| M7 | 实验与评估 | **第 6 章（3 组实验）** | 5 |
| M8 | 工程化与论文素材 | 全局收尾 | 3 |

---

## [M0] 工程地基 · 项目骨架与测试脚手架
<!-- section: 全局前置 -->

- [x] M0.1 建立目录骨架与包初始化 — 交付：`src/{llm,memory,rag,planning,agents,api,utils}`、`tests/`、`docs/`、`data/`、`scripts/`
- [x] M0.2 依赖清单与工具配置 — 交付：`requirements.txt` + `requirements-dev.txt`、`pytest.ini`、`.gitignore` 补全
- [x] M0.3 配置中心 — 交付：`src/config.py`，含 .env 加载、类型转换、缺键校验、单例访问
- [x] M0.4 统一日志模块 — 交付：`src/utils/logger.py`，控制台 + 文件双通道
- [x] M0.5 测试脚手架 — 交付：`tests/conftest.py` + 首批 91 条用例（配置中心边界 + 日志幂等 + 看板解析 + 骨架冒烟），全绿
- [x] M0.6 进度看板生成器 — 交付：`scripts/gen_progress.py` + `docs/progress.html`
- [x] M0.7 一键校验脚本 — 交付：`scripts/check.py`，串联 pytest 与看板生成

## [M1] LLM 接入层 · DeepSeek 统一调用
<!-- section: 第 4/5/6 章支撑 -->

- [x] M1.1 聊天模型工厂 — 交付：`src/llm/factory.py`，DeepSeek 统一入口（模型名 / 温度）
- [ ] M1.2 调用健壮性 — 交付：超时、指数退避重试、错误分类与可读报错
- [ ] M1.3 结构化输出解析器 — 交付：JSON 提取 + schema 校验 + 失败重试
- [ ] M1.4 Prompt 模板库 — 交付：`src/llm/prompts.py`，集中管理各 Agent 提示词
- [ ] M1.5 Mock 模型 — 交付：离线 MockLLM，使测试不依赖网络
- [ ] M1.6 测试与连通性验收 — 交付：单元测试全绿 + 一次真实 API 调用验证

## [M2] 分层记忆三件套 · 工作 / 情景 / 语义
<!-- section: 第 4 章（创新点 1） -->

- [x] M2.1 SQLite 存储层与建表迁移 — 交付：`src/memory/store.py`，连接管理 + 三表 schema 初始化
- [x] M2.2 工作记忆 WorkingMemory — 交付：`src/memory/working.py`，滑动窗口 + token 预算裁剪
- [x] M2.3 情景记忆 EpisodicMemory — 交付：`src/memory/episodic.py`，事件流写入 + 时间线/关键词检索
- [x] M2.4 语义记忆 SemanticMemory — 交付：`src/memory/semantic.py`，规则抽取 + 去重合并 + 冲突更新
- [x] M2.5 记忆管理器 MemoryManager — 交付：`src/memory/manager.py`，三件套统一门面与路由
- [x] M2.6 记忆上下文组装 — 交付：`src/memory/context.py`，分层检索 + 按预算压缩 + 优先级排序
- [x] M2.7 遗忘与巩固策略 — 交付：`src/memory/forgetting.py`（半衰期衰减）+ 归档与巩固（`episodic.py`）
- [x] M2.8 记忆模块测试与演示 — 交付：`tests/test_memory_e2e.py` + `examples/demo_memory.py`（含遗忘曲线演示）

## [M3] RAG 四件套 · 切分 / 向量化 / 检索 / 重排
<!-- section: 第 6 章（创新点 3） -->

- [x] M3.1 文档加载与清洗 — 交付：`src/rag/loader.py`，支持 md/txt/pdf
- [x] M3.2 三种切分策略 — 交付：`src/rag/splitter.py`，固定长度 / 递归 / 语义（embed_fn 由 M3.3 注入）
- [x] M3.3 向量化与 Chroma 索引 — 交付：`src/rag/vectorstore.py`，本地 ONNX 向量模型 + 持久化 + 哈希幂等增量
- [x] M3.4 检索器与重排 — 交付：`src/rag/retriever.py`，向量 / 关键词 / 混合（RRF 融合）+ 大模型重排
- [x] M3.5 知识库构建脚本 — 交付：`scripts/build_kb.py`，支持三种切分策略与 reset
- [x] M3.6 RAG 测试 — 交付：加载 / 切分 / 向量库 / 检索 / 构建，136 条用例全绿

## [M4] 学习者画像 · 动态规划 · 多 Agent 编排
<!-- section: 第 5 章（创新点 2） -->

- [x] M4.1 学习者画像模型 — 交付：`src/planning/profile.py`，掌握度/目标/偏好 + 持久化
- [x] M4.2 目标拆解器 — 交付：`src/planning/decomposer.py`，把学习目标拆为可执行知识点
- [x] M4.3 路径规划器 — 交付：`src/planning/planner.py`，产出带依赖关系的学习路径
- [x] M4.4 三类功能 Agent — 交付：导师 Agent / 出题 Agent / 评估 Agent，`src/agents/`
- [x] M4.5 LangGraph 状态图编排 — 交付：条件路由 + 循环 + 状态持久化，`src/planning/graph.py`
- [x] M4.6 动态重规划 — 交付：依据评估结果实时调整后续路径
- [x] M4.7 端到端串联与测试 — 交付：完整学-练-评-调闭环 demo + 测试全绿

## [M5] FastAPI 服务层 · 接口与流式输出
<!-- section: 第 6 章 -->

- [ ] M5.1 应用工厂与路由分层 — 交付：`src/api/app.py`、`routers/`
- [ ] M5.2 会话与对话接口 — 交付：SSE 流式对话端点
- [ ] M5.3 记忆 · 画像 · 进度查询接口 — 交付：状态可见性接口
- [ ] M5.4 服务层测试 — 交付：TestClient 用例全绿

## [M6] 前端交互界面
<!-- section: 第 6 章 -->

- [ ] M6.1 页面骨架 — 交付：对话区 + 学习路径图 + 记忆面板三栏布局
- [ ] M6.2 后端联调 — 交付：SSE 流式渲染接通真实接口
- [ ] M6.3 视觉打磨 — 交付：深色主题、响应式、加载态与异常态

## [M7] 实验与评估 · 论文 3 组实验
<!-- section: 第 6 章（3 组实验） -->

- [ ] M7.1 实验数据集与指标定义 — 交付：测试集 + 评价指标（记忆命中率、路径合理性、回答质量）
- [ ] M7.2 实验一：分层记忆消融 — 交付：无记忆 / 仅工作记忆 / 全分层记忆 对比
- [ ] M7.3 实验二：切分与检索策略对比 — 交付：三种切分 × 三种检索 交叉实验
- [ ] M7.4 实验三：动态规划 vs 静态路径 — 交付：学习效率与路径适应性对比
- [ ] M7.5 结果图表与实验报告 — 交付：论文可用图表 + 实验章节素材

## [M8] 工程化与论文素材
<!-- section: 全局收尾 -->

- [ ] M8.1 Docker 容器化 — 交付：`Dockerfile` + `docker-compose.yml`
- [ ] M8.2 部署与运行文档 — 交付：`docs/deployment.md`
- [ ] M8.3 论文素材归档 — 交付：架构图、流程图、伪代码、关键截图集中归档

---

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-16 | 建立计划表，划分 M0–M8 共 9 个模块、49 条子任务 |
| 2026-09-16 | M0 工程地基完成：目录骨架、配置中心、日志模块、91 条测试全绿、进度看板与一键校验脚本 |
| 2026-09-16 | M1.1 完成：聊天模型工厂 `src/llm/factory.py`，DeepSeek 统一入口 + 密钥防泄漏 |
| 2026-09-16 | **简化重构**：移除小米 MiMo 支持与「多提供方」抽象，收敛为 DeepSeek 单提供方；同步清理 config、.env.example、README、src/__init__ 与测试 |
| 2026-09-16 | **M1.1 二次简化**：改用 `langchain-deepseek` 的 `ChatDeepSeek`，代码为「load_dotenv → os.getenv → ChatDeepSeek」三步直写；默认模型 `deepseek-v4-flash`；`config.py` 职责收窄为「路径与端口」 |
| 2026-09-17 | M2.1 完成：`src/memory/store.py` 建起 working / episodic / semantic 三张表，含连接管理、索引、UNIQUE 约束与 `PRAGMA user_version` 版本号，14 条测试全绿 |
| 2026-09-17 | M2.2 完成：`src/memory/working.py` 工作记忆，滑动窗口（条数上限）+ token 预算裁剪（超限丢最老的），17 条测试全绿 |
| 2026-09-17 | M2.3 完成：`src/memory/episodic.py` 情景记忆，事件流写入 + 时间线（含同秒 id 兜底排序）+ 关键词检索 + 按类型筛选，23 条测试全绿 |
| 2026-09-17 | M2.4 完成：`src/memory/semantic.py` 语义记忆，规则抽取 + 去重合并 + 冲突更新（created / reinforced / updated / kept 四态），46 条测试全绿 |
| 2026-09-17 | 交付最小可运行 demo `examples/demo_memory.py`，用「同一提问、有无记忆」的对比展示效果 |
| 2026-09-17 | M2.5 完成：`src/memory/manager.py` 记忆管理器，三件套统一入口；区分会话与用户两个维度，14 条测试全绿 |
| 2026-09-17 | M2.6 完成：`src/memory/context.py` 上下文组装，按稳定性排序 + 预算裁剪，14 条测试全绿 |
| 2026-09-17 | M2.7 完成：`src/memory/forgetting.py` 半衰期衰减 + 情景记忆归档与巩固；表结构升级到 v2（新增 `archived` 字段，含老库自动迁移），24 条测试全绿 |
| 2026-09-17 | **M2 分层记忆模块全部完成（8/8）**：M2.8 端到端测试 + demo 升级为使用 MemoryManager，并新增遗忘曲线演示 |
| 2026-09-17 | 修复看板不幂等导致的重复提交（去掉生成时间戳）；M2 阶段共出现三对「只改时间戳」的冗余提交 |
| 2026-09-17 | **M3 开始**：M3.1 文档加载与清洗、M3.2 三种切分策略完成。**关键发现：DeepSeek 不提供 embedding 接口**（模型列表仅 deepseek-flash / deepseek-v4-pro），向量化改用 Chroma 自带的本地 ONNX 模型 |
| 2026-09-17 | M3.3 完成：`src/rag/vectorstore.py` 向量库（哈希幂等增量 + 持久化）。实测语义检索成功：「循环调用自己」命中「递归是函数自己调用自己」，字面零重叠 |
| 2026-09-17 | M3.4 完成：`src/rag/retriever.py` 三种检索模式（向量 / 关键词 / RRF 混合）+ 大模型重排，Retriever 统一入口 |
| 2026-09-17 | **M3 RAG 模块全部完成（6/6）**：M3.5 `scripts/build_kb.py` 构建脚本、M3.6 测试补齐（136 条）。**实测发现本地向量模型的中文能力边界**：能处理「停止 / 停下来」这类近义替换，但抓不住「循环调用自己 ↔ 自我引用」，已用 xfail 固定为已知局限 |
| 2026-09-17 | **M4 开始（论文第 5 章，创新点 2）**：M4.1 学习者画像（掌握度用指数移动平均更新 + 目标与偏好，表结构升 v3）、M4.2 目标拆解器（大模型拆解 + 悬空依赖/自依赖/成环清洗）、M4.3 路径规划器（拓扑分层 + 层内按掌握度排序）、M4.4 三类功能 Agent（导师 / 出题 / 评估）。**设计分工：大模型负责理解和生成，算法负责保证正确性** |
| 2026-09-17 | **M4 学习闭环全部完成（7/7）**：M4.5 LangGraph 状态图（条件路由 3 处 + 1 条回边 + checkpointer 存档，节点逻辑与图组装拆开以便脱离 LangGraph 单测）、M4.6 动态重规划（重排顺序之外增加**回退补前置**：在 B 上反复出错就回头补 A）、M4.7 端到端闭环 demo（`examples/demo_learning.py`）。实测真实模型：拆解出 3 个带依赖的知识点、讲解带类比、答错自动换讲法重讲、连续错两次后换知识点并回退补前置 |
