"""接口的入参与出参模型。

用 Pydantic 定义而不是直接收发字典，好处是**校验写在门口**：

    学习目标不能是空字符串
    最多拆几个知识点要在合理范围内
    答案不能是空白

漏掉这些校验的话，请求会带着空目标一路跑到大模型那儿，白花一次调用，
回来的还得是一句没头没脑的讲解。放在门口拦下，报错信息也直接指出是哪个字段。

出参模型同样有价值：它是一份**可执行的接口文档**（打开 ``/docs`` 就能看到），
也保证返回给前端的字段名不会因为内部改名而悄悄变化。
"""

from pydantic import BaseModel, Field, field_validator


class SessionCreateRequest(BaseModel):
    """开一场学习会话。"""

    goal: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="学习目标，越具体越好，例如「掌握递归」",
    )
    user_id: str = Field(
        "default-user",
        min_length=1,
        max_length=64,
        description="用户 id。画像按它区分，换个会话水平还记得",
    )
    session_id: str | None = Field(
        None,
        max_length=64,
        description="会话 id。不传就自动生成一个",
    )
    max_topics: int = Field(
        8,
        ge=1,
        le=20,
        description="最多拆出几个知识点。太少了学不透，太多了学不完",
    )

    @field_validator("goal")
    @classmethod
    def _goal_not_blank(cls, value: str) -> str:
        """挡掉「一串空格」这种看着非空、其实什么信息都没有的目标。

        ``min_length=1`` 拦不住它——空格也是字符。放进去的话，会拿着一个
        空白目标去调大模型，白花一次调用，回来的讲解也没法看。
        """
        value = value.strip()
        if not value:
            raise ValueError("学习目标不能是空白")
        return value


class SessionInfo(BaseModel):
    """会话的基本信息。"""

    session_id: str
    user_id: str
    goal: str
    stage: str = Field(description="need_topic / waiting / done")
    finished: bool


class AnswerRequest(BaseModel):
    """提交一次作答。"""

    answer: str = Field(..., min_length=1, max_length=2000, description="学生写的答案")

    @field_validator("answer")
    @classmethod
    def _answer_not_blank(cls, value: str) -> str:
        """同样挡掉「一串空格」。"""
        value = value.strip()
        if not value:
            raise ValueError("答案不能是空白")
        return value


class TopicOut(BaseModel):
    """一个知识点。"""

    topic: str
    depends_on: list[str] = Field(default_factory=list, description="前置知识点")
    reason: str = ""


class StepOut(BaseModel):
    """学习路径上的一步。"""

    order: int
    topic: str
    mastery: float = Field(description="掌握度 0~1")
    status: str = Field(description="focus 重点攻 / practice 多练习 / review 快速过")
    depends_on: list[str] = Field(default_factory=list)


class PathOut(BaseModel):
    """当前学习路径。每轮都会按最新画像重算，所以它一直在变。"""

    goal: str
    steps: list[StepOut] = Field(default_factory=list)


class QuestionOut(BaseModel):
    """当前该做的那道题。"""

    index: int = Field(description="第几题，从 0 开始")
    total: int
    question: str
    hint: str = ""


class GradeOut(BaseModel):
    """批改结果。"""

    correct: bool
    score: float = Field(description="0~1 的得分")
    feedback: str


class SessionState(BaseModel):
    """一次会话的完整快照——前端三栏布局全靠它。"""

    session_id: str
    user_id: str
    goal: str
    stage: str
    finished: bool
    topic: str = Field(description="正在学的知识点，没有则为空串")
    explanation: str = Field(description="导师最近一次讲解")
    question: QuestionOut | None = Field(None, description="当前该做的题，没有则为 null")
    path: PathOut
    progress: dict = Field(default_factory=dict, description="done / total / ratio")
    grade: GradeOut | None = Field(None, description="最近一次批改结果")
    log: list[str] = Field(default_factory=list, description="过程流水，给人看的")


class MemoryStats(BaseModel):
    """三层记忆各存了多少。"""

    messages: int = Field(description="工作记忆：这次会话聊过多少句")
    events: int = Field(description="情景记忆：发生过多少件事")
    facts: int = Field(description="语义记忆：沉淀了多少条事实")


class MemoryOut(BaseModel):
    """记忆面板要的全部内容。"""

    stats: MemoryStats
    context: str = Field(description="三层记忆组装出的简报，就是喂给模型的那段")
    facts: list[dict] = Field(default_factory=list, description="语义记忆明细")
    events: list[dict] = Field(default_factory=list, description="情景记忆明细")


class ProfileOut(BaseModel):
    """学习者画像。"""

    user_id: str
    goal: str
    snapshot: str = Field(description="给人看的画像简报")
    topics: list[dict] = Field(default_factory=list, description="每个知识点的掌握度与练习量")


class KnowledgeHit(BaseModel):
    """知识库里搜到的一段资料。"""

    text: str
    score: float
    source: str = ""
    sources: list[str] = Field(default_factory=list, description="被哪几路检索命中")


class HealthOut(BaseModel):
    """健康检查。"""

    status: str
    version: str
    sessions: int = Field(description="当前进程里缓存了几场会话")
