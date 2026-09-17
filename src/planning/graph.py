"""用 LangGraph 把各个节点连成一张图。

## 图长什么样

::

                          ┌─────────────────────────┐
                          │  入口（按 stage 分流）  │
                          └───────┬─────────┬───────┘
                    没答案│        │         │有答案
                          ▼        │         ▼
                     ┌────────┐    │    ┌──────────┐
                     │ choose │◄───┘    │ evaluate │
                     └───┬────┘         └────┬─────┘
                         │                   │
                         ▼      ┌────────────┼────────────┐
                     ┌────────┐ │重讲        │还有题      │练完
                     │ teach  │◄┘            ▼            ▼
                     └───┬────┘          （停下）    ┌────────┐
                         │                 等作答    │ update │
                         ▼                            └───┬────┘
                     ┌────────┐                         │
                     │  quiz  │                     回到 choose
                     └───┬────┘
                         │
                      （停下，等作答）

三个 LangGraph 的核心能力都用上了：

**条件路由**
    入口和 ``evaluate`` 之后都有分叉。往哪走取决于状态——有没有答案、
    答对没答对、还有没有题。这就是「动态」的来源。

**循环**
    ``evaluate`` 答错时会回到 ``teach``，形成一条回边。注意这条回边不会
    转不停：重讲次数由状态记着，用完就改走 ``update``。

**状态持久化**
    图不是一次跑完的。学生的答案来自屏幕那头，程序不可能干等，所以每来一个
    输入就让图走一段。``checkpointer`` 负责记住上次停在哪，下次调用把
    没传的字段从存档里恢复出来。

## 换一种持久化

这里默认用 ``MemorySaver``——状态存在内存里，进程一关就没了，适合开发和测试。
要真持久化，换个 ``SqliteSaver`` 就行：两者实现的是同一个接口，
``build_graph()`` 的 ``checkpointer`` 参数就是为这个留的口子。
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents import GraderAgent, QuizAgent, TutorAgent
from src.agents.base import TokenStream
from src.memory.manager import MemoryManager
from src.planning.decomposer import decompose
from src.planning.profile import LearnerProfile
from src.planning.session import (
    STAGE_DONE,
    LearningState,
    choose_topic_node,
    evaluate_node,
    new_state,
    quiz_node,
    route_after_evaluate,
    route_entry,
    summary,
    teach_node,
    update_node,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_graph(
    *,
    tutor: TutorAgent,
    quiz_agent: QuizAgent,
    grader: GraderAgent,
    profile: LearnerProfile,
    memory: MemoryManager,
    checkpointer: Any | None = None,
    token_stream: TokenStream | None = None,
) -> Any:
    """把节点组装成可执行的图。

    参数
    ----
    tutor, quiz_agent, grader:
        三个功能 Agent。
    profile:
        学习者画像——路径排序和掌握度更新都要用。
    memory:
        分层记忆——讲解时提供上下文，过程里记事件。
    checkpointer:
        状态存档方式。不传就用内存版。
    token_stream:
        讲解片段的出口，给 Web 层的 SSE 用。没人往里挂回调时
        ``TokenStream.active`` 为假，讲解就一次性生成，不走流式。

    返回
    ----
    编译好的图。用 ``.invoke(输入, config={"configurable": {"thread_id": ...}})``
    驱动，同一个 ``thread_id`` 的多次调用会接着上次的状态继续。
    """
    graph = StateGraph(LearningState)

    # 节点函数本身是 ``(state) -> dict`` 的形式（见 session.py），
    # 依赖靠这里绑进去。这样 session.py 不必知道图的存在，能单独测试。
    #
    # token_stream 传的是**对象**而不是函数：回调挂在对象上、随时可换，
    # 而节点每次执行都去问它「现在有没有人在听」。这样 Web 层可以先建会话、
    # 等 SSE 就绪了再挂回调，不必重建整张图。
    graph.add_node("choose", lambda state: choose_topic_node(state, profile=profile))
    graph.add_node(
        "teach",
        lambda state: teach_node(
            state,
            tutor=tutor,
            memory=memory,
            profile=profile,
            token_stream=token_stream,
        ),
    )
    graph.add_node(
        "quiz",
        lambda state: quiz_node(state, quiz_agent=quiz_agent, memory=memory, profile=profile),
    )
    graph.add_node(
        "evaluate",
        lambda state: evaluate_node(state, grader=grader, memory=memory, profile=profile),
    )
    graph.add_node("update", lambda state: update_node(state, profile=profile, memory=memory))

    graph.set_conditional_entry_point(
        route_entry,
        {"done": END, "evaluate": "evaluate", "choose": "choose"},
    )

    # 挑完知识点就往下讲。但整条路径走完时没有知识点可挑，得直接收工——
    # 少了这个判断，choose 会拿着空知识点去找导师讲，白花一次调用。
    graph.add_conditional_edges(
        "choose",
        lambda state: "done" if state.get("stage") == STAGE_DONE else "teach",
        {"done": END, "teach": "teach"},
    )
    graph.add_edge("teach", "quiz")
    graph.add_edge("quiz", END)  # 出完题停下，等学生作答

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"retry": "teach", "wait": END, "update": "update"},
    )
    graph.add_edge("update", "choose")  # 换下一个知识点，形成闭环

    app = graph.compile(checkpointer=checkpointer or MemorySaver())
    logger.info("学习图已编译完成 | 节点 5 个 | 条件路由 3 处 | 回边 1 条")
    return app


class LearningSession:
    """一场学习会话的高层封装：开课 → 逐题作答 → 收尾。

    直接操作图也行，但要自己拼 ``config``、自己管 ``thread_id``；
    这个类把这些收起来，业务代码只需要关心「开课」和「作答」。

    用法::

        session = LearningSession(goal="掌握递归", user_id="u1", session_id="s1")
        state = session.start()
        print(state["explanation"])
        while not session.finished:
            question = session.current_question()
            state = session.answer("学生写的答案")
    """

    def __init__(
        self,
        *,
        goal: str,
        user_id: str,
        session_id: str,
        model: Any | None = None,
        db_path: str | Path | None = None,
        max_topics: int = 8,
        checkpointer: Any | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> None:
        """开一场会话。

        参数
        ----
        goal:
            学习目标，例如「掌握递归」。
        user_id:
            用户 id。画像按它区分，换个会话水平还记得。
        session_id:
            会话 id。工作记忆和情景记忆按它区分。
        model:
            聊天模型。不传用默认 DeepSeek；测试时塞假模型就能离线跑。
        db_path:
            数据库路径，不传用 ``.env`` 里的配置。
        max_topics:
            最多拆出几个知识点。
        on_token:
            讲解时的逐字回调，供 Web 层做流式输出。开课前随时改
            ``self.tokens.callback`` 就能换掉它。
        """
        self.goal = goal
        self.user_id = user_id
        self.session_id = session_id

        # 讲解片段的出口。传对象而不是函数，是为了让节点能分辨
        # 「有人在听」和「没人听」——见 TokenStream 的说明
        self.tokens = TokenStream(on_token)

        self.profile = LearnerProfile(user_id, db_path=db_path)
        self.memory = MemoryManager(session_id, user_id=user_id, db_path=db_path)

        self.tutor = TutorAgent(model)
        self.quiz_agent = QuizAgent(model)
        self.grader = GraderAgent(model)

        self.max_topics = max_topics
        self.app = build_graph(
            tutor=self.tutor,
            quiz_agent=self.quiz_agent,
            grader=self.grader,
            profile=self.profile,
            memory=self.memory,
            checkpointer=checkpointer,
            token_stream=self.tokens,
        )
        self._config = {"configurable": {"thread_id": session_id}}
        self.state: LearningState = {}

    # ---------- 驱动 ----------

    def start(self) -> LearningState:
        """开课：拆解目标 → 选知识点 → 讲解 → 出题。

        拆解只在这里做一次。之后每轮都按最新画像重排路径，
        但知识点清单本身不重拆——目标没变，拆它没有意义，还白白多花一次调用。
        """
        topics = decompose(self.goal, model=self.tutor.model, max_topics=self.max_topics)
        initial = new_state(self.user_id, self.session_id, self.goal, topics)
        self.state = self.app.invoke(initial, config=self._config)
        return self.state

    def answer(self, text: str) -> LearningState:
        """学生对当前这道题的作答，返回推进后的状态。"""
        if not text.strip():
            raise ValueError("答案不能为空——空答案判定不出对错，还会污染画像")
        self.state = self.app.invoke({"answer": text}, config=self._config)
        return self.state

    # ---------- 查看 ----------

    @property
    def finished(self) -> bool:
        """整条路径是否走完了。"""
        return self.state.get("stage") == STAGE_DONE

    def current_question(self) -> dict | None:
        """当前该做的那道题。出完题、还没做完时才有值。"""
        questions = self.state.get("questions", [])
        index = self.state.get("index", 0)
        if index < len(questions):
            return questions[index]
        return None

    def summary(self) -> str:
        """收尾汇报。"""
        return summary(self.state, profile=self.profile)

    def close(self) -> None:
        """关掉数据库连接。"""
        self.profile.close()
        self.memory.close()
