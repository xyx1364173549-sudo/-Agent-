"""学习闭环（LangGraph 编排）测试。

这一层把前面所有模块串起来跑，所以测试也按**完整场景**写，而不是逐个函数：

    开课 → 答题 → 答错重讲 → 再答错 → 换知识点 → 一路答对 → 收工

全程用假模型，不联网、不花钱。假模型按提示词里的特征词分辨自己该扮演谁——
拆解器、导师、出题老师还是批改老师。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.planning.graph import LearningSession
from src.planning.planner import STATUS_REVIEW
from src.planning.session import MAX_RETRY, QUIZ_COUNT, STAGE_DONE, STAGE_WAITING


class FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content


class ScriptedModel:
    """按角色分派回答的假模型。

    判断依据是各提示词里的固定开头——这也是在间接验证提示词没被改跑偏：
    哪天真把导师的提示词改成了别的样子，这里就会立刻对不上。
    """

    def __init__(
        self,
        *,
        topics: list[dict],
        questions: list[dict] | None = None,
        grades: list[dict] | None = None,
    ) -> None:
        self.topics = topics
        self.questions = questions or [
            {"question": f"第{i}题", "answer": f"答案{i}", "hint": ""} for i in range(1, QUIZ_COUNT + 1)
        ]
        self._grades = list(grades or [])
        self.prompts: list[str] = []
        self.teach_count = 0

    def invoke(self, prompt: str) -> FakeReply:
        self.prompts.append(prompt)

        if "拆解成若干知识点" in prompt:
            return FakeReply(json.dumps(self.topics, ensure_ascii=False))

        if "一对一编程导师" in prompt:
            self.teach_count += 1
            return FakeReply(f"这是第 {self.teach_count} 次讲解。")

        if "请为下面的知识点出" in prompt:
            return FakeReply(json.dumps(self.questions, ensure_ascii=False))

        if "请批改下面这道题" in prompt:
            return FakeReply(json.dumps(self._grade_for(prompt), ensure_ascii=False))

        raise AssertionError(f"假模型没见过的提示词：{prompt[:60]}")

    def _grade_for(self, prompt: str) -> dict:
        """决定这次判对还是判错。

        两种方式，优先用「答案里带标记」：

        - 答案里写 ``【错】`` / ``【对】``，直接照着判。写场景测试时用这个，
          不必去数「第几次作答该对了」——数错了就会和队列错位，
          排查半天才发现是测试自己写歪的；
        - 否则按 ``grades`` 队列依次取，用完后一律判对。
        """
        if "【错】" in prompt:
            return {"correct": False, "feedback": "答错了"}
        if "【对】" in prompt:
            return {"correct": True, "feedback": "答对了"}

        if self._grades:
            return self._grades.pop(0)
        return {"correct": True, "feedback": "默认判对"}

    # 查看某一类提示词收过几次
    def count_kind(self, keyword: str) -> int:
        return sum(1 for prompt in self.prompts if keyword in prompt)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "session.db"


def make_session(db: Path, model: ScriptedModel, *, goal: str = "掌握递归", session_id: str = "s1") -> LearningSession:
    return LearningSession(
        goal=goal,
        user_id="u1",
        session_id=session_id,
        model=model,
        db_path=db,
    )


ONE_TOPIC = [{"topic": "递归", "depends_on": [], "reason": "基础"}]


# --------------------------------------------------------------------------
# 开课
# --------------------------------------------------------------------------


def test_start_teaches_and_asks(db: Path) -> None:
    """开课就该讲一段、出几道题，然后停下等作答。"""
    model = ScriptedModel(topics=ONE_TOPIC)
    session = make_session(db, model)

    state = session.start()

    assert state["topic"] == "递归"
    assert state["explanation"] == "这是第 1 次讲解。"
    assert len(state["questions"]) == QUIZ_COUNT
    assert state["stage"] == STAGE_WAITING
    assert session.finished is False
    session.close()


def test_start_decomposes_only_once(db: Path) -> None:
    """知识点清单只拆一次。

    目标没变，每轮重拆没有意义，还白白多花一次 API 调用。
    """
    model = ScriptedModel(topics=ONE_TOPIC)
    session = make_session(db, model)
    session.start()
    session.answer("答案")  # 答错会重讲；不管怎样都不该再拆一次

    assert model.count_kind("拆解成若干知识点") == 1
    session.close()


def test_start_puts_context_into_teach_prompt(db: Path) -> None:
    """讲解时要带上记忆与画像——这是「个性化」的落点。"""
    model = ScriptedModel(topics=ONE_TOPIC)
    session = make_session(db, model)
    session.start()

    teach_prompt = next(prompt for prompt in model.prompts if "一对一编程导师" in prompt)
    assert "【画像】" in teach_prompt
    session.close()


# --------------------------------------------------------------------------
# 答题
# --------------------------------------------------------------------------


def test_correct_answer_moves_to_next_question(db: Path) -> None:
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": True, "feedback": "对"}])
    session = make_session(db, model)
    session.start()

    state = session.answer("我的答案")

    assert state["index"] == 1
    assert state["grade"]["correct"] is True
    assert state["retries"] == 0
    session.close()


def test_answer_updates_mastery(db: Path) -> None:
    """答题结果要真的进画像，否则路径永远不会变。"""
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": True, "feedback": "对"}])
    session = make_session(db, model)
    session.start()
    session.answer("我的答案")

    assert session.profile.mastery_of("递归") > 0.0
    session.close()


def test_answer_records_episodic_event(db: Path) -> None:
    """每一次作答都是一条真实发生过的事件。"""
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": False, "feedback": "错在漏了终止条件"}])
    session = make_session(db, model)
    session.start()
    session.answer("我的答案")

    answers = session.memory.episodic.by_type("answered_question")
    assert answers
    assert "漏了终止条件" in answers[0]["content"]
    session.close()


def test_teaching_and_answering_go_into_working_memory(db: Path) -> None:
    """讲解和作答都要写进工作记忆。

    工作记忆是「这次会话聊过什么」的载体。不写的话，下一轮讲解时
    ``memory.context()`` 里没有刚才说过的话，导师就看不到学生上一轮的反应。
    """
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": True, "feedback": "对"}])
    session = make_session(db, model)
    session.start()
    after_start = session.memory.working.count()

    session.answer("我的答案")

    assert after_start >= 1, "讲解应当写进工作记忆"
    assert session.memory.working.count() == after_start + 1, "作答也要写进去"
    roles = [item["role"] for item in session.memory.working.messages()]
    assert "assistant" in roles and "user" in roles
    session.close()


def test_blank_answer_rejected(db: Path) -> None:
    """空答案直接拒绝，不进流程。

    放进去的话，批改 Agent 会抛错，而那时状态已经被推进了一半。
    """
    model = ScriptedModel(topics=ONE_TOPIC)
    session = make_session(db, model)
    session.start()

    with pytest.raises(ValueError, match="不能为空"):
        session.answer("   ")
    session.close()


def test_current_question_follows_index(db: Path) -> None:
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": True, "feedback": "对"}])
    session = make_session(db, model)
    session.start()

    assert session.current_question()["question"] == "第1题"
    session.answer("答案")
    assert session.current_question()["question"] == "第2题"
    session.close()


# --------------------------------------------------------------------------
# 答错 → 重讲（图里的那条回边）
# --------------------------------------------------------------------------


def test_wrong_answer_triggers_retry(db: Path) -> None:
    """答错一次会换个讲法重讲，并重新出题。"""
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": False, "feedback": "漏了终止条件"}])
    session = make_session(db, model)
    session.start()

    state = session.answer("错答案")

    assert state["retries"] == 1
    assert state["stage"] == STAGE_WAITING
    assert model.teach_count == 2, "应当重讲了一次"
    assert state["index"] == 0, "重讲后重新出题，进度归零"
    session.close()


def test_retry_tells_tutor_what_went_wrong(db: Path) -> None:
    """重讲时必须把上次错在哪告诉导师，否则它只会照原样再讲一遍。"""
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": False, "feedback": "漏了终止条件"}])
    session = make_session(db, model)
    session.start()
    session.answer("错答案")

    second_teach = [prompt for prompt in model.prompts if "一对一编程导师" in prompt][1]
    assert "漏了终止条件" in second_teach
    assert "换一种讲法" in second_teach
    session.close()


def test_second_wrong_answer_gives_up(db: Path) -> None:
    """重讲后还错，就先放下这个知识点，别再磨。

    一直卡在同一个点上，学生就跑掉了。
    """
    model = ScriptedModel(
        topics=ONE_TOPIC,
        grades=[
            {"correct": False, "feedback": "还是错"},
            {"correct": False, "feedback": "又错"},
        ],
    )
    session = make_session(db, model)
    session.start()
    session.answer("错答案1")
    session.answer("错答案2")

    assert MAX_RETRY == 1
    assert model.teach_count == 2, "只重讲一次，不该再讲第三次"
    assert session.finished is True, "只剩这一个知识点，放弃它就没得学了"
    assert "递归" in session.state["skipped"]
    session.close()


def test_giving_up_one_topic_moves_to_the_next(db: Path) -> None:
    """放弃一个知识点后要换下一个，而不是又挑回它。

    这条盯的是一个很容易踩的坑：放弃 A 之后如果还按原顺序挑，
    会立刻挑回 A，于是「讲 → 错 → 讲 → 错」转个不停，永远出不来。
    """
    model = ScriptedModel(
        topics=[
            {"topic": "A", "depends_on": []},
            {"topic": "B", "depends_on": []},
        ],
        grades=[
            {"correct": False, "feedback": "A 错"},
            {"correct": False, "feedback": "A 又错"},
            {"correct": True, "feedback": "对"},
        ],
    )
    session = make_session(db, model)
    session.start()
    assert session.state["topic"] == "A"

    session.answer("错1")
    session.answer("错2")

    assert session.state["topic"] == "B", "A 已被放弃，该去学 B 了"
    session.close()


def test_retry_context_not_leaked_to_next_topic(db: Path) -> None:
    """「上一次没讲明白」这段提示只能出现在同一个知识点的重讲里。

    串到下一个知识点的话，导师会对着一个新知识点说「你上次漏了终止条件」，
    学生一头雾水。

    注意区分：B 的提示词里**应该**出现 A 的历史（那是记忆上下文，是分层记忆
    该做的事），不该出现的只有「换一种讲法」这段**本轮**的重讲提示。
    """
    model = ScriptedModel(
        topics=[
            {"topic": "A", "depends_on": []},
            {"topic": "B", "depends_on": []},
        ],
        grades=[
            {"correct": False, "feedback": "A 的错法"},
            {"correct": False, "feedback": "A 又错"},
            # 之后全对，把 B 也练上去
            *[{"correct": True, "feedback": "对"} for _ in range(8)],
        ],
    )
    session = make_session(db, model)
    session.start()  # 学 A
    session.answer("错1")
    session.answer("错2")  # A 被放弃，自动换到 B

    assert session.state["topic"] == "B"

    teach_prompts = [prompt for prompt in model.prompts if "一对一编程导师" in prompt]
    b_teach = next(prompt for prompt in teach_prompts if "知识点：B" in prompt)
    assert "【上一次没讲明白】" not in b_teach, "重讲提示不该跨知识点"
    session.close()


# --------------------------------------------------------------------------
# 换知识点与收工
# --------------------------------------------------------------------------


def test_moves_to_next_topic_after_mastering(db: Path) -> None:
    """把一个知识点练到掌握，路径会自动换到下一个。

    这就是「动态规划」：顺序不是一开始定死的，而是每轮按最新画像重算。
    """
    model = ScriptedModel(
        topics=[
            {"topic": "A", "depends_on": []},
            {"topic": "B", "depends_on": ["A"]},
        ],
        grades=[{"correct": True, "feedback": "对"} for _ in range(20)],
    )
    session = make_session(db, model)
    session.start()
    assert session.state["topic"] == "A"

    for _ in range(12):
        session.answer("对答案")
        if session.state["topic"] == "B":
            break

    assert session.state["topic"] == "B", "A 已掌握，该轮到 B 了"
    session.close()


def test_finishes_when_everything_mastered(db: Path) -> None:
    """全部知识点都掌握后，图要自己收工。"""
    model = ScriptedModel(
        topics=ONE_TOPIC,
        grades=[{"correct": True, "feedback": "对"} for _ in range(30)],
    )
    session = make_session(db, model)
    session.start()

    for _ in range(20):
        if session.finished:
            break
        session.answer("对答案")

    assert session.finished is True
    assert session.state["stage"] == STAGE_DONE
    session.close()


def test_no_teach_after_finishing(db: Path) -> None:
    """收工之后不该再讲课。

    choose 挑不出知识点时会返回「完成」，此时若还往 teach 走，
    导师会拿着空知识点去问模型——白花一次调用，还可能讲出莫名其妙的内容。
    """
    model = ScriptedModel(
        topics=ONE_TOPIC,
        grades=[{"correct": True, "feedback": "对"} for _ in range(30)],
    )
    session = make_session(db, model)
    session.start()

    for _ in range(20):
        if session.finished:
            break
        session.answer("对答案")

    count_at_finish = model.teach_count
    session.answer("再来一题")  # 已经结束了，再喂答案

    assert model.teach_count == count_at_finish
    session.close()


def test_summary_after_finishing(db: Path) -> None:
    model = ScriptedModel(
        topics=ONE_TOPIC,
        grades=[{"correct": True, "feedback": "对"} for _ in range(30)],
    )
    session = make_session(db, model)
    session.start()
    for _ in range(20):
        if session.finished:
            break
        session.answer("对答案")

    text = session.summary()
    assert "掌握递归" in text
    assert "递归" in text
    session.close()


# --------------------------------------------------------------------------
# 状态持久化（checkpointer 的作用）
# --------------------------------------------------------------------------


def test_state_survives_across_calls(db: Path) -> None:
    """跨多次调用，状态要接得上。

    每次调用只传一个答案，其余字段（在学哪个知识点、做到第几题、
    前面讲过什么）都靠 checkpointer 从存档里恢复。
    """
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": True, "feedback": "对"}])
    session = make_session(db, model)
    first = session.start()

    session.answer("答案")  # 只传了答案

    assert session.state["topic"] == first["topic"]
    assert len(session.state["questions"]) == QUIZ_COUNT
    assert session.state["index"] == 1
    session.close()


def test_log_accumulates(db: Path) -> None:
    """过程日志要一条条攒起来，不能后一条盖掉前一条。

    ``log`` 字段声明成了可累加，节点只管往里丢一条。
    """
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": True, "feedback": "对"}])
    session = make_session(db, model)
    session.start()
    after_start = len(session.state["log"])

    session.answer("答案")

    assert len(session.state["log"]) > after_start
    assert any("批改" in item["entry"] for item in session.state["log"])
    session.close()


def test_explanation_reflects_latest_teach(db: Path) -> None:
    """重讲之后，state 里留的必须是新讲解，不能还是旧的。"""
    model = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": False, "feedback": "错"}])
    session = make_session(db, model)
    session.start()
    assert session.state["explanation"] == "这是第 1 次讲解。"

    session.answer("错答案")

    assert session.state["explanation"] == "这是第 2 次讲解。"
    session.close()


# --------------------------------------------------------------------------
# 动态重规划：卡住了就回退补前置
# --------------------------------------------------------------------------

TWO_TOPICS = [
    {"topic": "A", "depends_on": []},
    {"topic": "B", "depends_on": ["A"]},
]


def test_falls_back_to_prerequisite_when_stuck(db: Path) -> None:
    """在 B 上反复出错时，回头补它的前置 A。

    这是「动态重规划」比「重排顺序」更进一步的地方：重排只能调整**学什么**的
    次序，回退能承认**之前学得不牢**。在「尾递归」上反复摔跤，通常是
    「递归」本身没打稳——继续往前推只会在下一个坑里再摔一次。
    """
    model = ScriptedModel(topics=TWO_TOPICS)
    session = make_session(db, model)
    session.start()
    assert session.state["topic"] == "A"

    # 把 A 练到掌握（需要连续答对两轮，因为掌握度是一题一题涨上去的）
    for _ in range(12):
        if session.state["topic"] == "B":
            break
        session.answer("【对】A 的答案")
    assert session.state["topic"] == "B", "A 已掌握，该学 B 了"

    session.answer("【错】B 的答案")  # 第一次错 → 重讲
    session.answer("【错】B 的答案")  # 第二次错 → 放弃 B，回退补前置

    assert "B" in session.state["skipped"]
    assert "A" in session.state["needs_review"]
    assert session.state["topic"] == "A", "该回头补 A 了，而不是继续往前推"
    assert any("回头补前置" in item["entry"] for item in session.state["log"])
    session.close()


def test_needs_review_cleared_after_reviewing(db: Path) -> None:
    """补完前置就把它从待补名单里划掉。

    划不掉的话，下次挑知识点时还会优先挑它，卡在同一个地方反复补。
    """
    model = ScriptedModel(topics=TWO_TOPICS)
    session = make_session(db, model)
    session.start()
    for _ in range(12):
        if session.state["topic"] == "B":
            break
        session.answer("【对】A 的答案")
    session.answer("【错】B 的答案")
    session.answer("【错】B 的答案")
    assert session.state["topic"] == "A"

    # 补 A：把这一轮题做完
    for _ in range(4):
        if session.state["topic"] != "A":
            break
        session.answer("【对】A 的答案")

    assert "A" not in session.state["needs_review"]
    session.close()


def test_no_fallback_when_topic_has_no_prerequisite(db: Path) -> None:
    """没有前置知识点时不会乱拉东西回来补。"""
    model = ScriptedModel(
        topics=ONE_TOPIC,
        grades=[
            {"correct": False, "feedback": "错"},
            {"correct": False, "feedback": "又错"},
        ],
    )
    session = make_session(db, model)
    session.start()
    session.answer("错")
    session.answer("错")

    assert session.state["needs_review"] == []
    assert session.finished is True
    session.close()


# --------------------------------------------------------------------------
# 端到端：一场学习跑完，各个模块都要对得上
# --------------------------------------------------------------------------


def test_full_learning_writes_three_memories(db: Path) -> None:
    """一场学习跑完，三种记忆都该留下东西。

    这是 M4 与 M2 的接缝处——学习闭环不是自己转自己的，
    它必须持续往分层记忆里写，否则「记忆驱动」四个字就落不了地。
    """
    model = ScriptedModel(topics=ONE_TOPIC)
    session = make_session(db, model)
    session.start()
    session.answer("递归我还是不太会")

    stats = session.memory.stats()
    assert stats["messages"] >= 2, "工作记忆：讲解 + 学生作答"
    assert stats["events"] >= 2, "情景记忆：讲解事件 + 作答事件"
    assert stats["facts"] >= 1, "语义记忆：从学生的话里抽出的事实"
    session.close()


def test_answer_is_learned_as_semantic_fact(db: Path) -> None:
    """学生自己说的「不太会」要沉淀成语义记忆。

    画像记的是数值（掌握度 0.3），语义记忆记的是**他自己的说法**。
    两者用途不同：数值用来排序，原话用来在讲解时引用。
    """
    model = ScriptedModel(topics=ONE_TOPIC)
    session = make_session(db, model)
    session.start()
    session.answer("递归我还是不太会")

    facts = session.memory.semantic.all_facts()
    assert any(item["key"] == "递归" for item in facts)
    session.close()


def test_profile_persists_across_sessions(db: Path) -> None:
    """换一场会话，水平还认得出来。

    这是「会话」和「用户」分成两个维度的全部意义：会话可以关，
    用户不会变。
    """
    first = make_session(db, ScriptedModel(topics=ONE_TOPIC), session_id="s1")
    first.start()
    first.answer("【对】答案")
    mastery = first.profile.mastery_of("递归")
    assert mastery > 0.0
    first.close()

    second = make_session(db, ScriptedModel(topics=ONE_TOPIC), session_id="s2")
    second.start()
    assert second.profile.mastery_of("递归") == mastery
    second.close()


def test_learning_events_obey_forgetting(db: Path) -> None:
    """学习过程记下的事件也遵守遗忘规律。

    这条是 M4 与 M2.7 的接缝：学习闭环往情景记忆里写的事件，
    同样会被时间衰减和归档规则管着——不是两套互不相干的机制。
    """
    model = ScriptedModel(topics=ONE_TOPIC)
    session = make_session(db, model)
    session.start()
    session.answer("【错】答案")

    episodic = session.memory.episodic
    before = episodic.count()
    assert before >= 2

    future = datetime.now() + timedelta(days=60)
    archived = episodic.forget_weak(now=future)

    assert archived >= 1, "60 天后低重要度的事件应当被归档"
    assert episodic.count() < before
    session.close()


# --------------------------------------------------------------------------
# 会话隔离
# --------------------------------------------------------------------------


def test_sessions_do_not_share_working_memory(db: Path) -> None:
    """两场会话的工作记忆互不干扰，但同一个人的画像共享。

    这正是「会话」和「用户」分开两个维度的意义。
    """
    model_a = ScriptedModel(topics=ONE_TOPIC, grades=[{"correct": True, "feedback": "对"}])
    session_a = make_session(db, model_a, session_id="s1")
    session_a.start()
    session_a.answer("答案")

    model_b = ScriptedModel(topics=ONE_TOPIC)
    session_b = make_session(db, model_b, session_id="s2")
    session_b.start()

    assert session_b.memory.working.count() == 1, "新会话的工作记忆应该只有刚开课那一条"
    assert session_b.profile.mastery_of("递归") > 0.0, "同一个人换了会话，水平还认得出来"

    session_a.close()
    session_b.close()
