"""学习会话的状态与各节点逻辑：学 → 练 → 评 → 调 的闭环。

## 为什么把这个文件和 ``graph.py`` 分开

``graph.py`` 用 LangGraph 把节点连成图；本文件只写**每个节点干什么**。

分开的理由很实在：节点逻辑是业务，图是拼装方式。混在一起写的话，
单位测试就得先起一个 LangGraph 才能跑；分开之后，节点函数是普通函数，
传个字典进去、看传出来的字典对不对就行——**不装 LangGraph 也能测**。

## 一次调用只走一段

图不是「一次跑完」的。学生的回答来自屏幕那头，程序不可能一直等着，
所以每来一个输入就让图往前走一段，靠 checkpointer 记住上次停在哪：

    第一次调用：选知识点 → 讲解 → 出题 → 停（等作答）
    喂一个答案：  批改 → 停（还有题）/ 换讲法重讲 / 换下一个知识点

## 答错怎么办

给一次「换个讲法重讲」的机会，还错就先放下这个知识点、换个会的建立信心。
一直卡在同一个点上，学生就跑掉了。
"""

import operator
from typing import Annotated, TypedDict

from src.agents import GraderAgent, QuizAgent, TutorAgent
from src.memory.manager import MemoryManager
from src.planning.planner import STATUS_REVIEW, plan
from src.planning.profile import LearnerProfile, WEAK_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 一个知识点出几道题。太少了判不准，太多了学生不耐烦。
QUIZ_COUNT = 3

# 允许重讲几次。一次就够——第二次还讲不明白，说明该回头补前置知识，
# 而不是继续在这里磨。
MAX_RETRY = 1

# 阶段标记，决定了图从哪儿继续
STAGE_NEED_TOPIC = "need_topic"  # 该挑下一个知识点了
STAGE_WAITING = "waiting"  # 已出题，等学生作答
STAGE_DONE = "done"  # 整条路径走完


class LearningState(TypedDict, total=False):
    """一次学习会话的全部状态。

    ``total=False`` 表示字段可以缺——刚开始时只有目标和知识点，其余边走边填。

    ``log`` 字段用了 ``Annotated[list, operator.add]``：这样节点只要返回
    ``{"log": [新的一条]}``，框架就会把它**追加**到原列表后面，节点不必
    自己读一遍旧列表再拼回去。这是 LangGraph 声明「可累加字段」的标准写法。
    """

    user_id: str
    session_id: str
    goal: str
    topics: list[dict]  # 拆解出来的知识点
    path: dict  # 当前学习路径（每次都按最新画像重算）
    topic: str  # 正在学的知识点
    explanation: str  # 导师的讲解
    questions: list[dict]  # 当前这批题
    index: int  # 做到第几题
    answer: str  # 学生这次写的答案
    grade: dict  # 批改结果
    feedback: str  # 给学生的反馈（重讲时会带进提示词）
    retries: int  # 当前知识点已经重讲过几次
    retry_needed: bool  # 这次批改后要不要重讲（由批改节点算好，路由只读结论）
    skipped: list[str]  # 本次会话已经放弃的知识点（重讲后仍答错的）
    needs_review: list[str]  # 因为卡在后继知识点上、被拉回来重补的前置知识点
    results: list[bool]  # 当前知识点所有作答
    rounds: int  # 累计批改过多少次
    log: Annotated[list, operator.add]
    stage: str


def new_state(user_id: str, session_id: str, goal: str, topics: list[dict]) -> LearningState:
    """开一场新的学习会话。"""
    return {
        "user_id": user_id,
        "session_id": session_id,
        "goal": goal,
        "topics": topics,
        "path": {},
        "topic": "",
        "explanation": "",
        "questions": [],
        "index": 0,
        "answer": "",
        "grade": {},
        "feedback": "",
        "retries": 0,
        "retry_needed": False,
        "skipped": [],
        "needs_review": [],
        "results": [],
        "rounds": 0,
        "log": [],
        "stage": STAGE_NEED_TOPIC,
    }


# --------------------------------------------------------------------------
# 节点
# --------------------------------------------------------------------------


def choose_topic_node(state: LearningState, *, profile: LearnerProfile) -> dict:
    """挑下一个该学的知识点。

    每次都**按最新画像重排一遍路径**，而不是沿用上一轮的顺序——
    学生刚练完一个知识点，掌握度变了，最该学的可能就换人了。
    这正是「动态规划」里「动态」两个字的意思。

    已经放弃过的知识点（重讲后仍然答错）会被跳过。这一步不能省：
    只有单个知识点时，放弃它之后如果还按原顺序挑，会又挑回它，
    于是「讲 → 错 → 讲 → 错」转个不停。
    """
    result = plan(state["topics"], profile, goal=state["goal"])
    step = _pick_step(
        result,
        set(state.get("skipped", [])),
        set(state.get("needs_review", [])),
    )

    if step is None:
        logger.info("本次会话没有可继续的知识点了 | goal=%s", state["goal"])
        return {
            "path": result,
            "topic": "",
            "stage": STAGE_DONE,
            "log": [{"entry": "没有可继续的知识点了，本次学习结束"}],
        }

    # 换知识点时把「上一个知识点」的痕迹清掉：index、重讲次数、作答记录
    # 都是跟着知识点走的，不清会串到下一个头上
    return {
        "path": result,
        "topic": step["topic"],
        "stage": STAGE_NEED_TOPIC,
        "index": 0,
        "retries": 0,
        "results": [],
        "questions": [],
        "log": [
            {
                "entry": f"选出下一个知识点：{step['topic']}"
                f"（{step['status']}，掌握度 {step['mastery']:.2f}）"
            }
        ],
    }


def _pick_step(result: dict, skipped: set[str], needs_review: set[str]) -> dict | None:
    """挑下一个该学的知识点。

    优先级：**回退补前置** > 正常按掌握度排序。

    补前置要插队是有道理的：在「尾递归」上反复出错，很可能是「递归」本身
    没打牢。这时候按原顺序继续往前推，只会在下一个坑里再摔一次。
    """
    steps = result.get("steps", [])

    for step in steps:
        if step["topic"] in needs_review and step["topic"] not in skipped:
            return step

    for step in steps:
        if step["status"] == STATUS_REVIEW:
            continue  # 已掌握，不用专门学
        if step["topic"] in skipped:
            continue  # 这次会话已经放弃过
        return step
    return None


def _prerequisites(topic: str, topics: list[dict]) -> list[str]:
    """查一个知识点依赖哪些前置。"""
    for item in topics:
        if item["topic"] == topic:
            return list(item.get("depends_on", []))
    return []


def teach_node(
    state: LearningState,
    *,
    tutor: TutorAgent,
    memory: MemoryManager,
    profile: LearnerProfile,
) -> dict:
    """讲解当前知识点。

    上下文由两块拼成：**分层记忆**（最近发生过什么）+ **学习者画像**
    （这个人什么水平）。重讲时再把上次的错误反馈加上，让模型换个讲法。

    这就是「分层记忆驱动的个性化教学」在代码里的样子——
    同一个知识点，不同的人拿到不同的讲解。
    """
    topic = state["topic"]
    parts: list[str] = []

    memory_context = memory.context()
    if memory_context.strip():
        parts.append(f"【记忆】\n{memory_context}")

    snapshot = profile.snapshot()
    if snapshot.strip():
        parts.append(f"【画像】\n{snapshot}")

    retrying = state.get("retries", 0) > 0
    if retrying and state.get("feedback"):
        parts.append(
            "【上一次没讲明白】\n"
            f"学生上次作答暴露出：{state['feedback']}\n"
            "请换一种讲法或换个类比，不要重复上次的说法。"
        )

    explanation = tutor.explain(topic, context="\n\n".join(parts))
    memory.record_event("taught", f"讲解了《{topic}》", importance=0.4)
    # 讲解也进工作记忆：下一轮讲解时 memory.context() 会带上「刚才讲了什么」，
    # 学生说「还是没懂」时导师才知道该在哪个说法上再展开
    memory.add_message("assistant", explanation)

    return {
        "explanation": explanation,
        "log": [{"entry": f"导师讲解《{topic}》" + ("（换讲法重讲）" if retrying else "")}],
    }


def quiz_node(
    state: LearningState,
    *,
    quiz_agent: QuizAgent,
    memory: MemoryManager,
    profile: LearnerProfile,
) -> dict:
    """出题。

    传进画像是有意为之：出题 Agent 会避开学生已经掌握的内容，把题出在薄弱点上。
    没有这一步，个性化就只剩「讲」而没有「练」。
    """
    topic = state["topic"]
    questions = quiz_agent.generate(topic, count=QUIZ_COUNT, context=profile.snapshot())
    memory.record_event("assigned_quiz", f"为《{topic}》布置了 {len(questions)} 道题", importance=0.4)

    # index 归 0：重讲之后会重新出一批题，进度得从头数
    return {
        "questions": questions,
        "index": 0,
        "stage": STAGE_WAITING,
        "feedback": "",
        "log": [{"entry": f"出题 {len(questions)} 道"}],
    }


def evaluate_node(
    state: LearningState,
    *,
    grader: GraderAgent,
    memory: MemoryManager,
    profile: LearnerProfile,
) -> dict:
    """批改学生这一次的作答，并把结果写进画像。

    掌握度**当场就更新**，不等这个知识点练完。理由：每一次练习都真实反映了
    当时的水平，攒到最后一起算并不会更准，反而让「学到一半的画像」失去意义。
    指数移动平均本身就带着「近期权重更高」的性质，逐次更新正是它设计出来
    要解决的问题。

    批改结果同时写进情景记忆——「9 月 17 日答错了递归的终止条件」是一条真实
    发生过的事件，将来回看学习历程时要靠它。
    """
    question = state["questions"][state["index"]]
    answer = state["answer"]

    # 学生的话先进工作记忆，再批改。写工作记忆时会顺带过一遍事实抽取
    # （M2.5 那个能力），「递归我还是不太会」这类话会随之沉淀成语义记忆——
    # 于是画像之外还留着一份「他自己是怎么说的」
    memory.add_message("user", answer)

    result = grader.grade(
        question["question"],
        reference=question["answer"],
        answer=answer,
    )

    topic = state["topic"]
    mastery = profile.record_attempt(topic, result["correct"])

    # 「要不要重讲」在这里一次算清，写进 state 让路由函数照着走。
    # 不这么做的话，重讲次数得在路由和节点两处推断，很容易推出无限重讲——
    # 路由函数只能返回去向，改不了 state，所以判断依据必须由节点准备。
    retries = state.get("retries", 0)
    retry_needed = (not result["correct"]) and retries < MAX_RETRY

    # 重讲机会也用完了还是答错 —— 这个知识点本次会话先放下。
    # 必须记下来，否则「挑知识点」时又会挑回它，来回转不出去
    skipped = list(state.get("skipped", []))
    if not result["correct"] and not retry_needed and topic not in skipped:
        skipped.append(topic)

    # 顺带做一次**回退诊断**：卡住的时候，把他依赖的前置知识点重新拉回待学列表。
    # 「动态重规划」不只是重排顺序——在「尾递归」上反复出错，往往是「递归」
    # 没真的打牢。这时候继续往前推，只会在下一个坑里再摔一次
    needs_review = list(state.get("needs_review", []))
    pulled_back: list[str] = []
    if not result["correct"] and not retry_needed:
        for dep in _prerequisites(topic, state["topics"]):
            if dep not in needs_review:
                needs_review.append(dep)
                pulled_back.append(dep)

    memory.record_event(
        "answered_question",
        f"《{topic}》{'答对' if result['correct'] else '答错'}：{result['feedback']}",
        # 答错的事更值得记住——将来复盘「他卡在哪」要靠这些
        importance=0.7 if not result["correct"] else 0.5,
    )

    return {
        "grade": result,
        "feedback": result["feedback"],
        "index": state["index"] + 1,
        "rounds": state.get("rounds", 0) + 1,
        "retries": retries + (1 if retry_needed else 0),
        "retry_needed": retry_needed,
        "skipped": skipped,
        "needs_review": needs_review,
        "results": [*state.get("results", []), result["correct"]],
        "answer": "",
        "log": [
            {
                "entry": f"批改第 {state['index'] + 1} 题："
                f"{'对' if result['correct'] else '错'} —— {result['feedback'][:60]}"
                f"（{topic} 掌握度 → {mastery:.2f}）"
                + (f"；回头补前置：{'、'.join(pulled_back)}" if pulled_back else "")
            }
        ],
    }


def update_node(
    state: LearningState,
    *,
    profile: LearnerProfile,
    memory: MemoryManager,
) -> dict:
    """一个知识点练完了：记一笔事件，然后回到「挑下一个知识点」。

    掌握度已经由 ``evaluate_node`` 逐题更新过了，这里只做收尾：
    把本轮的整体表现记成一条情景事件，供以后复盘用。
    """
    topic = state["topic"]
    results = state.get("results", [])
    correct_count = sum(results)
    mastery = profile.mastery_of(topic)

    # 这个知识点补完了，从「待补前置」名单里划掉——否则下次选知识点时
    # 还会优先挑它，卡在同一个地方反复补
    needs_review = [name for name in state.get("needs_review", []) if name != topic]

    if not results:
        return {
            "stage": STAGE_NEED_TOPIC,
            "retries": 0,
            "results": [],
            "needs_review": needs_review,
        }

    # 掌握度低于阈值就记成「卡住了」。事件类型决定它将来会不会被翻出来，
    # 所以判据要和 planner 用的薄弱阈值保持一致，两处不能各定一个数
    event_type = "struggled" if mastery < WEAK_THRESHOLD else "progressed"
    memory.record_event(
        event_type,
        f"《{topic}》本轮 {len(results)} 题对 {correct_count} 题，掌握度更新为 {mastery:.2f}",
        importance=0.8 if event_type == "struggled" else 0.6,
    )

    return {
        "stage": STAGE_NEED_TOPIC,
        "retries": 0,
        "needs_review": needs_review,
        "results": [],
        "log": [
            {
                "entry": f"《{topic}》本轮答对 {correct_count}/{len(results)}，"
                f"掌握度 {mastery:.2f}，换下一个知识点"
            }
        ],
    }


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------


def route_entry(state: LearningState) -> str:
    """刚进入图时往哪走。

    带着答案进来就是「继续做题」，空手进来就是「开始学新东西」。
    """
    if state.get("stage") == STAGE_DONE:
        return "done"
    if state.get("stage") == STAGE_WAITING and state.get("answer"):
        return "evaluate"
    return "choose"


def route_after_evaluate(state: LearningState) -> str:
    """批改完之后往哪走。

    三种去向：

    ``retry``
        答错了，但还有重讲机会 —— 换个讲法再来。这条边把流程送回 ``teach``，
        构成图里唯一一条**回边**，也就是「循环」。
    ``wait``
        答对了、还有题没做 —— 停下来等学生作答。
    ``update``
        这个知识点本轮结束了（题做完了，或者答错且不能再重讲）——
        记成绩、换下一个。

    **答错且不能重讲时直接换知识点**，不让他硬着头皮做完剩下的题：
    第一题就不会，后面几题大概率也是错，继续做只是消耗耐心。
    先放下、换个会的建立信心，回头再补。
    """
    if state.get("retry_needed"):
        return "retry"

    grade = state.get("grade", {})
    if grade.get("correct", False) and state["index"] < len(state["questions"]):
        return "wait"

    return "update"


def summary(state: LearningState, *, profile: LearnerProfile) -> str:
    """把这次会话的结果讲成一段话，用于收尾汇报。"""
    if state.get("stage") != STAGE_DONE:
        return "这次的学习还没结束。"

    lines = [f"学习目标：{state.get('goal', '')}", "", "最终画像：", profile.snapshot(), ""]
    lines.append(f"一共批改 {state.get('rounds', 0)} 次，过程如下：")
    for item in state.get("log", []):
        lines.append(f"  - {item['entry']}")
    return "\n".join(lines)
