"""会话与对话接口。

## 接口一览

    POST   /api/sessions                  建一场会话（不跑图，等前端来拉）
    GET    /api/sessions/{id}             看当前状态
    GET    /api/sessions/{id}/start/stream 开课，SSE 流式讲解 + 出题
    POST   /api/sessions/{id}/answer      提交一次作答
    DELETE /api/sessions/{id}             删掉会话

## 为什么开课要用 SSE、作答却用普通请求

开课要生成 350 字讲解，等它写完要好几秒，屏幕上空着很难受，所以逐字推。
作答的批改结果本身很短，等一下就出来了，用普通 JSON 更好处理——
错误码、重试、调试都更直接。

**不要为了统一而统一**：哪种形式更贴合这一步的实际需要，就用哪种。
"""

import threading
from collections.abc import Iterator
from datetime import datetime
from queue import Queue

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    AnswerRequest,
    GradeOut,
    PathOut,
    QuestionOut,
    SessionCreateRequest,
    SessionInfo,
    SessionState,
    StepOut,
)
from src.api.sse import (
    EVENT_DONE,
    EVENT_ERROR,
    EVENT_STATE,
    EVENT_STATUS,
    EVENT_TOKEN,
    sse_event,
    sse_headers,
)
from src.api.store import SessionNotFound, SessionStore
from src.planning.graph import LearningSession
from src.planning.planner import plan as build_plan
from src.planning.planner import progress as path_progress
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# --------------------------------------------------------------------------
# 内部辅助
# --------------------------------------------------------------------------


def _store(request: Request) -> SessionStore:
    """从应用状态里取会话缓存。"""
    return request.app.state.sessions


def get_session_or_404(request: Request, session_id: str) -> LearningSession:
    """取会话，找不到就返回 404（而不是 500）。

    跨模块用，所以名字不带下划线：``routers/inspect.py`` 也调它。
    """
    try:
        return _store(request).require(session_id)
    except SessionNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"没有找到会话 {session_id}。可能是 id 写错了，或者服务重启过——"
            "会话缓存放在内存里，重启就没了（画像和记忆还在数据库里）",
        ) from None


def build_state(session: LearningSession) -> SessionState:
    """把会话的内部状态整理成接口返回的形状。

    内部状态是个扁平字典（LangGraph 要求那样），直接丢给前端会暴露一堆
    实现细节。这里做一次整理：该嵌套的嵌套、该给默认值的给默认值。

    **路径每次都按最新画像重算**，而不是读开课时存下的那份快照。原因很实际：
    作答会立刻改变掌握度，而 ``path`` 是上一轮「挑知识点」时算出来的，
    中间答的这几道题不会重新走那一步。不重算的话，前端就会看到
    「刚答对了，掌握度却还是 0.00」这种自相矛盾的数据。

    重算很便宜——纯本地计算，不调模型，也不碰网络。
    """
    state = session.state

    fresh_path = build_plan(
        state.get("topics", []),
        session.profile,
        goal=session.goal,
    )
    steps = [
        StepOut(
            order=item["order"],
            topic=item["topic"],
            mastery=item["mastery"],
            status=item["status"],
            depends_on=list(item.get("depends_on", [])),
        )
        for item in fresh_path.get("steps", [])
    ]

    questions = state.get("questions", [])
    index = state.get("index", 0)
    question = None
    if index < len(questions):
        item = questions[index]
        question = QuestionOut(
            index=index,
            total=len(questions),
            question=item["question"],
            hint=item.get("hint", ""),
        )

    raw_grade = state.get("grade") or {}
    grade = None
    if raw_grade:
        grade = GradeOut(
            correct=bool(raw_grade.get("correct")),
            score=float(raw_grade.get("score", 0.0)),
            feedback=str(raw_grade.get("feedback", "")),
        )

    return SessionState(
        session_id=session.session_id,
        user_id=session.user_id,
        goal=session.goal,
        stage=state.get("stage", "need_topic"),
        finished=session.finished,
        topic=state.get("topic", ""),
        explanation=state.get("explanation", ""),
        question=question,
        path=PathOut(goal=fresh_path.get("goal", session.goal), steps=steps),
        progress=path_progress(fresh_path),
        grade=grade,
        log=[item.get("entry", "") for item in state.get("log", [])],
    )


# --------------------------------------------------------------------------
# 接口
# --------------------------------------------------------------------------


@router.post("/sessions", response_model=SessionInfo, status_code=201, summary="建一场学习会话")
def create_session(payload: SessionCreateRequest, request: Request) -> SessionInfo:
    """建会话但**不立刻开课**。

    拆开来是有意的：开课要调好几次大模型、花好几秒，放在创建接口里会让
    前端一直转圈。分开之后，前端可以先拿到 session_id 把界面搭好，
    再去拉 SSE 慢慢等内容出来。
    """
    store = _store(request)
    session_id = payload.session_id or _make_session_id(payload.user_id)

    try:
        session = store.create(
            goal=payload.goal,
            user_id=payload.user_id,
            session_id=session_id,
            max_topics=payload.max_topics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    return SessionInfo(
        session_id=session.session_id,
        user_id=session.user_id,
        goal=session.goal,
        stage="need_topic",
        finished=False,
    )


@router.get("/sessions/{session_id}", response_model=SessionState, summary="看会话当前状态")
def read_session(session_id: str, request: Request) -> SessionState:
    return build_state(get_session_or_404(request, session_id))


@router.get(
    "/sessions/{session_id}/start/stream",
    summary="开课：流式讲解 + 出题（SSE）",
    response_class=StreamingResponse,
)
def start_session_stream(session_id: str, request: Request) -> StreamingResponse:
    """开课。用 SSE 把导师讲解逐字推出去。

    用 GET 而不是 POST，是为了让浏览器能直接用 ``EventSource`` 连——
    它只支持 GET。参数都在 URL 里，本来就没什么请求体要传。
    """
    session = get_session_or_404(request, session_id)

    if session.state:
        raise HTTPException(
            status_code=409,
            detail="这场会话已经开过课了，直接查状态或提交答案即可",
        )

    return StreamingResponse(
        _stream_start(session),
        media_type="text/event-stream",
        headers=sse_headers(),
    )


def _stream_start(session: LearningSession) -> Iterator[str]:
    """开课的事件流。

    结构是**一个线程 + 一个队列**：

        后台线程跑 session.start()，讲解每生成一段就调 on_token，把文字塞进队列；
        主线程（SSE 生成器）不停从队列取，取到什么就推给浏览器。

    为什么不用「直接在大模型回调里 yield」？因为 ``yield`` 只能出现在生成器函数里，
    而模型回调是模型那边调用的，不在生成器的调用栈上。队列把两边解耦开，
    一个只管生产、一个只管消费。
    """
    queue: Queue = Queue()

    def on_token(text: str) -> None:
        queue.put((EVENT_TOKEN, {"text": text}))

    # 挂上出口。挂上之后 TokenStream.active 变真，讲解就走流式；
    # 请求结束（或客户端断开）时在 finally 里摘掉，避免回调残留在会话上
    session.tokens.callback = on_token

    def run() -> None:
        try:
            session.start()
            queue.put((EVENT_STATE, build_state(session).model_dump()))
        except Exception as exc:  # noqa: BLE001 - 任何失败都要让前端知道，不能静默断流
            logger.exception("开课失败 | session=%s", session.session_id)
            queue.put((EVENT_ERROR, {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            queue.put(None)  # 结束哨兵

    # daemon=True：客户端断开后这个线程不会拦住进程退出。
    # 它最多把这一轮跑完，代价是一次没用的模型调用——比让进程卡住划算
    threading.Thread(target=run, daemon=True).start()

    yield sse_event(EVENT_STATUS, {"message": "正在拆解学习目标……"})

    try:
        while True:
            item = queue.get()
            if item is None:
                break
            event, data = item
            yield sse_event(event, data)
    finally:
        # 客户端中途断开时生成器会被关掉，这里保证回调不会残留——
        # 否则它还会往一个没人听的队列里塞文字
        session.tokens.callback = None

    yield sse_event(EVENT_DONE, {"session_id": session.session_id})


@router.post("/sessions/{session_id}/answer", response_model=SessionState, summary="提交一次作答")
def submit_answer(session_id: str, payload: AnswerRequest, request: Request) -> SessionState:
    """批改学生这一次的作答，推进到下一步。"""
    session = get_session_or_404(request, session_id)

    if not session.state:
        raise HTTPException(status_code=409, detail="这场会话还没开课，先拉一次开课接口")
    if session.finished:
        raise HTTPException(status_code=409, detail="这条学习路径已经走完了")
    if session.current_question() is None:
        raise HTTPException(status_code=409, detail="当前没有待作答的题目")

    try:
        session.answer(payload.answer)
    except ValueError as exc:
        # 空答案之类的问题属于请求本身的毛病，400 比 500 更贴切
        raise HTTPException(status_code=400, detail=str(exc)) from None

    return build_state(session)


@router.delete("/sessions/{session_id}", summary="删掉会话")
def delete_session(session_id: str, request: Request) -> dict[str, bool]:
    removed = _store(request).drop(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"没有找到会话 {session_id}")
    return {"removed": True}


def _make_session_id(user_id: str) -> str:
    """自动生成会话 id。

    用「用户 + 时间戳」而不是随机串：出问题时看日志一眼就知道是谁的哪一场。
    """
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_user = "".join(ch for ch in user_id if ch.isalnum() or ch in "-_") or "user"
    return f"{safe_user}-{stamp}"
