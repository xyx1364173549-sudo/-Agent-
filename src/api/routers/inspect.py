"""记忆、画像与知识库的查询接口。

这些接口回答的是「系统现在知道你什么」，对应前端右侧那个记忆面板。

## 为什么要把内部状态暴露出来

一个只吐答案的系统和一个能解释自己判断的系统，可信度完全不同。学生看到
「本次学习重点是这三块」配上「上一轮递归练了 5 题对了 2 题」，才会相信
这个安排是有依据的，而不是随机抽的。

对开发同样有用：调不对的时候，看一眼记忆里到底存了什么，比读日志快得多。
"""

import threading

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.routers.session import get_session_or_404
from src.api.schemas import KnowledgeHit, MemoryOut, MemoryStats, ProfileOut
from src.planning.profile import LearnerProfile
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 知识库懒加载：构造 VectorStore 会拉起 Chroma 并加载本地向量模型，
# 要几秒钟。放在模块导入时做，会让服务启动变慢，还会拖累没用到知识库的接口。
_kb_lock = threading.Lock()
_kb_store = None


@router.get("/sessions/{session_id}/memory", response_model=MemoryOut, summary="看三层记忆")
def read_memory(session_id: str, request: Request) -> MemoryOut:
    """把三层记忆的内容摊开给人看。"""
    session = get_session_or_404(request, session_id)
    memory = session.memory

    stats = memory.stats()
    facts = [
        {
            "category": item["category"],
            "key": item["key"],
            "value": item["value"],
            "confidence": item["confidence"],
        }
        for item in memory.semantic.all_facts()
    ]
    events = [
        {
            "event_type": item["event_type"],
            "content": item["content"],
            "importance": item["importance"],
            "created_at": item["created_at"],
        }
        for item in memory.episodic.recent(20)
    ]

    return MemoryOut(
        stats=MemoryStats(
            messages=stats["messages"],
            events=stats["events"],
            facts=stats["facts"],
        ),
        context=memory.context(),
        facts=facts,
        events=events,
    )


@router.get("/sessions/{session_id}/profile", response_model=ProfileOut, summary="看学习者画像")
def read_profile(session_id: str, request: Request) -> ProfileOut:
    """画像跟着**用户**走，不跟着会话走。

    所以这场会话即使是刚开的，只要这个用户以前学过，画像里就已经有东西了。
    这也正是「换个会话还记得你」的实现方式。
    """
    session = get_session_or_404(request, session_id)
    profile: LearnerProfile = session.profile

    return ProfileOut(
        user_id=session.user_id,
        goal=profile.goal(),
        snapshot=profile.snapshot(),
        topics=profile.all_topics(),
    )


@router.get("/kb/search", response_model=list[KnowledgeHit], summary="搜知识库")
def search_knowledge(
    q: str = Query(..., min_length=1, max_length=200, description="查询词"),
    top_k: int = Query(5, ge=1, le=20, description="返回几条"),
    mode: str = Query("hybrid", pattern="^(vector|keyword|hybrid)$", description="检索模式"),
) -> list[KnowledgeHit]:
    """按语义/关键词检索学习资料。

    知识库是空的（没跑过 ``scripts/build_kb.py``）时返回空列表，而不是报错——
    「还没建库」是个正常状态，不该让接口挂掉。
    """
    store = _knowledge_store()
    if store.count() == 0:
        return []

    from src.rag.retriever import Retriever

    try:
        hits = Retriever(store).search(q, mode=mode, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - 检索失败不该让整个服务出错
        logger.exception("知识库检索失败 | q=%s", q)
        raise HTTPException(status_code=500, detail=f"检索失败：{exc}") from None

    return [
        KnowledgeHit(
            text=item["text"],
            score=float(item.get("score", 0.0)),
            source=item.get("source", ""),
            sources=list(item.get("sources", [])),
        )
        for item in hits
    ]


def _knowledge_store():
    """懒加载向量库（第一次调用时会拉起 Chroma 并加载本地向量模型）。"""
    global _kb_store
    if _kb_store is not None:
        return _kb_store

    with _kb_lock:
        if _kb_store is None:
            from src.rag.vectorstore import VectorStore

            logger.info("正在打开知识库……")
            _kb_store = VectorStore()
            logger.info("知识库已就绪 | 共 %d 个块", _kb_store.count())
    return _kb_store
