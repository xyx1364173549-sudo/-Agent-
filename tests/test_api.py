"""服务层测试（M5.4）。

用 FastAPI 自带的 ``TestClient`` 发真实请求，但**模型是假的**——
整套接口离线跑通，不花 API 费用、结果稳定可复现。

覆盖三类东西：

1. **正常流程**：建会话 → 流式开课 → 作答 → 查记忆与画像；
2. **错误路径**：会话不存在、重复创建、答案为空、没开课就作答。
   这些最容易被漏测，而它们恰恰是线上最常被触发的；
3. **SSE 报文格式**：事件名和分隔空行都要对。少一个空行，浏览器会一直等
   这条消息的剩余部分，表现成「卡住没反应」。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.planning.session import QUIZ_COUNT


class FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeModel:
    """按提示词里的特征词分辨角色的假模型。"""

    def __init__(self, *, topics: list[dict] | None = None) -> None:
        self.topics = topics or [{"topic": "递归", "depends_on": [], "reason": "基础"}]
        self.questions = [
            {"question": f"第{i}题", "answer": f"答案{i}", "hint": "提示"}
            for i in range(1, QUIZ_COUNT + 1)
        ]

    def _reply_text(self, prompt: str) -> str:
        if "拆解成若干知识点" in prompt:
            return json.dumps(self.topics, ensure_ascii=False)
        if "一对一编程导师" in prompt:
            return "递归就是函数自己调用自己。"
        if "请为下面的知识点出" in prompt:
            return json.dumps(self.questions, ensure_ascii=False)
        if "请批改下面这道题" in prompt:
            wrong = "【错】" in prompt
            return json.dumps(
                {"correct": not wrong, "score": 0.0 if wrong else 1.0, "feedback": "评语"},
                ensure_ascii=False,
            )
        raise AssertionError(f"假模型没见过的提示词：{prompt[:60]}")

    def invoke(self, prompt: str) -> FakeReply:
        return FakeReply(self._reply_text(prompt))

    def stream(self, prompt: str):
        """按字推，模拟真实模型的流式输出。"""
        for char in self._reply_text(prompt):
            yield FakeReply(char)


@pytest.fixture
def client(tmp_path: Path):
    """一个独立的测试客户端，数据库落在临时目录。"""
    app = create_app(db_path=tmp_path / "api.db", model=FakeModel())
    with TestClient(app) as test_client:
        yield test_client


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体拆成 (事件名, 数据) 列表。

    自己解析一遍，等于顺手验证了报文的格式确实符合规范——
    直接断言字符串相等太脆，看不懂哪里错了。
    """
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        assert event_name is not None, f"这条报文没有事件名：{block!r}"
        assert data is not None, f"这条报文没有数据：{block!r}"
        events.append((event_name, data))
    return events


# --------------------------------------------------------------------------
# 基础
# --------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sessions"] == 0


def test_index_points_to_docs(client: TestClient) -> None:
    """根路径给指引，免得直接访问看到 404 一脸茫然。"""
    body = client.get("/").json()
    assert body["docs"] == "/docs"


def test_openapi_schema_is_available(client: TestClient) -> None:
    """接口文档能生成，说明所有出参模型都是合法的。"""
    schema = client.get("/openapi.json").json()
    assert "/api/sessions" in schema["paths"]
    assert "/api/sessions/{session_id}/answer" in schema["paths"]


# --------------------------------------------------------------------------
# 建会话
# --------------------------------------------------------------------------


def test_create_session(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"goal": "掌握递归", "user_id": "u1"})

    assert response.status_code == 201
    body = response.json()
    assert body["goal"] == "掌握递归"
    assert body["session_id"], "没传 session_id 时应当自动生成"
    assert body["finished"] is False


def test_create_session_with_explicit_id(client: TestClient) -> None:
    body = client.post(
        "/api/sessions", json={"goal": "掌握递归", "user_id": "u1", "session_id": "my-session"}
    ).json()

    assert body["session_id"] == "my-session"


def test_duplicate_session_id_rejected(client: TestClient) -> None:
    """同一个 id 建两次要报 409。

    悄悄复用会让「我以为开了新课，结果接着上次的进度」这种困惑很难排查。
    """
    payload = {"goal": "掌握递归", "user_id": "u1", "session_id": "dup"}
    assert client.post("/api/sessions", json=payload).status_code == 201
    assert client.post("/api/sessions", json=payload).status_code == 409


@pytest.mark.parametrize("goal", ["", "   "])
def test_blank_goal_rejected(client: TestClient, goal: str) -> None:
    """空目标在门口就拦下，别带着它去调大模型。"""
    response = client.post("/api/sessions", json={"goal": goal, "user_id": "u1"})

    assert response.status_code == 422


def test_too_many_topics_rejected(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"goal": "掌握递归", "max_topics": 999})

    assert response.status_code == 422


# --------------------------------------------------------------------------
# 开课（SSE）
# --------------------------------------------------------------------------


def test_start_stream_emits_all_event_kinds(client: TestClient) -> None:
    """开课的事件流要有：状态提示、讲解片段、完整状态、结束标记。"""
    client.post("/api/sessions", json={"goal": "掌握递归", "user_id": "u1", "session_id": "s1"})

    response = client.get("/api/sessions/s1/start/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    names = [name for name, _ in events]

    assert names[0] == "status", "先告诉前端在忙什么"
    assert "token" in names, "讲解要逐字推"
    assert names[-1] == "done", "最后要有结束标记"
    assert "state" in names


def test_start_stream_tokens_rebuild_explanation(client: TestClient) -> None:
    """把推出去的片段拼起来，应当等于最终的讲解全文。

    这条盯的是「流式」和「完整文本」必须一致——两边不一致的话，
    学生看到的和有存进记忆的就不是同一段话。
    """
    client.post("/api/sessions", json={"goal": "掌握递归", "user_id": "u1", "session_id": "s1"})
    events = parse_sse(client.get("/api/sessions/s1/start/stream").text)

    pieces = "".join(data["text"] for name, data in events if name == "token")
    state = next(data for name, data in events if name == "state")

    assert pieces
    assert pieces == state["explanation"]


def test_start_stream_state_has_path_and_question(client: TestClient) -> None:
    """开课完成后，前端要能拿到路径和第一道题。"""
    client.post("/api/sessions", json={"goal": "掌握递归", "user_id": "u1", "session_id": "s1"})
    events = parse_sse(client.get("/api/sessions/s1/start/stream").text)
    state = next(data for name, data in events if name == "state")

    assert state["topic"] == "递归"
    assert state["stage"] == "waiting"
    assert len(state["path"]["steps"]) == 1
    assert state["path"]["steps"][0]["topic"] == "递归"
    assert state["question"]["question"] == "第1题"
    assert state["question"]["total"] == QUIZ_COUNT


def test_start_stream_twice_rejected(client: TestClient) -> None:
    """重复开课报 409——否则会重置进度、白花几次模型调用。"""
    client.post("/api/sessions", json={"goal": "掌握递归", "user_id": "u1", "session_id": "s1"})
    client.get("/api/sessions/s1/start/stream")

    assert client.get("/api/sessions/s1/start/stream").status_code == 409


def test_start_unknown_session_returns_404(client: TestClient) -> None:
    response = client.get("/api/sessions/nope/start/stream")

    assert response.status_code == 404
    assert "没有找到会话" in response.json()["detail"]


# --------------------------------------------------------------------------
# 作答
# --------------------------------------------------------------------------


def _started_client(client: TestClient, session_id: str = "s1") -> None:
    client.post(
        "/api/sessions", json={"goal": "掌握递归", "user_id": "u1", "session_id": session_id}
    )
    client.get(f"/api/sessions/{session_id}/start/stream")


def test_answer_correct(client: TestClient) -> None:
    _started_client(client)

    response = client.post("/api/sessions/s1/answer", json={"answer": "我的答案"})

    assert response.status_code == 200
    body = response.json()
    assert body["grade"]["correct"] is True
    assert body["question"]["index"] == 1, "答完一题该推进到下一题"


def test_answer_wrong_triggers_retry(client: TestClient) -> None:
    _started_client(client)

    body = client.post("/api/sessions/s1/answer", json={"answer": "【错】我不知道"}).json()

    assert body["grade"]["correct"] is False
    assert body["question"]["index"] == 0, "重讲后重新出题，进度归零"
    assert body["explanation"] == "递归就是函数自己调用自己。"


def test_blank_answer_rejected_by_schema(client: TestClient) -> None:
    """空答案在 Pydantic 那层就被拦下（422），根本进不了业务流程。"""
    _started_client(client)

    assert client.post("/api/sessions/s1/answer", json={"answer": ""}).status_code == 422


def test_answer_before_start_rejected(client: TestClient) -> None:
    """没开课就作答要报 409，而不是 500。"""
    client.post("/api/sessions", json={"goal": "掌握递归", "user_id": "u1", "session_id": "s1"})

    response = client.post("/api/sessions/s1/answer", json={"answer": "答案"})

    assert response.status_code == 409
    assert "还没开课" in response.json()["detail"]


def test_answer_unknown_session_returns_404(client: TestClient) -> None:
    assert client.post("/api/sessions/nope/answer", json={"answer": "答案"}).status_code == 404


def test_answer_updates_mastery(client: TestClient) -> None:
    """作答要真的影响画像——否则路径永远不会变。"""
    _started_client(client)
    client.post("/api/sessions/s1/answer", json={"answer": "我的答案"})

    profile = client.get("/api/sessions/s1/profile").json()
    mastery = {item["topic"]: item["mastery"] for item in profile["topics"]}

    assert mastery["递归"] > 0.0


def test_path_mastery_is_not_stale(client: TestClient) -> None:
    """作答之后，接口返回的路径里掌握度必须跟着变。

    这条守着一个真实出过的问题：路径是「挑知识点」那一步算出来的，
    中间答的几道题不会重新走那一步。如果直接把存档的路径原样返回，
    前端就会看到「刚答对了，掌握度还是 0.00」这种自相矛盾的数据。
    """
    _started_client(client)
    before = client.get("/api/sessions/s1").json()["path"]["steps"][0]["mastery"]
    assert before == 0.0

    after = client.post("/api/sessions/s1/answer", json={"answer": "我的答案"}).json()

    assert after["path"]["steps"][0]["mastery"] > before
    assert after["path"]["steps"][0]["status"] != "focus" or after["path"]["steps"][0]["mastery"] > 0


# --------------------------------------------------------------------------
# 查状态
# --------------------------------------------------------------------------


def test_read_session(client: TestClient) -> None:
    _started_client(client)

    body = client.get("/api/sessions/s1").json()

    assert body["session_id"] == "s1"
    assert body["user_id"] == "u1"
    assert body["goal"] == "掌握递归"
    assert body["log"], "过程日志要能拿出来看"


def test_read_memory(client: TestClient) -> None:
    _started_client(client)
    client.post("/api/sessions/s1/answer", json={"answer": "递归我还是不太会"})

    body = client.get("/api/sessions/s1/memory").json()

    assert body["stats"]["messages"] >= 2, "讲解 + 作答"
    assert body["stats"]["events"] >= 2
    assert body["context"], "组装好的记忆简报要能拿到"
    assert body["events"]


def test_read_profile(client: TestClient) -> None:
    _started_client(client)

    body = client.get("/api/sessions/s1/profile").json()

    assert body["user_id"] == "u1"
    assert body["snapshot"], "画像简报不能是空的"


def test_kb_search_on_empty_store(client: TestClient) -> None:
    """知识库没建时返回空列表，而不是报错。

    「还没建库」是个正常状态，不该让接口挂掉。
    """
    response = client.get("/api/kb/search", params={"q": "递归"})

    assert response.status_code == 200
    assert response.json() == []


def test_kb_search_requires_query(client: TestClient) -> None:
    assert client.get("/api/kb/search").status_code == 422


# --------------------------------------------------------------------------
# 删除
# --------------------------------------------------------------------------


def test_delete_session(client: TestClient) -> None:
    _started_client(client)

    assert client.delete("/api/sessions/s1").json() == {"removed": True}
    assert client.get("/api/sessions/s1").status_code == 404
    assert client.get("/api/health").json()["sessions"] == 0


def test_delete_unknown_session_returns_404(client: TestClient) -> None:
    assert client.delete("/api/sessions/nope").status_code == 404


# --------------------------------------------------------------------------
# 会话隔离
# --------------------------------------------------------------------------


def test_sessions_are_independent(client: TestClient) -> None:
    """两场会话互不干扰，但同一个用户的画像共享。"""
    _started_client(client, "s1")
    client.post("/api/sessions/s1/answer", json={"answer": "我的答案"})

    _started_client(client, "s2")

    first = client.get("/api/sessions/s1").json()
    second = client.get("/api/sessions/s2").json()

    assert first["session_id"] != second["session_id"]
    # s2 是新会话，工作记忆从零开始
    assert client.get("/api/sessions/s2/memory").json()["stats"]["messages"] < 3
    # 但同一个用户，画像带着上次的水平
    assert client.get("/api/sessions/s2/profile").json()["topics"]
