"""FastAPI 应用组装。

## 为什么用「应用工厂」而不是直接写一个模块级 app

``create_app()`` 每次调用都返回一个**独立的应用实例**，各自的会话缓存互不干扰。
这样测试才能并行、才能重复跑：如果全局只有一个 app，上一个用例开的会话会
留在缓存里，下一个用例「建同名会话」就会撞车。

模块底部仍然留了一句 ``app = create_app()``，那是给 ``uvicorn`` 用的——
它需要一个现成的对象。两件事不冲突。

## 启动方式

::

    .venv\\Scripts\\python.exe -m uvicorn src.api.app:app --reload

然后打开 http://127.0.0.1:8000/docs 就能看到自动生成的接口文档，
每个字段的含义都在上面——这是 Pydantic 模型白送的一份文档。
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import inspect_router, session_router
from src.api.schemas import HealthOut
from src.api.store import SessionStore
from src.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

VERSION = "0.6.0"

# 前端目录。纯静态页面，不需要构建步骤——双击也能开，
# 论文答辩时现场演示方便
WEB_DIR = PROJECT_ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动与停止时各做一件事。

    停止时把会话里的数据库连接关掉。不做的话进程退出时会有若干 SQLite
    连接悬着——单机开发看不出问题，但反复重启调试时可能撞上文件锁。
    """
    yield
    app.state.sessions.close_all()


def create_app(
    *,
    db_path: str | Path | None = None,
    model: Any | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    """组装应用。

    参数
    ----
    db_path:
        数据库路径，传给所有新会话。不传用 ``.env`` 里的配置。
    model:
        聊天模型。不传时每个会话各自创建默认的 DeepSeek 模型；
        **测试时塞一个假模型进来**，整套接口就能离线跑通，
        既不花钱，结果也稳定可复现。
    store:
        直接传入会话缓存。一般不用，留给需要复用缓存的场景（比如测试里
        想手工预置一场会话）。

    返回
    ----
    配置好的 ``FastAPI`` 实例。
    """
    app = FastAPI(
        title="分层记忆驱动的学习路径规划 Agent",
        description=(
            "毕业论文项目：用分层记忆（工作 / 情景 / 语义）与动态任务规划"
            "（LangGraph 多 Agent 编排）实现个性化学习路径规划。"
        ),
        version=VERSION,
        lifespan=lifespan,
    )

    # 前端本地开发时端口不同（比如 5173），浏览器会拦截跨域请求。
    # 开发阶段放开即可；真上线时要改成具体的域名白名单
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.sessions = store or SessionStore(db_path=db_path, model=model)
    app.state.db_path = db_path
    app.state.model = model

    app.include_router(session_router, prefix="/api", tags=["会话与对话"])
    app.include_router(inspect_router, prefix="/api", tags=["记忆与画像"])

    @app.get("/api/health", response_model=HealthOut, tags=["系统"], summary="健康检查")
    def health() -> HealthOut:
        """给部署脚本和监控用的：服务活着就返回 200。"""
        return HealthOut(
            status="ok",
            version=VERSION,
            sessions=app.state.sessions.count(),
        )

    # 前端页面挂到根路径，**必须在所有 API 路由之后**——
    # mount("/") 是个通配匹配，先挂它会把 /api/... 也一并吞掉。
    #
    # 前后端同源的好处：不用处理跨域，前端里直接写 fetch('/api/...') 就行，
    # 演示时也只需要起一个服务。
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
        logger.info("前端已挂载 | %s", WEB_DIR)
    else:
        logger.warning("没找到前端目录 %s，只有接口可用", WEB_DIR)

    logger.info("应用已组装 | 版本 %s | 接口文档 /docs", VERSION)
    return app


# uvicorn 需要的模块级实例
app = create_app()
