from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mellow_chat_runtime import app_state
from mellow_chat_runtime.config.settings import get_settings
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.domain_lookup_store import get_domain_store
from mellow_chat_runtime.core.orchestrator import Orchestrator
from mellow_chat_runtime.infra.database import init_db
from mellow_chat_runtime.routers.admin import router as admin_router
from mellow_chat_runtime.routers.chat import router as chat_router
from mellow_chat_runtime.routers.models import router as model_router
from mellow_chat_runtime.routers.runtime import router as runtime_router
from mellow_chat_runtime.services.llm_service import create_llm_service
from mellow_chat_runtime.services.vector_retrieval_service import VectorRetrievalService

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
        force=True,
    )


async def startup() -> None:
    settings = get_settings()
    app_state.settings = settings

    init_db()

    domain_store = get_domain_store(
        data_path=settings.domain_data_file,
        backend="json",
        vectordb_lore_search_url=settings.vectordb_lore_search_url,
        vectordb_timeout_sec=settings.vectordb_timeout_sec,
    )
    vector_retrieval_service = VectorRetrievalService(
        domain_store=domain_store,
        index_path=settings.vector_index_file,
    )
    dispatcher = DomainLookupDispatcher(domain_store)

    llm = create_llm_service(
        host=settings.ollama_host,
        port=settings.ollama_port,
        timeout=settings.ollama_timeout,
        models={
            "fast": settings.fast_model,
            "thinking": settings.thinking_model,
            "research": settings.research_model,
        },
    )

    orchestrator = Orchestrator(lookup_dispatcher=dispatcher)
    orchestrator.register_service("llm", llm)
    await orchestrator.initialize()

    app_state.llm_service = llm
    app_state.orchestrator = orchestrator
    app_state.vector_retrieval_service = vector_retrieval_service

    logger.info("mellow_chat_runtime startup complete")


async def shutdown() -> None:
    if app_state.orchestrator:
        await app_state.orchestrator.shutdown()
    app_state.orchestrator = None
    app_state.llm_service = None
    app_state.vector_retrieval_service = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(
    title="mellow_chat_runtime",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(model_router)
app.include_router(runtime_router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict:
    orch = app_state.orchestrator
    if orch is None:
        return {"ok": False, "detail": "orchestrator unavailable"}
    return await orch.health_check()


def main() -> None:
    import uvicorn

    settings = get_settings()
    configure_logging()
    uvicorn.run(
        "mellow_chat_runtime.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )


if __name__ == "__main__":
    main()
