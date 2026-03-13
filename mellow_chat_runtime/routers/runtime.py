from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from mellow_chat_runtime import app_state
from mellow_chat_runtime.runtime import ErrorBody, ErrorDetail, TurnRequest, TurnResponse, get_runtime_adapter
from mellow_chat_runtime.runtime.engine_backed_adapter import _new_trace_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runtime", tags=["Runtime"])


def _get_adapter():
    return get_runtime_adapter(impl="engine-backed", orchestrator=app_state.orchestrator, llm_service=app_state.llm_service)


@router.post("/turn", response_model=TurnResponse)
async def runtime_turn(req: TurnRequest):
    trace_id = _new_trace_id()
    adapter = _get_adapter()
    try:
        return await adapter.turn(req, trace_id=trace_id)
    except Exception as e:
        logger.exception("runtime turn failed: %s", e)
        return JSONResponse(
            status_code=503,
            content=ErrorBody(error=ErrorDetail(code="SERVICE_UNAVAILABLE", message=str(e), trace_id=trace_id)).model_dump(),
        )


@router.get("/status")
async def runtime_status():
    adapter = _get_adapter()
    return await adapter.status()
