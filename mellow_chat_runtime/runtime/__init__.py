from mellow_chat_runtime.runtime.adapter import RuntimeAdapter
from mellow_chat_runtime.runtime.engine_backed_adapter import EngineBackedAdapter, _new_trace_id
from mellow_chat_runtime.runtime.llm_only import LLMOnlyAdapter
from mellow_chat_runtime.runtime.schemas import (
    ErrorBody,
    ErrorDetail,
    StatusResponse,
    TurnRequest,
    TurnResponse,
)


def get_runtime_adapter(impl: str = "engine-backed", orchestrator=None, llm_service=None):
    if impl == "llm-only":
        return LLMOnlyAdapter(llm_service=llm_service)
    return EngineBackedAdapter(orchestrator=orchestrator)


__all__ = [
    "RuntimeAdapter",
    "EngineBackedAdapter",
    "LLMOnlyAdapter",
    "TurnRequest",
    "TurnResponse",
    "StatusResponse",
    "ErrorBody",
    "ErrorDetail",
    "get_runtime_adapter",
    "_new_trace_id",
]
