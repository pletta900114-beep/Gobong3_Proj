from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TurnRequestUser(BaseModel):
    id: str


class TurnRequestInput(BaseModel):
    text: str
    locale: Optional[str] = None


class TurnRequestContext(BaseModel):
    character_id: Optional[str] = "default"
    model_tier_requested: Optional[str] = "free"
    metadata: Optional[Dict[str, Any]] = None


class TurnRequest(BaseModel):
    session_id: str
    user: TurnRequestUser
    input: TurnRequestInput
    context: Optional[TurnRequestContext] = None


class TurnPayload(BaseModel):
    id: str
    speech: str = ""
    passage: Optional[str] = None


class TurnState(BaseModel):
    session_id: str
    state_version: int = 1
    system_state: str = "IDLE"
    model_tier_effective: str = "free"


class TurnMeta(BaseModel):
    trace_id: str
    runtime_impl: str
    latency_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TurnResponse(BaseModel):
    turn: TurnPayload
    state: TurnState
    meta: TurnMeta


class StatusRuntime(BaseModel):
    impl: str
    version: str
    uptime_sec: float


class StatusHealth(BaseModel):
    system_state: str
    last_error: Optional[str] = None


class StatusResponse(BaseModel):
    runtime: StatusRuntime
    health: StatusHealth
    time: datetime = Field(default_factory=datetime.utcnow)


class ErrorDetail(BaseModel):
    code: str
    message: str
    trace_id: Optional[str] = None


class ErrorBody(BaseModel):
    error: ErrorDetail


class RetrievalDebug(BaseModel):
    query: Optional[str] = None
    lore_source: Optional[Literal["vector", "fallback", "none", "canonical"]] = None
    memory_source: Optional[Literal["vector", "fallback", "none", "canonical"]] = None
    relationship_source: Optional[Literal["vector", "fallback", "none", "canonical"]] = None
    lore_hit_ids: List[str] = Field(default_factory=list)
    memory_hit_ids: List[str] = Field(default_factory=list)
    relationship_hit_ids: List[str] = Field(default_factory=list)
    lore_scores: Dict[str, float] = Field(default_factory=dict)
    memory_scores: Dict[str, float] = Field(default_factory=dict)
    relationship_scores: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    fallback_used: Optional[bool] = None


class RPDebug(BaseModel):
    validator_passed: Optional[bool] = None
    fallback_used: Optional[bool] = None
    retry_count: Optional[int] = None
    final_verdict: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_reasons: List[str] = Field(default_factory=list)


class ChatAskResponseModel(BaseModel):
    response: str
    session_id: int
    message_id: int
    speaker_id: Optional[str] = None
    speaker_type: Optional[str] = None
    model_provider: str
    model_name: str
    selected_mode: str
    processing_time_ms: int
    used_context: Dict[str, Any] = Field(default_factory=dict)
    model: Dict[str, Any] = Field(default_factory=dict)
    request_id: str


class ChatAskAdminResponseModel(ChatAskResponseModel):
    retrieval_debug: Optional[RetrievalDebug] = None
    rp_debug: Optional[RPDebug] = None


class VectorReindexResponse(BaseModel):
    entity_type: Literal["lore", "memory", "relationship"]
    entity_id: str
    status: Literal["queued", "reindexed"]
