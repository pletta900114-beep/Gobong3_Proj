from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

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
