from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Optional

from mellow_chat_runtime.runtime.adapter import RuntimeAdapter
from mellow_chat_runtime.runtime.schemas import (
    StatusHealth,
    StatusResponse,
    StatusRuntime,
    TurnMeta,
    TurnPayload,
    TurnRequest,
    TurnResponse,
    TurnState,
)


def _new_trace_id() -> str:
    return f"trc_{datetime.utcnow().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


class EngineBackedAdapter(RuntimeAdapter):
    def __init__(self, orchestrator=None):
        self._orchestrator = orchestrator
        self._started = datetime.utcnow()

    async def turn(self, req: TurnRequest, trace_id: Optional[str] = None) -> TurnResponse:
        if self._orchestrator is None:
            raise RuntimeError("orchestrator unavailable")

        t0 = time.perf_counter()
        trace_id = trace_id or _new_trace_id()

        ctx = req.context or None
        character_id = (ctx.character_id if ctx else "default") or "default"
        metadata = (ctx.metadata if ctx else {}) or {}

        result = await self._orchestrator.run_agent(
            user_input=req.input.text,
            history=[],
            mode="fast",
            persona_id=str(metadata.get("persona_id", "default")),
            user_profile_id=str(req.user.id or "default"),
            lore_topic=str(metadata.get("lore_topic", "default")),
            character_id=character_id,
            world_id=str(metadata.get("world_id", "default")),
            scene_id=str(metadata.get("scene_id", "default")),
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        return TurnResponse(
            turn=TurnPayload(id=f"turn_{uuid.uuid4().hex[:8]}", speech=result.answer, passage=None),
            state=TurnState(session_id=req.session_id, system_state=self._orchestrator.get_state().value, model_tier_effective="free"),
            meta=TurnMeta(trace_id=trace_id, runtime_impl="engine-backed", latency_ms=round(latency_ms, 2)),
        )

    async def status(self) -> StatusResponse:
        state = "IDLE"
        if self._orchestrator is not None:
            state = self._orchestrator.get_state().value
        uptime = (datetime.utcnow() - self._started).total_seconds()
        return StatusResponse(runtime=StatusRuntime(impl="engine-backed", version="0.1", uptime_sec=uptime), health=StatusHealth(system_state=state))
