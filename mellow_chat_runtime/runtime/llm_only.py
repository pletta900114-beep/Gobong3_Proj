from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from mellow_chat_runtime.runtime.adapter import RuntimeAdapter
from mellow_chat_runtime.runtime.schemas import StatusHealth, StatusResponse, StatusRuntime, TurnMeta, TurnPayload, TurnRequest, TurnResponse, TurnState


class LLMOnlyAdapter(RuntimeAdapter):
    def __init__(self, llm_service=None):
        self._llm = llm_service
        self._started = datetime.utcnow()

    async def turn(self, req: TurnRequest, trace_id: Optional[str] = None) -> TurnResponse:
        trace_id = trace_id or f"trc_{uuid.uuid4().hex[:8]}"
        text = req.input.text
        if self._llm is not None:
            result = await self._llm.generate(prompt=text, mode="fast")
            text = result.content
        return TurnResponse(
            turn=TurnPayload(id=f"turn_{uuid.uuid4().hex[:8]}", speech=text, passage=None),
            state=TurnState(session_id=req.session_id, system_state="IDLE", model_tier_effective="free"),
            meta=TurnMeta(trace_id=trace_id, runtime_impl="llm-only", latency_ms=0.0),
        )

    async def status(self) -> StatusResponse:
        uptime = (datetime.utcnow() - self._started).total_seconds()
        return StatusResponse(runtime=StatusRuntime(impl="llm-only", version="0.1", uptime_sec=uptime), health=StatusHealth(system_state="IDLE"))
