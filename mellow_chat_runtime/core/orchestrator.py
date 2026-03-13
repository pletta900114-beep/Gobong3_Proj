from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from mellow_chat_runtime.core.agent_brain import AgentBrain, AgentResult
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.states import SystemState, TransitionResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """Text-only orchestrator for minimal chatbot runtime."""

    def __init__(self, lookup_dispatcher: DomainLookupDispatcher) -> None:
        self.current_state: SystemState = SystemState.IDLE
        self._services: Dict[str, Any] = {}
        self._gpu_lock = asyncio.Lock()
        self._lookup_dispatcher = lookup_dispatcher
        self.agent: Optional[AgentBrain] = None
        self.llm_service: Optional[Any] = None

    def register_service(self, name: str, service: Any) -> None:
        self._services[name] = service
        if name == "llm":
            self.llm_service = service
            self._init_agent_if_possible()

    def get_service(self, name: str) -> Optional[Any]:
        return self._services.get(name)

    def _init_agent_if_possible(self) -> None:
        if self.agent is not None:
            return
        llm = self.llm_service or self._services.get("llm")
        if llm is None:
            return
        self.agent = AgentBrain(llm_service=llm, lookup_dispatcher=self._lookup_dispatcher)

    async def initialize(self) -> None:
        llm = self._services.get("llm")
        if llm and hasattr(llm, "connect"):
            await llm.connect()
        self._init_agent_if_possible()

    async def shutdown(self) -> None:
        llm = self._services.get("llm")
        if llm and hasattr(llm, "disconnect"):
            await llm.disconnect()

    def get_state(self) -> SystemState:
        return self.current_state

    async def request_state_change(self, target_state: SystemState, reason: str = "") -> TransitionResult:
        if self.current_state == target_state:
            return TransitionResult.SUCCESS
        valid = {
            SystemState.IDLE: {SystemState.TEXT},
            SystemState.TEXT: {SystemState.IDLE},
        }
        if target_state not in valid.get(self.current_state, set()):
            return TransitionResult.INVALID_TRANSITION
        self.current_state = target_state
        logger.debug("State changed to %s reason=%s", target_state, reason)
        return TransitionResult.SUCCESS

    async def run_agent(
        self,
        user_input: str,
        history: Optional[list] = None,
        mode: str = "fast",
        selected_model: Optional[str] = None,
        persona_id: str = "default",
        user_profile_id: str = "default",
        lore_topic: str = "default",
        character_id: str = "default",
        world_id: str = "default",
        scene_id: str = "default",
    ) -> AgentResult:
        self._init_agent_if_possible()
        if self.agent is None:
            raise RuntimeError("AgentBrain is not initialized")

        async with self._gpu_lock:
            return await self.agent.run(
                user_input=user_input,
                context=history or [],
                persona_id=persona_id,
                user_profile_id=user_profile_id,
                lore_topic=lore_topic,
                character_id=character_id,
                world_id=world_id,
                scene_id=scene_id,
                mode=mode,
                selected_model=selected_model,
            )

    async def health_check(self) -> Dict[str, Any]:
        llm_ok = False
        if self.llm_service and hasattr(self.llm_service, "health_check"):
            llm_ok = await self.llm_service.health_check()
        return {
            "state": self.current_state.value,
            "llm_available": llm_ok,
            "agent_initialized": self.agent is not None,
        }
