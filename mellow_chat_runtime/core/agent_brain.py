from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mellow_chat_runtime.core.dialogue_policy import apply_dialogue_weighting
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.prompt_builder import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    tool: str
    args: Dict[str, Any]


@dataclass
class AgentStep:
    turn: int
    thought: str
    action: Optional[AgentAction] = None
    observation: str = ""


@dataclass
class AgentResult:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    total_turns: int = 0
    finish_reason: str = ""


class AgentBrain:
    """Minimal loop for text chatbot with lookup-only context gathering."""

    def __init__(self, llm_service: Any, lookup_dispatcher: DomainLookupDispatcher, max_turns: int = 2, context_window: int = 8) -> None:
        self._llm = llm_service
        self._lookup = lookup_dispatcher
        self._max_turns = max_turns
        self._context_window = context_window

    async def run(
        self,
        user_input: str,
        context: Optional[List[Dict[str, str]]] = None,
        persona_id: str = "default",
        user_profile_id: str = "default",
        lore_topic: str = "default",
        character_id: str = "default",
        world_id: str = "default",
        scene_id: str = "default",
        mode: str = "fast",
        selected_model: Optional[str] = None,
    ) -> AgentResult:
        steps: List[AgentStep] = []
        history = self._trim_history(context or [])

        # Turn 1: Collect domain lookups deterministically.
        persona = self._lookup.execute("lookup_persona", {"persona_id": persona_id}).payload
        user_profile = self._lookup.execute("lookup_user_profile", {"profile_id": user_profile_id}).payload
        lore = self._lookup.execute("lookup_lorebook", {"topic": lore_topic}).payload
        memories = self._lookup.execute("lookup_memories_possessions", {"character_id": character_id}).payload
        world_state = self._lookup.execute("lookup_world_state", {"world_id": world_id}).payload
        scene_state = self._lookup.execute("lookup_scene_state", {"scene_id": scene_id}).payload
        dialogue_priority = self._lookup.execute("lookup_dialogue_priority", {"scene_id": scene_id}).payload

        steps.append(
            AgentStep(
                turn=1,
                thought="Collected required domain context via lookup dispatcher.",
                action=AgentAction(tool="domain_lookup_bundle", args={
                    "persona_id": persona_id,
                    "user_profile_id": user_profile_id,
                    "lore_topic": lore_topic,
                    "character_id": character_id,
                    "world_id": world_id,
                    "scene_id": scene_id,
                }),
                observation=self._cap_observation_size({
                    "persona": persona,
                    "user_profile": user_profile,
                    "lore": lore,
                    "memories": memories,
                    "world_state": world_state,
                    "scene_state": scene_state,
                    "dialogue_priority": dialogue_priority,
                }),
            )
        )

        system_prompt = build_system_prompt(persona=persona, dialogue_priority=dialogue_priority)
        user_prompt = build_user_prompt(
            user_text=user_input,
            user_profile=user_profile,
            lore=lore,
            memories=memories,
            world_state=world_state,
            scene_state=scene_state,
            history=history,
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        resolved_model = selected_model or self._llm.get_model_for_mode(mode)
        response = await self._llm.chat(messages=messages, model=resolved_model)
        answer = response.text.strip() if response.text else ""

        if self._is_empty_llm_response(answer):
            fallback = await self._llm.generate(prompt=user_prompt, system_prompt=system_prompt, mode=mode)
            answer = (fallback.content or "").strip()

        answer = apply_dialogue_weighting(answer, dialogue_priority)

        if self._is_empty_llm_response(answer):
            answer = "I need a bit more context to respond consistently with the current world state."

        return AgentResult(
            answer=answer,
            steps=steps,
            total_turns=1,
            finish_reason="completed",
        )

    def _is_empty_llm_response(self, text: Any) -> bool:
        return not isinstance(text, str) or not text.strip()

    def _cap_observation_size(self, observation: Any, max_chars: int = 1200) -> str:
        try:
            obs = json.dumps(observation, ensure_ascii=False, default=str)
        except Exception:
            obs = str(observation)
        if len(obs) <= max_chars:
            return obs
        return obs[:max_chars] + f"\n[TRUNCATED_OBS original_len={len(obs)}]"

    def _trim_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= self._context_window:
            return messages
        return messages[-self._context_window :]
