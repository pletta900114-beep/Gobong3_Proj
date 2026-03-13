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
        active_character = self._resolve_active_character(character_id)
        counterpart_ids = self._resolve_counterpart_ids(scene_state, character_id)
        relationships = self._lookup.execute(
            "lookup_relationships",
            {"character_id": character_id, "counterpart_ids": counterpart_ids},
        ).payload

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
                    "active_character": active_character,
                    "relationships": relationships,
                }),
            )
        )

        system_prompt = build_system_prompt(
            persona=persona,
            dialogue_priority=dialogue_priority,
            active_character=active_character,
            relationships=relationships if isinstance(relationships, list) else [],
        )
        user_prompt = build_user_prompt(
            user_text=user_input,
            user_profile=user_profile if isinstance(user_profile, dict) else {},
            lore=lore if isinstance(lore, dict) else {},
            memories=self._prioritize_memories(memories if isinstance(memories, dict) else {}),
            world_state=world_state if isinstance(world_state, dict) else {},
            scene_state=scene_state if isinstance(scene_state, dict) else {},
            relationships=relationships if isinstance(relationships, list) else [],
            history=history,
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        resolved_model = selected_model or self._llm.get_model_for_mode(mode)
        response = await self._llm.chat(messages=messages, model=resolved_model)
        answer = response.text.strip() if response.text else ""

        if self._is_empty_llm_response(answer):
            fallback = await self._llm.generate(prompt=user_prompt, system_prompt=system_prompt, mode=mode)
            answer = (fallback.content or "").strip()

        answer = apply_dialogue_weighting(answer, dialogue_priority if isinstance(dialogue_priority, dict) else {})

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

    def _resolve_active_character(self, character_id: str) -> Dict[str, Any]:
        cleaned_character_id = (character_id or "").strip()
        if not cleaned_character_id or cleaned_character_id == "default":
            return {"name": "Narrator", "type": "bot", "role": "narrator", "speech_style": {"tone": "neutral", "forbidden": []}}

        bot_character = self._lookup.execute("lookup_bot_character", {"character_id": cleaned_character_id}).payload
        if bot_character:
            return bot_character

        user_character = self._lookup.execute("lookup_user_character", {"character_id": cleaned_character_id}).payload
        if user_character:
            return user_character

        return {
            "id": cleaned_character_id,
            "name": cleaned_character_id,
            "type": "bot",
            "role": "character",
            "speech_style": {"tone": "neutral", "forbidden": []},
        }

    def _resolve_counterpart_ids(self, scene_state: Dict[str, Any], character_id: str) -> List[str]:
        participants = scene_state.get("participants", []) if isinstance(scene_state, dict) else []
        if not isinstance(participants, list):
            return []
        return [str(item).strip() for item in participants if str(item).strip() and str(item).strip() != character_id]

    def _prioritize_memories(self, memories: Dict[str, Any]) -> Dict[str, Any]:
        items = memories.get("important_memories", []) if isinstance(memories, dict) else []
        if not isinstance(items, list):
            items = []
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        possessions = memories.get("possessions", []) if isinstance(memories.get("possessions", []), list) else []
        return {
            "character_id": memories.get("character_id", ""),
            "important_memories": cleaned[-5:],
            "possessions": [str(item).strip() for item in possessions if str(item).strip()][:5],
        }
