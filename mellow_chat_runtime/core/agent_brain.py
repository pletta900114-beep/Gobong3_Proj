from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mellow_chat_runtime.core.dialogue_policy import apply_dialogue_weighting
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.prompt_builder import build_system_prompt, build_user_prompt
from mellow_chat_runtime.core.rp_parser import ParsedSceneEvent
from mellow_chat_runtime.core.text_sanitizer import (
    STOP_SEQUENCES,
    has_forbidden_output_markers,
    sanitize_assistant_text,
    sanitize_history_text,
    sanitize_memory_text,
)

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
    observation: str = ''


@dataclass
class AgentResult:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    total_turns: int = 0
    finish_reason: str = ''


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
        persona_id: str = 'default',
        user_profile_id: str = 'default',
        lore_topic: str = 'default',
        character_id: str = 'default',
        world_id: str = 'default',
        scene_id: str = 'default',
        mode: str = 'fast',
        selected_model: Optional[str] = None,
        scene_event: Optional[ParsedSceneEvent] = None,
        target_character_hint: Optional[str] = None,
    ) -> AgentResult:
        steps: List[AgentStep] = []
        history = self._trim_history(self._sanitize_history(context or []))

        persona = self._lookup.execute('lookup_persona', {'persona_id': persona_id}).payload
        user_profile = self._lookup.execute('lookup_user_profile', {'profile_id': user_profile_id}).payload
        lore = self._lookup.execute('lookup_lorebook', {'topic': lore_topic}).payload
        memories = self._lookup.execute('lookup_memories_possessions', {'character_id': character_id}).payload
        world_state = self._lookup.execute('lookup_world_state', {'world_id': world_id}).payload
        scene_state = self._lookup.execute('lookup_scene_state', {'scene_id': scene_id}).payload
        dialogue_priority = self._lookup.execute('lookup_dialogue_priority', {'scene_id': scene_id}).payload
        active_character = self._resolve_active_character(character_id)
        counterpart_ids = self._resolve_counterpart_ids(scene_state, character_id)
        relationships = self._lookup.execute(
            'lookup_relationships',
            {'character_id': character_id, 'counterpart_ids': counterpart_ids},
        ).payload

        steps.append(
            AgentStep(
                turn=1,
                thought='Collected required domain context via lookup dispatcher.',
                action=AgentAction(tool='domain_lookup_bundle', args={
                    'persona_id': persona_id,
                    'user_profile_id': user_profile_id,
                    'lore_topic': lore_topic,
                    'character_id': character_id,
                    'world_id': world_id,
                    'scene_id': scene_id,
                    'target_character_hint': target_character_hint,
                }),
                observation=self._cap_observation_size({
                    'persona': persona,
                    'user_profile': user_profile,
                    'lore': lore,
                    'memories': memories,
                    'world_state': world_state,
                    'scene_state': scene_state,
                    'dialogue_priority': dialogue_priority,
                    'active_character': active_character,
                    'relationships': relationships,
                    'scene_event': scene_event.__dict__ if scene_event else {},
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
            scene_event=scene_event,
            target_character_hint=target_character_hint,
        )

        resolved_model = selected_model or self._llm.get_model_for_mode(mode)
        answer = await self._generate_rp_answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=resolved_model,
            mode=mode,
            active_character=active_character,
        )

        answer = apply_dialogue_weighting(answer, dialogue_priority if isinstance(dialogue_priority, dict) else {})
        if not self._is_valid_rp_output(answer):
            answer = self._fallback_rp_output(active_character)

        return AgentResult(
            answer=answer,
            steps=steps,
            total_turns=1,
            finish_reason='completed',
        )

    async def _generate_rp_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        mode: str,
        active_character: Dict[str, Any],
    ) -> str:
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self._llm.chat(messages=messages, model=model, options={'stop': STOP_SEQUENCES})
        raw_answer = response.text or ''
        answer = sanitize_assistant_text(raw_answer)

        if self._is_empty_llm_response(answer):
            if not str(raw_answer).strip():
                fallback = await self._llm.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    mode=mode,
                    options={'stop': STOP_SEQUENCES},
                )
                answer = sanitize_assistant_text(fallback.content or '')
            else:
                answer = str(raw_answer).strip()

        if self._is_valid_rp_output(answer):
            return sanitize_assistant_text(answer)

        repair_messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': sanitize_assistant_text(answer) or str(answer).strip()},
            {
                'role': 'user',
                'content': 'Rewrite into final in-character RP output with one narration paragraph and one quoted spoken line only.',
            },
        ]
        repaired = await self._llm.chat(messages=repair_messages, model=model, options={'stop': STOP_SEQUENCES})
        repaired_answer = sanitize_assistant_text(repaired.text or '')
        if self._is_valid_rp_output(repaired_answer):
            return repaired_answer

        return self._fallback_rp_output(active_character)

    def _is_valid_rp_output(self, text: Any) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        cleaned = sanitize_assistant_text(text)
        if not cleaned:
            return False
        if has_forbidden_output_markers(cleaned):
            return False
        quote_match = re.search(r'["“].+?["”]', cleaned, re.DOTALL)
        if not quote_match:
            return False
        narration = re.sub(r'["“].+?["”]', '', cleaned, flags=re.DOTALL).strip()
        if len(narration) < 2:
            return False
        return True

    def _fallback_rp_output(self, active_character: Dict[str, Any]) -> str:
        name = str(active_character.get('name') or '그는').strip() or '그는'
        return f'{name}은 잠시 숨을 고르며 상대의 눈을 똑바로 바라본다.\n\n"지금은 서두르지 말자. 이 장면 안에서 차분히 이어가면 돼."'

    def _is_empty_llm_response(self, text: Any) -> bool:
        return not isinstance(text, str) or not text.strip()

    def _cap_observation_size(self, observation: Any, max_chars: int = 1200) -> str:
        try:
            obs = json.dumps(observation, ensure_ascii=False, default=str)
        except Exception:
            obs = str(observation)
        if len(obs) <= max_chars:
            return obs
        return obs[:max_chars] + f'\n[TRUNCATED_OBS original_len={len(obs)}]'

    def _trim_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(messages) <= self._context_window:
            return messages
        return messages[-self._context_window :]

    def _sanitize_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        sanitized: List[Dict[str, str]] = []
        for message in messages:
            role = str(message.get('role', 'user'))
            content = sanitize_history_text(role, str(message.get('content', '')))
            if not content:
                continue
            sanitized.append({'role': role, 'content': content})
        return sanitized

    def _resolve_active_character(self, character_id: str) -> Dict[str, Any]:
        cleaned_character_id = (character_id or '').strip()
        if not cleaned_character_id or cleaned_character_id == 'default':
            return {'name': 'Narrator', 'type': 'bot', 'role': 'narrator', 'speech_style': {'tone': 'neutral', 'forbidden': []}}

        bot_character = self._lookup.execute('lookup_bot_character', {'character_id': cleaned_character_id}).payload
        if bot_character:
            return bot_character

        user_character = self._lookup.execute('lookup_user_character', {'character_id': cleaned_character_id}).payload
        if user_character:
            return user_character

        return {
            'id': cleaned_character_id,
            'name': cleaned_character_id,
            'type': 'bot',
            'role': 'character',
            'speech_style': {'tone': 'neutral', 'forbidden': []},
        }

    def _resolve_counterpart_ids(self, scene_state: Dict[str, Any], character_id: str) -> List[str]:
        participants = scene_state.get('participants', []) if isinstance(scene_state, dict) else []
        if not isinstance(participants, list):
            return []
        return [str(item).strip() for item in participants if str(item).strip() and str(item).strip() != character_id]

    def _prioritize_memories(self, memories: Dict[str, Any]) -> Dict[str, Any]:
        items = memories.get('important_memories', []) if isinstance(memories, dict) else []
        if not isinstance(items, list):
            items = []
        cleaned = [sanitize_memory_text(str(item)) for item in items if str(item).strip()]
        cleaned = [item for item in cleaned if item]
        possessions = memories.get('possessions', []) if isinstance(memories.get('possessions', []), list) else []
        return {
            'character_id': memories.get('character_id', ''),
            'important_memories': cleaned[-5:],
            'possessions': [str(item).strip() for item in possessions if str(item).strip()][:5],
        }
