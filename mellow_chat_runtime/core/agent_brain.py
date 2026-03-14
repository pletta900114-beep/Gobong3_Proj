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

_DIALOGUE_RE = re.compile(r'["“](.+?)["”]', re.DOTALL)
_NARRATION_FIRST_PERSON_TOKENS = (
    '내 시선',
    '내 눈빛',
    '내게',
    '나를',
    '우리의',
    '우리',
)


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
    success: bool = True
    error_code: str = ''
    message: str = ''
    validator_passed: bool = False
    fallback_used: bool = False
    retry_count: int = 0
    final_verdict: str = 'PASS'
    failure_reason: str = ''
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class RPGenerationOutcome:
    answer: str = ''
    validator_passed: bool = False
    fallback_used: bool = False
    retry_count: int = 0
    failure_reasons: List[str] = field(default_factory=list)


class AgentBrain:
    """Minimal loop for text chatbot with lookup-only context gathering."""

    def __init__(self, llm_service: Any, lookup_dispatcher: DomainLookupDispatcher, max_turns: int = 2, context_window: int = 8) -> None:
        self._llm = llm_service
        self._lookup = lookup_dispatcher
        self._max_turns = max_turns
        self._context_window = context_window

    def _rp_chat_kwargs(self, *, for_repair: bool = False) -> Dict[str, Any]:
        options: Dict[str, Any] = {'stop': STOP_SEQUENCES}
        if for_repair:
            options['num_predict'] = 180
        return {'options': options, 'think': False}

    async def run(
        self,
        user_input: str,
        context: Optional[List[Dict[str, str]]] = None,
        retrieval_context: Optional[Dict[str, Any]] = None,
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
        request_id: Optional[str] = None,
        audience: str = 'user',
    ) -> AgentResult:
        steps: List[AgentStep] = []
        history = self._trim_history(self._sanitize_history(context or []))
        retrieval_context = retrieval_context or {}

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
                    'retrieval_context': retrieval_context,
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
            retrieval_context=retrieval_context,
        )

        resolved_model = selected_model or self._llm.get_model_for_mode(mode)
        expected_language = self._detect_primary_language(scene_event.raw_text if scene_event is not None else user_input)
        outcome = await self._generate_rp_answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=resolved_model,
            mode=mode,
            active_character=active_character,
            request_id=request_id,
            scene_event=scene_event,
            expected_language=expected_language,
            audience=audience,
        )
        answer = apply_dialogue_weighting(outcome.answer, dialogue_priority if isinstance(dialogue_priority, dict) else {}) if outcome.answer else ''
        post_reasons: List[str] = list(outcome.failure_reasons)
        if answer:
            post_ok, post_reasons = self._validate_rp_output(
                answer,
                active_character=active_character,
                expected_language=expected_language,
            )
            outcome.validator_passed = post_ok and not outcome.fallback_used
            if not post_ok:
                outcome.failure_reasons = post_reasons
                if audience == 'admin' and not outcome.fallback_used:
                    answer = self._fallback_rp_output(active_character)
                    outcome.fallback_used = True
                elif audience != 'admin':
                    answer = ''
        final_verdict = 'PASS' if outcome.validator_passed and not outcome.fallback_used else 'FAIL'
        success = bool(answer) if audience == 'admin' else bool(answer) and outcome.validator_passed and not outcome.fallback_used
        failure_reason = ','.join(outcome.failure_reasons)
        error_code = ''
        message = ''
        if not success:
            error_code = self._derive_failure_code(outcome.failure_reasons)
            message = 'RP 응답 품질 검증에 실패했습니다. 잠시 후 다시 시도해 주세요.'

        return AgentResult(
            answer=answer,
            steps=steps,
            total_turns=1,
            finish_reason='completed' if success else 'validation_failed',
            success=success,
            error_code=error_code,
            message=message,
            validator_passed=outcome.validator_passed,
            fallback_used=outcome.fallback_used,
            retry_count=outcome.retry_count,
            final_verdict=final_verdict,
            failure_reason=failure_reason,
            failure_reasons=list(outcome.failure_reasons),
        )

    async def _generate_rp_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        mode: str,
        active_character: Dict[str, Any],
        request_id: Optional[str] = None,
        scene_event: Optional[ParsedSceneEvent] = None,
        expected_language: Optional[str] = None,
        audience: str = 'user',
    ) -> RPGenerationOutcome:
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        response = await self._llm.chat(messages=messages, model=model, **self._rp_chat_kwargs())
        raw_answer = response.text or ''
        raw_thinking = getattr(response, 'thinking', '') or ''
        answer = sanitize_assistant_text(raw_answer)
        outcome = RPGenerationOutcome(answer=answer)

        if self._is_empty_llm_response(answer) and str(raw_thinking).strip():
            logger.warning(
                'rp.output.thinking_only_detected request_id=%s character=%s model=%s input_mode=%s thinking_preview=%s',
                request_id,
                active_character.get('id') or active_character.get('name'),
                model,
                getattr(scene_event, 'input_mode', None),
                self._preview(raw_thinking),
            )

        first_ok, first_reasons = self._validate_rp_output(
            answer,
            active_character=active_character,
            expected_language=expected_language,
        )
        if first_ok:
            logger.info(
                'rp.output.first_pass_valid request_id=%s character=%s model=%s input_mode=%s preview=%s',
                request_id,
                active_character.get('id') or active_character.get('name'),
                model,
                getattr(scene_event, 'input_mode', None),
                self._preview(answer),
            )
            outcome.answer = sanitize_assistant_text(answer)
            outcome.validator_passed = True
            return outcome

        logger.warning(
            'rp.output.first_pass_invalid request_id=%s character=%s model=%s input_mode=%s reasons=%s raw_preview=%s sanitized_preview=%s',
            request_id,
            active_character.get('id') or active_character.get('name'),
            model,
            getattr(scene_event, 'input_mode', None),
            ','.join(first_reasons),
            self._preview(raw_answer),
            self._preview(answer),
        )

        repaired_outcome = await self._repair_rp_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            active_character=active_character,
            request_id=request_id,
            scene_event=scene_event,
            expected_language=expected_language,
            prior_answer=answer,
            failure_reasons=first_reasons,
        )
        repaired_outcome.retry_count = 1
        if repaired_outcome.answer:
            return repaired_outcome

        if audience != 'admin':
            outcome.answer = ''
            outcome.failure_reasons = repaired_outcome.failure_reasons or first_reasons
            return outcome

        fallback_answer = self._fallback_rp_output(active_character)
        logger.warning(
            'rp.output.fallback_used request_id=%s character=%s model=%s first_pass_reasons=%s repair_reasons=%s fallback_preview=%s',
            request_id,
            active_character.get('id') or active_character.get('name'),
            model,
            ','.join(first_reasons),
            ','.join(repaired_outcome.failure_reasons),
            self._preview(fallback_answer),
        )
        return RPGenerationOutcome(
            answer=fallback_answer,
            validator_passed=False,
            fallback_used=True,
            retry_count=1,
            failure_reasons=repaired_outcome.failure_reasons or first_reasons,
        )

    async def _repair_rp_output(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        active_character: Dict[str, Any],
        request_id: Optional[str],
        scene_event: Optional[ParsedSceneEvent],
        expected_language: Optional[str],
        prior_answer: str,
        failure_reasons: List[str],
    ) -> RPGenerationOutcome:
        scene_text = ''
        if scene_event is not None:
            scene_text = scene_event.raw_text or scene_event.user_dialogue or scene_event.user_narration
        repair_instruction = self._build_repair_instruction(failure_reasons)
        repair_messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
        if prior_answer.strip():
            repair_messages.append({'role': 'assistant', 'content': prior_answer})
        repair_messages.append({'role': 'user', 'content': f'{repair_instruction}\n장면:\n{scene_text}'.strip()})
        repaired = await self._llm.chat(messages=repair_messages, model=model, **self._rp_chat_kwargs(for_repair=True))
        repaired_raw = repaired.text or ''
        repaired_answer = sanitize_assistant_text(repaired.text or '')
        repaired_ok, repaired_reasons = self._validate_rp_output(
            repaired_answer,
            active_character=active_character,
            expected_language=expected_language,
        )
        if repaired_ok:
            logger.info(
                'rp.output.repair_pass_valid request_id=%s character=%s model=%s reasons_cleared=%s preview=%s',
                request_id,
                active_character.get('id') or active_character.get('name'),
                model,
                ','.join(failure_reasons),
                self._preview(repaired_answer),
            )
            return RPGenerationOutcome(answer=repaired_answer, validator_passed=True, failure_reasons=[])
        logger.warning(
            'rp.output.repair_pass_invalid request_id=%s character=%s model=%s reasons=%s raw_preview=%s sanitized_preview=%s',
            request_id,
            active_character.get('id') or active_character.get('name'),
            model,
            ','.join(repaired_reasons),
            self._preview(repaired_raw),
            self._preview(repaired_answer),
        )
        return RPGenerationOutcome(answer='', validator_passed=False, failure_reasons=repaired_reasons or failure_reasons)

    def _build_repair_instruction(self, failure_reasons: List[str]) -> str:
        if 'narration_not_third_person' in failure_reasons:
            return (
                '최종 RP 답변만 다시 써라.\n'
                '서술은 엄격한 3인칭이어야 한다.\n'
                '서술에 내 / 나 / 내게 / 우리 / 나를 / 내 시선 / 내 눈빛을 쓰지 마라.\n'
                '사용자는 상대 / 사용자 / 멜로우로만 가리켜라.\n'
                '짧은 서술 한 문단과 따옴표 대사 한 줄만 출력하라.\n'
                '코드블록, JSON, 메타 텍스트, 분석을 쓰지 마라.'
            )
        return (
            '최종 RP 답변만 출력하라.\n'
            '짧은 서술 한 문단과 따옴표 대사 한 줄만 출력하라.\n'
            '코드블록, JSON, 메타 텍스트, 분석, 생각을 쓰지 마라.\n'
            '빈 응답을 쓰지 마라.\n'
            '사용자 입력의 주 언어를 유지하라.\n'
            '서술은 선택된 캐릭터의 3인칭으로만 쓴다.'
        )

    def _validate_rp_output(
        self,
        text: Any,
        active_character: Optional[Dict[str, Any]] = None,
        expected_language: Optional[str] = None,
    ) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        if not isinstance(text, str) or not text.strip():
            return False, ['empty']
        cleaned = sanitize_assistant_text(text)
        if not cleaned:
            return False, ['sanitized_empty']
        if has_forbidden_output_markers(cleaned):
            reasons.append('meta_or_token_leak')
        narration, dialogue = self._split_rp_sections(cleaned)
        if not dialogue:
            reasons.append('missing_quoted_dialogue')
        if not narration:
            reasons.append('missing_narration')
        if expected_language == 'ko' and self._detect_primary_language(cleaned) == 'en':
            reasons.append('language_drift')
        if narration and not self._is_third_person_narration(narration, active_character):
            reasons.append('narration_not_third_person')
        return not reasons, reasons

    def _is_valid_rp_output(
        self,
        text: Any,
        active_character: Optional[Dict[str, Any]] = None,
        expected_language: Optional[str] = None,
    ) -> bool:
        ok, _ = self._validate_rp_output(text, active_character=active_character, expected_language=expected_language)
        return ok

    def _fallback_rp_output(self, active_character: Dict[str, Any]) -> str:
        character_name = str(active_character.get('name') or '').strip()
        if character_name == 'Sunday':
            return '선데이는 숨을 고른 뒤 흔들림 없는 시선으로 상대를 바라본다.\n\n"지금 드린 말은 가볍게 꺼낸 것이 아닙니다. 이 장면에서는 차분히, 끝까지 책임 있게 답하겠습니다."'
        if character_name == 'Aventurine':
            return '어벤츄린은 한쪽 입꼬리를 옅게 올린 채 판을 다시 가늠하듯 상대를 바라본다.\n\n"성급하게 판을 접을 생각은 없어. 네 판단도 포함해서, 이번 수는 끝까지 계산해 보자."'
        subject = self._fallback_subject(active_character)
        return f'{subject} 잠시 숨을 고르며 상대의 눈을 똑바로 바라본다.\n\n"지금은 서두르지 말자. 이 장면 안에서 차분히 이어가면 돼."'

    def _derive_failure_code(self, reasons: List[str]) -> str:
        if 'narration_not_third_person' in reasons:
            return 'POV_RULE_FAILED'
        if 'missing_narration' in reasons or 'missing_quoted_dialogue' in reasons:
            return 'NARRATION_RULE_FAILED'
        if 'language_drift' in reasons or 'meta_or_token_leak' in reasons:
            return 'CHARACTER_STYLE_FAILED'
        return 'RP_VALIDATION_FAILED'

    def _is_empty_llm_response(self, text: Any) -> bool:
        return not isinstance(text, str) or not text.strip()

    def _detect_primary_language(self, text: str) -> str:
        hangul_count = len(re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]', text or ''))
        latin_count = len(re.findall(r'[A-Za-z]', text or ''))
        if hangul_count >= max(8, latin_count * 2):
            return 'ko'
        if latin_count >= max(12, hangul_count * 2):
            return 'en'
        return 'same-as-input'

    def _is_third_person_narration(self, narration: str, active_character: Optional[Dict[str, Any]]) -> bool:
        text = (narration or '').strip()
        if not text:
            return False
        if re.search(r'\b(I|my|me|mine|we|our|us)\b', text, re.IGNORECASE):
            return False
        if any(token in text for token in _NARRATION_FIRST_PERSON_TOKENS):
            return False
        if re.search(r'(^|[\s"“”(])(?:내|나|우리)(?=$|[\s"“”.,!?)]|\n)', text):
            return False
        if re.search(r'(^|[\s(])(?:나는|난|내가|나에게|저는|제가|저를|제게|우리는|우린|우리가|우리를|우리에게)\b', text):
            return False
        character_name = str((active_character or {}).get('name') or '').strip()
        third_person_markers = ['그는', '그녀는', '그가', '그녀가']
        if character_name:
            third_person_markers.extend([
                character_name,
                f'{character_name}은',
                f'{character_name}는',
                f'{character_name}이',
                f'{character_name}가',
            ])
        if any(marker in text for marker in third_person_markers):
            return True
        if re.search(r'(^|[\n\s])([^\s"“”]{2,20})(은|는|이|가)\s*', text):
            return True
        return False

    def _split_rp_sections(self, text: str) -> tuple[str, str]:
        cleaned = sanitize_assistant_text(text)
        dialogue_chunks = [chunk.strip() for chunk in _DIALOGUE_RE.findall(cleaned) if chunk.strip()]
        narration_text = _DIALOGUE_RE.sub('', cleaned)
        narration_text = re.sub(r'\n{3,}', '\n\n', narration_text).strip()
        dialogue_text = '\n'.join(dialogue_chunks).strip()
        return narration_text, dialogue_text

    def _fallback_subject(self, active_character: Optional[Dict[str, Any]]) -> str:
        character_name = str((active_character or {}).get('name') or '').strip()
        if character_name and re.search(r'[\u3131-\u318E\uAC00-\uD7A3]', character_name):
            if character_name.endswith(('은', '는', '이', '가')):
                return character_name
            return f'{character_name}는'
        return '그는'


    def _preview(self, text: Any, max_chars: int = 160) -> str:
        if not isinstance(text, str):
            return ''
        cleaned = text.replace('\n', '\\n').strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars] + '...'

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
