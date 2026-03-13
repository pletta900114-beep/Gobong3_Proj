from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from mellow_chat_runtime.core.rp_parser import ParsedSceneEvent


def build_system_prompt(
    persona: Dict[str, Any],
    dialogue_priority: Dict[str, Any],
    active_character: Optional[Dict[str, Any]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
) -> str:
    persona_desc = persona.get('description', '')
    priority_rules = dialogue_priority.get('rules', '')
    active_character = active_character or {}
    relationships = relationships or []
    character_name = active_character.get('name') or 'Unknown Character'
    speech_style = active_character.get('speech_style', {}) if isinstance(active_character.get('speech_style'), dict) else {}
    tone = speech_style.get('tone') or active_character.get('tone') or 'neutral'
    forbidden = speech_style.get('forbidden', [])
    if not isinstance(forbidden, list):
        forbidden = []
    forbidden_text = ', '.join(str(item).strip() for item in forbidden if str(item).strip()) or 'none'
    relationship_keys = active_character.get('relationship_keys', [])
    if not isinstance(relationship_keys, list):
        relationship_keys = []
    relationship_text = ', '.join(str(item).strip() for item in relationship_keys if str(item).strip()) or 'none'
    role_context = active_character.get('profile') or active_character.get('role') or active_character.get('type') or 'character'
    relationship_lines: List[str] = []
    for item in relationships[:4]:
        target_id = item.get('target_id', 'unknown')
        summary = item.get('summary', '')
        rel_tone = item.get('tone', 'neutral')
        boundaries = item.get('boundaries', [])
        if not isinstance(boundaries, list):
            boundaries = []
        boundary_text = ', '.join(str(boundary).strip() for boundary in boundaries if str(boundary).strip()) or 'none'
        relationship_lines.append(f'- {target_id}: {summary} | tone={rel_tone} | boundaries={boundary_text}')
    relationship_block = '\n'.join(relationship_lines) if relationship_lines else '- none'
    return (
        f'당신은 {character_name}이다.\n\n'
        '역할:\n'
        f'{role_context}\n\n'
        '말투:\n'
        f'{tone}\n\n'
        '입력 해석 규칙:\n'
        '- 사용자 입력에는 서술과 대사가 함께 들어올 수 있다\n'
        '- 서술은 장면과 행동 맥락으로 해석한다\n'
        '- 따옴표 안의 문장은 사용자의 실제 발화로 해석한다\n'
        '- 설명이 아니라 장면 안의 즉각적인 반응으로 답한다\n\n'
        '출력 형식:\n'
        '1. 짧은 서술 또는 행동 문단 하나\n'
        '2. 따옴표로 감싼 대사 한 줄\n\n'
        '핵심 규칙:\n'
        '- 항상 캐릭터를 유지한다\n'
        '- 제공된 세계관 정보와 모순되지 않게 답한다\n'
        '- 설명보다 행동과 대사를 우선한다\n'
        '- 메타 발화나 4벽 깨기를 하지 않는다\n'
        '- 선택된 캐릭터를 3인칭 서술로 묘사한다\n'
        '- 사용자 입력의 주 언어와 같은 언어로 답한다\n'
        '- 사용자 입력이 한국어이면 서술과 대사를 모두 한국어로 유지하고 영어로 전환하지 않는다\n\n'
        '우선순위:\n'
        '1. 현재 장면 규칙과 장면 목표\n'
        '2. 세계 상태 제약과 연속성\n'
        '3. 캐릭터 기억과 관계 맥락\n'
        '4. 로어북 사실과 용어\n\n'
        '금지 요소:\n'
        f'{forbidden_text}\n\n'
        '정체성 맥락:\n'
        f'관계 키: {relationship_text}\n'
        f'페르소나: {persona_desc}\n'
        f'대화 정책: {priority_rules}\n\n'
        '관계 맥락:\n'
        f'{relationship_block}\n\n'
        '출력 제한:\n'
        '- 사용자에게 보여줄 최종 RP 답변만 출력한다\n'
        '- 답변 설명, 지시문 언급, 분석, 계획, 체크리스트, 역할 태그를 쓰지 않는다\n'
        '- <|im_start|>, <|im_end|>, <|endoftext|>, <think> 같은 토큰을 출력하지 않는다\n'
        '- assistant 식 메타 문장을 쓰지 않는다\n'
        '- 끝까지 캐릭터를 유지한다\n\n'
        '아래에 대화 맥락과 기억 정보가 이어진다.\n'
        '도메인 데이터가 주어졌다면 외부 사실을 임의로 지어내지 않는다.'
    )


def build_user_prompt(
    user_text: str,
    user_profile: Dict[str, Any],
    lore: Dict[str, Any],
    memories: Dict[str, Any],
    world_state: Dict[str, Any],
    scene_state: Dict[str, Any],
    relationships: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    scene_event: Optional[ParsedSceneEvent] = None,
    target_character_hint: Optional[str] = None,
) -> str:
    parts: List[str] = []
    primary_language = _detect_primary_language(scene_event.raw_text if scene_event is not None else user_text)
    language_label = '한국어' if primary_language == 'ko' else '영어' if primary_language == 'en' else '입력과 동일한 언어'
    relationships = relationships or []
    prioritized_memories = memories.get('important_memories', []) if isinstance(memories, dict) else []
    if not isinstance(prioritized_memories, list):
        prioritized_memories = []
    prioritized_memories = [str(item).strip() for item in prioritized_memories if str(item).strip()][:5]
    world_facts = world_state.get('facts', []) if isinstance(world_state, dict) else []
    if not isinstance(world_facts, list):
        world_facts = []
    world_facts = [str(item).strip() for item in world_facts if str(item).strip()][:5]
    relationship_summary = []
    for item in relationships[:2]:
        summary = str(item.get('summary', '')).strip()
        tone = str(item.get('tone', 'neutral')).strip() or 'neutral'
        boundaries = item.get('boundaries', [])
        if not isinstance(boundaries, list):
            boundaries = []
        boundary = ', '.join(str(boundary).strip() for boundary in boundaries[:1] if str(boundary).strip()) or 'none'
        relationship_summary.append(f'tone={tone}; boundary={boundary}; summary={summary}')

    if history:
        recent = history[-6:]
        parts.append('최근 대화:\n' + '\n'.join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent))

    if scene_event is not None:
        parts.append(
            '파싱된 사용자 장면 이벤트:\n'
            f'원문 입력: {scene_event.raw_text or user_text}\n'
            f'사용자 서술: {scene_event.user_narration or "(없음)"}\n'
            f'사용자 대사: {scene_event.user_dialogue or "(없음)"}\n'
            f'입력 모드: {scene_event.input_mode}\n'
            f'대상 힌트: {target_character_hint or scene_event.target_character_hint or "(없음)"}'
        )

    parts.append(
        '출력 제약:\n'
        f'- 응답 주 언어: {language_label}\n'
        '- 사용자 입력의 주 언어와 같은 언어를 유지한다\n'
        '- 한국어 입력이면 서술과 대사를 모두 한국어로 유지하고 영어로 전환하지 않는다\n'
        f'- 서술은 선택된 캐릭터를 3인칭으로 묘사한다\n\n'
        '우선 맥락:\n'
        f'장면 우선: {json.dumps(scene_state, ensure_ascii=False)}\n'
        f'세계 제약: {json.dumps({"facts": world_facts, "location": world_state.get("location"), "time": world_state.get("time"), "state": world_state.get("state")}, ensure_ascii=False)}\n'
        f'캐릭터 기억: {json.dumps({"important_memories": prioritized_memories, "possessions": memories.get("possessions", [])}, ensure_ascii=False)}\n'
        f'관계 맥락: {json.dumps(relationship_summary, ensure_ascii=False)}\n'
        f'로어 참고: {json.dumps(lore, ensure_ascii=False)}'
    )
    parts.append('사용자 프로필:\n' + json.dumps(user_profile, ensure_ascii=False))
    parts.append('현재 사용자 메시지:\n' + user_text)
    return '\n\n'.join(parts)


def _detect_primary_language(text: str) -> str:
    hangul_count = len(re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]', text or ''))
    latin_count = len(re.findall(r'[A-Za-z]', text or ''))
    if hangul_count >= max(8, latin_count * 2):
        return 'ko'
    if latin_count >= max(12, hangul_count * 2):
        return 'en'
    return 'same-as-input'
