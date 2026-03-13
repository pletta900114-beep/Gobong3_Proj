from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from mellow_chat_runtime.core.rp_parser import ParsedSceneEvent

_DIRECT_ADDRESS_TEMPLATE = r'(^|\n|["“])\s*{alias}\s*(?:,|，|:|야|아|님|씨|!|\?)'
_TARGET_CONTEXT_KEYWORDS = (
    '바라',
    '보며',
    '보았',
    '시선',
    '쪽으로',
    '향해',
    '몸을 기울',
    '다가',
)

DIRECT_VOCATIVE_BONUS = 1.5
NARRATION_TARGET_BONUS = 1.0
DIALOGUE_ALIAS_BONUS = 0.75
RELATIONSHIP_SCENE_BONUS = 0.25


def build_speaker_relevance(
    parsed_scene_event: ParsedSceneEvent,
    characters: Iterable[Dict[str, Any]],
    scene_state: Optional[Dict[str, Any]] = None,
    relationships: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, float]:
    alias_map = _build_alias_map(characters)
    scores: Dict[str, float] = {}

    direct_target = _find_direct_vocative_target(parsed_scene_event.user_dialogue, alias_map)
    if direct_target:
        scores[direct_target] = scores.get(direct_target, 0.0) + DIRECT_VOCATIVE_BONUS

    narration_target = _find_narration_target(parsed_scene_event.user_narration, alias_map)
    if narration_target and narration_target != direct_target:
        scores[narration_target] = scores.get(narration_target, 0.0) + NARRATION_TARGET_BONUS

    for alias_target in _find_dialogue_alias_mentions(parsed_scene_event.user_dialogue, alias_map):
        if alias_target == direct_target:
            continue
        scores[alias_target] = scores.get(alias_target, 0.0) + DIALOGUE_ALIAS_BONUS

    if direct_target:
        if _is_scene_participant(direct_target, scene_state):
            scores[direct_target] = scores.get(direct_target, 0.0) + RELATIONSHIP_SCENE_BONUS
        if relationships and relationships.get(direct_target):
            scores[direct_target] = scores.get(direct_target, 0.0) + RELATIONSHIP_SCENE_BONUS

    return scores


def _build_alias_map(characters: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for character in characters:
        if not isinstance(character, dict):
            continue
        character_id = str(character.get('id', '')).strip()
        if not character_id:
            continue
        names: List[str] = []
        raw_name = str(character.get('name', '')).strip()
        if raw_name:
            names.append(raw_name)
        aliases = character.get('aliases', [])
        if isinstance(aliases, list):
            names.extend(str(alias).strip() for alias in aliases if str(alias).strip())
        names.append(character_id)
        for name in names:
            alias_map.setdefault(name.lower(), character_id)
    return alias_map


def _find_direct_vocative_target(dialogue: str, alias_map: Dict[str, str]) -> Optional[str]:
    lowered_dialogue = (dialogue or '').lower()
    for alias, character_id in alias_map.items():
        if re.search(_DIRECT_ADDRESS_TEMPLATE.format(alias=re.escape(alias)), lowered_dialogue, re.IGNORECASE):
            return character_id
    return None


def _find_narration_target(narration: str, alias_map: Dict[str, str]) -> Optional[str]:
    lowered_narration = (narration or '').lower()
    for alias, character_id in alias_map.items():
        alias_index = lowered_narration.find(alias)
        if alias_index < 0:
            continue
        start = max(0, alias_index - 16)
        end = min(len(lowered_narration), alias_index + len(alias) + 20)
        window = lowered_narration[start:end]
        if any(keyword in window for keyword in _TARGET_CONTEXT_KEYWORDS):
            return character_id
    return None


def _find_dialogue_alias_mentions(dialogue: str, alias_map: Dict[str, str]) -> List[str]:
    lowered_dialogue = (dialogue or '').lower()
    found: List[str] = []
    for alias, character_id in alias_map.items():
        if alias in lowered_dialogue and character_id not in found:
            found.append(character_id)
    return found


def _is_scene_participant(character_id: str, scene_state: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(scene_state, dict):
        return False
    participants = scene_state.get('participants', [])
    if not isinstance(participants, list):
        return False
    return character_id in {str(item).strip() for item in participants if str(item).strip()}
