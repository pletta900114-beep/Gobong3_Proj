from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ParsedSceneEvent:
    raw_text: str
    user_narration: str
    user_dialogue: str
    target_character_hint: Optional[str]
    input_mode: str

    @property
    def has_narration(self) -> bool:
        return bool(self.user_narration.strip())

    @property
    def has_dialogue(self) -> bool:
        return bool(self.user_dialogue.strip())


_QUOTED_TEXT_RE = re.compile(r'["“](.*?)["”]', re.DOTALL)
_NARRATION_HINT_RE = re.compile(
    r'(?:나는|난|그는|그녀는|천천히|조용히|가만히|고개를|시선을|몸을|손끝|웃으며|웃었다|바라보|숨을|앉아|일어나|다가가|다가섰|기울였|기울였다|내리깔|침묵|멀리|향해)',
    re.IGNORECASE,
)
_DIRECT_ADDRESS_TEMPLATE = r'(^|\n|["“])\s*{alias}\s*(?:,|，|:|야|아|님|씨|!|\?)'
_TARGET_CONTEXT_KEYWORDS = (
    '바라',
    '보며',
    '보았',
    '보자',
    '쪽으로',
    '향해',
    '다가',
    '시선',
    '몸을 기울',
    '이름을 불',
)


def parse_scene_event(raw_text: str, characters: Iterable[Dict[str, Any]]) -> ParsedSceneEvent:
    normalized = _normalize_text(raw_text)
    dialogue_segments = [segment.strip() for segment in _QUOTED_TEXT_RE.findall(normalized) if segment.strip()]
    narration_candidate = _QUOTED_TEXT_RE.sub(' ', normalized)
    narration_candidate = re.sub(r'\n{3,}', '\n\n', narration_candidate)
    narration_candidate = re.sub(r'[ \t]{2,}', ' ', narration_candidate).strip()
    dialogue = '\n'.join(dialogue_segments).strip()
    narration = narration_candidate.strip()

    if dialogue and narration:
        input_mode = 'mixed'
    elif dialogue:
        input_mode = 'dialogue_only'
    elif _looks_like_narration(normalized):
        input_mode = 'narration_only'
        narration = normalized
    else:
        input_mode = 'dialogue_only'
        dialogue = normalized
        narration = ''

    target_hint = _extract_target_character_hint(normalized, dialogue, characters)
    return ParsedSceneEvent(
        raw_text=normalized,
        user_narration=narration.strip(),
        user_dialogue=dialogue.strip(),
        target_character_hint=target_hint,
        input_mode=input_mode,
    )


def _normalize_text(text: str) -> str:
    cleaned = (text or '').replace('\r\n', '\n').replace('\r', '\n')
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _looks_like_narration(text: str) -> bool:
    if not text.strip():
        return False
    return bool(_NARRATION_HINT_RE.search(text))


def _extract_target_character_hint(raw_text: str, dialogue: str, characters: Iterable[Dict[str, Any]]) -> Optional[str]:
    alias_map = _build_alias_map(characters)
    lowered_raw = raw_text.lower()
    lowered_dialogue = dialogue.lower()

    for alias, character_id in alias_map.items():
        if re.search(_DIRECT_ADDRESS_TEMPLATE.format(alias=re.escape(alias)), lowered_dialogue, re.IGNORECASE):
            return character_id
        if re.search(_DIRECT_ADDRESS_TEMPLATE.format(alias=re.escape(alias)), lowered_raw, re.IGNORECASE):
            return character_id

    for alias, character_id in alias_map.items():
        alias_index = lowered_raw.find(alias)
        if alias_index < 0:
            continue
        start = max(0, alias_index - 16)
        end = min(len(lowered_raw), alias_index + len(alias) + 20)
        window = lowered_raw[start:end]
        if any(keyword in window for keyword in _TARGET_CONTEXT_KEYWORDS):
            return character_id

    return None


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
