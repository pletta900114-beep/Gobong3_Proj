from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpeakerParticipant:
    character_id: str
    is_major: bool = True
    can_speak: bool = True
    weight: float = 1.0


def select_next_speaker(
    participants: List[SpeakerParticipant],
    recent_speaker_history: Optional[List[str]] = None,
    dialogue_priority: Optional[Dict[str, float]] = None,
    scene_rules: Optional[Dict[str, object]] = None,
    target_character_hint: Optional[str] = None,
    speaker_relevance: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Deterministic speaker selection.
    Rules:
    - Optional force/include/exclude from scene rules.
    - Target hint is a strong preference after scene force.
    - 1:1 alternates when possible.
    - Multi-party uses major/minor weighting plus recency penalties and relevance bonuses.
    """
    if not participants:
        return None

    history = list(recent_speaker_history or [])
    rules = dict(scene_rules or {})
    force_speaker = _as_str(rules.get('force_speaker_id'))
    if force_speaker and _is_eligible(force_speaker, participants, rules):
        return force_speaker

    eligible = [p for p in participants if _is_eligible(p.character_id, participants, rules)]
    if not eligible:
        return None

    hinted_speaker = _as_str(target_character_hint)
    if hinted_speaker and any(item.character_id == hinted_speaker for item in eligible):
        return hinted_speaker

    if len(eligible) == 1:
        return eligible[0].character_id

    if len(eligible) == 2 and not speaker_relevance:
        last_speaker = history[-1] if history else None
        for participant in sorted(eligible, key=lambda item: item.character_id):
            if participant.character_id != last_speaker:
                return participant.character_id
        return sorted(eligible, key=lambda item: item.character_id)[0].character_id

    return _select_multi(eligible, history, dialogue_priority or {}, speaker_relevance or {})


def _select_multi(
    participants: List[SpeakerParticipant],
    history: List[str],
    dialogue_priority: Dict[str, float],
    speaker_relevance: Dict[str, float],
) -> Optional[str]:
    major_weight = float(dialogue_priority.get('major_weight', 1.0) or 1.0)
    minor_weight = float(dialogue_priority.get('minor_weight', 0.5) or 0.5)
    recency_penalty = float(dialogue_priority.get('recency_penalty', 0.25) or 0.25)
    max_consecutive_turns = int(dialogue_priority.get('max_consecutive_turns', 1) or 1)

    recent_window = history[-6:]
    consecutive_speaker = _consecutive_tail_speaker(history)
    scored: List[tuple[float, str]] = []

    for participant in participants:
        base_weight = major_weight if participant.is_major else minor_weight
        score = base_weight * max(participant.weight, 0.0)
        score += float(speaker_relevance.get(participant.character_id, 0.0) or 0.0)

        if participant.character_id == consecutive_speaker and _consecutive_tail_count(history) >= max_consecutive_turns:
            score -= 1.0

        for index, speaker_id in enumerate(reversed(recent_window), start=1):
            if speaker_id == participant.character_id:
                score -= recency_penalty / float(index)

        scored.append((score, participant.character_id))

    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored:
        return None
    return scored[0][1]


def _is_eligible(character_id: str, participants: List[SpeakerParticipant], scene_rules: Dict[str, object]) -> bool:
    participant = next((p for p in participants if p.character_id == character_id), None)
    if participant is None or not participant.can_speak:
        return False

    include = set(_as_str_list(scene_rules.get('include_speakers')))
    exclude = set(_as_str_list(scene_rules.get('exclude_speakers')))
    if include and character_id not in include:
        return False
    if character_id in exclude:
        return False
    return True


def _consecutive_tail_speaker(history: List[str]) -> Optional[str]:
    if not history:
        return None
    return history[-1]


def _consecutive_tail_count(history: List[str]) -> int:
    if not history:
        return 0
    tail = history[-1]
    count = 0
    for speaker_id in reversed(history):
        if speaker_id != tail:
            break
        count += 1
    return count


def _as_str(value: object) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _as_str_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out
