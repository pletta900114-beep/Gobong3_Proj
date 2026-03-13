from __future__ import annotations

import re
from typing import Dict, List

from mellow_chat_runtime.core.domain_lookup_store import DomainLookupStore


class MemoryPromotionService:
    """Promote selected short-term turn content into long-term character memories."""

    KEYWORDS = (
        "remember",
        "important",
        "promise",
        "promised",
        "decide",
        "decided",
        "plan",
        "planned",
        "기억",
        "중요",
        "약속",
        "결정",
        "계획",
    )

    def __init__(self, domain_store: DomainLookupStore, max_items: int = 20) -> None:
        self._domain_store = domain_store
        self._max_items = max(1, int(max_items or 20))

    def promote_from_text(self, character_id: str, text: str) -> List[str]:
        cleaned_character_id = (character_id or "").strip()
        if not cleaned_character_id:
            return []

        candidates = self._extract_candidates(text)
        if not candidates:
            return []

        memory_state = self._domain_store.get_memory_and_possessions(cleaned_character_id)
        existing_items = self._clean_memory_list(memory_state.get("important_memories", []))
        existing_lower = {item.lower(): item for item in existing_items}

        promoted: List[str] = []
        merged = list(existing_items)
        for candidate in candidates:
            lowered = candidate.lower()
            if lowered in existing_lower:
                continue
            merged.append(candidate)
            existing_lower[lowered] = candidate
            promoted.append(candidate)

        if not promoted:
            return []

        updated_memory: Dict[str, object] = {
            "character_id": cleaned_character_id,
            "important_memories": merged[-self._max_items :],
            "possessions": self._clean_memory_list(memory_state.get("possessions", [])),
        }
        self._domain_store.upsert("memories", cleaned_character_id, updated_memory)
        return promoted

    def _extract_candidates(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return []

        parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
        candidates: List[str] = []
        seen = set()
        for raw_part in parts:
            candidate = raw_part.strip(" -:;,")
            if len(candidate) < 12 or len(candidate) > 220:
                continue
            lowered = candidate.lower()
            if not any(keyword in lowered for keyword in self.KEYWORDS):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            candidates.append(candidate)
            if len(candidates) >= 2:
                break
        return candidates

    def _clean_memory_list(self, values: object) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for value in values:
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
        return out
