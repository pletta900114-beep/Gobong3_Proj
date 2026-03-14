from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mellow_chat_runtime.services import retrieval_scoring_config as scoring_config


class RetrievalReranker:
    def rerank_lore(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
        canonical_items: Dict[str, Dict[str, Any]],
        lore_topics: Optional[List[str]] = None,
        top_k: int = scoring_config.LORE_TOP_K,
    ) -> List[Dict[str, Any]]:
        ranked = self._ranked(
            candidates=candidates,
            scorer=lambda item: self._score_lore(
                query=query,
                candidate=item,
                canonical_item=canonical_items.get(str(item.get("id", "")), {}),
                lore_topics=lore_topics or [],
            ),
            threshold=scoring_config.LORE_THRESHOLD,
            top_k=top_k,
        )
        return ranked

    def rerank_memories(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
        active_speaker_id: str,
        participant_ids: List[str],
        top_k: int = scoring_config.MEMORY_TOP_K,
    ) -> List[Dict[str, Any]]:
        ranked = self._ranked(
            candidates=candidates,
            scorer=lambda item: self._score_memory(
                query=query,
                candidate=item,
                active_speaker_id=active_speaker_id,
                participant_ids=participant_ids,
            ),
            threshold=scoring_config.MEMORY_THRESHOLD,
            top_k=top_k,
        )
        return ranked

    def rerank_relationships(
        self,
        *,
        query: str,
        candidates: List[Dict[str, Any]],
        source_id: str,
        participant_ids: List[str],
        top_k: int = scoring_config.RELATIONSHIP_TOP_K,
    ) -> List[Dict[str, Any]]:
        ranked = self._ranked(
            candidates=candidates,
            scorer=lambda item: self._score_relationship(
                query=query,
                candidate=item,
                source_id=source_id,
                participant_ids=participant_ids,
            ),
            threshold=scoring_config.RELATIONSHIP_THRESHOLD,
            top_k=top_k,
        )
        return ranked

    def _ranked(self, *, candidates: List[Dict[str, Any]], scorer: Any, threshold: float, top_k: int) -> List[Dict[str, Any]]:
        scored: List[tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            score = scorer(candidate)
            if score < threshold:
                continue
            updated = dict(candidate)
            updated["score"] = round(score, 4)
            scored.append((score, updated))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in scored[: max(1, int(top_k or 1))]]

    def _score_lore(
        self,
        *,
        query: str,
        candidate: Dict[str, Any],
        canonical_item: Dict[str, Any],
        lore_topics: List[str],
    ) -> float:
        query_tokens = self._tokenize(query)
        vector_score = float(candidate.get("vector_score", 0.0))
        topic = str(canonical_item.get("topic") or candidate.get("topic") or "").strip()
        aliases = self._clean_list(canonical_item.get("aliases", []))
        priority = int(canonical_item.get("priority") or candidate.get("priority") or 0)
        topic_match = scoring_config.LORE_TOPIC_EXACT_BOOST if self._phrase_match(query, topic) else 0.0
        alias_match = scoring_config.LORE_ALIAS_MATCH_BOOST if any(self._phrase_match(query, alias) for alias in aliases) else 0.0
        lore_topic_match = scoring_config.LORE_TOPIC_HINT_BOOST if any(self._same_text(topic, item) or any(self._same_text(alias, item) for alias in aliases) for item in lore_topics) else 0.0
        world_match = scoring_config.LORE_WORLD_MATCH_BOOST if len(set(query_tokens) & set(self._tokenize(str(candidate.get("summary_text", ""))))) >= 2 else 0.0
        priority_boost = min(priority, scoring_config.LORE_MAX_PRIORITY_STEPS) * scoring_config.LORE_PRIORITY_BOOST_PER_STEP
        unrelated_penalty = scoring_config.LORE_UNRELATED_PENALTY if vector_score <= 1.0 and topic_match == 0.0 and alias_match == 0.0 and lore_topic_match == 0.0 else 0.0
        return vector_score + topic_match + alias_match + lore_topic_match + world_match + priority_boost + unrelated_penalty

    def _score_memory(
        self,
        *,
        query: str,
        candidate: Dict[str, Any],
        active_speaker_id: str,
        participant_ids: List[str],
    ) -> float:
        vector_score = float(candidate.get("vector_score", 0.0))
        memory_text = str(candidate.get("memory_text") or candidate.get("summary_text") or "").strip()
        character_id = str(candidate.get("character_id") or "").strip()
        importance_boost = scoring_config.MEMORY_IMPORTANCE_BOOST if any(keyword in memory_text.lower() for keyword in scoring_config.IMPORTANT_MEMORY_KEYWORDS) else 0.0
        recency_boost = min(self._memory_index(candidate) * scoring_config.MEMORY_RECENCY_BOOST_PER_STEP, scoring_config.MEMORY_RECENCY_BOOST_CAP)
        active_speaker_boost = scoring_config.MEMORY_ACTIVE_SPEAKER_BOOST if character_id == active_speaker_id else 0.0
        participant_boost = scoring_config.MEMORY_PARTICIPANT_BOOST if character_id in participant_ids else 0.0
        scene_match = scoring_config.MEMORY_SCENE_MATCH_BOOST if len(set(self._tokenize(query)) & set(self._tokenize(memory_text))) >= 2 else 0.0
        stale_penalty = scoring_config.MEMORY_STALE_PENALTY if vector_score < 1.0 and scene_match == 0.0 and character_id != active_speaker_id else 0.0
        return vector_score + importance_boost + recency_boost + active_speaker_boost + participant_boost + scene_match + stale_penalty

    def _score_relationship(
        self,
        *,
        query: str,
        candidate: Dict[str, Any],
        source_id: str,
        participant_ids: List[str],
    ) -> float:
        vector_score = float(candidate.get("vector_score", 0.0))
        target_id = str(candidate.get("target_id") or "").strip()
        character_id = str(candidate.get("character_id") or "").strip()
        exact_boost = scoring_config.RELATIONSHIP_SOURCE_TARGET_EXACT_BOOST if character_id == source_id and target_id in participant_ids else 0.0
        participant_boost = scoring_config.RELATIONSHIP_PARTICIPANT_PRESENCE_BOOST if target_id in participant_ids else 0.0
        tone_relevance = scoring_config.RELATIONSHIP_TONE_SCENE_RELEVANCE_BOOST if len(set(self._tokenize(query)) & set(self._tokenize(str(candidate.get("summary_text", ""))))) >= 2 else 0.0
        unrelated_penalty = scoring_config.RELATIONSHIP_UNRELATED_PENALTY if target_id and target_id not in participant_ids and vector_score < 1.0 else 0.0
        return vector_score + exact_boost + participant_boost + tone_relevance + unrelated_penalty

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+|[가-힣]{2,}", (text or "").lower())

    def _clean_list(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    def _phrase_match(self, query: str, phrase: str) -> bool:
        normalized_phrase = str(phrase or "").strip().lower()
        return bool(normalized_phrase) and normalized_phrase in str(query or "").lower()

    def _same_text(self, left: str, right: str) -> bool:
        return str(left or "").strip().lower() == str(right or "").strip().lower()

    def _memory_index(self, candidate: Dict[str, Any]) -> int:
        raw_id = str(candidate.get("id") or candidate.get("source_id") or "")
        if ":" not in raw_id:
            return 0
        try:
            return int(raw_id.rsplit(":", 1)[1]) + 1
        except ValueError:
            return 0
