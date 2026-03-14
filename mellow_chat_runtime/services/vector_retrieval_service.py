from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mellow_chat_runtime.core.domain_lookup_store import DomainLookupStore
from mellow_chat_runtime.infra.vector_index_store import JsonVectorIndexStore
from mellow_chat_runtime.services import retrieval_scoring_config as scoring_config
from mellow_chat_runtime.services.retrieval_reranker import RetrievalReranker
from mellow_chat_runtime.services.summary_formatter import (
    build_lore_summary,
    build_memory_summary,
    build_relationship_summary,
    searchable_fields_changed,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQueryContext:
    query: str
    active_speaker_id: str = ""
    participant_ids: List[str] = field(default_factory=list)
    lore_topics: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    items: List[Dict[str, Any]]
    source: str
    fallback_used: bool = False
    error: str = ""


class VectorRetrievalService:
    def __init__(self, domain_store: DomainLookupStore, index_path: Path) -> None:
        self._domain_store = domain_store
        self._index_store = JsonVectorIndexStore(index_path)
        self._reranker = RetrievalReranker()

    def search_lore(self, query: str, lore_topics: Optional[List[str]] = None, top_k: int = scoring_config.LORE_TOP_K) -> RetrievalResult:
        fallback_items: List[Dict[str, Any]] = []
        for topic in lore_topics or []:
            item = self._domain_store.get_lore(topic)
            if item:
                fallback_items.append(item)
        canonical_items = self._domain_store.list_section("lorebook")
        return self._search_collection(
            collection="lore_entries",
            query=query,
            top_k=top_k,
            fallback_items=fallback_items,
            rerank=lambda candidates: self._reranker.rerank_lore(
                query=query,
                candidates=candidates,
                canonical_items=canonical_items,
                lore_topics=lore_topics or [],
                top_k=top_k,
            ),
        )

    def search_memories(self, query: str, character_ids: List[str], top_k: int = scoring_config.MEMORY_TOP_K) -> RetrievalResult:
        allowed_ids = {item.strip() for item in character_ids if isinstance(item, str) and item.strip()}
        fallback_items = self._canonical_memory_entries(list(allowed_ids))
        return self._search_collection(
            "memory_entries",
            query,
            top_k,
            fallback_items=fallback_items,
            allowed_ids=allowed_ids,
            id_field="character_id",
            rerank=lambda candidates: self._reranker.rerank_memories(
                query=query,
                candidates=candidates,
                active_speaker_id=character_ids[0] if character_ids else "",
                participant_ids=[item for item in character_ids if item],
                top_k=top_k,
            ),
        )

    def search_relationships(
        self,
        query: str,
        source_id: str,
        counterpart_ids: Optional[List[str]] = None,
        top_k: int = scoring_config.RELATIONSHIP_TOP_K,
    ) -> RetrievalResult:
        fallback_items = self._domain_store.get_relationships(source_id, counterpart_ids=counterpart_ids)
        allowed_pairs = {
            f"{source_id}:{target_id.strip()}"
            for target_id in (counterpart_ids or [])
            if isinstance(target_id, str) and target_id.strip()
        }
        return self._search_collection(
            "relationship_entries",
            query,
            top_k,
            fallback_items=fallback_items,
            allowed_ids=allowed_pairs if allowed_pairs else None,
            id_field="pair_id",
            rerank=lambda candidates: self._reranker.rerank_relationships(
                query=query,
                candidates=candidates,
                source_id=source_id,
                participant_ids=[item for item in (counterpart_ids or []) if item],
                top_k=top_k,
            ),
        )

    def build_context(self, context: RetrievalQueryContext) -> Dict[str, Any]:
        relationship_targets = [item for item in context.participant_ids if item and item != context.active_speaker_id]
        lore = self.search_lore(query=context.query, lore_topics=context.lore_topics, top_k=scoring_config.LORE_TOP_K)
        memories = self.search_memories(
            query=context.query,
            character_ids=[context.active_speaker_id, *context.participant_ids],
            top_k=scoring_config.MEMORY_TOP_K,
        )
        relationships = self.search_relationships(
            query=context.query,
            source_id=context.active_speaker_id,
            counterpart_ids=relationship_targets,
            top_k=scoring_config.RELATIONSHIP_TOP_K,
        )
        return {
            "lore": lore.items[: scoring_config.LORE_TOP_K],
            "memories": memories.items[: scoring_config.MEMORY_TOP_K],
            "relationships": relationships.items[: scoring_config.RELATIONSHIP_TOP_K],
            "debug": {
                "lore_source": lore.source,
                "memory_source": memories.source,
                "relationship_source": relationships.source,
                "fallback_used": lore.fallback_used or memories.fallback_used or relationships.fallback_used,
                "errors": [item for item in [lore.error, memories.error, relationships.error] if item],
            },
        }

    def reindex(self) -> Dict[str, int]:
        lore_entries = self._build_lore_entries()
        memory_entries = self._build_memory_entries()
        relationship_entries = self._build_relationship_entries()
        self._index_store.replace_collection("lore_entries", lore_entries)
        self._index_store.replace_collection("memory_entries", memory_entries)
        self._index_store.replace_collection("relationship_entries", relationship_entries)
        return {
            "lore_entries": len(lore_entries),
            "memory_entries": len(memory_entries),
            "relationship_entries": len(relationship_entries),
        }

    def mark_dirty_if_needed(
        self,
        section: str,
        key: str,
        payload: Dict[str, Any],
        existing: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        existing = existing if isinstance(existing, dict) else self._domain_store.get_section_item(section, key)
        updated = dict(payload)
        if searchable_fields_changed(section, existing, updated):
            updated["embedding_status"] = "dirty"
        else:
            updated["embedding_status"] = str(existing.get("embedding_status", "clean") or "clean")
        return updated

    def _build_lore_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for lore_id, payload in self._domain_store.list_section("lorebook").items():
            summary_text = build_lore_summary(payload)
            updated = dict(payload)
            updated["summary_text"] = summary_text
            updated["embedding_status"] = "clean"
            self._domain_store.upsert("lorebook", lore_id, updated)
            entries.append(
                {
                    "id": lore_id,
                    "source_id": lore_id,
                    "topic": updated.get("topic", ""),
                    "summary_text": summary_text,
                    "priority": updated.get("priority", 0),
                }
            )
        return entries

    def _build_memory_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for character_id, payload in self._domain_store.list_section("memories").items():
            updated = dict(payload)
            updated["summary_text"] = build_memory_summary(updated)
            updated["embedding_status"] = "clean"
            self._domain_store.upsert("memories", character_id, updated)
            for index, memory_text in enumerate(self._clean_list(updated.get("important_memories", []))):
                entries.append(
                    {
                        "id": f"{character_id}:{index}",
                        "source_id": f"{character_id}:{index}",
                        "character_id": character_id,
                        "summary_text": memory_text,
                        "memory_text": memory_text,
                    }
                )
        return entries

    def _build_relationship_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for source_id, relationship_map in self._domain_store.list_section("relationships").items():
            updated_map: Dict[str, Any] = {}
            for target_id, payload in relationship_map.items():
                if not isinstance(payload, dict):
                    continue
                updated = dict(payload)
                updated["target_id"] = target_id
                updated["summary_text"] = build_relationship_summary(updated)
                updated["embedding_status"] = "clean"
                updated_map[target_id] = updated
                entries.append(
                    {
                        "id": f"{source_id}:{target_id}",
                        "source_id": f"{source_id}:{target_id}",
                        "pair_id": f"{source_id}:{target_id}",
                        "character_id": source_id,
                        "target_id": target_id,
                        "summary_text": updated["summary_text"],
                    }
                )
            self._domain_store.upsert("relationships", source_id, updated_map)
        return entries

    def _search_collection(
        self,
        collection: str,
        query: str,
        top_k: int,
        fallback_items: List[Dict[str, Any]],
        allowed_ids: Optional[set[str]] = None,
        id_field: str = "source_id",
        rerank: Optional[Any] = None,
    ) -> RetrievalResult:
        try:
            candidates = self._vector_candidates(
                collection=collection,
                query=query,
                allowed_ids=allowed_ids,
                id_field=id_field,
                candidate_limit=max(6, max(1, int(top_k or 1)) * 4),
            )
            items = rerank(candidates) if rerank is not None else candidates[: max(1, int(top_k or 1))]
            if items:
                return RetrievalResult(items=items, source="vector")
            return RetrievalResult(items=fallback_items[: max(1, int(top_k or 1))], source="canonical", fallback_used=True)
        except Exception as exc:
            logger.warning("vector retrieval failed collection=%s error=%s", collection, exc)
            return RetrievalResult(
                items=fallback_items[: max(1, int(top_k or 1))],
                source="canonical",
                fallback_used=True,
                error=str(exc),
            )

    def _vector_candidates(
        self,
        *,
        collection: str,
        query: str,
        allowed_ids: Optional[set[str]],
        id_field: str,
        candidate_limit: int,
    ) -> List[Dict[str, Any]]:
        ranked: List[tuple[float, Dict[str, Any]]] = []
        for entry in self._index_store.list_entries(collection):
            if allowed_ids is not None and str(entry.get(id_field, "")).strip() not in allowed_ids:
                continue
            score = self._score(query, str(entry.get("summary_text", "")))
            if score <= 0:
                continue
            candidate = dict(entry)
            candidate["vector_score"] = float(score)
            ranked.append((score, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in ranked[:candidate_limit]]

    def _canonical_memory_entries(self, character_ids: List[str]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for character_id in character_ids:
            payload = self._domain_store.get_memory_and_possessions(character_id)
            for index, memory_text in enumerate(self._clean_list(payload.get("important_memories", []))):
                items.append(
                    {
                        "id": f"{character_id}:{index}",
                        "character_id": character_id,
                        "summary_text": memory_text,
                        "memory_text": memory_text,
                    }
                )
        return items

    def _score(self, query: str, text: str) -> float:
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)
        if not query_tokens or not text_tokens:
            return 0.0
        overlap = sum(1 for token in query_tokens if token in text_tokens)
        return float(overlap)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+|[가-힣]{2,}", (text or "").lower())

    def _clean_list(self, values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]
