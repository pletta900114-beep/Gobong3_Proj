from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

import mellow_chat_runtime.app_state as app_state
import mellow_chat_runtime.core.domain_lookup_store as domain_lookup_store_module
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.orchestrator import Orchestrator
from mellow_chat_runtime.infra.database import init_db
from mellow_chat_runtime.routers.admin import router as admin_router
from mellow_chat_runtime.routers.chat import router as chat_router
from mellow_chat_runtime.services.retrieval_reranker import RetrievalReranker
from mellow_chat_runtime.services.vector_retrieval_service import VectorRetrievalService


class RecordingLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.chat_calls: list[list[dict[str, str]]] = []

    def get_model_for_mode(self, mode: str) -> str:
        return "qwen3.5:9b"

    async def chat(self, messages, model=None, **kwargs):
        self.chat_calls.append(messages)
        return SimpleNamespace(text=self.response_text, thinking="", model=model or "qwen3.5:9b")


def _reset_domain_store() -> None:
    domain_lookup_store_module._global_store = None


def _build_client(tmp_path: Path, llm: RecordingLLM) -> tuple[TestClient, object, VectorRetrievalService]:
    _reset_domain_store()
    domain_file = tmp_path / "domain_data.json"
    vector_index_file = tmp_path / "vector_index.json"
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    dispatcher = DomainLookupDispatcher(store)
    orchestrator = Orchestrator(lookup_dispatcher=dispatcher)
    orchestrator.register_service("llm", llm)
    vector_service = VectorRetrievalService(domain_store=store, index_path=vector_index_file)

    app_state.settings = SimpleNamespace(
        domain_data_file=domain_file,
        vector_index_file=vector_index_file,
        memory_promotion_enabled=False,
        memory_promotion_max_items=20,
    )
    app_state.orchestrator = orchestrator
    app_state.llm_service = llm
    app_state.vector_retrieval_service = vector_service

    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(admin_router)
    return TestClient(app), store, vector_service


def test_lore_reranker_prefers_topic_alias_and_priority() -> None:
    reranker = RetrievalReranker()

    ranked = reranker.rerank_lore(
        query="IPC audit route update",
        lore_topics=["IPC"],
        candidates=[
            {"id": "lore_001", "topic": "Interastral Peace Corporation", "summary_text": "IPC contracts route", "priority": 10, "vector_score": 1.0},
            {"id": "lore_999", "topic": "Tea Ceremony", "summary_text": "teacup etiquette", "priority": 10, "vector_score": 1.0},
        ],
        canonical_items={
            "lore_001": {"topic": "Interastral Peace Corporation", "aliases": ["IPC"], "priority": 10},
            "lore_999": {"topic": "Tea Ceremony", "aliases": ["tea"], "priority": 10},
        },
        top_k=3,
    )

    assert [item["id"] for item in ranked] == ["lore_001"]


def test_memory_reranker_prefers_active_speaker_and_filters_stale_hits() -> None:
    reranker = RetrievalReranker()

    ranked = reranker.rerank_memories(
        query="remember the station plan",
        candidates=[
            {"id": "bot_char_01:2", "character_id": "bot_char_01", "summary_text": "Remember the station plan before the meeting", "memory_text": "Remember the station plan before the meeting", "vector_score": 1.0},
            {"id": "bot_char_02:0", "character_id": "bot_char_02", "summary_text": "Old ballroom dance rehearsal", "memory_text": "Old ballroom dance rehearsal", "vector_score": 1.0},
        ],
        active_speaker_id="bot_char_01",
        participant_ids=["bot_char_01", "user_char_01"],
        top_k=4,
    )

    assert [item["id"] for item in ranked] == ["bot_char_01:2"]


def test_relationship_reranker_prefers_present_counterpart_and_filters_unrelated_pair() -> None:
    reranker = RetrievalReranker()

    ranked = reranker.rerank_relationships(
        query="respond to user_char_01 calmly",
        candidates=[
            {"id": "bot_char_01:user_char_01", "character_id": "bot_char_01", "target_id": "user_char_01", "summary_text": "calm strategic trust", "vector_score": 1.0},
            {"id": "bot_char_01:bot_char_77", "character_id": "bot_char_01", "target_id": "bot_char_77", "summary_text": "old unrelated rivalry", "vector_score": 1.0},
        ],
        source_id="bot_char_01",
        participant_ids=["user_char_01"],
        top_k=2,
    )

    assert [item["id"] for item in ranked] == ["bot_char_01:user_char_01"]


def test_memory_ranking_changes_when_active_speaker_changes() -> None:
    reranker = RetrievalReranker()
    candidates = [
        {"id": "bot_char_01:1", "character_id": "bot_char_01", "summary_text": "Remember the station plan before departure", "memory_text": "Remember the station plan before departure", "vector_score": 1.0},
        {"id": "bot_char_02:1", "character_id": "bot_char_02", "summary_text": "Remember the station plan before departure", "memory_text": "Remember the station plan before departure", "vector_score": 1.0},
    ]

    ranked_for_bot_01 = reranker.rerank_memories(
        query="remember the station plan",
        candidates=candidates,
        active_speaker_id="bot_char_01",
        participant_ids=["bot_char_01", "bot_char_02"],
    )
    ranked_for_bot_02 = reranker.rerank_memories(
        query="remember the station plan",
        candidates=candidates,
        active_speaker_id="bot_char_02",
        participant_ids=["bot_char_01", "bot_char_02"],
    )

    assert ranked_for_bot_01[0]["id"] == "bot_char_01:1"
    assert ranked_for_bot_02[0]["id"] == "bot_char_02:1"


def test_relationship_ranking_changes_with_participant_composition() -> None:
    reranker = RetrievalReranker()
    candidates = [
        {"id": "bot_char_01:user_char_01", "character_id": "bot_char_01", "target_id": "user_char_01", "summary_text": "calm trust with user", "vector_score": 1.0},
        {"id": "bot_char_01:bot_char_02", "character_id": "bot_char_01", "target_id": "bot_char_02", "summary_text": "formal teamwork with sunday", "vector_score": 1.0},
    ]

    ranked_with_user = reranker.rerank_relationships(
        query="respond to user_char_01 calmly",
        candidates=candidates,
        source_id="bot_char_01",
        participant_ids=["user_char_01"],
    )
    ranked_with_sunday = reranker.rerank_relationships(
        query="coordinate with bot_char_02 formally",
        candidates=candidates,
        source_id="bot_char_01",
        participant_ids=["bot_char_02"],
    )

    assert ranked_with_user[0]["id"] == "bot_char_01:user_char_01"
    assert ranked_with_sunday[0]["id"] == "bot_char_01:bot_char_02"


def test_lore_reranker_prefers_exact_topic_and_alias_when_lore_topics_are_given() -> None:
    reranker = RetrievalReranker()

    ranked = reranker.rerank_lore(
        query="Need IPC protocol summary",
        lore_topics=["Interastral Peace Corporation"],
        candidates=[
            {"id": "lore_001", "topic": "Interastral Peace Corporation", "summary_text": "IPC protocol and contract order", "priority": 8, "vector_score": 1.0},
            {"id": "lore_002", "topic": "Stonehearts", "summary_text": "strategic executives", "priority": 8, "vector_score": 1.0},
        ],
        canonical_items={
            "lore_001": {"topic": "Interastral Peace Corporation", "aliases": ["IPC"], "priority": 8},
            "lore_002": {"topic": "Stonehearts", "aliases": ["Ten Stonehearts"], "priority": 8},
        },
    )

    assert [item["id"] for item in ranked] == ["lore_001"]


def test_unrelated_hits_drop_below_threshold_and_are_excluded() -> None:
    reranker = RetrievalReranker()

    lore_ranked = reranker.rerank_lore(
        query="IPC route audit",
        lore_topics=["IPC"],
        candidates=[
            {"id": "lore_noise", "topic": "Tea Ceremony", "summary_text": "cup sugar etiquette", "priority": 10, "vector_score": 1.0},
        ],
        canonical_items={
            "lore_noise": {"topic": "Tea Ceremony", "aliases": ["tea"], "priority": 10},
        },
    )
    relationship_ranked = reranker.rerank_relationships(
        query="speak with user_char_01 calmly",
        candidates=[
            {"id": "bot_char_01:bot_char_77", "character_id": "bot_char_01", "target_id": "bot_char_77", "summary_text": "old rivalry in another scene", "vector_score": 0.5},
        ],
        source_id="bot_char_01",
        participant_ids=["user_char_01"],
    )

    assert lore_ranked == []
    assert relationship_ranked == []


def test_chat_ask_uses_reranked_retrieval_and_suppresses_unrelated_lore(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM('어벤츄린은 짧게 고개를 끄덕인다.\n\n"관련 정보만 정리해 두었어."')
    client, store, vector_service = _build_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    store.upsert(
        "lorebook",
        "lore_irrelevant",
        {
            "id": "lore_irrelevant",
            "topic": "Tea Ceremony",
            "aliases": ["tea"],
            "content": "Ceramic cup etiquette and sugar timing.",
            "priority": 10,
            "summary_text": "Ceramic cup etiquette and sugar timing.",
            "embedding_status": "dirty",
        },
    )
    vector_service.reindex()

    response = client.post(
        "/chat/ask",
        json={
            "question": "IPC contract route와 station plan만 정리해.",
            "stream": False,
            "mode": "fast",
            "audience": "admin",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
            "lore_topics": ["IPC"],
        },
        headers={"x-user": username},
    )

    assert response.status_code == 200
    body = response.json()
    user_prompt = llm.chat_calls[-1][1]["content"]
    assert "Interastral Peace Corporation" in user_prompt
    assert "Tea Ceremony" not in user_prompt
    assert "station plan" in user_prompt.lower()
    assert body["retrieval_debug"]["lore_scores"]
    assert body["retrieval_debug"]["memory_scores"]
