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


def _build_client(tmp_path: Path, llm: RecordingLLM) -> tuple[TestClient, VectorRetrievalService]:
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
    return TestClient(app), vector_service


def test_vector_retrieval_injects_relevant_context_without_unrelated_noise(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM('어벤츄린은 짧게 상황을 정리한다.\n\n"이번 건은 내가 기억하고 있어."')
    client, vector_service = _build_client(tmp_path, llm)
    vector_service.reindex()
    username = f"user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/chat/ask",
        json={
            "question": "IPC 규약과 station negotiation 이야기를 기억해?",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
            "lore_topics": ["IPC"],
        },
        headers={"x-user": username},
    )

    assert response.status_code == 200
    user_prompt = llm.chat_calls[-1][1]["content"]
    assert "relationship summaries" in user_prompt
    assert "important memories" in user_prompt
    assert "lore support" in user_prompt
    assert "Interastral Peace Corporation" in user_prompt
    assert "Debt negotiation at station nine" in user_prompt
    assert "Public oath in Penacony assembly hall" not in user_prompt


def test_vector_retrieval_falls_back_to_canonical_and_reports_debug(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM('어벤츄린은 차분히 고개를 든다.\n\n"기존 기억을 기준으로 답할게."')
    client, _ = _build_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/chat/ask",
        json={
            "question": "대화 기록 기준으로 답해.",
            "stream": False,
            "mode": "fast",
            "audience": "admin",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
            "lore_topic": "IPC",
        },
        headers={"x-user": username},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_debug"]["fallback_used"] is True
    assert body["retrieval_debug"]["lore_source"] == "canonical"
    assert body["retrieval_debug"]["query"]
    assert body["retrieval_debug"]["lore_hit_ids"]
    assert body["retrieval_debug"]["lore_scores"] == {}
    assert body["retrieval_debug"]["memory_scores"] == {}
    assert body["retrieval_debug"]["relationship_scores"] == {}
    assert body["retrieval_debug"]["errors"] == []


def test_admin_dirty_mark_and_reindex_flow(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM('선데이는 조용히 시선을 든다.\n\n"재색인이 끝났어."')
    client, vector_service = _build_client(tmp_path, llm)

    update = client.put(
        "/admin/lore/lore_001",
        json={
            "data": {
                "topic": "Interastral Peace Corporation",
                "aliases": ["IPC"],
                "content": "A galaxy-scale finance and contracts organization with strict lounge protocol.",
                "priority": 10,
            }
        },
    )
    assert update.status_code == 200
    assert update.json()["item"]["embedding_status"] == "dirty"

    reindex = client.post("/admin/vector/reindex")
    assert reindex.status_code == 200
    assert reindex.json()["indexed"]["lore_entries"] >= 1

    refreshed = vector_service.search_lore("strict lounge protocol", lore_topics=["IPC"])
    assert refreshed.source == "vector"
    assert refreshed.items[0]["topic"] == "Interastral Peace Corporation"
