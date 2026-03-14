from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import mellow_chat_runtime.app_state as app_state
import mellow_chat_runtime.core.domain_lookup_store as domain_lookup_store_module
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.orchestrator import Orchestrator
from mellow_chat_runtime.infra.vector_index_store import JsonVectorIndexStore
from mellow_chat_runtime.routers.admin import router as admin_router
from mellow_chat_runtime.routers.chat import router as chat_router
from mellow_chat_runtime.services.vector_retrieval_service import VectorRetrievalService


class StubLLM:
    def get_model_for_mode(self, mode: str) -> str:
        return "qwen3.5:9b"


def _reset_domain_store() -> None:
    domain_lookup_store_module._global_store = None


def _build_client(tmp_path: Path) -> tuple[TestClient, object, VectorRetrievalService, Path]:
    _reset_domain_store()
    domain_file = tmp_path / "domain_data.json"
    vector_index_file = tmp_path / "vector_index.json"
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    dispatcher = DomainLookupDispatcher(store)
    orchestrator = Orchestrator(lookup_dispatcher=dispatcher)
    orchestrator.register_service("llm", StubLLM())
    vector_service = VectorRetrievalService(domain_store=store, index_path=vector_index_file)

    app_state.settings = SimpleNamespace(
        domain_data_file=domain_file,
        vector_index_file=vector_index_file,
        memory_promotion_enabled=False,
        memory_promotion_max_items=20,
    )
    app_state.orchestrator = orchestrator
    app_state.llm_service = orchestrator.llm_service
    app_state.vector_retrieval_service = vector_service

    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(admin_router)
    return TestClient(app), store, vector_service, vector_index_file


def test_lore_reindex_smoke(tmp_path: Path) -> None:
    # arrange
    client, store, vector_service, vector_index_file = _build_client(tmp_path)
    lore_id = "lore_001"
    updated_content = "A galaxy-scale organization handling finance, contracts, and aurora vault ledgers."

    # act
    update_response = client.put(
        f"/admin/lore/{lore_id}",
        json={
            "data": {
                "topic": "Interastral Peace Corporation",
                "aliases": ["IPC", "aurora vault"],
                "content": updated_content,
                "priority": 10,
            }
        },
    )
    dirty_item = store.get_section_item("lorebook", lore_id)

    reindex_response = client.post("/admin/vector/reindex")
    ready_item = store.get_section_item("lorebook", lore_id)
    index_entries = JsonVectorIndexStore(vector_index_file).list_entries("lore_entries")
    retrieval = vector_service.search_lore("aurora vault ledgers", lore_topics=["IPC"])

    # assert
    assert update_response.status_code == 200
    assert dirty_item["embedding_status"] == "dirty"

    assert reindex_response.status_code == 200
    assert reindex_response.json()["success"] is True
    assert reindex_response.json()["indexed"]["lore_entries"] >= 1

    # Current runtime marks successful reindex completion as clean.
    assert ready_item["embedding_status"] == "clean"
    assert any(entry["id"] == lore_id and "aurora vault ledgers" in entry["summary_text"] for entry in index_entries)

    assert retrieval.source == "vector"
    assert any(item["id"] == lore_id for item in retrieval.items)
