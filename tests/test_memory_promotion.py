from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mellow_chat_runtime.app_state as app_state
import mellow_chat_runtime.core.domain_lookup_store as domain_lookup_store_module
from mellow_chat_runtime.core.agent_brain import AgentResult
from mellow_chat_runtime.core.states import TransitionResult
from mellow_chat_runtime.infra.database import init_db
from mellow_chat_runtime.routers.chat import router as chat_router
from mellow_chat_runtime.services.memory_promotion_service import MemoryPromotionService


class FakeLLM:
    def get_model_for_mode(self, mode: str) -> str:
        return "qwen3.5:9b"


class FakeOrchestrator:
    async def request_state_change(self, target_state, reason: str = ""):
        return TransitionResult.SUCCESS

    async def run_agent(self, **kwargs):
        return AgentResult(answer="This is important: we decided to meet again tomorrow.")


def _reset_domain_store() -> None:
    domain_lookup_store_module._global_store = None


def test_memory_promotion_service_promotes_keyword_sentences(tmp_path: Path) -> None:
    _reset_domain_store()
    domain_file = tmp_path / "domain_data.json"
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    service = MemoryPromotionService(domain_store=store, max_items=20)

    promoted = service.promote_from_text("user_char_01", "This is important: keep the station meeting plan.")

    assert promoted == ["This is important: keep the station meeting plan."]
    memory_state = store.get_memory_and_possessions("user_char_01")
    assert "This is important: keep the station meeting plan." in memory_state["important_memories"]


def test_chat_ask_promotes_user_and_bot_memories(tmp_path: Path) -> None:
    _reset_domain_store()
    domain_file = tmp_path / "domain_data.json"
    init_db()

    app_state.settings = SimpleNamespace(
        domain_data_file=domain_file,
        memory_promotion_enabled=True,
        memory_promotion_max_items=20,
    )
    app_state.llm_service = FakeLLM()
    app_state.orchestrator = FakeOrchestrator()

    app = FastAPI()
    app.include_router(chat_router)
    client = TestClient(app)

    session_response = client.post(
        "/chat/ask",
        json={
            "question": "Important: remember the station meeting plan.",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": "memory_test_user"},
    )

    assert session_response.status_code == 200

    _reset_domain_store()
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    user_memory = store.get_memory_and_possessions("user_char_01")
    bot_memory = store.get_memory_and_possessions("bot_char_01")

    assert any("remember the station meeting plan" in item.lower() for item in user_memory["important_memories"])
    assert any("we decided to meet again tomorrow" in item.lower() for item in bot_memory["important_memories"])

