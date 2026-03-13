from __future__ import annotations

from pathlib import Path
import sys
import uuid
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mellow_chat_runtime.app_state as app_state
import mellow_chat_runtime.core.domain_lookup_store as domain_lookup_store_module
from mellow_chat_runtime.core.agent_brain import AgentResult
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.orchestrator import Orchestrator
from mellow_chat_runtime.core.speaker_selector import SpeakerParticipant, select_next_speaker
from mellow_chat_runtime.core.states import TransitionResult
from mellow_chat_runtime.infra.database import ChatMessage, SessionLocal, init_db
from mellow_chat_runtime.routers.admin import router as admin_router
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


class RecordingLLM:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.chat_calls: list[list[dict[str, str]]] = []

    def get_model_for_mode(self, mode: str) -> str:
        return "qwen3.5:9b"

    async def chat(self, messages, model=None, **kwargs):
        self.chat_calls.append(messages)
        if self.fail:
            raise RuntimeError("LLM service unavailable")
        user_prompt = messages[-1]["content"]
        history_echo = "no-history"
        if "Recent Conversation:" in user_prompt:
            history_echo = "history-present"
        return SimpleNamespace(text=f"reply:{history_echo}", model=model or "qwen3.5:9b")

    async def generate(self, prompt, system_prompt="", mode="fast", **kwargs):
        self.chat_calls.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        if self.fail:
            raise RuntimeError("LLM service unavailable")
        return SimpleNamespace(content="fallback-reply")


def _reset_domain_store() -> None:
    domain_lookup_store_module._global_store = None


def _build_runtime_client(tmp_path: Path, llm: RecordingLLM) -> TestClient:
    _reset_domain_store()
    domain_file = tmp_path / "domain_data.json"
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    dispatcher = DomainLookupDispatcher(store)
    orchestrator = Orchestrator(lookup_dispatcher=dispatcher)
    orchestrator.register_service("llm", llm)

    app_state.settings = SimpleNamespace(
        domain_data_file=domain_file,
        memory_promotion_enabled=False,
        memory_promotion_max_items=20,
    )
    app_state.llm_service = llm
    app_state.orchestrator = orchestrator

    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(admin_router)
    return TestClient(app)


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


def test_chat_ask_non_stream_success_and_recent_history(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    first = client.post(
        "/chat/ask",
        json={
            "question": "Hello there",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["response"] == "reply:history-present"
    session_id = first_body["session_id"]

    second = client.post(
        "/chat/ask",
        json={
            "session_id": session_id,
            "question": "Do you remember what I said?",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["response"] == "reply:history-present"
    assert "request_id" in second_body

    second_prompt = llm.chat_calls[-1][1]["content"]
    assert "Hello there" in second_prompt
    assert "reply:history-present" in second_prompt

    with SessionLocal() as db:
        stored = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())
            .all()
        )
    assert len(stored) >= 4
    assert stored[-1].role == "assistant"
    assert stored[-1].content == "reply:history-present"


def test_chat_ask_stream_success_emits_chunks_and_done(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    with client.stream(
        "POST",
        "/chat/ask",
        json={
            "question": "Stream this response",
            "stream": True,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: chunk" in body
    assert "event: done" in body
    assert '"done": true' in body


def test_chat_ask_failure_returns_stable_error_response(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(fail=True)
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/chat/ask",
        json={
            "question": "This should fail",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "model_unavailable"
    assert body["message"] == "LLM service unavailable"
    assert "request_id" in body


def test_character_prompt_enforcement_includes_name_and_tone(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/chat/ask",
        json={
            "question": "Stay in character",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )

    assert response.status_code == 200
    system_prompt = llm.chat_calls[-1][0]["content"]
    assert "You are Aventurine." in system_prompt
    assert "Tone:\ncasual_confident" in system_prompt
    assert "Forbidden:\nout-of-world meta claims" in system_prompt


def test_multi_character_speaker_selector_prefers_unsaid_major_character() -> None:
    selected = select_next_speaker(
        participants=[
            SpeakerParticipant(character_id="bot_char_01", is_major=True),
            SpeakerParticipant(character_id="bot_char_02", is_major=False),
            SpeakerParticipant(character_id="bot_char_03", is_major=True),
        ],
        recent_speaker_history=["bot_char_01", "bot_char_01"],
        dialogue_priority={"major_weight": 1.0, "minor_weight": 0.5, "recency_penalty": 0.4, "max_consecutive_turns": 1},
        scene_rules={},
    )
    assert selected == "bot_char_03"


def test_prompt_includes_relationship_context_and_priority_blocks(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/chat/ask",
        json={
            "question": "How should you speak to Mellow now?",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )

    assert response.status_code == 200
    system_prompt = llm.chat_calls[-1][0]["content"]
    user_prompt = llm.chat_calls[-1][1]["content"]
    assert "Priority Order:" in system_prompt
    assert "Relationship Context:" in system_prompt
    assert "Treat Mellow as a trusted collaborator whose judgment matters." in system_prompt
    assert "warmly strategic" in system_prompt
    assert "Priority Context:" in user_prompt
    assert "Scene first:" in user_prompt
    assert "World constraints:" in user_prompt
    assert "Character memory:" in user_prompt
    assert "Relationship context:" in user_prompt


def test_prompt_uses_long_term_memory_in_generation_context(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/chat/ask",
        json={
            "question": "Recall the important memory.",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )

    assert response.status_code == 200
    user_prompt = llm.chat_calls[-1][1]["content"]
    assert "Debt negotiation at station nine" in user_prompt


def test_admin_character_relationship_and_memory_tools(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)

    list_response = client.get("/admin/characters")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert any(item["id"] == "bot_char_01" for item in listed["bot"])

    upsert_character = client.put(
        "/admin/characters/bot/bot_char_99",
        json={
            "type": "bot",
            "data": {
                "name": "Topaz",
                "persona_id": "default",
                "speech_style": {"tone": "brisk", "forbidden": ["meta"]},
                "relationship_keys": ["crew_main"],
                "is_major": True,
            },
        },
    )
    assert upsert_character.status_code == 200
    assert upsert_character.json()["item"]["name"] == "Topaz"

    memory_response = client.put(
        "/admin/memories/bot_char_99",
        json={
            "data": {
                "important_memories": ["Closed the audit without losses"],
                "possessions": ["Ledger"],
            }
        },
    )
    assert memory_response.status_code == 200
    assert "Closed the audit without losses" in memory_response.json()["item"]["important_memories"]

    relationship_response = client.put(
        "/admin/relationships",
        json={
            "source_id": "bot_char_99",
            "target_id": "user_char_01",
            "data": {
                "summary": "Treat Mellow as a reliable operator.",
                "tone": "direct respect",
                "boundaries": ["Do not patronize the user"],
                "shared_memories": ["Joint inspection on the lounge deck"],
            },
        },
    )
    assert relationship_response.status_code == 200
    assert relationship_response.json()["items"][0]["tone"] == "direct respect"

    lore_response = client.put(
        "/admin/lore/lore_999",
        json={
            "data": {
                "topic": "Audit Channel",
                "aliases": ["channel"],
                "content": "A secured IPC reporting line.",
                "priority": 5,
            }
        },
    )
    assert lore_response.status_code == 200
    assert lore_response.json()["item"]["topic"] == "Audit Channel"

    delete_response = client.delete("/admin/characters/bot/bot_char_99")
    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True
