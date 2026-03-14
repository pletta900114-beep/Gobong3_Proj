from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import uuid
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mellow_chat_runtime.app_state as app_state
import mellow_chat_runtime.core.domain_lookup_store as domain_lookup_store_module
from mellow_chat_runtime.core.agent_brain import AgentBrain, AgentResult
from mellow_chat_runtime.core.domain_lookup_dispatcher import DomainLookupDispatcher
from mellow_chat_runtime.core.orchestrator import Orchestrator
from mellow_chat_runtime.core.rp_parser import parse_scene_event
from mellow_chat_runtime.core.speaker_relevance import build_speaker_relevance
from mellow_chat_runtime.core.speaker_selector import SpeakerParticipant, select_next_speaker
from mellow_chat_runtime.core.states import TransitionResult
from mellow_chat_runtime.core.text_sanitizer import sanitize_assistant_text
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
    def __init__(
        self,
        fail: bool = False,
        response_text: str | None = None,
        response_texts: list[str] | None = None,
        response_payloads: list[dict[str, str]] | None = None,
    ) -> None:
        self.fail = fail
        self.response_text = response_text
        self.response_texts = list(response_texts or [])
        self.response_payloads = list(response_payloads or [])
        self.chat_calls: list[list[dict[str, str]]] = []
        self.chat_call_options: list[dict[str, object]] = []

    def get_model_for_mode(self, mode: str) -> str:
        return "qwen3.5:9b"

    def _next_response(self, default_text: str) -> tuple[str, str]:
        if self.response_payloads:
            payload = self.response_payloads.pop(0)
            return str(payload.get("text", "")), str(payload.get("thinking", ""))
        if self.response_texts:
            return self.response_texts.pop(0), ""
        if self.response_text is not None:
            return self.response_text, ""
        return default_text, ""

    async def chat(self, messages, model=None, **kwargs):
        self.chat_calls.append(messages)
        self.chat_call_options.append(dict(kwargs))
        if self.fail:
            raise RuntimeError("LLM service unavailable")
        user_prompt = messages[-1]["content"]
        history_echo = "no-history"
        if "Recent Conversation:" in user_prompt or "최근 대화:" in user_prompt:
            history_echo = "history-present"
        default_text = f'그는 숨을 골라 상황을 정리한다.\n\n"reply:{history_echo}"'
        text, thinking = self._next_response(default_text)
        return SimpleNamespace(text=text, thinking=thinking, model=model or "qwen3.5:9b")

    async def generate(self, prompt, system_prompt="", mode="fast", **kwargs):
        self.chat_calls.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        if self.fail:
            raise RuntimeError("LLM service unavailable")
        return SimpleNamespace(content='그는 짧게 숨을 고른다.\n\n"fallback-reply"')


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
    assert "reply:history-present" in first_body["response"]
    assert "\"reply:history-present\"" in first_body["response"]
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
    assert "reply:history-present" in second_body["response"]
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
    assert "reply:history-present" in stored[-1].content


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
    assert "당신은 Aventurine이다." in system_prompt
    assert "말투:\ncasual_confident" in system_prompt
    assert "금지 요소:\nout-of-world meta claims" in system_prompt
    assert "사용자 입력이 한국어이면 서술과 대사를 모두 한국어로 유지하고 영어로 전환하지 않는다" in system_prompt


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
    assert "우선순위:" in system_prompt
    assert "관계 맥락:" in system_prompt
    assert "Treat Mellow as a trusted collaborator whose judgment matters." in system_prompt
    assert "warmly strategic" in system_prompt
    assert "출력 제약:" in user_prompt
    assert "우선 맥락:" in user_prompt
    assert "장면 우선:" in user_prompt
    assert "세계 제약:" in user_prompt
    assert "캐릭터 기억:" in user_prompt
    assert "관계 맥락:" in user_prompt


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


def test_sanitize_assistant_output_removes_hidden_reasoning_tokens() -> None:
    contaminated = (
        "Final answer line.\n"
        "<|endoftext|><|im_start|>assistant\n"
        "<think>\ninternal reasoning\n</think>\n"
        "assistant\nMore draft text"
    )
    cleaned = sanitize_assistant_text(contaminated)
    assert cleaned == "Final answer line."


def test_chat_ask_stores_only_sanitized_assistant_output(tmp_path: Path) -> None:
    init_db()
    contaminated = (
        "어벤츄린은 고개를 기울이며 상황을 차분히 정리한다.\n\n\"말로우, 안전하게 진행하자.\"\n"
        "<|endoftext|><|im_start|>assistant\n"
        "<think>\nprivate draft\n</think>\n"
        "assistant"
    )
    llm = RecordingLLM(response_text=contaminated)
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/chat/ask",
        json={
            "question": "오염 없는 답변만 줘.",
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
    body = response.json()
    assert "<|im_start|>" not in body["response"]
    assert "<think>" not in body["response"]
    assert body["response"]

    with SessionLocal() as db:
        stored = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == body["session_id"], ChatMessage.role == "assistant")
            .order_by(ChatMessage.id.desc())
            .first()
        )
    assert stored is not None
    assert stored.content == body["response"]
    assert "<|im_start|>" not in stored.content


def test_chat_ask_uses_sanitized_history_when_session_is_contaminated(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    first = client.post(
        "/chat/ask",
        json={
            "question": "첫 질문",
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
    session_id = first.json()["session_id"]

    with SessionLocal() as db:
        contaminated = db.query(ChatMessage).filter(ChatMessage.session_id == session_id, ChatMessage.role == "assistant").first()
        assert contaminated is not None
        contaminated.content = (
            "정상 앞부분\n<|endoftext|><|im_start|>assistant\n<think>\ninternal\n</think>\nuser\nbad tail"
        )
        db.commit()

    second = client.post(
        "/chat/ask",
        json={
            "session_id": session_id,
            "question": "이전 흐름을 이어줘",
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

    user_prompt = llm.chat_calls[-1][1]["content"]
    assert "정상 앞부분" in user_prompt
    assert "<|im_start|>" not in user_prompt
    assert "<think>" not in user_prompt
    assert "bad tail" not in user_prompt


def test_new_session_does_not_inherit_contaminated_history_from_other_session(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f"user_{uuid.uuid4().hex[:8]}"

    contaminated_session = client.post(
        "/chat/ask",
        json={
            "question": "오염 세션 시작",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )
    assert contaminated_session.status_code == 200
    contaminated_session_id = contaminated_session.json()["session_id"]

    with SessionLocal() as db:
        contaminated = db.query(ChatMessage).filter(ChatMessage.session_id == contaminated_session_id, ChatMessage.role == "assistant").first()
        assert contaminated is not None
        contaminated.content = "dirty\n<|im_start|>assistant\n<think>bad</think>"
        db.commit()

    fresh_session = client.post(
        "/chat/ask",
        json={
            "question": "신규 세션 테스트",
            "stream": False,
            "mode": "fast",
            "user_profile_id": "user_char_01",
            "character_id": "bot_char_01",
            "scene_id": "scene_default",
            "world_id": "default",
        },
        headers={"x-user": username},
    )
    assert fresh_session.status_code == 200

    user_prompt = llm.chat_calls[-1][1]["content"]
    assert "dirty" not in user_prompt
    assert "<|im_start|>" not in user_prompt



def test_parse_scene_event_narration_only() -> None:
    parsed = parse_scene_event('나는 조용히 소파에 앉아 숨을 골랐다.', [])
    assert parsed.input_mode == 'narration_only'
    assert parsed.user_narration == '나는 조용히 소파에 앉아 숨을 골랐다.'
    assert parsed.user_dialogue == ''
    assert parsed.target_character_hint is None


def test_parse_scene_event_dialogue_only_and_direct_target_hint(tmp_path: Path) -> None:
    _reset_domain_store()
    domain_file = tmp_path / 'domain_data.json'
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    parsed = parse_scene_event('"선데이, 그 말 진심이야?"', list(store.list_section('bot_characters').values()))
    assert parsed.input_mode == 'dialogue_only'
    assert parsed.user_dialogue == '선데이, 그 말 진심이야?'
    assert parsed.target_character_hint == 'bot_char_02'


def test_parse_scene_event_mixed_and_narration_target_hint(tmp_path: Path) -> None:
    _reset_domain_store()
    domain_file = tmp_path / 'domain_data.json'
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    parsed = parse_scene_event(
        '나는 어벤츄린 쪽으로 몸을 기울였다.\n"이번 판, 네가 보기엔 어때?"',
        list(store.list_section('bot_characters').values()),
    )
    assert parsed.input_mode == 'mixed'
    assert parsed.user_narration == '나는 어벤츄린 쪽으로 몸을 기울였다.'
    assert parsed.user_dialogue == '이번 판, 네가 보기엔 어때?'
    assert parsed.target_character_hint == 'bot_char_01'


def test_selector_prefers_target_hint_when_eligible() -> None:
    selected = select_next_speaker(
        participants=[
            SpeakerParticipant(character_id='bot_char_01', is_major=True),
            SpeakerParticipant(character_id='bot_char_02', is_major=False),
        ],
        recent_speaker_history=['bot_char_01'],
        dialogue_priority={'major_weight': 1.0, 'minor_weight': 0.5, 'recency_penalty': 0.4, 'max_consecutive_turns': 1},
        scene_rules={},
        target_character_hint='bot_char_02',
    )
    assert selected == 'bot_char_02'


def test_selector_scene_rules_override_target_hint() -> None:
    selected = select_next_speaker(
        participants=[
            SpeakerParticipant(character_id='bot_char_01', is_major=True),
            SpeakerParticipant(character_id='bot_char_02', is_major=False),
        ],
        recent_speaker_history=['bot_char_01'],
        dialogue_priority={'major_weight': 1.0, 'minor_weight': 0.5, 'recency_penalty': 0.4, 'max_consecutive_turns': 1},
        scene_rules={'force_speaker_id': 'bot_char_01'},
        target_character_hint='bot_char_02',
    )
    assert selected == 'bot_char_01'


def test_prompt_includes_parsed_scene_event_block(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 어벤츄린 쪽으로 몸을 기울였다.\n"이번 판, 네가 보기엔 어때?"',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    user_prompt = llm.chat_calls[-1][1]['content']
    assert '파싱된 사용자 장면 이벤트:' in user_prompt
    assert '사용자 서술: 나는 어벤츄린 쪽으로 몸을 기울였다.' in user_prompt
    assert '사용자 대사: 이번 판, 네가 보기엔 어때?' in user_prompt
    assert '대상 힌트: bot_char_01' in user_prompt


def test_chat_ask_non_stream_rp_output_and_target_selection(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_text='선데이는 잠시 침묵한 채 시선을 맞춘다.\n\n"그래. 지금은 거짓 없이 말하고 있어."')
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 그 말 진심이야?"',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['speaker_id'] == 'bot_char_02'
    assert '"그래. 지금은 거짓 없이 말하고 있어."' in body['response']
    assert 'rp' not in body['used_context']


def test_chat_ask_stream_rp_output_success(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_text='어벤츄린은 손끝에서 카드를 뒤집어 보이며 미소 짓는다.\n\n"계산은 이미 끝났지, 친구."')
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    with client.stream(
        'POST',
        '/chat/ask',
        json={
            'question': '나는 어벤츄린 쪽으로 몸을 기울이며 낮게 웃었다.\n"이번 판, 네 계산은 어디까지 끝났어?"',
            'stream': True,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    ) as response:
        assert response.status_code == 200
        body = ''.join(response.iter_text())

    assert 'event: chunk' in body
    assert 'event: done' in body
    assert '<think>' not in body
    assert 'assistant:' not in body


def test_chat_ask_filters_contaminated_memory_from_prompt(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '기억을 바탕으로 말해줘.',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_id': 'bot_char_01',
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    user_prompt = llm.chat_calls[-1][1]['content']
    assert 'Debt negotiation at station nine' in user_prompt
    assert '<|im_start|>' not in user_prompt
    assert 'prompt in Turn' not in user_prompt
    assert "I'll focus on" not in user_prompt


def test_chat_ask_repairs_invalid_first_output_into_rp(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_texts=[
        '요청하신 대로 차분하고 정중한 말투로 답변드리겠습니다.',
        '선데이는 조용히 숨을 내쉰다.\n\n"그래. 지금은 그렇게 말할 수 있어."',
    ])
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '"선데이, 그 말 진심이야?"',
            'audience': 'admin',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert '"그래. 지금은 그렇게 말할 수 있어."' in body['response']
    assert '요청하신 대로' not in body['response']
    assert len(llm.chat_calls) >= 2



def test_build_speaker_relevance_direct_mention_conflict_prefers_dialogue_target(tmp_path: Path) -> None:
    _reset_domain_store()
    domain_file = tmp_path / 'domain_data.json'
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    characters = list(store.list_section('bot_characters').values())
    parsed = parse_scene_event(
        '나는 어벤츄린 쪽으로 몸을 기울였지만 시선은 선데이에게 옮겼다.\n"선데이, 네 생각은?"',
        characters,
    )
    relevance = build_speaker_relevance(parsed, characters, scene_state=store.get_scene_state('scene_default'))
    assert relevance['bot_char_02'] > relevance.get('bot_char_01', 0.0)

    selected = select_next_speaker(
        participants=[
            SpeakerParticipant(character_id='bot_char_01', is_major=True),
            SpeakerParticipant(character_id='bot_char_02', is_major=False),
        ],
        recent_speaker_history=['bot_char_01'],
        dialogue_priority={'major_weight': 1.0, 'minor_weight': 0.5, 'recency_penalty': 0.4, 'max_consecutive_turns': 1},
        scene_rules={},
        target_character_hint=parsed.target_character_hint,
        speaker_relevance=relevance,
    )
    assert selected == 'bot_char_02'


def test_build_speaker_relevance_false_relevance_guard(tmp_path: Path) -> None:
    _reset_domain_store()
    domain_file = tmp_path / 'domain_data.json'
    store = domain_lookup_store_module.get_domain_store(data_path=domain_file)
    characters = list(store.list_section('bot_characters').values())
    parsed = parse_scene_event('어벤츄린 때와는 다르게 이번엔 조용히 정리하고 싶어.', characters)
    relevance = build_speaker_relevance(parsed, characters, scene_state=store.get_scene_state('scene_default'))
    assert relevance == {}


def test_chat_ask_hides_rp_used_context_by_default(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_text='선데이는 짧게 숨을 고른다.\n\n"네 생각을 듣고 있어."')
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 네 생각은?"',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    assert 'rp' not in response.json()['used_context']



def test_chat_ask_thinking_only_response_uses_final_answer_retry(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_payloads=[
        {"text": "", "thinking": "I should answer in character."},
        {"text": "선데이는 잠시 시선을 내렸다가 다시 마주 본다.\n\n\"그래. 적어도 지금은 거짓 없이 말하고 있어.\"", "thinking": ""},
    ])
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 그 말 진심이야?"',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert '"그래. 적어도 지금은 거짓 없이 말하고 있어."' in body['response']
    assert '서두르지 말자' not in body['response']
    assert len(llm.chat_calls) == 2
    assert llm.chat_call_options[0]['options']['stop'] == ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '</think>', '<think>']
    assert llm.chat_call_options[0]['think'] is False
    assert llm.chat_call_options[1]['options']['stop'] == ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '</think>', '<think>']
    assert llm.chat_call_options[1]['options']['num_predict'] == 180
    assert llm.chat_call_options[1]['think'] is False


def test_chat_ask_thinking_only_retry_stops_after_single_repair_then_fallback(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_payloads=[
        {"text": "", "thinking": "I should answer in character."},
        {"text": "", "thinking": "Still thinking."},
    ])
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 잠깐 숨을 멈추고 선데이를 바라봤다.\n"정말 솔직하게 말해줘."',
            'audience': 'admin',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['rp_debug']['fallback_used'] is True
    assert body['rp_debug']['final_verdict'] == 'FAIL'
    assert len(llm.chat_calls) == 2
    assert llm.chat_call_options[1]['options']['num_predict'] == 180
    assert llm.chat_call_options[1]['think'] is False


def test_chat_ask_repairs_english_drift_in_korean_session(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_texts=[
        'He watches you in silence.\n\n"Yes. I meant it."',
        '선데이는 숨을 고른 뒤 조용히 시선을 맞춘다.\n\n"그래. 그 말은 진심이야."',
    ])
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 그 말 진심이야?"',
            'audience': 'admin',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_id': 'bot_char_02',
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert '선데이는' in body['response']
    assert '"그래. 그 말은 진심이야."' in body['response']
    assert 'He watches you in silence.' not in body['response']
    assert len(llm.chat_calls) == 2


def test_retry_path_repairs_english_drift_in_korean_session(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_payloads=[
        {"text": "", "thinking": "I should answer in character."},
        {"text": "선데이는 시선을 잠시 내렸다가 다시 들어 올린다.\n\n\"그래. 그 말은 진심이야.\"", "thinking": ""},
    ])
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 조용히 숨을 골랐다.\n"선데이, 이번엔 피하지 마."',
            'audience': 'admin',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_ids': ['bot_char_01', 'bot_char_02'],
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert '선데이는' in body['response']
    assert '"그래. 그 말은 진심이야."' in body['response']
    assert len(llm.chat_calls) == 2
    assert llm.chat_call_options[1]['options']['stop'] == ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '</think>', '<think>']


def test_validate_rp_output_rejects_first_person_narration() -> None:
    brain = AgentBrain(llm_service=None, lookup_dispatcher=None)

    ok, reasons = brain._validate_rp_output(
        '나는 천천히 그를 바라봤다.\n\n"그래."',
        active_character={'name': '선데이'},
        expected_language='ko',
    )

    assert not ok
    assert 'narration_not_third_person' in reasons

    ok, reasons = brain._validate_rp_output(
        '선데이는 천천히 상대를 바라봤다.\n\n"그래."',
        active_character={'name': '선데이'},
        expected_language='ko',
    )

    assert ok
    assert reasons == []


def test_validate_rp_output_catches_english_drift_for_korean_input() -> None:
    brain = AgentBrain(llm_service=None, lookup_dispatcher=None)

    ok, reasons = brain._validate_rp_output(
        'He looks at you in silence.\n\n"I meant it."',
        active_character={'name': '선데이'},
        expected_language='ko',
    )

    assert not ok
    assert 'language_drift' in reasons


def test_chat_ask_user_audience_returns_validation_failure_without_fallback(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_text='나는 천천히 시선을 들었다.\n\n"그래."')
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 그 말 진심이야?"',
            'audience': 'user',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_id': 'bot_char_02',
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 422
    body = response.json()
    assert body['success'] is False
    assert body['error_code'] == 'POV_RULE_FAILED'
    assert 'fallback' not in body['message'].lower()


def test_chat_ask_user_audience_returns_normal_response_on_valid_output(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_text='선데이는 천천히 시선을 들어 상대를 마주 본다.\n\n"그래. 그 말은 진심이야."')
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 그 말 진심이야?"',
            'audience': 'user',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_id': 'bot_char_02',
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['response'].startswith('선데이는')
    assert 'rp_debug' not in body


def test_chat_ask_admin_audience_keeps_fallback_and_marks_fail(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM(response_texts=[
        'He watches you in silence.\n\n"Yes. I meant it."',
        '나는 시선을 잠시 내렸다.\n\n"그래."',
    ])
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '나는 천천히 고개를 들고 선데이를 바라봤다.\n"선데이, 그 말 진심이야?"',
            'audience': 'admin',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_id': 'bot_char_02',
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    body = response.json()
    assert body['rp_debug']['fallback_used'] is True
    assert body['rp_debug']['final_verdict'] == 'FAIL'
    assert body['rp_debug']['validator_passed'] is False


def test_rp_qa_report_json_and_markdown_share_final_verdict() -> None:
    spec = importlib.util.spec_from_file_location('rp_qa_smoke', Path(r'D:\Gobong3_Proj\scripts\rp_qa_smoke.py'))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    result = module.ScenarioResult(
        generated_at='2026-03-14T03:31:24',
        run_id='run_20260314_033124_01',
        name='sunday_sincerity',
        ok=False,
        validator_passed=False,
        final_verdict='FAIL',
        failure_reason='fallback_used',
        speaker_id='bot_char_02',
        input_language='ko',
        output_language='ko',
        quoted_dialogue=True,
        third_person=True,
        english_drift=False,
        fallback_used=True,
        retry_used=True,
        processing_time_ms=46100,
        reasons=['fallback_used'],
        response_preview='그는 잠시 숨을 고르며 상대의 눈을 똑바로 바라본다.\n\n"지금은 서두르지 말자. 이 장면 안에서 차분히 이어가면 돼."',
    )
    payload = module._build_result_payload([result])
    markdown = module._render_markdown(payload)

    assert payload['results'][0]['final_verdict'] == 'FAIL'
    assert '- final_verdict: FAIL' in markdown
    assert 'generated_at' in markdown
    assert 'run_id' in markdown


def test_character_prompt_includes_style_and_franchise_anchors(tmp_path: Path) -> None:
    init_db()
    llm = RecordingLLM()
    client = _build_runtime_client(tmp_path, llm)
    username = f'user_{uuid.uuid4().hex[:8]}'

    response = client.post(
        '/chat/ask',
        json={
            'question': '선데이, 그 말 진심이야?',
            'stream': False,
            'mode': 'fast',
            'user_profile_id': 'user_char_01',
            'character_id': 'bot_char_02',
            'scene_id': 'scene_default',
            'world_id': 'default',
        },
        headers={'x-user': username},
    )

    assert response.status_code == 200
    system_prompt = llm.chat_calls[-1][0]['content']
    assert '표기 앵커:' in system_prompt
    assert '스타일 앵커:' in system_prompt
    assert '프랜차이즈 앵커:' in system_prompt
    assert 'Arknights' in system_prompt or 'Genshin' in system_prompt


def test_sanitize_assistant_text_salvages_json_and_tokens() -> None:
    contaminated = '```json\n{"action": "선데이는 숨을 고른다.", "dialogue": "그래. 그 말은 진심이야."}\n```<|endoftext|><|im_start|>user'

    cleaned = sanitize_assistant_text(contaminated)

    assert cleaned == '선데이는 숨을 고른다.\n\n"그래. 그 말은 진심이야."'


def test_validate_rp_output_rejects_naege_and_uri_in_narration() -> None:
    brain = AgentBrain(llm_service=None, lookup_dispatcher=None)

    ok, reasons = brain._validate_rp_output(
        '선데이는 내게 시선을 두고 잠시 침묵했다.\n\n"그래."',
        active_character={'name': '선데이'},
        expected_language='ko',
    )
    assert not ok
    assert 'narration_not_third_person' in reasons

    ok, reasons = brain._validate_rp_output(
        '어벤츄린은 우리를 훑어보듯 미소 지었다.\n\n"흥미로운 판이네."',
        active_character={'name': '어벤츄린'},
        expected_language='ko',
    )
    assert not ok
    assert 'narration_not_third_person' in reasons


def test_fallback_output_is_character_aware() -> None:
    brain = AgentBrain(llm_service=None, lookup_dispatcher=None)

    sunday = brain._fallback_rp_output({'name': 'Sunday'})
    aventurine = brain._fallback_rp_output({'name': 'Aventurine'})

    assert '선데이는' in sunday
    assert '책임 있게 답하겠습니다' in sunday
    assert '어벤츄린은' in aventurine
    assert '이번 수는 끝까지 계산해 보자' in aventurine


def test_validate_rp_output_bans_first_person_only_in_narration() -> None:
    brain = AgentBrain(llm_service=None, lookup_dispatcher=None)

    ok, reasons = brain._validate_rp_output(
        '선데이는 내 시선을 마주하며 잠시 숨을 골랐다.\n\n"그래. 내가 그렇게 말한 이유는 분명해."',
        active_character={'name': '선데이'},
        expected_language='ko',
    )

    assert not ok
    assert 'narration_not_third_person' in reasons


def test_validate_rp_output_allows_first_person_in_dialogue() -> None:
    brain = AgentBrain(llm_service=None, lookup_dispatcher=None)

    ok, reasons = brain._validate_rp_output(
        '선데이는 상대를 가만히 바라보며 말을 고른다.\n\n"내가 그렇게 말한 이유는 분명해. 우리 둘 다 이미 알고 있잖아."',
        active_character={'name': '선데이'},
        expected_language='ko',
    )

    assert ok
    assert reasons == []
