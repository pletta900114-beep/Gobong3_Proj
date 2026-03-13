# mellow_chat_runtime

웹서비스 연동을 전제로 한 텍스트 캐릭터 챗봇 백엔드입니다.  
현재 범위는 최소 에이전트 루프 + 도메인 조회 + 세션 단위 모델 선택 + 세션 참가자 관리입니다.

## 현재 포함 기능
- `POST /chat/ask` (stream/non-stream)
- 세션 단위 모델 선택 API
- 세션 참가자(user/bot 캐릭터) API
- 도메인 데이터 조회(persona, lore, memory, scene, world, dialogue priority)
- 1:1 / 다자 대화를 위한 초기 `speaker_selector.py` 적용

## 프로젝트 구조
- `mellow_chat_runtime/`: FastAPI 앱 및 런타임 코드
- `mellow_chat_runtime_data/`: SQLite DB, 도메인 데이터/시드
- `service_backend_design_for_codex.md`: 설계 기준 문서

## 빠른 실행
1. Python 3.10+ 환경 준비
2. 필요 패키지 설치

```bash
pip install fastapi uvicorn sqlalchemy aiohttp pydantic pydantic-settings
```

3. 서버 실행

```bash
python -m mellow_chat_runtime.main
```

기본 주소: `http://127.0.0.1:8010`

## 환경 변수(.env) 예시
```env
API_HOST=127.0.0.1
API_PORT=8010
API_DEBUG=false

OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_TIMEOUT=60

FAST_MODEL=qwen3.5:9b
THINKING_MODEL=qwen3.5:9b
RESEARCH_MODEL=qwen3.5:9b

DOMAIN_LOOKUP_BACKEND=json
# DOMAIN_LOOKUP_BACKEND=vector
# VECTORDB_LORE_SEARCH_URL=http://localhost:9000/lore/search
# VECTORDB_TIMEOUT_SEC=2.0
```

## 주요 API

### Health / Runtime
- `GET /health`
- `GET /runtime/status`

### 모델 선택
- `POST /models/select`
- `GET /models/sessions/{session_id}`

요청 예시:
```json
{
  "session_id": 1,
  "selection": {
    "provider": "ollama",
    "model": "qwen3.5:9b",
    "mode": "fast"
  }
}
```

### 세션 참가자
- `POST /sessions/{session_id}/participants`
- `GET /sessions/{session_id}/participants`

요청 예시:
```json
{
  "user_character_ids": ["user_char_01"],
  "bot_character_ids": ["bot_char_01", "bot_char_02"]
}
```

### 채팅
- `POST /chat/ask`
- `GET /chat/sessions`
- `GET /chat/sessions/{session_id}/messages`

`/chat/ask` 요청 예시:
```json
{
  "session_id": 1,
  "question": "Who should speak next?",
  "stream": false,
  "mode": "fast",
  "persona_id": "default",
  "user_profile_id": "user_char_01",
  "character_ids": ["bot_char_01", "bot_char_02"],
  "scene_id": "scene_default",
  "world_id": "default",
  "lore_topics": ["Interastral Peace Corporation", "Stonehearts"]
}
```

`/chat/ask` 응답 메타데이터(비스트리밍 기준):
- `session_id`
- `message_id`
- `speaker_id`
- `speaker_type`
- `model_provider`
- `model_name`
- `selected_mode`
- `processing_time_ms`
- `used_context` (`persona_id`, `user_profile_id`, `character_ids`, `scene_id`, `world_id`, `lore_keys`)

## 데이터 파일
- SQLite: `mellow_chat_runtime_data/chatbot.db`
- 도메인 시드 예시: `mellow_chat_runtime_data/domain_data.seed.json`
- 런타임 도메인 데이터 기본 경로: `mellow_chat_runtime_data/domain_data.json`

## 메모리 저장
- 단기기억: 세션 메시지가 `mellow_chat_runtime_data/chatbot.db`의 `chat_messages`에 저장되고, `/chat/ask` 시 최근 대화 일부가 컨텍스트로 사용됩니다.
- 장기기억: 캐릭터별 메모리가 `domain_data.json`의 `memories` 섹션에 저장됩니다.
- 메모리 승격: `/chat/ask` 성공 후 현재 턴 텍스트에서 `important/remember/기억/중요/약속/결정/계획` 계열 문장이 감지되면 `important_memories`로 승격됩니다.

관련 설정:
- `MEMORY_PROMOTION_ENABLED=true`
- `MEMORY_PROMOTION_MAX_ITEMS=20`

## VectorDB 연동 (Phase 1)
현재 VectorDB 연동 범위는 `lorebook` 조회만입니다.  
`persona/user_profile/character/memory/scene/world/dialogue_priority`는 기존 JSON 조회를 그대로 사용합니다.

설정:
- `DOMAIN_LOOKUP_BACKEND=json` (기본)
- `DOMAIN_LOOKUP_BACKEND=vector` (벡터 lore 조회 활성화)
- `VECTORDB_LORE_SEARCH_URL=http://localhost:9000/lore/search`
- `VECTORDB_TIMEOUT_SEC=2.0`

동작 우선순위:
1. `DOMAIN_LOOKUP_BACKEND=vector` 이고 `VECTORDB_LORE_SEARCH_URL`가 있으면 lore를 벡터 검색 시도
2. 벡터 검색 실패/타임아웃/응답 파싱 실패 시 JSON lore로 fallback

벡터 lore API 기대 형태 (`POST VECTORDB_LORE_SEARCH_URL`):
요청:
```json
{
  "query": "Interastral Peace Corporation",
  "top_k": 1
}
```

허용 응답 예시 1:
```json
{
  "item": {
    "id": "lore_001",
    "topic": "Interastral Peace Corporation",
    "content": "..."
  }
}
```

허용 응답 예시 2:
```json
{
  "results": [
    {
      "item": {
        "id": "lore_001",
        "topic": "Interastral Peace Corporation",
        "content": "..."
      }
    }
  ]
}
```

## 현재 범위에서 의도적으로 제외
- VTuber, RAG, media pipeline, scheduler, evolution, guardian
- diagnostics/checkpoint/recovery 프레임워크
- 범용 툴 스택 재도입

