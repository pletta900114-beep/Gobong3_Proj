# mellow_chat_runtime

텍스트 기반 캐릭터 챗봇 런타임 백엔드입니다.

현재 범위는 다음까지 포함합니다.

- 채팅 세션 기반 `/chat/ask` 런타임
- active character prompt injection
- recent history 재사용
- multi-character speaker selection
- lore / scene / world / memory priority prompt
- long-term memory promotion and retrieval use
- relationship context injection
- session-scoped model selection
- character / memory / relationship / lore admin API
- runtime integration tests

## 현재 상태

현재 구현은 Phase 1 기본 마감 + Phase 2 확장 마감의 최소 범위를 반영합니다.

- Phase 1
  - single active character
  - prompt injection
  - recent history
  - stable chat endpoint
  - request-scoped logging
  - E2E runtime tests
- Phase 2
  - multi-character speaker selector
  - lore / scene / world priority
  - long-term memory usage
  - richer relationship context
  - character admin tools

## 프로젝트 구조

- `mellow_chat_runtime/`
  - FastAPI 앱, 라우터, 오케스트레이터, 프롬프트/도메인 로직
- `mellow_chat_runtime_data/`
  - SQLite DB, 런타임 도메인 데이터 파일
- `tests/`
  - 통합 테스트 포함
- `service_backend_design_for_codex.md`
  - 초기 설계 참고 문서
- `admin_to_user_nn_transition_design.md`
  - `admin -> user_NN` 전환 설계안

## 빠른 실행

1. Python 3.10+ 준비
2. 의존성 설치

```bash
pip install fastapi uvicorn sqlalchemy aiohttp pydantic pydantic-settings pytest
```

3. 서버 실행

```bash
python -m mellow_chat_runtime.main
```

기본 주소:

- `http://127.0.0.1:8010`

## 환경 변수

예시:

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

MEMORY_PROMOTION_ENABLED=true
MEMORY_PROMOTION_MAX_ITEMS=20
```

## 핵심 동작

### 1. Chat Runtime

`POST /chat/ask`

지원:

- non-stream JSON 응답
- stream SSE 응답
- `audience=user|admin` 실행 경로 분리
- recent history 반영
- active character 선택 및 prompt injection
- 관계 컨텍스트 주입
- long-term memory 주입
- request-scoped logging
- structured error response

non-stream 예시:

```json
{
  "question": "How should you answer now?",
  "audience": "user",
  "stream": false,
  "mode": "fast",
  "persona_id": "default",
  "user_profile_id": "user_char_01",
  "character_ids": ["bot_char_01", "bot_char_02"],
  "scene_id": "scene_default",
  "world_id": "default",
  "lore_topics": ["Interastral Peace Corporation"]
}
```

`audience` 정책:

- `user`
  - 기본값
  - validator 통과 시에만 정상 응답 반환
  - fallback 응답 생성 금지
  - validator 실패 시 명시적 실패 응답 반환
- `admin`
  - retry / repair / fallback / 상세 판정 정보 유지
  - 디버깅 및 QA 용도
  - 응답에 `rp_debug` 포함

RP 생성 안정화 정책:

- RP 생성 요청은 Ollama chat 경로에서 `think=false`를 사용해 `message.content`에 최종 답변이 직접 들어가도록 유도합니다.
- runtime retry 정책은 `agent_brain`에만 둡니다.
- 최대 체인:
  - main generation 1회
  - 필요 시 repair generation 1회
  - `audience=admin`에서만 fallback
- `llm_service`는 empty-content safe retry를 수행하지 않습니다.
- repair 프롬프트는 최종 출력 전용으로 짧게 유지합니다.
  - 짧은 서술 1문단
  - 따옴표 대사 1줄
  - 코드블록 / JSON / 메타 텍스트 / 분석 금지

출력 salvage:

- assistant 출력에서 다음 오염을 repair 전에 정리합니다.
  - `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
  - fenced code block
  - JSON-like RP 출력
- `action` / `narration` / `dialogue` / `speech` / `line` 필드가 있으면 RP 텍스트로 복원합니다.
- 복원 가능한 경우 repair 없이 first-pass validation으로 바로 통과시킵니다.

non-stream 성공 응답 주요 필드:

- `response`
- `session_id`
- `message_id`
- `speaker_id`
- `speaker_type`
- `model_provider`
- `model_name`
- `selected_mode`
- `processing_time_ms`
- `used_context`
- `request_id`

`audience=admin` 추가 필드:

- `rp_debug.validator_passed`
- `rp_debug.fallback_used`
- `rp_debug.retry_count`
- `rp_debug.final_verdict`
- `rp_debug.failure_reason`

non-stream 실패 응답 예시:

```json
{
  "error": "model_unavailable",
  "message": "LLM service unavailable",
  "request_id": "chat_1234567890"
}
```

RP 품질 실패 예시(`audience=user`):

```json
{
  "success": false,
  "error_code": "RP_VALIDATION_FAILED",
  "message": "RP 응답 품질 검증에 실패했습니다. 잠시 후 다시 시도해 주세요.",
  "request_id": "chat_1234567890",
  "failure_reason": "narration_not_third_person"
}
```

stream 응답 특성:

- `event: chunk`
- `event: done`
- `event: error`

### 2. Speaker Selection

다자 대화 시 다음 화자를 선택합니다.

반영 요소:

- include / exclude scene rules
- major / minor weight
- recency penalty
- max consecutive turns

관련 코드:

- `mellow_chat_runtime/core/speaker_selector.py`

### 3. Prompt Priority

프롬프트에는 현재 우선순위가 명시됩니다.

우선순위:

1. current scene rules and scene goal
2. world-state constraints and continuity
3. character memories and relationship context
4. lorebook facts for support and terminology

### 4. Memory

- short-term memory
  - 세션 메시지가 SQLite `chat_messages`에 저장
  - 최근 대화가 prompt에 재주입됨
- long-term memory
  - 캐릭터별 `memories`가 domain data에 저장
  - generation prompt에서 중요한 메모리 상위 항목을 사용
- memory promotion
  - 중요 문장을 감지해 `important_memories`로 승격

### 5. Relationship Context

active character 기준으로 scene participant와의 관계를 조회해서 prompt에 넣습니다.

예시 정보:

- 관계 요약
- 관계 톤
- boundary
- shared memory

### 6. RP QA Verification

QA 스모크:

```bash
python scripts/rp_qa_smoke.py --max-scenarios 3 --audience admin
```

로그에서 확인할 항목:

- `llm.chat.response`
  - `content_len > 0`
  - `thinking_len = 0`
  - `thinking_only=False`
- `rp.output.first_pass_valid`
  - repair 없이 통과한 경우
- `rp.output.repair_pass_valid`
  - 1회 repair로 복구된 경우
- `rp.output.fallback_used`
  - `audience=admin`에서만 허용

현재 QA 기준:

- `fallback_used = false` 이고 `validator_passed = true` 이면 `PASS`
- `fallback_used = true` 이면 `FAIL`
- `audience=user`에서 validator 실패로 명시적 실패 반환 시 `FAIL`

검증 예시:

- fenced JSON + trailing token 출력이 와도 sanitize 후 1회 호출만으로 통과 가능
- smoke 실행 시 request latency는 repair 미사용 케이스에서 대체로 1초대, repair 사용 시 2초대 수준으로 확인

## 주요 API

### Health / Runtime

- `GET /health`
- `GET /runtime/status`

### Model Selection

- `POST /models/select`
- `GET /models/sessions/{session_id}`

### Session Participants

- `GET /sessions/{session_id}/participants`
- `POST /sessions/{session_id}/participants`

### Chat

- `POST /chat/ask`
- `GET /chat/sessions`
- `GET /chat/sessions/{session_id}/messages`
- `DELETE /chat/sessions/{session_id}`
- `POST /chat/messages/{message_id}/feedback`

### Admin

- `GET /admin/characters`
- `GET /admin/characters/{user|bot}/{character_id}`
- `PUT /admin/characters/{user|bot}/{character_id}`
- `DELETE /admin/characters/{user|bot}/{character_id}`
- `GET /admin/memories/{character_id}`
- `PUT /admin/memories/{character_id}`
- `GET /admin/relationships/{source_id}`
- `PUT /admin/relationships`
- `GET /admin/lore/{lore_id}`
- `PUT /admin/lore/{lore_id}`

## 데이터 파일

- SQLite DB
  - `mellow_chat_runtime_data/chatbot.db`
- 런타임 도메인 데이터
  - `mellow_chat_runtime_data/domain_data.json`
- 시드 예시
  - `mellow_chat_runtime_data/domain_data.seed.json`

## 테스트

실행:

```bash
pytest -q
```

현재 포함 테스트 범위:

- memory promotion
- `/chat/ask` non-stream success
- `/chat/ask` stream success
- `/chat/ask` failure path
- character prompt enforcement
- multi-character speaker selection
- relationship context prompt injection
- long-term memory prompt usage
- admin API flow

QA 스모크 실행 예시:

```bash
python scripts/rp_qa_smoke.py --max-scenarios 1 --audience admin
python scripts/rp_qa_smoke.py --max-scenarios 1 --audience admin --save mellow_chat_runtime_data/qa/rp_qa_result.json --save-format json
python scripts/rp_qa_smoke.py --max-scenarios 1 --audience admin --save mellow_chat_runtime_data/qa/rp_qa_result.md --save-format md
```

QA 리포트 판정 기준:

- `fallback_used = false` 이고 `validator_passed = true` 이면 `PASS`
- `fallback_used = true` 이면 `FAIL`
- `audience=user` 경로에서 validator 실패로 명시적 실패 반환 시 `FAIL`

저장 파일은 실행 시각 기반 `run_id`가 붙어 저장됩니다.

예:

- `rp_qa_result_20260314_040738.json`
- `rp_qa_result_20260314_040738.md`

## admin -> user_NN 확장

현재 `admin` API는 운영자 수정용입니다.

이후 일반 사용자용 `user_NN` 구조로 확장하려면:

- `user_NN`은 account id
- `user_char_*`는 character id
- ownership은 DB에서 관리
- 일반 사용자용 `/me/*` 계층 추가

상세 설계:

- `admin_to_user_nn_transition_design.md`

## 현재 범위에서 의도적으로 제외

- auth / permission framework 전면 도입
- media / VTuber pipeline
- scheduler / evolution / guardian
- 대규모 아키텍처 재설계
