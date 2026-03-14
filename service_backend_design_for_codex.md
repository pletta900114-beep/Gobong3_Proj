# 서비스 백엔드 설계 초안
_대상: Codex 참고용 / 웹서비스 연동 전제 / 캐릭터 대화 엔진 기반_

## 1. 목표

이 백엔드는 단순한 채팅 API가 아니라, **웹서비스에 붙는 캐릭터 대화 엔진 백엔드**를 목표로 한다.

핵심 요구사항:

1. 유저가 기본 LLM/모델을 선택할 수 있어야 함
2. `user_char_NN : bot_char_NN` 구조의 1:1 대화와 다대다 대화를 지원해야 함
3. 로어북 조회가 가능해야 함
4. 캐릭터별 기억, 소지품, 현재 위치/시간/장면 상태를 반영해야 함
5. 주요 캐릭터와 비주요 캐릭터의 발화 비중을 제어할 수 있어야 함
6. 웹 프론트엔드가 붙기 쉬운 API 형태여야 함

---

## 2. 시스템 성격

이 시스템은 아래 두 가지를 결합한 구조로 본다.

### A. 챗봇 API 서버
- 세션 관리
- 메시지 저장
- 스트리밍 응답
- 인증
- 모델 선택
- API 응답 스펙 제공

### B. 캐릭터 대화 엔진
- 캐릭터 프로필
- 유저 캐릭터 / 봇 캐릭터
- 로어북
- 장면 상태
- 기억 / 소지품
- 발화 우선순위
- 누가 말할지 결정하는 오케스트레이션

즉 최종 구조는:

```text
웹 프론트
  ↓
서비스 API 서버
  ↓
캐릭터 대화 엔진
  ↓
선택된 LLM Provider / Model
```

---

## 3. 아키텍처 개요

```text
[Frontend / Web Client]
  ↓
[API Layer]
  - sessions
  - chat
  - runtime
  - model selection
  - character selection
  ↓
[Application Layer]
  - ChatService
  - SessionService
  - CharacterService
  - LoreService
  - SceneService
  - ModelRoutingService
  ↓
[Orchestration Layer]
  - Orchestrator
  - AgentBrain
  - SpeakerSelector
  - PromptBuilder
  - DialoguePolicy
  - DomainLookupDispatcher
  ↓
[Domain Data Layer]
  - canonical domain store (source of truth)
  - vector index (support retrieval only)
  - user characters
  - bot characters
  - lorebook
  - memories
  - possessions
  - world state
  - scene state
  - dialogue priority
  ↓
[LLM Layer]
  - provider adapter
  - model registry
  - model routing
  ↓
[Persistence Layer]
  - chat/session db
  - canonical domain store
  - vector index store
  - config store
```

---

## 4. 권장 모듈 구조

```text
mellow_chat_runtime/
  api/
    chat.py
    sessions.py
    runtime.py
    models.py
    characters.py

  core/
    orchestrator.py
    agent_brain.py
    prompt_builder.py
    speaker_selector.py
    dialogue_policy.py
    domain_lookup_dispatcher.py

  services/
    llm_service.py
    model_routing_service.py
    character_service.py
    lore_service.py
    scene_service.py
    memory_service.py
    chat_service.py

  domain/
    character_models.py
    lore_models.py
    scene_models.py
    memory_models.py
    message_models.py

  infra/
    database.py
    repositories/
      session_repository.py
      message_repository.py
      character_repository.py
      lore_repository.py
      scene_repository.py
      memory_repository.py

  config/
    settings.py
    model_registry.py

  prompts/
    default_system_prompt.txt
    personas/
    scenes/

mellow_chat_runtime_data/
  personas/
  user_profiles/
  bot_profiles/
  lorebook/
  scenes/
  world_state/
  memories/
  dialogue_policy/
```

---

## 5. 핵심 도메인 모델

## 5.1 Session
```json
{
  "id": 1,
  "user_id": "alice",
  "title": "New Chat",
  "selected_model": "qwen3.5:9b",
  "created_at": "2026-03-06T05:13:11"
}
```

## 5.2 Message
```json
{
  "id": 10,
  "session_id": 1,
  "speaker_id": "bot_char_01",
  "speaker_type": "bot",
  "role": "assistant",
  "content": "......",
  "selected_mode": "fast",
  "processing_time_ms": 423,
  "created_at": "2026-03-06T05:15:00"
}
```

## 5.3 Character
```json
{
  "id": "bot_char_01",
  "type": "bot",
  "name": "Aventurine",
  "persona_id": "aventurine_default",
  "speech_style": {
    "tone": "casual_confident",
    "forbidden": ["과도한 메타발언"]
  },
  "relationship_keys": ["mellow"]
}
```

## 5.4 User Character
```json
{
  "id": "user_char_01",
  "type": "user",
  "name": "Mellow",
  "profile": "UI/UX 디자이너, 프로그래머",
  "traits": ["설계 중심", "캐릭터 몰입 중요"]
}
```

## 5.5 Lorebook Entry
```json
{
  "id": "lore_001",
  "topic": "스타피스 컴퍼니",
  "aliases": ["IPC", "Interastral Peace Corporation"],
  "content": "......",
  "priority": 10,
  "summary_text": "......",
  "embedding_status": "dirty"
}
```

## 5.6 Scene State
```json
{
  "id": "scene_default",
  "location": "라운지",
  "time": "저녁",
  "participants": ["user_char_01", "bot_char_01"],
  "goal": "상황 정리 및 대화",
  "mood": "calm"
}
```

## 5.7 Memory / Possession
```json
{
  "character_id": "user_char_01",
  "important_memories": ["과거 프로젝트 사건", "관계 변화"],
  "possessions": ["카드 케이스", "메모장"],
  "summary_text": "과거 프로젝트 사건 | 관계 변화",
  "embedding_status": "dirty"
}
```

## 5.8 Relationship Context
```json
{
  "target_id": "bot_char_01",
  "summary": "신뢰 가능한 협업 상대",
  "tone": "warmly strategic",
  "boundaries": ["기존 합의를 함부로 무시하지 않음"],
  "shared_memories": ["긴장된 협상을 함께 안정시켰다"],
  "summary_text": "신뢰 가능한 협업 상대 | warmly strategic | 긴장된 협상을 함께 안정시켰다",
  "embedding_status": "dirty"
}
```

---

## 6. 모델 선택 구조

유저가 기본 LLM을 바꿀 수 있어야 하므로, 모델 선택은 세션 단위 또는 유저 단위로 관리한다.

### 우선순위
1. 요청 본문에서 지정한 모델
2. 세션에 저장된 기본 모델
3. 유저 기본 모델
4. 시스템 기본 모델

### 예시
```json
{
  "provider": "ollama",
  "model": "qwen3.5:9b",
  "mode": "fast"
}
```

### Model Registry 예시
```json
{
  "default": {
    "provider": "ollama",
    "model": "qwen3.5:9b"
  },
  "available_models": [
    {"id": "qwen3.5:9b", "provider": "ollama", "supports_tools": true},
    {"id": "llama3.1:8b", "provider": "ollama", "supports_tools": true}
  ]
}
```

---

## 7. 1:1 / 다대다 대화 구조

## 7.1 1:1
- 유저 캐릭터 1명
- 봇 캐릭터 1명
- speaker selection 불필요하거나 단순함

## 7.2 다대다
- 참가자 목록 존재
- 발화 비중 계산 필요
- scene 상태에 따라 speaker selection 필요
- 같은 캐릭터 연속 발화 제한 규칙 필요 가능

### SpeakerSelector 책임
- 이번 턴의 발화자 결정
- major/minor 비중 반영
- 최근 발화 이력 반영
- 장면 규칙 반영

### DialoguePolicy 책임
- 주요 캐릭터 우선권
- 비주요 캐릭터 발화 비율
- scene별 가중치
- 특수 규칙 (특정 캐릭터는 필요한 경우만 발화)

---

## 8. 로어북 / 기억 / 세계 상태 조회

현재 요구사항상 범용 도구는 최소화하고, **도메인 조회 함수**만 유지한다.

### 권장 조회 함수
- `lookup_persona(persona_id)`
- `lookup_user_profile(user_id or character_id)`
- `lookup_lorebook(topic or entity or location)`
- `lookup_memories_possessions(character_id)`
- `lookup_world_state(scene_id or world_id)`
- `lookup_dialogue_priority(scene_id, participants)`

### 원칙
- 웹 검색 없음
- 파일시스템 수정 없음
- 외부 셸 실행 없음
- 내부 저장소 조회만 허용
- canonical store가 source of truth
- vector index는 lore / memory / relationship에 대한 보조 검색층
- reindex 상태 전이 용어는 현재 `dirty -> clean`

---

## 9. 요청 처리 흐름

```text
POST /chat/ask
  ↓
Session 확인 / 생성
  ↓
User message 저장
  ↓
ModelRoutingService가 모델 결정
  ↓
Scene/Character/Lore/Memory canonical 조회
  ↓
SpeakerSelector가 발화자 결정
  ↓
Vector retrieval이 lore / memory / relationship 보조 컨텍스트 조회
  - 실패 시 canonical fallback
  - admin audience에서는 retrieval_debug로 source / hit ids / errors 노출
  ↓
PromptBuilder가 최종 프롬프트 조합
  ↓
AgentBrain이 최소 루프 실행
  ↓
LLMService 호출
  ↓
Assistant message 저장
  ↓
응답 반환 (SSE 또는 JSON)
```

---

## 10. API 초안

## 10.1 Health
`GET /health`

응답 예시:
```json
{
  "state": "IDLE",
  "llm_available": true,
  "agent_initialized": true
}
```

## 10.2 Runtime Status
`GET /runtime/status`

## 10.3 Session List
`GET /chat/sessions`

## 10.4 Session Messages
`GET /chat/sessions/{id}/messages`

## 10.5 Ask
`POST /chat/ask`

예시 요청:
```json
{
  "question": "다음엔 누가 먼저 말해야 해?",
  "mode": "fast",
  "stream": true,
  "model": "qwen3.5:9b",
  "persona_id": "aventurine_default",
  "user_profile_id": "user_char_01",
  "character_ids": ["bot_char_01", "bot_char_02"],
  "scene_id": "scene_default",
  "world_id": "world_default",
  "lore_topics": ["스타피스 컴퍼니", "스톤하트"]
}
```

예시 응답:
```json
{
  "response": "......",
  "session_id": 1,
  "message_id": 11,
  "speaker_id": "bot_char_01",
  "model": "qwen3.5:9b",
  "used_context": {
    "persona_id": "aventurine_default",
    "scene_id": "scene_default",
    "lore_keys": ["스타피스 컴퍼니"]
  }
}
```

`audience=admin` 응답에는 아래 retrieval debug가 추가될 수 있다.

```json
{
  "retrieval_debug": {
    "query": "원문 질문 + active speaker + participants + scene goal/mood + recent turns",
    "lore_source": "vector",
    "memory_source": "canonical",
    "relationship_source": "none",
    "lore_hit_ids": ["lore_001"],
    "memory_hit_ids": ["bot_char_01:0"],
    "relationship_hit_ids": [],
    "errors": []
  }
}
```

## 10.6 Model Selection
`POST /models/select`

예시:
```json
{
  "session_id": 1,
  "model": "qwen3.5:9b"
}
```

## 10.7 Character Selection
`POST /sessions/{id}/participants`

예시:
```json
{
  "user_character_ids": ["user_char_01"],
  "bot_character_ids": ["bot_char_01", "bot_char_02"]
}
```

## 10.8 Vector Reindex
`POST /admin/vector/reindex`

현재 구현은 lore / memory / relationship canonical payload를 재요약하고,
vector index를 다시 만들며,
canonical `embedding_status`를 `dirty -> clean`으로 갱신한다.

---

## 11. 1차 구현 우선순위

### Phase 1
- 현재 `mellow_chat_runtime` 안정화
- health / sessions / ask / messages 보강
- seed data 구조 확정
- basic model selection 추가

### Phase 2
- character / lore / memory / world_state API 추가
- prompt_builder 강화
- lookup 품질 개선

### Phase 3
- speaker_selector 추가
- 다대다 발화 제어
- dialogue priority 강화

### Phase 4
- 웹 프론트 통합
- 사용자 설정 UI
- 운영 로그 / admin 기능

### 테스트 메모
- pytest 실행 시에는 fixture 기반 test DB override를 사용한다.
- 운영 DB는 테스트 중 source of truth 검증 대상이 아니며, 각 테스트는 별도 SQLite test DB를 사용한다.

---

## 12. Codex 작업 시 유의사항

1. broad package import를 피하고 concrete import를 사용할 것
2. generalized tool stack을 다시 들여오지 말 것
3. VTuber, RAG, media, scheduler, guardian, evolution 관련 경로를 재도입하지 말 것
4. 최소 agent loop는 유지하되, domain lookup 중심으로 유지할 것
5. 새 기능을 추가할 때는 먼저 API 스펙과 도메인 모델부터 정의할 것
6. 1차 목표는 웹서비스용 텍스트 캐릭터 챗봇 백엔드이며, 기능 과잉 확장은 금지

---

## 13. Reality Boundary Rule

- ✅ **확정(verified)**: 현재 `mellow_chat_runtime`는 독립 실행 가능한 텍스트 챗봇 런타임이다.
- ✅ **확정(verified)**: 웹서비스 연동을 위해서는 모델 선택, 캐릭터 데이터, 로어북, scene state, 발화 정책이 필요하다.
- ⚠️ **조건부(possible)**: 현재 구조는 1:1 대화에 더 가까우며, 다대다 speaker selection은 추가 설계가 필요할 수 있다.
- ❌ **가정(hypothetical)**: 지금 상태만으로 자연스러운 다대다 캐릭터 대화가 완성된 것은 아니다.

---

## 14. 다음 Codex 작업 추천

1. `model selection` API와 세션별 모델 저장 추가
2. `character / lore / scene / memory` seed schema 정의
3. `speaker_selector.py` 초안 작성
4. `/chat/ask` 응답에 `speaker_id`, `model`, `used_context` 메타데이터 추가
5. README 및 curl 예시 작성

