# admin -> user_NN 대응 구조 설계안

## 목적
현재 `admin` 중심의 도메인 수정 구조를 유지하면서, 이후 일반 사용자도 `user_NN` 식별자 기준으로 자기 캐릭터/메모리/관계 데이터를 다룰 수 있게 전환하는 최소 구조를 정의한다.

이 문서는 현재 코드베이스를 기준으로 한다.

- 채팅 런타임: `mellow_chat_runtime/routers/chat.py`
- 관리 API: `mellow_chat_runtime/routers/admin.py`
- 세션/사용자 DB: `mellow_chat_runtime/infra/database.py`
- 도메인 저장소: `mellow_chat_runtime/core/domain_lookup_store.py`

## 현재 상태
현재 식별자 계층은 아래처럼 섞여 있다.

- 실제 로그인/세션 사용자 식별: `x-user`, `users.username`, `chat_sessions.user_id`
- user character 식별: `user_char_01`
- bot character 식별: `bot_char_01`
- 운영 수정 경로: `/admin/*`

즉, 지금은:
- `User`는 세션 소유자
- `user_char_*`는 세계관/대화상 사용자 캐릭터
- `admin`은 운영자 수정 API

아직 없는 것은:
- 일반 사용자용 소유권 모델
- `user_NN`과 캐릭터/메모리/관계의 연결 규칙
- admin API와 user API의 권한 분리

## 목표 상태
전환 후에는 식별자를 아래처럼 분리한다.

1. Account ID
- 형식 예시: `user_01`, `user_02`, `user_17`
- 역할: 실제 서비스 사용자 계정 식별자
- 저장 위치: `users.username`
- 용도: 세션 소유권, 수정 권한, 개인 자산 소유권 판단

2. Character ID
- 형식 예시: `user_char_01`, `bot_char_01`
- 역할: 대화 내 페르소나/캐릭터 식별자
- 저장 위치: domain store
- 용도: prompt injection, memory, relationship, scene participants

3. Session ID
- 역할: 채팅 세션 식별자
- 저장 위치: `chat_sessions.id`
- 용도: 최근 대화, 모델 선택, 참가자 연결

핵심 원칙:
- `user_NN`은 계정 ID다.
- `user_char_NN`은 캐릭터 ID다.
- 둘은 동일 개념이 아니다.

## 권장 매핑 규칙
최소 규칙은 아래와 같다.

- 한 `user_NN`은 0개 이상의 `user_char_*`를 소유할 수 있다.
- `bot_char_*`는 기본적으로 시스템 소유다.
- `memories`는 캐릭터 단위로 저장하되, 수정 권한은 소유자 기준으로 판정한다.
- `relationships`는 source character의 소유권 기준으로 수정 권한을 판정한다.

예시:
- `user_01` 소유 캐릭터: `user_char_01`, `user_char_07`
- 시스템 소유 캐릭터: `bot_char_01`, `bot_char_02`

## 최소 데이터 모델 확장안
현재 DB와 domain store를 크게 바꾸지 않고 아래만 추가하면 된다.

### 1. DB에 account-character ownership 매핑 추가
신규 테이블 제안:

`user_character_ownership`
- `id`
- `user_id` (FK -> users.id)
- `character_id` (string, unique optional)
- `role` (`owner` / `editor`)
- `created_at`

목적:
- 어떤 `user_NN`이 어떤 `user_char_*`를 소유하는지 명확히 기록
- admin 권한과 user 권한을 분리

### 2. domain store 항목에 owner 메타를 선택적으로 허용
예시:
- `user_characters[user_char_01].owner_user_id = "user_01"`
- 선택 사항이지만, 빠른 조회에는 유리함

권장 순서:
- 권한 판정의 정본은 DB ownership table
- domain data의 `owner_user_id`는 캐시/보조 정보

## API 계층 분리안
현재 `/admin/*`는 유지한다.
그 위에 일반 사용자용 `/me/*` 또는 `/users/me/*` 계층을 추가한다.

### 유지: 운영자 API
- `/admin/characters/*`
- `/admin/memories/*`
- `/admin/relationships/*`
- `/admin/lore/*`

역할:
- 전체 데이터 수정 가능
- seed/domain 운영용

### 추가: 일반 사용자 API
권장 prefix:
- `/me/characters`
- `/me/memories`
- `/me/relationships`

예시:
- `GET /me/characters`
  - 현재 로그인 사용자(`x-user` 또는 auth subject)가 소유한 character만 반환
- `PUT /me/characters/{character_id}`
  - 본인 소유 캐릭터만 수정 가능
- `GET /me/memories/{character_id}`
  - 본인 소유 character memory만 조회/수정 가능
- `PUT /me/relationships/{source_id}/{target_id}`
  - source character가 본인 소유일 때만 수정 가능

중요:
- 일반 사용자 API는 `bot_char_*` 수정 권한이 없어야 한다.
- bot character에 대한 수정은 admin only로 유지하는 게 안전하다.

## 권한 판정 규칙
최소 권한 규칙은 아래처럼 단순하게 유지한다.

1. Admin
- 모든 character, memory, relationship, lore 수정 가능

2. Normal user (`user_NN`)
- 본인 소유 `user_char_*`만 수정 가능
- 본인 소유 character의 memory만 수정 가능
- 본인 소유 character를 source로 하는 relationship만 수정 가능
- `bot_char_*`, lore, global scene/world는 수정 불가

3. Chat runtime
- 읽기 자체는 현재처럼 가능
- 단, session participants에 user-owned character를 붙일 때는 세션 사용자와 ownership이 일치하는지 검증 가능

## 현재 코드에 대응하는 구현 단계
### Step 1. user_NN을 공식 account id로 고정
현재 `x-user`를 그대로 `users.username`에 넣고 있다.

권장:
- `x-user: user_01` 형식을 공식 규칙으로 채택
- 지금 구조에서는 추가 수정 없이도 가능

즉, 바로 시작 가능한 규칙:
- 로그인 사용자 헤더는 `user_NN`
- `users.username == user_NN`

### Step 2. ownership table 추가
`infra/database.py`에 `UserCharacterOwnership` 테이블 추가

필드:
- `user_id`
- `character_id`
- `role`

필요 함수:
- `assign_character_owner(db, username, character_id)`
- `list_owned_characters(db, username)`
- `user_owns_character(db, username, character_id)`

### Step 3. admin API는 유지하고 me API 추가
`routers/me.py` 신규 추가 권장

최소 엔드포인트:
- `GET /me/characters`
- `GET /me/characters/{character_id}`
- `PUT /me/characters/{character_id}`
- `GET /me/memories/{character_id}`
- `PUT /me/memories/{character_id}`
- `GET /me/relationships/{source_id}`
- `PUT /me/relationships/{source_id}/{target_id}`

각 endpoint에서:
- `x-user` -> `user_NN`
- DB ownership 확인
- 권한 있으면 기존 `DomainLookupStore.upsert()` 재사용

### Step 4. chat runtime에 ownership 검증 연결
현재 `chat.py`는 session participants를 그대로 저장한다.

추가 권장 검증:
- `user_character_ids`를 세션에 넣을 때, 그 character가 현재 세션 사용자 소유인지 확인
- 소유하지 않은 `user_char_*`는 거부

bot character는 그대로 허용 가능

### Step 5. 생성 규칙 정의
일반 사용자가 새 캐릭터를 만들 경우 규칙은 아래처럼 단순하게 유지한다.

- account id: `user_01`
- default owned character id: `user_char_01_main` 또는 `user_char_user_01_01`

권장:
- display용 이름과 id 규칙 분리
- id는 기계적 규칙 유지

## 추천 ID 규칙
### 계정 ID
- `user_01`
- `user_02`
- `user_15`

### 유저 캐릭터 ID
권장 2안 중 하나:

1. 단순 번호형
- `user_char_01`
- `user_char_02`

장점:
- 현재 구조와 가장 잘 맞음

단점:
- 어떤 user가 소유하는지 id만으로는 모름

2. 소유자 포함형
- `user_char_user_01_01`
- `user_char_user_01_02`

장점:
- 소유자 추적이 쉬움

단점:
- id가 길어짐

현재 구조 기준 추천:
- ID는 기존처럼 `user_char_*` 유지
- ownership은 DB로 관리

## migration 전략
운영 중단 없이 가려면 아래 순서가 안전하다.

1. `x-user` 입력값을 `user_NN` 규칙으로 통일
2. ownership table 추가
3. 기존 `user_char_*`를 적절한 `user_NN`에 매핑하는 backfill 수행
4. `/me/*` API 추가
5. UI/클라이언트를 admin 경로 대신 me 경로로 전환
6. admin은 운영자 전용으로만 남김

## 최소 구현 범위 제안
지금 코드베이스에서 가장 작은 다음 단계는 이것이다.

- [ ] `users.username`에 `user_NN` 규칙 적용
- [ ] `UserCharacterOwnership` 테이블 추가
- [ ] `routers/me.py` 추가
- [ ] `/sessions/{session_id}/participants`에 user character ownership 검증 추가
- [ ] `/me/characters`, `/me/memories`, `/me/relationships` 구현

## 하지 말아야 할 것
현 단계에서는 아래는 과하다.

- auth 시스템 전면 도입
- RBAC 프레임워크 추가
- bot/user character 스토리지를 완전히 분리
- character engine 전체 재설계

현재 구조에서는 소유권 테이블 + `/me/*` 계층만 추가해도 충분하다.

## 최종 결론
가능 여부 기준으로는 이미 준비돼 있다.

현재 구조는:
- `admin`을 운영자 수정 API로 유지하면서
- `user_NN`을 계정 식별자로 삼고
- 그 아래 `user_char_*` ownership을 붙여
- 일반 사용자용 `/me/*` API를 얹는 방식으로
무리 없이 확장 가능하다.

가장 중요한 원칙은 하나다.

- `user_NN`은 account
- `user_char_*`는 character
- ownership은 DB로 관리
- admin과 user API는 분리
