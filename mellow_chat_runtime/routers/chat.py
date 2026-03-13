from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from mellow_chat_runtime import app_state
from mellow_chat_runtime.core.domain_lookup_store import get_domain_store
from mellow_chat_runtime.core.rp_parser import ParsedSceneEvent, parse_scene_event
from mellow_chat_runtime.core.speaker_selector import SpeakerParticipant, select_next_speaker
from mellow_chat_runtime.core.states import SystemState, TransitionResult
from mellow_chat_runtime.core.text_sanitizer import sanitize_assistant_text, sanitize_history_text
from mellow_chat_runtime.infra.database import (
    ChatMessage,
    ChatSession,
    MessageFeedback,
    get_db,
    get_or_create_session,
    get_or_create_user,
)
from mellow_chat_runtime.services.memory_promotion_service import MemoryPromotionService
from mellow_chat_runtime.services.model_routing_service import ModelRoutingService

router = APIRouter(tags=['Chat'])
model_router = ModelRoutingService(default_provider='ollama')
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(...)
    mode: str = Field('fast')
    provider: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[int] = None
    stream: bool = True
    persona_id: str = 'default'
    user_profile_id: str = 'default'
    lore_topic: str = 'default'
    lore_topics: Optional[List[str]] = None
    character_id: str = 'default'
    character_ids: Optional[List[str]] = None
    world_id: str = 'default'
    scene_id: str = 'default'


class SessionParticipantsRequest(BaseModel):
    user_character_ids: List[str] = Field(default_factory=list)
    bot_character_ids: List[str] = Field(default_factory=list)


class SessionParticipantsResponse(BaseModel):
    session_id: int
    user_character_ids: List[str] = Field(default_factory=list)
    bot_character_ids: List[str] = Field(default_factory=list)


class ChatErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str


def _user_from_header(x_user: Optional[str]) -> str:
    return (x_user or 'default_user').strip() or 'default_user'


def _parse_json_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for item in data:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned)
    return out


def _compact_unique(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _resolve_lore_keys(request: ChatRequest) -> List[str]:
    keys: List[str] = []
    if request.lore_topics:
        keys.extend(request.lore_topics)
    if request.lore_topic and request.lore_topic != 'default':
        keys.append(request.lore_topic)
    return _compact_unique(keys)


def _get_participants_from_session(session: ChatSession) -> Dict[str, List[str]]:
    return {
        'user_character_ids': _parse_json_list(session.user_character_ids_json),
        'bot_character_ids': _parse_json_list(session.bot_character_ids_json),
    }


def _new_request_id() -> str:
    return f'chat_{uuid.uuid4().hex[:10]}'


def _classify_chat_error(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else 'request_failed'
        return exc.status_code, 'request_failed', detail
    message = str(exc).strip() or 'Chat request failed'
    lowered = message.lower()
    if 'llm service unavailable' in lowered or 'orchestrator not initialized' in lowered:
        return 503, 'model_unavailable', message
    return 500, 'generation_failed', message


def _non_stream_error_response(exc: Exception, request_id: str) -> JSONResponse:
    status_code, error_code, message = _classify_chat_error(exc)
    return JSONResponse(
        status_code=status_code,
        content=ChatErrorResponse(error=error_code, message=message, request_id=request_id).model_dump(),
    )


def _stream_error_event(exc: Exception, request_id: str) -> str:
    _, error_code, message = _classify_chat_error(exc)
    payload = ChatErrorResponse(error=error_code, message=message, request_id=request_id).model_dump()
    return f'event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


def _build_sanitized_history(rows: List[ChatMessage], max_items: int = 8) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows[-max_items:]:
        content = sanitize_history_text(row.role, row.content)
        if not content:
            continue
        out.append({'role': row.role, 'content': content})
    return out


def _list_known_characters(domain_store: Any) -> List[Dict[str, Any]]:
    user_characters = list(domain_store.list_section('user_characters').values())
    bot_characters = list(domain_store.list_section('bot_characters').values())
    return [item for item in user_characters + bot_characters if isinstance(item, dict)]


def _build_speaker_relevance(parsed_scene_event: ParsedSceneEvent) -> Dict[str, float]:
    if parsed_scene_event.target_character_hint:
        return {parsed_scene_event.target_character_hint: 2.0}
    return {}


@router.get('/chat/sessions')
async def get_chat_sessions(x_user: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id, ChatSession.is_active == True)
        .order_by(ChatSession.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            'id': s.id,
            'title': s.title,
            'created_at': s.created_at.isoformat(),
            'selected_model': {
                'provider': s.selected_model_provider,
                'model': s.selected_model_name,
                'mode': s.selected_model_mode,
            },
            'participants': _get_participants_from_session(s),
        }
        for s in sessions
    ]


@router.get('/chat/sessions/{session_id}/messages')
async def get_session_messages(session_id: int, x_user: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')

    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.asc()).all()
    feedbacks = db.query(MessageFeedback).all()
    feedback_map = {f.message_id: f.is_positive for f in feedbacks}

    return [
        {
            'id': m.id,
            'role': m.role,
            'speaker_id': m.speaker_id,
            'speaker_type': m.speaker_type,
            'content': m.content,
            'selected_mode': m.selected_mode,
            'processing_time': m.processing_time,
            'feedback_positive': feedback_map.get(m.id),
            'created_at': m.timestamp.isoformat(),
        }
        for m in messages
    ]


@router.delete('/chat/sessions/{session_id}')
async def delete_chat_session(session_id: int, x_user: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    session.is_active = False
    db.commit()
    return {'success': True, 'deleted_id': session_id}


@router.post('/chat/messages/{message_id}/feedback')
async def submit_message_feedback(message_id: int, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    is_positive = body.get('is_positive')
    if is_positive is None:
        raise HTTPException(status_code=400, detail='is_positive is required')

    existing = db.query(MessageFeedback).filter(MessageFeedback.message_id == message_id).first()
    if existing:
        existing.is_positive = bool(is_positive)
    else:
        db.add(MessageFeedback(message_id=message_id, is_positive=bool(is_positive)))
    db.commit()
    return {'success': True, 'message_id': message_id, 'positive': bool(is_positive)}


@router.get('/sessions/{session_id}/participants', response_model=SessionParticipantsResponse)
async def get_session_participants(session_id: int, x_user: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id, ChatSession.is_active == True).first()
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')
    participants = _get_participants_from_session(session)
    return SessionParticipantsResponse(
        session_id=session.id,
        user_character_ids=participants['user_character_ids'],
        bot_character_ids=participants['bot_character_ids'],
    )


@router.post('/sessions/{session_id}/participants', response_model=SessionParticipantsResponse)
async def upsert_session_participants(
    session_id: int,
    request: SessionParticipantsRequest,
    x_user: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id, ChatSession.is_active == True).first()
    if not session:
        raise HTTPException(status_code=404, detail='Session not found')

    user_character_ids = _compact_unique(request.user_character_ids)
    bot_character_ids = _compact_unique(request.bot_character_ids)
    session.user_character_ids_json = json.dumps(user_character_ids, ensure_ascii=False)
    session.bot_character_ids_json = json.dumps(bot_character_ids, ensure_ascii=False)
    db.commit()
    db.refresh(session)

    return SessionParticipantsResponse(
        session_id=session.id,
        user_character_ids=user_character_ids,
        bot_character_ids=bot_character_ids,
    )


@router.post('/chat/ask')
async def chat_ask(request: ChatRequest, http_request: Request, x_user: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    request_id = _new_request_id()
    if not request.question.strip():
        raise HTTPException(status_code=400, detail='Question is required')

    if app_state.orchestrator is None:
        raise HTTPException(status_code=503, detail='Orchestrator not initialized')
    if app_state.llm_service is None:
        raise HTTPException(status_code=503, detail='LLM service not initialized')

    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    session = get_or_create_session(db=db, user_id=user.id, session_id=request.session_id)
    session_participants = _get_participants_from_session(session)
    lore_keys = _resolve_lore_keys(request)
    requested_character_ids = _compact_unique(request.character_ids or [])
    if not requested_character_ids and request.character_id and request.character_id != 'default':
        requested_character_ids = [request.character_id]

    domain_store = get_domain_store(data_path=app_state.settings.domain_data_file if app_state.settings else None)
    known_characters = _list_known_characters(domain_store)
    parsed_scene_event = parse_scene_event(request.question, known_characters)
    speaker_relevance = _build_speaker_relevance(parsed_scene_event)

    active_character_ids = _compact_unique(
        requested_character_ids
        + session_participants['user_character_ids']
        + session_participants['bot_character_ids']
    )
    effective_lore_topic = lore_keys[0] if lore_keys else request.lore_topic

    user_speaker_id = request.user_profile_id
    if session_participants['user_character_ids']:
        user_speaker_id = session_participants['user_character_ids'][0]
    user_msg = ChatMessage(
        session_id=session.id,
        role='user',
        speaker_id=user_speaker_id,
        speaker_type='user',
        content=request.question,
    )
    db.add(user_msg)
    db.commit()

    state_result = await app_state.orchestrator.request_state_change(SystemState.TEXT, reason='chat ask')
    if state_result == TransitionResult.INVALID_TRANSITION:
        raise HTTPException(status_code=409, detail='Invalid state transition')

    history_rows = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.timestamp.asc()).all()
    history = _build_sanitized_history(history_rows, max_items=8)
    recent_speaker_history = [str(row.speaker_id) for row in history_rows if row.speaker_id]
    selection = model_router.resolve(
        session=session,
        llm_service=app_state.llm_service,
        mode=request.mode,
        request_provider=request.provider,
        request_model=request.model,
    )
    memory_promotion_enabled = bool(getattr(app_state.settings, 'memory_promotion_enabled', True))
    memory_promotion_service = MemoryPromotionService(
        domain_store=domain_store,
        max_items=int(getattr(app_state.settings, 'memory_promotion_max_items', 20)),
    )
    scene_state = domain_store.get_scene_state(request.scene_id)
    scene_rules: Dict[str, Any] = {}
    if isinstance(scene_state.get('rules'), dict):
        scene_rules = dict(scene_state.get('rules', {}))
    if isinstance(scene_state.get('rules'), list):
        for item in scene_state.get('rules', []):
            if isinstance(item, dict):
                key = item.get('key')
                value = item.get('value')
                if isinstance(key, str):
                    scene_rules[key] = value
    bot_participants: List[SpeakerParticipant] = []
    bot_ids = _compact_unique(requested_character_ids + session_participants['bot_character_ids'])
    for character_id in bot_ids:
        bot_data = domain_store.get_bot_character(character_id)
        if not bot_data:
            continue
        bot_participants.append(
            SpeakerParticipant(
                character_id=character_id,
                is_major=bool(bot_data.get('is_major', True)),
            )
        )
    if not bot_participants and request.character_id and request.character_id != 'default':
        bot_participants.append(SpeakerParticipant(character_id=request.character_id, is_major=True))
    selected_speaker_id = select_next_speaker(
        participants=bot_participants,
        recent_speaker_history=recent_speaker_history,
        dialogue_priority=domain_store.get_dialogue_priority(request.scene_id),
        scene_rules=scene_rules,
        target_character_hint=parsed_scene_event.target_character_hint,
        speaker_relevance=speaker_relevance,
    )
    if selected_speaker_id is None and bot_ids:
        selected_speaker_id = bot_ids[0]
    speaker_type = 'bot'
    if selected_speaker_id and domain_store.get_user_character(selected_speaker_id):
        speaker_type = 'user'
    used_context = {
        'persona_id': request.persona_id,
        'user_profile_id': request.user_profile_id,
        'character_ids': active_character_ids,
        'scene_id': request.scene_id,
        'world_id': request.world_id,
        'lore_keys': lore_keys,
        'rp': {
            'input_mode': parsed_scene_event.input_mode,
            'target_character_hint': parsed_scene_event.target_character_hint,
            'has_narration': parsed_scene_event.has_narration,
            'has_dialogue': parsed_scene_event.has_dialogue,
        },
    }
    logger.info(
        'chat.ask.start request_id=%s session_id=%s user=%s selected_model=%s selected_speaker=%s stream=%s input_mode=%s target_hint=%s',
        request_id,
        session.id,
        username,
        selection.model,
        selected_speaker_id or request.character_id,
        request.stream,
        parsed_scene_event.input_mode,
        parsed_scene_event.target_character_hint,
    )

    async def stream_generator():
        started = time.time()
        assistant_id: Optional[int] = None
        try:
            result = await app_state.orchestrator.run_agent(
                user_input=request.question,
                history=history,
                mode=request.mode,
                persona_id=request.persona_id,
                user_profile_id=request.user_profile_id,
                lore_topic=effective_lore_topic,
                character_id=selected_speaker_id or request.character_id,
                world_id=request.world_id,
                scene_id=request.scene_id,
                selected_model=selection.model,
                scene_event=parsed_scene_event,
                target_character_hint=parsed_scene_event.target_character_hint,
            )

            full = sanitize_assistant_text(result.answer or '')
            for i in range(0, len(full), 200):
                yield f'event: chunk\ndata: {json.dumps({"chunk": full[i:i+200], "request_id": request_id}, ensure_ascii=False)}\n\n'

            elapsed_ms = int((time.time() - started) * 1000)
            assistant = ChatMessage(
                session_id=session.id,
                role='assistant',
                speaker_id=selected_speaker_id,
                speaker_type=speaker_type,
                content=full,
                selected_mode=request.mode,
                processing_time=elapsed_ms,
            )
            db.add(assistant)
            db.commit()
            db.refresh(assistant)
            assistant_id = assistant.id
            if memory_promotion_enabled:
                memory_promotion_service.promote_from_text(user_speaker_id, request.question)
                if selected_speaker_id:
                    memory_promotion_service.promote_from_text(selected_speaker_id, full)

            done_payload = {
                'done': True,
                'session_id': session.id,
                'message_id': assistant.id,
                'speaker_id': selected_speaker_id,
                'speaker_type': speaker_type,
                'model_provider': selection.provider,
                'model_name': selection.model,
                'selected_mode': request.mode,
                'processing_time_ms': elapsed_ms,
                'used_context': used_context,
                'model': {
                    'provider': selection.provider,
                    'model': selection.model,
                    'mode': selection.mode,
                    'source': selection.source,
                },
                'request_id': request_id,
            }
            logger.info(
                'chat.ask.end request_id=%s session_id=%s message_id=%s success=true latency_ms=%s',
                request_id,
                session.id,
                assistant_id,
                elapsed_ms,
            )
            yield f'event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n'
        except Exception as e:
            elapsed_ms = int((time.time() - started) * 1000)
            logger.exception(
                'chat.ask.error request_id=%s session_id=%s latency_ms=%s error=%s',
                request_id,
                session.id,
                elapsed_ms,
                str(e),
            )
            yield _stream_error_event(e, request_id)
        finally:
            await app_state.orchestrator.request_state_change(SystemState.IDLE, reason='chat ask done')

    if request.stream:
        return StreamingResponse(stream_generator(), media_type='text/event-stream')

    started = time.time()
    try:
        result = await app_state.orchestrator.run_agent(
            user_input=request.question,
            history=history,
            mode=request.mode,
            persona_id=request.persona_id,
            user_profile_id=request.user_profile_id,
            lore_topic=effective_lore_topic,
            character_id=selected_speaker_id or request.character_id,
            world_id=request.world_id,
            scene_id=request.scene_id,
            selected_model=selection.model,
            scene_event=parsed_scene_event,
            target_character_hint=parsed_scene_event.target_character_hint,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        cleaned_response = sanitize_assistant_text(result.answer or '')
        assistant = ChatMessage(
            session_id=session.id,
            role='assistant',
            speaker_id=selected_speaker_id,
            speaker_type=speaker_type,
            content=cleaned_response,
            selected_mode=request.mode,
            processing_time=elapsed_ms,
        )
        db.add(assistant)
        db.commit()
        db.refresh(assistant)
        if memory_promotion_enabled:
            memory_promotion_service.promote_from_text(user_speaker_id, request.question)
            if selected_speaker_id:
                memory_promotion_service.promote_from_text(selected_speaker_id, cleaned_response)
        logger.info(
            'chat.ask.end request_id=%s session_id=%s message_id=%s success=true latency_ms=%s',
            request_id,
            session.id,
            assistant.id,
            elapsed_ms,
        )
        return {
            'response': cleaned_response,
            'session_id': session.id,
            'message_id': assistant.id,
            'speaker_id': selected_speaker_id,
            'speaker_type': speaker_type,
            'model_provider': selection.provider,
            'model_name': selection.model,
            'selected_mode': request.mode,
            'processing_time_ms': elapsed_ms,
            'used_context': used_context,
            'model': {
                'provider': selection.provider,
                'model': selection.model,
                'mode': selection.mode,
                'source': selection.source,
            },
            'request_id': request_id,
        }
    except Exception as e:
        elapsed_ms = int((time.time() - started) * 1000)
        logger.exception(
            'chat.ask.error request_id=%s session_id=%s latency_ms=%s error=%s',
            request_id,
            session.id,
            elapsed_ms,
            str(e),
        )
        return _non_stream_error_response(e, request_id)
    finally:
        await app_state.orchestrator.request_state_change(SystemState.IDLE, reason='chat ask done')
