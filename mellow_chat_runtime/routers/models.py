from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from mellow_chat_runtime.infra.database import ChatSession, get_db, get_or_create_session, get_or_create_user

router = APIRouter(tags=["Models"])


class ModelDescriptor(BaseModel):
    provider: str = Field(..., min_length=1, max_length=80)
    model: str = Field(..., min_length=1, max_length=120)
    mode: Optional[str] = Field(default=None, max_length=50)


class SelectModelRequest(BaseModel):
    session_id: Optional[int] = None
    selection: ModelDescriptor


class SelectModelResponse(BaseModel):
    session_id: int
    selected: ModelDescriptor
    source: str = "session"


def _user_from_header(x_user: Optional[str]) -> str:
    return (x_user or "default_user").strip() or "default_user"


@router.post("/models/select", response_model=SelectModelResponse)
async def select_model(
    request: SelectModelRequest,
    x_user: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    session = get_or_create_session(db=db, user_id=user.id, session_id=request.session_id)

    session.selected_model_provider = request.selection.provider.strip()
    session.selected_model_name = request.selection.model.strip()
    if request.selection.mode is not None:
        session.selected_model_mode = request.selection.mode.strip() or None
    db.commit()
    db.refresh(session)

    return SelectModelResponse(
        session_id=session.id,
        selected=ModelDescriptor(
            provider=session.selected_model_provider or "ollama",
            model=session.selected_model_name or "",
            mode=session.selected_model_mode,
        ),
    )


@router.get("/models/sessions/{session_id}", response_model=SelectModelResponse)
async def get_selected_model(
    session_id: int,
    x_user: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    username = _user_from_header(x_user)
    user = get_or_create_user(db, username)
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id, ChatSession.is_active == True).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.selected_model_provider or not session.selected_model_name:
        raise HTTPException(status_code=404, detail="No model selected for this session")

    return SelectModelResponse(
        session_id=session.id,
        selected=ModelDescriptor(
            provider=session.selected_model_provider,
            model=session.selected_model_name,
            mode=session.selected_model_mode,
        ),
    )
