from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

DATA_DIR = Path("./mellow_chat_runtime_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "chatbot.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(120), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    sessions = relationship("ChatSession", back_populates="user")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(300), nullable=False, default="New Chat")
    is_active = Column(Boolean, default=True, nullable=False)
    selected_model_provider = Column(String(80), nullable=True)
    selected_model_name = Column(String(120), nullable=True)
    selected_model_mode = Column(String(50), nullable=True)
    user_character_ids_json = Column(Text, nullable=True)
    bot_character_ids_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    speaker_id = Column(String(120), nullable=True)
    speaker_type = Column(String(20), nullable=True)
    content = Column(Text, nullable=False)
    selected_mode = Column(String(50), nullable=True)
    processing_time = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="messages")


class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False, index=True)
    is_positive = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_chat_session_columns()
    _ensure_chat_message_columns()


def _ensure_chat_session_columns() -> None:
    with engine.begin() as conn:
        result = conn.execute(text("PRAGMA table_info(chat_sessions)"))
        columns = {str(row[1]) for row in result.fetchall()}
        if "selected_model_provider" not in columns:
            conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN selected_model_provider VARCHAR(80)"))
        if "selected_model_name" not in columns:
            conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN selected_model_name VARCHAR(120)"))
        if "selected_model_mode" not in columns:
            conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN selected_model_mode VARCHAR(50)"))
        if "user_character_ids_json" not in columns:
            conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN user_character_ids_json TEXT"))
        if "bot_character_ids_json" not in columns:
            conn.execute(text("ALTER TABLE chat_sessions ADD COLUMN bot_character_ids_json TEXT"))


def _ensure_chat_message_columns() -> None:
    with engine.begin() as conn:
        result = conn.execute(text("PRAGMA table_info(chat_messages)"))
        columns = {str(row[1]) for row in result.fetchall()}
        if "speaker_id" not in columns:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN speaker_id VARCHAR(120)"))
        if "speaker_type" not in columns:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN speaker_type VARCHAR(20)"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if user:
        return user
    user = User(username=username)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_or_create_session(db: Session, user_id: int, session_id: int | None = None) -> ChatSession:
    if session_id:
        found = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id, ChatSession.is_active == True).first()
        if found:
            return found
    session = ChatSession(user_id=user_id, title="New Chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session
