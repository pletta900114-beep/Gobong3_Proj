from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SpeechStyle(BaseModel):
    tone: str = "neutral"
    forbidden: List[str] = Field(default_factory=list)


class UserCharacter(BaseModel):
    id: str
    type: str = "user"
    name: str
    profile: str = ""
    traits: List[str] = Field(default_factory=list)
    relationship_keys: List[str] = Field(default_factory=list)
    is_major: bool = True


class BotCharacter(BaseModel):
    id: str
    type: str = "bot"
    name: str
    persona_id: str = "default"
    speech_style: SpeechStyle = Field(default_factory=SpeechStyle)
    relationship_keys: List[str] = Field(default_factory=list)
    is_major: bool = True


class LorebookEntry(BaseModel):
    id: str
    topic: str
    aliases: List[str] = Field(default_factory=list)
    content: str
    priority: int = 0
    summary_text: Optional[str] = None
    embedding_status: Optional[Literal["pending", "dirty", "ready", "failed"]] = None


class SceneRule(BaseModel):
    key: str
    value: Any


class SceneState(BaseModel):
    id: str
    location: str
    time: str
    participants: List[str] = Field(default_factory=list)
    goal: str = ""
    mood: str = "neutral"
    rules: List[SceneRule] = Field(default_factory=list)


class WorldState(BaseModel):
    id: str
    location: str = ""
    time: str = ""
    state: str = "stable"
    facts: List[str] = Field(default_factory=list)


class MemoryPossession(BaseModel):
    character_id: str
    important_memories: List[str] = Field(default_factory=list)
    possessions: List[str] = Field(default_factory=list)
    summary_text: Optional[str] = None
    embedding_status: Optional[Literal["pending", "dirty", "ready", "failed"]] = None


class RelationshipContext(BaseModel):
    target_id: str
    summary: str = ""
    tone: str = "neutral"
    boundaries: List[str] = Field(default_factory=list)
    shared_memories: List[str] = Field(default_factory=list)
    summary_text: Optional[str] = None
    embedding_status: Optional[Literal["pending", "dirty", "ready", "failed"]] = None


class DialoguePriority(BaseModel):
    scene_id: str
    major_weight: float = 1.0
    minor_weight: float = 0.5
    recency_penalty: float = 0.25
    max_consecutive_turns: int = 1
    rules: str = ""


class DomainDataBundle(BaseModel):
    personas: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    user_characters: Dict[str, UserCharacter] = Field(default_factory=dict)
    bot_characters: Dict[str, BotCharacter] = Field(default_factory=dict)
    lorebook: Dict[str, LorebookEntry] = Field(default_factory=dict)
    scene_state: Dict[str, SceneState] = Field(default_factory=dict)
    world_state: Dict[str, WorldState] = Field(default_factory=dict)
    memories: Dict[str, MemoryPossession] = Field(default_factory=dict)
    relationships: Dict[str, Dict[str, RelationshipContext]] = Field(default_factory=dict)
    dialogue_priority: Dict[str, DialoguePriority] = Field(default_factory=dict)
    user_profiles: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)


