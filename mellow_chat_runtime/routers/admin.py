from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mellow_chat_runtime import app_state
from mellow_chat_runtime.core.domain_lookup_store import get_domain_store

router = APIRouter(prefix="/admin", tags=["Admin"])

CHARACTER_SECTION_MAP = {
    "user": "user_characters",
    "bot": "bot_characters",
}


class CharacterUpsertRequest(BaseModel):
    type: Literal["user", "bot"]
    data: Dict[str, Any]


class MemoryUpsertRequest(BaseModel):
    data: Dict[str, Any]


class RelationshipUpsertRequest(BaseModel):
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    data: Dict[str, Any]


class LoreUpsertRequest(BaseModel):
    data: Dict[str, Any]


def _store():
    settings = app_state.settings
    data_path = getattr(settings, "domain_data_file", None) if settings else None
    return get_domain_store(data_path=data_path)


@router.get("/characters")
async def list_characters() -> Dict[str, List[Dict[str, Any]]]:
    store = _store()
    return {
        "user": list(store.list_section("user_characters").values()),
        "bot": list(store.list_section("bot_characters").values()),
    }


@router.get("/characters/{character_type}/{character_id}")
async def get_character(character_type: Literal["user", "bot"], character_id: str) -> Dict[str, Any]:
    store = _store()
    section = CHARACTER_SECTION_MAP[character_type]
    item = store.get_section_item(section, character_id)
    if not item:
        raise HTTPException(status_code=404, detail="Character not found")
    return item


@router.put("/characters/{character_type}/{character_id}")
async def upsert_character(character_type: Literal["user", "bot"], character_id: str, request: CharacterUpsertRequest) -> Dict[str, Any]:
    if request.type != character_type:
        raise HTTPException(status_code=400, detail="Character type mismatch")
    store = _store()
    section = CHARACTER_SECTION_MAP[character_type]
    payload = dict(request.data)
    payload["id"] = character_id
    payload["type"] = character_type
    store.upsert(section, character_id, payload)
    return {"success": True, "section": section, "item": store.get_section_item(section, character_id)}


@router.delete("/characters/{character_type}/{character_id}")
async def delete_character(character_type: Literal["user", "bot"], character_id: str) -> Dict[str, Any]:
    store = _store()
    section = CHARACTER_SECTION_MAP[character_type]
    deleted = store.delete(section, character_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Character not found")
    return {"success": True, "section": section, "deleted_id": character_id}


@router.get("/memories/{character_id}")
async def get_memory(character_id: str) -> Dict[str, Any]:
    store = _store()
    return store.get_memory_and_possessions(character_id)


@router.put("/memories/{character_id}")
async def upsert_memory(character_id: str, request: MemoryUpsertRequest) -> Dict[str, Any]:
    store = _store()
    payload = dict(request.data)
    payload["character_id"] = character_id
    store.upsert("memories", character_id, payload)
    return {"success": True, "item": store.get_section_item("memories", character_id)}


@router.get("/relationships/{source_id}")
async def list_relationships(source_id: str, target_id: Optional[str] = None) -> Dict[str, Any]:
    store = _store()
    counterpart_ids = [target_id] if target_id else None
    return {"items": store.get_relationships(source_id, counterpart_ids=counterpart_ids)}


@router.put("/relationships")
async def upsert_relationship(request: RelationshipUpsertRequest) -> Dict[str, Any]:
    store = _store()
    relationship_section = store.list_section("relationships")
    source_map = relationship_section.get(request.source_id, {})
    if not isinstance(source_map, dict):
        source_map = {}
    source_map[request.target_id] = {**request.data, "target_id": request.target_id}
    store.upsert("relationships", request.source_id, source_map)
    return {"success": True, "items": store.get_relationships(request.source_id, counterpart_ids=[request.target_id])}


@router.get("/lore/{lore_id}")
async def get_lore_item(lore_id: str) -> Dict[str, Any]:
    store = _store()
    item = store.get_section_item("lorebook", lore_id)
    if not item:
        raise HTTPException(status_code=404, detail="Lore not found")
    return item


@router.put("/lore/{lore_id}")
async def upsert_lore_item(lore_id: str, request: LoreUpsertRequest) -> Dict[str, Any]:
    store = _store()
    payload = dict(request.data)
    payload["id"] = lore_id
    store.upsert("lorebook", lore_id, payload)
    return {"success": True, "item": store.get_section_item("lorebook", lore_id)}
