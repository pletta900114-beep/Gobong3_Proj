from __future__ import annotations

from typing import Any, Dict, List


SEARCHABLE_SECTIONS = {
    "lorebook": {"topic", "aliases", "content", "summary_text"},
    "memories": {"important_memories", "summary_text"},
    "relationships": {"summary", "tone", "boundaries", "shared_memories", "summary_text"},
}


def build_lore_summary(payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("summary_text", "")).strip()
    if explicit:
        return explicit
    aliases = _clean_list(payload.get("aliases", []))
    parts = [str(payload.get("topic", "")).strip(), *aliases, str(payload.get("content", "")).strip()]
    return " | ".join(part for part in parts if part)


def build_memory_summary(payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("summary_text", "")).strip()
    if explicit:
        return explicit
    return " | ".join(_clean_list(payload.get("important_memories", [])))


def build_relationship_summary(payload: Dict[str, Any]) -> str:
    explicit = str(payload.get("summary_text", "")).strip()
    if explicit:
        return explicit
    parts = [
        str(payload.get("summary", "")).strip(),
        str(payload.get("tone", "")).strip(),
        " ".join(_clean_list(payload.get("shared_memories", []))),
        " ".join(_clean_list(payload.get("boundaries", []))),
    ]
    return " | ".join(part for part in parts if part)


def searchable_fields_changed(section: str, existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    keys = SEARCHABLE_SECTIONS.get(section, set())
    for key in keys:
        if _normalized(existing.get(key)) != _normalized(incoming.get(key)):
            return True
    return False


def prepare_searchable_payload(section: str, key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(payload)
    if section == "lorebook":
        updated.setdefault("id", key)
        updated["summary_text"] = build_lore_summary(updated)
    elif section == "memories":
        updated.setdefault("character_id", key)
        updated["summary_text"] = build_memory_summary(updated)
    elif section == "relationships":
        updated["summary_text"] = build_relationship_summary(updated)
    return updated


def _clean_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _normalized(value: Any) -> Any:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return ""
    return str(value).strip()
