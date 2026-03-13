from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_system_prompt(
    persona: Dict[str, Any],
    dialogue_priority: Dict[str, Any],
    active_character: Optional[Dict[str, Any]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
) -> str:
    persona_desc = persona.get("description", "")
    priority_rules = dialogue_priority.get("rules", "")
    active_character = active_character or {}
    relationships = relationships or []
    character_name = active_character.get("name") or "Unknown Character"
    speech_style = active_character.get("speech_style", {}) if isinstance(active_character.get("speech_style"), dict) else {}
    tone = speech_style.get("tone") or active_character.get("tone") or "neutral"
    forbidden = speech_style.get("forbidden", [])
    if not isinstance(forbidden, list):
        forbidden = []
    forbidden_text = ", ".join(str(item).strip() for item in forbidden if str(item).strip()) or "none"
    relationship_keys = active_character.get("relationship_keys", [])
    if not isinstance(relationship_keys, list):
        relationship_keys = []
    relationship_text = ", ".join(str(item).strip() for item in relationship_keys if str(item).strip()) or "none"
    role_context = active_character.get("profile") or active_character.get("role") or active_character.get("type") or "character"
    relationship_lines: List[str] = []
    for item in relationships[:4]:
        target_id = item.get("target_id", "unknown")
        summary = item.get("summary", "")
        rel_tone = item.get("tone", "neutral")
        boundaries = item.get("boundaries", [])
        if not isinstance(boundaries, list):
            boundaries = []
        boundary_text = ", ".join(str(boundary).strip() for boundary in boundaries if str(boundary).strip()) or "none"
        relationship_lines.append(f"- {target_id}: {summary} | tone={rel_tone} | boundaries={boundary_text}")
    relationship_block = "\n".join(relationship_lines) if relationship_lines else "- none"
    return (
        f"You are {character_name}.\n\n"
        "Role:\n"
        f"{role_context}\n\n"
        "Tone:\n"
        f"{tone}\n\n"
        "Rules:\n"
        "- Stay in character\n"
        "- Keep responses coherent with provided world data\n"
        "- Do not break the fourth wall\n"
        "- Avoid forbidden behaviors\n\n"
        "Priority Order:\n"
        "1. Current scene rules and scene goal\n"
        "2. World-state constraints and continuity\n"
        "3. Character memories and relationship context\n"
        "4. Lorebook facts for support and terminology\n\n"
        "Forbidden:\n"
        f"{forbidden_text}\n\n"
        "Identity Context:\n"
        f"Relationship keys: {relationship_text}\n"
        f"Persona: {persona_desc}\n"
        f"Dialogue policy: {priority_rules}\n\n"
        "Relationship Context:\n"
        f"{relationship_block}\n\n"
        "Conversation context and memory follow.\n"
        "Do not invent external facts when domain data is provided."
    )


def build_user_prompt(
    user_text: str,
    user_profile: Dict[str, Any],
    lore: Dict[str, Any],
    memories: Dict[str, Any],
    world_state: Dict[str, Any],
    scene_state: Dict[str, Any],
    relationships: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    parts: List[str] = []
    relationships = relationships or []
    prioritized_memories = memories.get("important_memories", []) if isinstance(memories, dict) else []
    if not isinstance(prioritized_memories, list):
        prioritized_memories = []
    prioritized_memories = [str(item).strip() for item in prioritized_memories if str(item).strip()][:5]
    world_facts = world_state.get("facts", []) if isinstance(world_state, dict) else []
    if not isinstance(world_facts, list):
        world_facts = []
    world_facts = [str(item).strip() for item in world_facts if str(item).strip()][:5]
    relationship_summary = []
    for item in relationships[:4]:
        target_id = item.get("target_id", "unknown")
        summary = item.get("summary", "")
        tone = item.get("tone", "neutral")
        relationship_summary.append(f"{target_id}: {summary} (tone={tone})")

    if history:
        recent = history[-6:]
        parts.append("Recent Conversation:\n" + "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent))

    parts.append(
        "Priority Context:\n"
        f"Scene first: {json.dumps(scene_state, ensure_ascii=False)}\n"
        f"World constraints: {json.dumps({'facts': world_facts, 'location': world_state.get('location'), 'time': world_state.get('time'), 'state': world_state.get('state')}, ensure_ascii=False)}\n"
        f"Character memory: {json.dumps({'important_memories': prioritized_memories, 'possessions': memories.get('possessions', [])}, ensure_ascii=False)}\n"
        f"Relationship context: {json.dumps(relationship_summary, ensure_ascii=False)}\n"
        f"Lore support: {json.dumps(lore, ensure_ascii=False)}"
    )
    parts.append("User Profile:\n" + json.dumps(user_profile, ensure_ascii=False))
    parts.append("Current User Message:\n" + user_text)
    return "\n\n".join(parts)
