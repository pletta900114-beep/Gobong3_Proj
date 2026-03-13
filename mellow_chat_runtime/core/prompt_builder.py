from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_system_prompt(persona: Dict[str, Any], dialogue_priority: Dict[str, Any]) -> str:
    persona_desc = persona.get("description", "")
    priority_rules = dialogue_priority.get("rules", "")
    return (
        "You are a text-only roleplay chatbot. Keep responses coherent with provided world data.\n"
        f"Persona: {persona_desc}\n"
        f"Dialogue policy: {priority_rules}\n"
        "Do not invent external facts when domain data is provided."
    )


def build_user_prompt(
    user_text: str,
    user_profile: Dict[str, Any],
    lore: Dict[str, Any],
    memories: Dict[str, Any],
    world_state: Dict[str, Any],
    scene_state: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    parts: List[str] = []
    if history:
        recent = history[-6:]
        parts.append("Recent Conversation:\n" + "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent))

    parts.append("User Profile:\n" + json.dumps(user_profile, ensure_ascii=False))
    parts.append("Lorebook:\n" + json.dumps(lore, ensure_ascii=False))
    parts.append("Important Memories & Possessions:\n" + json.dumps(memories, ensure_ascii=False))
    parts.append("Current World State:\n" + json.dumps(world_state, ensure_ascii=False))
    parts.append("Current Scene State:\n" + json.dumps(scene_state, ensure_ascii=False))
    parts.append("Current User Message:\n" + user_text)
    return "\n\n".join(parts)
