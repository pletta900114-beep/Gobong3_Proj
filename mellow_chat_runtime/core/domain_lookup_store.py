from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


class DomainLookupStore:
    """Base lookup store API for chatbot context lookup."""

    def get_persona(self, persona_id: str = "default") -> Dict[str, Any]:
        raise NotImplementedError

    def get_user_profile(self, profile_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_user_character(self, character_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_bot_character(self, character_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_lore(self, topic: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_memory_and_possessions(self, character_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_relationships(self, character_id: str, counterpart_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_world_state(self, world_id: str = "default") -> Dict[str, Any]:
        raise NotImplementedError

    def get_scene_state(self, scene_id: str = "scene_default") -> Dict[str, Any]:
        raise NotImplementedError

    def get_dialogue_priority(self, scene_id: str = "default") -> Dict[str, Any]:
        raise NotImplementedError

    def list_section(self, section: str) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def get_section_item(self, section: str, key: str) -> Dict[str, Any]:
        raise NotImplementedError

    def upsert(self, section: str, key: str, value: Dict[str, Any]) -> None:
        raise NotImplementedError

    def delete(self, section: str, key: str) -> bool:
        raise NotImplementedError


class JsonDomainLookupStore(DomainLookupStore):
    """Small JSON-backed domain store for chatbot context lookup."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self._data_path = data_path
        self._lock = RLock()
        self._data: Dict[str, Any] = {
            "personas": {
                "default": {
                    "id": "default",
                    "name": "Narrator",
                    "description": "A concise in-world guide who keeps continuity and tone.",
                }
            },
            "user_profiles": {
                "user_char_01": {
                    "id": "user_char_01",
                    "name": "Mellow",
                    "profile": "UI/UX designer and programmer focused on character immersion.",
                    "traits": ["design-driven", "continuity-focused"],
                }
            },
            "user_characters": {
                "user_char_01": {
                    "id": "user_char_01",
                    "type": "user",
                    "name": "Mellow",
                    "profile": "UI/UX designer and programmer",
                    "traits": ["design-driven", "character immersion"],
                    "relationship_keys": ["crew_main"],
                    "is_major": True,
                }
            },
            "bot_characters": {
                "bot_char_01": {
                    "id": "bot_char_01",
                    "type": "bot",
                    "name": "Aventurine",
                    "persona_id": "default",
                    "speech_style": {
                        "tone": "casual_confident",
                        "forbidden": ["out-of-world meta claims"],
                    },
                    "relationship_keys": ["crew_main"],
                    "is_major": True,
                },
                "bot_char_02": {
                    "id": "bot_char_02",
                    "type": "bot",
                    "name": "Sunday",
                    "persona_id": "default",
                    "speech_style": {
                        "tone": "measured_formal",
                        "forbidden": [],
                    },
                    "relationship_keys": ["crew_support"],
                    "is_major": False,
                },
            },
            "relationships": {
                "bot_char_01": {
                    "user_char_01": {
                        "target_id": "user_char_01",
                        "summary": "Treat Mellow as a trusted collaborator whose judgment matters.",
                        "tone": "warmly strategic",
                        "boundaries": ["Do not humiliate the user", "Do not dismiss prior agreements"],
                        "shared_memories": ["They stabilized a tense station negotiation together."],
                    },
                    "bot_char_02": {
                        "target_id": "bot_char_02",
                        "summary": "Respect Sunday as a careful ally with formal expectations.",
                        "tone": "polite but competitive",
                        "boundaries": ["Avoid open mockery in shared scenes"],
                        "shared_memories": ["They coordinated during a Penacony diplomatic review."],
                    },
                },
                "bot_char_02": {
                    "user_char_01": {
                        "target_id": "user_char_01",
                        "summary": "Treat Mellow as capable and sincerity-driven.",
                        "tone": "measured respect",
                        "boundaries": ["Avoid manipulative baiting"],
                        "shared_memories": ["They discussed public duty after the assembly hall oath."],
                    }
                },
            },
            "lorebook": {
                "lore_001": {
                    "id": "lore_001",
                    "topic": "Interastral Peace Corporation",
                    "aliases": ["IPC", "스타피스 컴퍼니"],
                    "content": "A galaxy-scale organization handling finance and contracts.",
                    "priority": 10,
                },
                "lore_002": {
                    "id": "lore_002",
                    "topic": "Stonehearts",
                    "aliases": ["스톤하트", "Ten Stonehearts"],
                    "content": "Senior strategic figures aligned with high-value IPC operations.",
                    "priority": 8,
                },
                "lore_003": {
                    "id": "lore_003",
                    "topic": "Astral Lounge Protocol",
                    "aliases": ["라운지 규약", "Lounge protocol"],
                    "content": "Negotiations in shared lounges prioritize calm tone and turn-taking.",
                    "priority": 6,
                },
            },
            "world_state": {
                "default": {
                    "id": "default",
                    "location": "Astral lounge deck",
                    "time": "Evening shift",
                    "state": "Stable",
                    "facts": ["Travel routes are open", "No immediate external crisis"],
                }
            },
            "scene_state": {
                "scene_default": {
                    "id": "scene_default",
                    "location": "Lounge",
                    "time": "Evening",
                    "participants": ["user_char_01", "bot_char_01", "bot_char_02"],
                    "goal": "Synchronize character plans",
                    "mood": "calm",
                    "rules": [
                        {"key": "include_speakers", "value": ["bot_char_01", "bot_char_02"]},
                        {"key": "exclude_speakers", "value": []},
                    ],
                }
            },
            "memories": {
                "user_char_01": {
                    "character_id": "user_char_01",
                    "important_memories": ["Past project incident", "Recent trust shift"],
                    "possessions": ["Card case", "Notebook"],
                },
                "bot_char_01": {
                    "character_id": "bot_char_01",
                    "important_memories": ["Debt negotiation at station nine"],
                    "possessions": ["Gemstone token"],
                },
                "bot_char_02": {
                    "character_id": "bot_char_02",
                    "important_memories": ["Public oath in Penacony assembly hall"],
                    "possessions": ["Ceremonial notebook"],
                },
            },
            "dialogue_priority": {
                "default": {
                    "major_weight": 1.0,
                    "minor_weight": 0.65,
                    "recency_penalty": 0.35,
                    "max_consecutive_turns": 1,
                    "rules": "Major characters lead the turn, but minor characters should still enter regularly.",
                }
            },
        }
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._data_path:
            return
        if not self._data_path.exists():
            return
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            self._data.update(raw)

    def _save_to_disk(self) -> None:
        if not self._data_path:
            return
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_persona(self, persona_id: str = "default") -> Dict[str, Any]:
        return dict(self._data.get("personas", {}).get(persona_id, self._data["personas"]["default"]))

    def get_user_profile(self, profile_id: str) -> Dict[str, Any]:
        return dict(self._data.get("user_profiles", {}).get(profile_id, {}))

    def get_user_character(self, character_id: str) -> Dict[str, Any]:
        return dict(self._data.get("user_characters", {}).get(character_id, {}))

    def get_bot_character(self, character_id: str) -> Dict[str, Any]:
        return dict(self._data.get("bot_characters", {}).get(character_id, {}))

    def get_lore(self, topic: str) -> Dict[str, Any]:
        lorebook = self._data.get("lorebook", {})
        if topic in lorebook:
            return dict(lorebook.get(topic, {}))
        normalized_topic = topic.strip().lower()
        for entry in lorebook.values():
            entry_topic = str(entry.get("topic", "")).strip().lower()
            aliases = [str(alias).strip().lower() for alias in entry.get("aliases", [])]
            if normalized_topic and (entry_topic == normalized_topic or normalized_topic in aliases):
                return dict(entry)
        return {}

    def get_memory_and_possessions(self, character_id: str) -> Dict[str, Any]:
        return dict(
            self._data.get("memories", {}).get(
                character_id,
                {"character_id": character_id, "important_memories": [], "possessions": []},
            )
        )

    def get_relationships(self, character_id: str, counterpart_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        section = self._data.get("relationships", {})
        raw = section.get(character_id, {})
        if not isinstance(raw, dict):
            return []
        allowed = {item.strip() for item in (counterpart_ids or []) if isinstance(item, str) and item.strip()}
        results: List[Dict[str, Any]] = []
        for target_id, payload in raw.items():
            if allowed and target_id not in allowed:
                continue
            if isinstance(payload, dict):
                item = dict(payload)
                item.setdefault("target_id", target_id)
                results.append(item)
        results.sort(key=lambda item: str(item.get("target_id", "")))
        return results

    def get_world_state(self, world_id: str = "default") -> Dict[str, Any]:
        return dict(self._data.get("world_state", {}).get(world_id, self._data["world_state"]["default"]))

    def get_scene_state(self, scene_id: str = "scene_default") -> Dict[str, Any]:
        section = self._data.get("scene_state", {})
        fallback = section.get("scene_default", {})
        return dict(section.get(scene_id, fallback))

    def get_dialogue_priority(self, scene_id: str = "default") -> Dict[str, Any]:
        return dict(self._data.get("dialogue_priority", {}).get(scene_id, self._data["dialogue_priority"]["default"]))

    def list_section(self, section: str) -> Dict[str, Dict[str, Any]]:
        raw = self._data.get(section, {})
        if not isinstance(raw, dict):
            return {}
        return {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}

    def get_section_item(self, section: str, key: str) -> Dict[str, Any]:
        raw = self._data.get(section, {})
        if not isinstance(raw, dict):
            return {}
        value = raw.get(key, {})
        return dict(value) if isinstance(value, dict) else {}

    def upsert(self, section: str, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            target = self._data.setdefault(section, {})
            if isinstance(target, dict):
                target[key] = value
                self._save_to_disk()

    def delete(self, section: str, key: str) -> bool:
        with self._lock:
            target = self._data.get(section, {})
            if not isinstance(target, dict) or key not in target:
                return False
            del target[key]
            self._save_to_disk()
            return True


class VectorDomainLookupStore(DomainLookupStore):
    """
    Vector-backed lookup adapter.
    Phase 1 scope: only lore lookup is routed to vector search, everything else
    delegates to JSON store to keep the current minimal runtime behavior.
    """

    def __init__(self, fallback_store: JsonDomainLookupStore, lore_search_url: Optional[str], timeout_sec: float = 2.0) -> None:
        self._fallback = fallback_store
        self._lore_search_url = (lore_search_url or "").strip() or None
        self._timeout_sec = float(timeout_sec or 2.0)

    def get_persona(self, persona_id: str = "default") -> Dict[str, Any]:
        return self._fallback.get_persona(persona_id)

    def get_user_profile(self, profile_id: str) -> Dict[str, Any]:
        return self._fallback.get_user_profile(profile_id)

    def get_user_character(self, character_id: str) -> Dict[str, Any]:
        return self._fallback.get_user_character(character_id)

    def get_bot_character(self, character_id: str) -> Dict[str, Any]:
        return self._fallback.get_bot_character(character_id)

    def get_lore(self, topic: str) -> Dict[str, Any]:
        result = self._vector_search_lore(topic)
        if result:
            return result
        return self._fallback.get_lore(topic)

    def get_memory_and_possessions(self, character_id: str) -> Dict[str, Any]:
        return self._fallback.get_memory_and_possessions(character_id)

    def get_relationships(self, character_id: str, counterpart_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self._fallback.get_relationships(character_id, counterpart_ids=counterpart_ids)

    def get_world_state(self, world_id: str = "default") -> Dict[str, Any]:
        return self._fallback.get_world_state(world_id)

    def get_scene_state(self, scene_id: str = "scene_default") -> Dict[str, Any]:
        return self._fallback.get_scene_state(scene_id)

    def get_dialogue_priority(self, scene_id: str = "default") -> Dict[str, Any]:
        return self._fallback.get_dialogue_priority(scene_id)

    def list_section(self, section: str) -> Dict[str, Dict[str, Any]]:
        return self._fallback.list_section(section)

    def get_section_item(self, section: str, key: str) -> Dict[str, Any]:
        return self._fallback.get_section_item(section, key)

    def upsert(self, section: str, key: str, value: Dict[str, Any]) -> None:
        self._fallback.upsert(section=section, key=key, value=value)

    def delete(self, section: str, key: str) -> bool:
        return self._fallback.delete(section=section, key=key)

    def _vector_search_lore(self, topic: str) -> Dict[str, Any]:
        if not self._lore_search_url:
            return {}
        payload = {"query": topic, "top_k": 1}
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url=self._lore_search_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except (URLError, OSError, TimeoutError, ValueError):
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        if isinstance(parsed, dict):
            direct = parsed.get("item")
            if isinstance(direct, dict):
                return direct
            results = parsed.get("results")
            if isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    if isinstance(first.get("item"), dict):
                        return dict(first["item"])
                    return first
        return {}


_global_store: Optional[DomainLookupStore] = None


def get_domain_store(
    data_path: Optional[Path] = None,
    backend: str = "json",
    vectordb_lore_search_url: Optional[str] = None,
    vectordb_timeout_sec: float = 2.0,
) -> DomainLookupStore:
    global _global_store
    if _global_store is None:
        json_store = JsonDomainLookupStore(data_path=data_path)
        if (backend or "json").strip().lower() == "vector":
            _global_store = VectorDomainLookupStore(
                fallback_store=json_store,
                lore_search_url=vectordb_lore_search_url,
                timeout_sec=vectordb_timeout_sec,
            )
        else:
            _global_store = json_store
    return _global_store
