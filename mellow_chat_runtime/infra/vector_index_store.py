from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List


class JsonVectorIndexStore:
    def __init__(self, data_path: Path) -> None:
        self._data_path = data_path
        self._lock = RLock()
        self._data: Dict[str, List[Dict[str, Any]]] = {
            "lore_entries": [],
            "memory_entries": [],
            "relationship_entries": [],
        }
        self._load()

    def _load(self) -> None:
        if not self._data_path.exists():
            return
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return
        for collection in self._data:
            values = raw.get(collection)
            if isinstance(values, list):
                self._data[collection] = [dict(item) for item in values if isinstance(item, dict)]

    def replace_collection(self, collection: str, entries: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._data[collection] = [dict(item) for item in entries]
            self._save()

    def list_entries(self, collection: str) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._data.get(collection, [])]

    def _save(self) -> None:
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
