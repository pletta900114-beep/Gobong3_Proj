from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mellow_chat_runtime.infra.database import ChatSession


@dataclass
class EffectiveModelSelection:
    provider: str
    model: str
    mode: str
    source: str


class ModelRoutingService:
    """Resolve model with simple precedence while staying session-centric."""

    def __init__(self, default_provider: str = "ollama") -> None:
        self._default_provider = default_provider

    def resolve(
        self,
        session: ChatSession,
        llm_service: object,
        mode: str = "fast",
        request_provider: Optional[str] = None,
        request_model: Optional[str] = None,
    ) -> EffectiveModelSelection:
        cleaned_mode = (mode or "fast").strip().lower() or "fast"
        if request_model:
            return EffectiveModelSelection(
                provider=(request_provider or self._default_provider).strip(),
                model=request_model.strip(),
                mode=cleaned_mode,
                source="request",
            )
        if session.selected_model_provider and session.selected_model_name:
            return EffectiveModelSelection(
                provider=session.selected_model_provider,
                model=session.selected_model_name,
                mode=session.selected_model_mode or cleaned_mode,
                source="session",
            )
        model_name = llm_service.get_model_for_mode(cleaned_mode)  # type: ignore[attr-defined]
        return EffectiveModelSelection(
            provider=self._default_provider,
            model=model_name,
            mode=cleaned_mode,
            source="system",
        )
