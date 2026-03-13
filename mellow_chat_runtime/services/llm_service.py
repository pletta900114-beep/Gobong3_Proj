from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class LLMStatus(Enum):
    DISCONNECTED = auto()
    CONNECTED = auto()
    GENERATING = auto()
    ERROR = auto()


class ModelType(Enum):
    FAST = "fast"
    THINKING = "thinking"
    RESEARCH = "research"


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatContext:
    system_prompt: str = ""
    messages: List[ChatMessage] = field(default_factory=list)
    max_history: int = 20

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]

    def get_messages(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        out.extend([m.to_dict() for m in self.messages])
        return out


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_generated: int = 0
    generation_time_ms: float = 0.0
    is_complete: bool = True
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class GenerationResult:
    content: str
    model: str
    eval_count: int = 0
    eval_duration_ms: float = 0.0
    prompt_eval_count: int = 0


class LLMServiceError(Exception):
    pass


class LLMService:
    DEFAULT_MODELS = {
        ModelType.FAST: "qwen3.5:9b",
        ModelType.THINKING: "qwen3.5:9b",
        ModelType.RESEARCH: "qwen3.5:9b",
    }

    def __init__(self, host: str = "localhost", port: int = 11434, timeout: float = 60.0, models: Optional[Dict[ModelType, str]] = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._base_url = f"http://{host}:{port}"
        self._models = models or self.DEFAULT_MODELS.copy()
        self._status = LLMStatus.DISCONNECTED
        self._session: Optional[aiohttp.ClientSession] = None
        self._contexts: Dict[str, ChatContext] = {}
        self._default_context = ChatContext()
        self._reconnect_lock = asyncio.Lock()

    async def connect(self) -> bool:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        try:
            async with self._session.get(f"{self._base_url}/api/tags") as resp:
                if resp.status != 200:
                    raise LLMServiceError(f"Ollama returned {resp.status}")
            self._status = LLMStatus.CONNECTED
            return True
        except Exception as e:
            self._status = LLMStatus.ERROR
            raise LLMServiceError(str(e))

    async def _cleanup_session(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def disconnect(self) -> None:
        async with self._reconnect_lock:
            await self._cleanup_session()
            self._status = LLMStatus.DISCONNECTED

    async def _ensure_connected(self) -> bool:
        if self._session and not self._session.closed and self._status == LLMStatus.CONNECTED:
            return True
        try:
            await self.connect()
            return True
        except Exception:
            return False

    async def health_check(self) -> bool:
        if not self._session or self._session.closed:
            return False
        try:
            async with self._session.get(f"{self._base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_status(self) -> LLMStatus:
        return self._status

    def is_ready(self) -> bool:
        return self._status == LLMStatus.CONNECTED

    def is_available(self) -> bool:
        return self._status in (LLMStatus.CONNECTED, LLMStatus.GENERATING)

    def get_model_for_mode(self, mode: str) -> str:
        m = (mode or "fast").strip().lower()
        if m == "thinking":
            return self._models[ModelType.THINKING]
        if m == "research":
            return self._models[ModelType.RESEARCH]
        return self._models[ModelType.FAST]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if not await self._ensure_connected():
            raise LLMServiceError("LLM service unavailable")

        use_model = model or self.get_model_for_mode("fast")
        payload: Dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools

        self._status = LLMStatus.GENERATING
        try:
            assert self._session is not None
            async with self._session.post(f"{self._base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    raise LLMServiceError(await resp.text())
                data = await resp.json()
            message = data.get("message", {})
            text = message.get("content") or ""
            tool_calls = message.get("tool_calls")
            return LLMResponse(
                text=text,
                model=use_model,
                tokens_generated=int(data.get("eval_count", 0) or 0),
                generation_time_ms=float((data.get("eval_duration", 0) or 0) / 1_000_000),
                is_complete=True,
                tool_calls=tool_calls,
            )
        finally:
            self._status = LLMStatus.CONNECTED

    async def generate(self, prompt: str, system_prompt: str = "", mode: str = "fast", context_id: Optional[str] = None, **kwargs: Any) -> GenerationResult:
        context = self._get_context(context_id)
        if system_prompt:
            context.system_prompt = system_prompt
        messages = context.get_messages() + [{"role": "user", "content": prompt}]
        response = await self.chat(messages=messages, model=self.get_model_for_mode(mode), **kwargs)
        context.add_message("user", prompt)
        context.add_message("assistant", response.text)
        return GenerationResult(
            content=response.text,
            model=response.model,
            eval_count=response.tokens_generated,
            eval_duration_ms=response.generation_time_ms,
            prompt_eval_count=0,
        )

    def _get_context(self, context_id: Optional[str]) -> ChatContext:
        if context_id is None:
            return self._default_context
        if context_id not in self._contexts:
            self._contexts[context_id] = ChatContext()
        return self._contexts[context_id]


def create_llm_service(host: str = "localhost", port: int = 11434, timeout: float = 60.0, models: Optional[Dict[str, str]] = None) -> LLMService:
    model_mapping = None
    if models:
        model_mapping = {
            ModelType.FAST: models.get("fast", LLMService.DEFAULT_MODELS[ModelType.FAST]),
            ModelType.THINKING: models.get("thinking", LLMService.DEFAULT_MODELS[ModelType.THINKING]),
            ModelType.RESEARCH: models.get("research", LLMService.DEFAULT_MODELS[ModelType.RESEARCH]),
        }
    return LLMService(host=host, port=port, timeout=timeout, models=model_mapping)

