from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from mellow_chat_runtime.runtime.schemas import StatusResponse, TurnRequest, TurnResponse


class RuntimeAdapter(ABC):
    @abstractmethod
    async def turn(self, req: TurnRequest, trace_id: Optional[str] = None) -> TurnResponse:
        ...

    @abstractmethod
    async def status(self) -> StatusResponse:
        ...
