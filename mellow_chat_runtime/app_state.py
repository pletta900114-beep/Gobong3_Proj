from __future__ import annotations

import asyncio
from typing import Optional

settings = None
orchestrator = None
llm_service = None
vector_retrieval_service = None

SESSION_BUSY = set()
SESSION_BUSY_LOCK = asyncio.Lock()
