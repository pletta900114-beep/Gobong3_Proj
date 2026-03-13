from __future__ import annotations

from enum import Enum


class SystemState(str, Enum):
    IDLE = "IDLE"
    TEXT = "TEXT"


class TransitionResult(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_TRANSITION = "INVALID_TRANSITION"
