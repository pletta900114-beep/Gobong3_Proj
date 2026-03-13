from __future__ import annotations

from typing import Any, Dict


def apply_dialogue_weighting(reply: str, dialogue_priority: Dict[str, Any]) -> str:
    """Minimal post-processing hook for major/minor speaking priority policy."""
    major = float(dialogue_priority.get("major_weight", 1.0) or 1.0)
    minor = float(dialogue_priority.get("minor_weight", 0.5) or 0.5)
    if major >= minor:
        return reply
    # If minor would outweigh major, keep output concise to reduce over-dominance.
    return reply[:1200]
