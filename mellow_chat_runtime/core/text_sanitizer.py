from __future__ import annotations

import re
from typing import List

STOP_SEQUENCES: List[str] = [
    '<|im_start|>',
    '<|im_end|>',
    '<|endoftext|>',
    '</think>',
    '<think>',
]

_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r'<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>', re.IGNORECASE)
_ROLE_LINE_RE = re.compile(r'^(assistant|user|system)\s*:?\s*$', re.IGNORECASE)
_META_PREFIXES = (
    '**Constraints Checklist',
    '**Mental Sandbox Simulation',
    '**Draft:',
    'Draft:',
    'Checklist:',
    'Final Response:',
    'Wait,',
    '요청하신 대로',
    '답변드리겠습니다',
    '말투로 말씀드리겠습니다',
    '이 상황에서는',
    '사용자의 입력을 고려하면',
)
_META_PATTERNS = (
    'as an ai',
    'assistant:',
    'user:',
    'system:',
    'analysis:',
    'mental sandbox',
    'final response:',
    'checklist:',
    'draft:',
    'prompt in turn',
    "i'll focus on",
    'the user seems',
    'internal reasoning',
)


def sanitize_assistant_text(text: str) -> str:
    cleaned = _normalize_text(text)
    if not cleaned:
        return ''

    cleaned = _THINK_BLOCK_RE.sub('', cleaned)
    cleaned = _SPECIAL_TOKEN_RE.split(cleaned)[0].strip()

    lines = [line.rstrip() for line in cleaned.splitlines()]
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append('')
            continue
        if _ROLE_LINE_RE.match(stripped):
            break
        if any(stripped.startswith(prefix) for prefix in _META_PREFIXES):
            break
        kept.append(line)

    cleaned = '\n'.join(kept).strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def sanitize_history_text(role: str, text: str) -> str:
    if (role or '').strip().lower() == 'assistant':
        return sanitize_assistant_text(text)
    return normalize_user_text(text)


def normalize_user_text(text: str) -> str:
    return _normalize_text(text)


def sanitize_memory_text(text: str) -> str:
    cleaned = sanitize_assistant_text(text)
    if not cleaned:
        return ''
    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in _META_PATTERNS):
        return ''
    return cleaned


def has_forbidden_output_markers(text: str) -> bool:
    lowered = (text or '').lower()
    if any(token.lower() in lowered for token in STOP_SEQUENCES):
        return True
    return any(pattern in lowered for pattern in _META_PATTERNS)


def _normalize_text(text: str) -> str:
    cleaned = (text or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
