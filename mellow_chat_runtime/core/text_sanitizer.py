from __future__ import annotations

import re
import json
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
_FENCED_BLOCK_RE = re.compile(r"```(?:json|text|markdown)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
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
    cleaned = _SPECIAL_TOKEN_RE.sub('', cleaned).strip()
    cleaned = _unwrap_fenced_blocks(cleaned)
    cleaned = _salvage_structured_rp(cleaned)
    cleaned = _extract_last_rp_block(cleaned)

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
    cleaned = _SPECIAL_TOKEN_RE.sub('', cleaned).strip()
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


def _unwrap_fenced_blocks(text: str) -> str:
    match = _FENCED_BLOCK_RE.search(text or '')
    if not match:
        return text
    inner = match.group(1).strip()
    if inner:
        return inner
    return _FENCED_BLOCK_RE.sub('', text or '').strip()


def _salvage_structured_rp(text: str) -> str:
    candidate = (text or '').strip()
    if not candidate:
        return ''
    if not candidate.startswith(('{', '[')):
        return candidate
    try:
        data = json.loads(candidate)
    except Exception:
        return candidate
    rp_text = _structured_to_rp(data)
    return rp_text or candidate


def _structured_to_rp(data: object) -> str:
    if isinstance(data, list):
        for item in reversed(data):
            rp_text = _structured_to_rp(item)
            if rp_text:
                return rp_text
        return ''
    if not isinstance(data, dict):
        return ''

    narration = str(
        data.get('narration')
        or data.get('action')
        or data.get('scene')
        or ''
    ).strip()
    dialogue = str(
        data.get('dialogue')
        or data.get('speech')
        or data.get('line')
        or ''
    ).strip()
    if not narration and not dialogue:
        return ''
    if dialogue and not re.match(r'^[\"“].*[\"”]$', dialogue, re.DOTALL):
        dialogue = f'"{dialogue}"'
    if narration and dialogue:
        return f'{narration}\n\n{dialogue}'
    return narration or dialogue


def _extract_last_rp_block(text: str) -> str:
    cleaned = (text or '').strip()
    if not cleaned:
        return ''
    quote_positions = [m.start() for m in re.finditer(r'[\"“]', cleaned)]
    if not quote_positions:
        return cleaned
    last_quote = quote_positions[-1]
    narration_start = cleaned.rfind('\n\n', 0, last_quote)
    if narration_start == -1:
        candidate = cleaned[:].strip()
    else:
        prev_break = cleaned.rfind('\n\n', 0, narration_start)
        candidate = cleaned[(prev_break + 2 if prev_break != -1 else 0):].strip()
    return candidate or cleaned
