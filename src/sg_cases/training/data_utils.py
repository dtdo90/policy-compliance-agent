"""Helpers for preparing synthetic training rows."""

from __future__ import annotations

from typing import Any


DEFAULT_SENTENCE_TRANSFORMER_BASE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_BASE = "cross-encoder/ms-marco-MiniLM-L12-v2"

def _clean_anchor_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_prompt_index(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_training_anchor_text(entry: dict[str, Any], disclosures: dict[str, Any]) -> str | None:
    explicit_anchor = entry.get("anchor")
    if isinstance(explicit_anchor, str):
        text = explicit_anchor.strip()
        if text:
            return text

    disclaimer_id = str(entry.get("disclaimer_id", "")).strip()
    if not disclaimer_id:
        return None

    disclosure = disclosures.get(disclaimer_id, {})
    if not isinstance(disclosure, dict):
        return None

    anchor_data = disclosure.get("anchor", "")
    if isinstance(anchor_data, str):
        text = anchor_data.strip()
        return text if text else None

    if isinstance(anchor_data, list):
        ordered_anchors = _clean_anchor_values(anchor_data)
        if not ordered_anchors:
            return None

        prompt_index = _normalize_prompt_index(entry.get("prompt_index"))
        if prompt_index is None:
            return ordered_anchors[0] if len(ordered_anchors) == 1 else None
        if not (0 <= prompt_index < len(ordered_anchors)):
            return None
        return ordered_anchors[prompt_index]

    if not isinstance(anchor_data, dict):
        return None

    ordered_anchors = _clean_anchor_values(anchor_data.get("mandatory")) + _clean_anchor_values(anchor_data.get("standard"))
    if not ordered_anchors:
        return None

    prompt_index = _normalize_prompt_index(entry.get("prompt_index"))
    if prompt_index is None:
        return None

    if not (0 <= prompt_index < len(ordered_anchors)):
        return None

    return ordered_anchors[prompt_index]
