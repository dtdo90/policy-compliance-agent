"""Disclosure loading helpers."""

from __future__ import annotations

import json
from typing import Iterable

from .models import Disclaimer
from .paths import resolve_project_path


def load_disclaimers(file_path: str) -> list[Disclaimer]:
    with resolve_project_path(file_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Disclosure file must be a JSON object: {file_path}")

    disclaimers: list[Disclaimer] = []
    for disclaimer_id, item in data.items():
        if not isinstance(item, dict):
            continue
        disclaimers.append(
            Disclaimer(
                id=str(disclaimer_id),
                theme=item.get("Theme", ""),
                description=item.get("Description", ""),
                purpose_of_control=item.get("Purpose_of_control", ""),
                anchor=item.get("anchor", item.get("Criteria", "")),
                criteria=item.get("Criteria", ""),
                keywords=item.get("Keywords", []),
            )
        )
    return disclaimers


def filter_disclaimers(
    disclaimers: Iterable[Disclaimer],
    include_rule_ids: Iterable[int | str] | None = None,
    exclude_rule_ids: Iterable[int | str] | None = None,
) -> list[Disclaimer]:
    include = {str(rule_id) for rule_id in include_rule_ids} if include_rule_ids else None
    exclude = {str(rule_id) for rule_id in exclude_rule_ids} if exclude_rule_ids else set()

    filtered: list[Disclaimer] = []
    for disclaimer in disclaimers:
        if include is not None and disclaimer.id not in include:
            continue
        if disclaimer.id in exclude:
            continue
        filtered.append(disclaimer)
    return filtered
