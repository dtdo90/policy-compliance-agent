"""JSON helpers."""

from __future__ import annotations

import json
from typing import Any

from .paths import ensure_parent_dir, resolve_project_path


def safe_json_load(path: str, default: Any) -> Any:
    try:
        with resolve_project_path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: str, data: Any) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
