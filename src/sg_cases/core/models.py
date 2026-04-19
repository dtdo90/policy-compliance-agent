"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Disclaimer:
    id: str
    theme: str
    description: str
    purpose_of_control: str
    anchor: Any
    criteria: str
    keywords: list[Any] = field(default_factory=list)
