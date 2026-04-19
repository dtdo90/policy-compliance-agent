"""Optional metadata-enriched inference kept separate from semantic app inference."""

from __future__ import annotations

from typing import Any

from ..core.config import load_config
from .semantic import run_semantic_inference


def run_enriched_inference(
    input_path: str | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Temporary compatibility wrapper.

    The standardized project keeps metadata tooling under ``sg_cases.metadata`` and
    keeps the app-facing inference path semantic-only. Until metadata verifiers are
    reintroduced on top of the packaged semantic analyzer, this wrapper delegates to
    semantic inference.
    """

    config = config or load_config(config_path)
    return run_semantic_inference(input_path=input_path, config=config)


def main(config_path: str | None = None, input_path: str | None = None) -> dict[str, dict[str, Any]]:
    return run_enriched_inference(input_path=input_path, config_path=config_path)
