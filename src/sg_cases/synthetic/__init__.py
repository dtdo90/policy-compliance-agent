"""Synthetic data generation exports."""

from __future__ import annotations

from typing import Any

from ..core.config import load_config


def generate(config: dict[str, Any] | None = None, config_path: str | None = None) -> str:
    config = config or load_config(config_path)
    backend = str(config.get("synthetic", {}).get("backend", "transformers_qwen")).strip().lower()

    if backend in {"", "transformers_qwen", "qwen", "local_qwen"}:
        from .qwen_generation import generate as _generate

        return _generate(config=config)

    if backend in {"external_api", "api"}:
        from .external_api_generation import generate as _generate

        return _generate(config=config)

    raise ValueError(f"Unsupported synthetic backend: {backend}")


def generate_synthetic_data(*args, **kwargs):
    from .synthetic_data_generation import generate_synthetic_data as _generate_synthetic_data

    return _generate_synthetic_data(*args, **kwargs)


def generate_transcripts(*args, **kwargs):
    from .synthetic_script_generation import generate_transcripts as _generate_transcripts

    return _generate_transcripts(*args, **kwargs)

__all__ = [
    "generate",
    "generate_synthetic_data",
    "generate_transcripts",
]
