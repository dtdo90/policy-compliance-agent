"""CLI for synthetic generation."""

from __future__ import annotations

import argparse

from ..synthetic import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic RM snippets with Qwen.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    args = parser.parse_args()
    output_path = generate(config_path=args.config)
    print(output_path)
