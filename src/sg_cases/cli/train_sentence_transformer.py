"""CLI for sentence-transformer training."""

from __future__ import annotations

import argparse

from ..training import train_sentence_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the semantic sentence-transformer.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    args = parser.parse_args()
    output_dir = train_sentence_transformer(config_path=args.config)
    print(output_dir)
