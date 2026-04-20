"""CLI for cross-encoder training."""

from __future__ import annotations

import argparse

from ..training import train_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the semantic cross-encoder.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    args = parser.parse_args()
    output_dir = train_cross_encoder(config_path=args.config)
    print(output_dir)
