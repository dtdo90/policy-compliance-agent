"""CLI for semantic inference."""

from __future__ import annotations

import argparse

from ..inference import run_semantic_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic compliance inference over transcript txt files.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--input", default=None, help="Optional transcript folder override.")
    args = parser.parse_args()
    run_semantic_inference(input_path=args.input, config_path=args.config)
