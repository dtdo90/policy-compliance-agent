"""CLI for the Gradio demo app."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the privacy-safe demo app.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Demo config path.")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share links.")
    args = parser.parse_args()

    from ..demo.app import launch_demo_app

    launch_demo_app(
        config_path=args.config,
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
