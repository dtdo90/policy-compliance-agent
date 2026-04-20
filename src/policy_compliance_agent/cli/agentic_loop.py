"""CLI for the local-LLM-assisted agentic loop."""

from __future__ import annotations

import argparse
import json

from ..agentic import run_local_agentic_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference/review/retrain on incoming transcripts against a fixed holdout set.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Config path.")
    parser.add_argument("--transcripts", default=None, help="Optional incoming transcript source override.")
    parser.add_argument("--holdout", default=None, help="Optional fixed holdout dataset override.")
    approval_group = parser.add_mutually_exclusive_group()
    approval_group.add_argument("--auto-approve-llm", dest="auto_approve_llm", action="store_true", help="Trust LLM labels automatically.")
    approval_group.add_argument("--require-human-review", dest="auto_approve_llm", action="store_false", help="Stop after LLM review and wait for human approval.")
    parser.set_defaults(auto_approve_llm=None)
    args = parser.parse_args()

    summary = run_local_agentic_loop(
        config_path=args.config,
        transcripts_source=args.transcripts,
        holdout_source=args.holdout,
        auto_approve_llm=args.auto_approve_llm,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
