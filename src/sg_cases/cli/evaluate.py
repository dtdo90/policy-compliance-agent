"""CLI for report evaluation."""

from __future__ import annotations

import argparse
import json

from ..evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a report against ground truth metadata.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--report", required=True, help="Path to the report JSON.")
    parser.add_argument("--truth", required=True, help="Path to the truth/metadata JSON.")
    args = parser.parse_args()
    results, missed_cases = evaluate(report_path=args.report, truth_path=args.truth, config_path=args.config)
    print(json.dumps(results, indent=2))
