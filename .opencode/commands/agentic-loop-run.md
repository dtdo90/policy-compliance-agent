---
description: Run the local deterministic Agentic Loop and summarize the result.
agent: agentic-supervisor
---

Run the local Agentic Loop with any extra CLI arguments the user supplied after this command.

Use:

```bash
uv run policy-agentic-loop --config configs/demo.yaml $ARGUMENTS
```

After the command finishes, inspect:

- `data/results/demo/agentic/loop_summary.json`
- `data/results/demo/agentic/synthetic_gate_metrics.json`
- `data/results/demo/agentic/synthetic_quality_report.json`
- `data/results/demo/agentic/coverage_analysis.json`
- `data/results/demo/agentic/holdout_metrics_before.json`
- `data/results/demo/agentic/holdout_metrics_after.json`

Return a concise operator summary:

- Stage status: synthetic gate, incoming review, coverage augmentation, retrain, holdout evaluation.
- How many incoming transcripts and review items were processed.
- Which examples were added or skipped.
- Whether the candidate model was promoted.
- Any risks or next actions.
