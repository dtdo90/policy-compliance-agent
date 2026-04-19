---
description: Audit before/after holdout metrics and model promotion safety.
agent: holdout-auditor
---

Audit the latest candidate promotion decision.

Inspect:

- `configs/demo.yaml`
- `data/results/demo/agentic/loop_summary.json`
- `data/results/demo/agentic/holdout_metrics_before.json`
- `data/results/demo/agentic/holdout_metrics_after.json`
- `data/results/demo/agentic/holdout_predictions_before.json`
- `data/results/demo/agentic/holdout_predictions_after.json`

Return:

- Promotion decision: promote, do not promote, or inconclusive.
- Metric comparison using the configured promotion metric.
- Rule-level changes.
- Any examples that look risky despite the aggregate metric.
