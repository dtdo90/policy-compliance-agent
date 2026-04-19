---
description: Explain the latest Agentic Loop artifacts in plain English.
agent: agentic-supervisor
---

Explain the latest Agentic Loop run from local artifacts. Do not rerun the pipeline unless the user explicitly asks.

Inspect:

- `data/results/demo/agentic/loop_summary.json`
- `data/results/demo/agentic/review_items.json`
- `data/results/demo/agentic/coverage_analysis.json`
- `data/results/demo/agentic/holdout_metrics_before.json`
- `data/results/demo/agentic/holdout_metrics_after.json`

Answer in plain English:

- What happened in the run?
- Which phrases or anchors were corrected by review?
- Did coverage augmentation add anything useful?
- Did the holdout metrics improve or regress?
- What should the human reviewer check next?
