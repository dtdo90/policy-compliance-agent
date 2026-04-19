---
description: Review synthetic coverage gaps after the latest loop run.
agent: coverage-analyst
---

Review synthetic coverage for the latest Agentic Loop run.

Inspect:

- `data/results/demo/agentic/review_items.json`
- `data/results/demo/agentic/coverage_analysis.json`
- `data/results/demo/agentic/synthetic_extensions.json`
- `data/results/demo/demo_synthetic_dataset.json`

Focus on corrected verifier mistakes and answer:

- Which anchor and label needed augmentation?
- Was the case already covered or a true coverage gap?
- Were generated variants appropriate, or could they confuse the model?
- What should be added, removed, or human-reviewed before retraining again?
