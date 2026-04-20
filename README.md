# Policy Compliance Agent

Policy Compliance Agent is a demo for agentic policy compliance on transcripts. It shows a practical workflow for:

1. running policy inference with a retriever and verifier model
2. reviewing pass and borderline evidence with a local LLM
3. approving new training phrases with human oversight
4. retraining on the original synthetic dataset plus the latest approved phrases
5. comparing before/after scores and diagnosing regressions

The demo models two transcript controls as semantic anchors. Rule 101 checks whether a helpdesk agent makes identity verification mandatory before an account reset or unlock, using one anchor for "verify identity before reset/unlock." Rule 102 checks whether a travel agent discloses both required booking-change costs before confirmation, using two mandatory anchors: one for the change fee and one for the fare difference or travel-credit impact. Rule 101 passes only when verification is clearly required before reset/unlock; Rule 102 passes only when both cost disclosures appear before the booking change is confirmed.

## Project Layout

- `configs/demo.yaml`: demo configuration
- `resources/demo_disclosures.json`: synthetic demo policy anchors
- `data/results/demo/demo_synthetic_dataset.json`: original synthetic training dataset
- `data/results/demo/transcript_scripts/`: sample transcript text files
- `src/policy_compliance_agent/demo/`: Gradio app and app-facing services
- `src/policy_compliance_agent/agentic/`: deterministic agentic loop orchestration
- `src/policy_compliance_agent/inference/`: retriever and verifier inference
- `src/policy_compliance_agent/training/`: sentence-transformer and cross-encoder training
- `tests/`: regression tests for the demo workflow

The source root is `src/`; the Python package imports as `policy_compliance_agent`.

## Setup

Install `uv`, then sync the locked project environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra demo --extra dev
```

Install OpenCode if you want to use the agentic wrapper:

```bash
curl -fsSL https://opencode.ai/install | bash
opencode --version
```

Optional local LLM support for app chat and labeling:

```bash
ollama pull qwen3:4b
ollama serve
```

The app still runs without Ollama for deterministic inference, but LLM-assisted labeling and chat require Ollama.

## First-Time Model Setup

The trained demo models are generated locally and are not committed to git.

```bash
uv run policy-train-retriever --config configs/demo.yaml
uv run policy-train-verifier --config configs/demo.yaml
uv run python -c "from policy_compliance_agent.demo import freeze_current_demo_baseline; freeze_current_demo_baseline(config_path='configs/demo.yaml')"
```

## Run The App

```bash
uv run policy-demo-app --config configs/demo.yaml
```

Open:

```text
http://127.0.0.1:7860
```

## Run The Agentic Loop CLI

```bash
uv run policy-agentic-loop --config configs/demo.yaml
```

With a folder of transcript `.txt` files:

```bash
uv run policy-agentic-loop \
  --config configs/demo.yaml \
  --transcripts data/results/demo/transcript_scripts
```

Require manual review instead of auto-approval:

```bash
uv run policy-agentic-loop \
  --config configs/demo.yaml \
  --require-human-review
```

## Run Tests

```bash
uv run pytest
```

## Retraining Data

- The original synthetic dataset is kept unchanged.
- Each retraining run uses `original synthetic dataset + latest approved phrases`.
- Approved phrases are overwritten per run to avoid mixing unrelated experiments.

## OpenCode

OpenCode is optional. It provides an agent-style way to run and inspect the same deterministic Python loop.

```bash
uv sync --extra demo --extra dev
opencode
```

Useful commands inside OpenCode:

```text
/agentic-loop-run
/agentic-loop-explain
/agentic-loop-coverage
/agentic-loop-audit
```
