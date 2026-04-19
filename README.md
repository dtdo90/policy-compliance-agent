# Policy Compliance Agent

Policy Compliance Agent is a personal demo for agentic policy compliance on transcripts. It shows a practical workflow for:

- running policy inference with a retriever and verifier model
- reviewing pass and borderline evidence with a local LLM
- approving new training phrases with human oversight
- retraining on the original synthetic dataset plus the latest approved phrases
- comparing before/after scores and diagnosing regressions

The bundled examples use fictional policies and transcripts.

## Project Layout

- `configs/demo.yaml`: demo configuration
- `resources/demo_disclosures.json`: synthetic demo policy anchors
- `data/results/demo/demo_synthetic_dataset.json`: original synthetic training dataset
- `data/results/demo/transcript_scripts/`: sample transcript text files
- `src/sg_cases/demo/`: Gradio app and app-facing services
- `src/sg_cases/agentic/`: deterministic agentic loop orchestration
- `src/sg_cases/inference/`: retriever and verifier inference
- `src/sg_cases/training/`: sentence-transformer and cross-encoder training
- `tests/`: regression tests for the demo workflow

## Setup

Create a local environment:

```bash
python3 -m venv .venv-local
source .venv-local/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[demo,dev]"
```

Optional local LLM support:

```bash
ollama pull qwen3:4b
ollama serve
```

The app still runs without Ollama for deterministic inference, but LLM-assisted labeling and chat require Ollama.

## First-Time Model Setup

The trained demo models are generated locally and are not committed to git.

```bash
source .venv-local/bin/activate
python -m sg_cases.cli.train_sentence_transformer --config configs/demo.yaml
python -m sg_cases.cli.train_cross_encoder --config configs/demo.yaml
python -c "from sg_cases.demo import freeze_current_demo_baseline; freeze_current_demo_baseline(config_path='configs/demo.yaml')"
```

## Run The App

```bash
source .venv-local/bin/activate
python -m sg_cases.cli.demo_app --config configs/demo.yaml
```

Open:

```text
http://127.0.0.1:7860
```

## Run The Agentic Loop CLI

```bash
source .venv-local/bin/activate
python -m sg_cases.cli.agentic_loop --config configs/demo.yaml
```

With a folder of transcript `.txt` files:

```bash
python -m sg_cases.cli.agentic_loop \
  --config configs/demo.yaml \
  --transcripts data/results/demo/transcript_scripts
```

Require manual review instead of auto-approval:

```bash
python -m sg_cases.cli.agentic_loop \
  --config configs/demo.yaml \
  --require-human-review
```

## Run Tests

```bash
source .venv-local/bin/activate
python -m pytest
```

## Data Policy

- Keep the original synthetic dataset unchanged.
- Each retraining run should use only `original synthetic dataset + latest approved phrases`.
- Avoid accumulating approved phrases across unrelated runs.
- Keep examples synthetic and easy to review.

## OpenCode

OpenCode is optional. It provides an agent-style way to run and inspect the same deterministic Python loop.

```bash
opencode
```

Useful commands inside OpenCode:

```text
/agentic-loop-run
/agentic-loop-explain
/agentic-loop-coverage
/agentic-loop-audit
```
