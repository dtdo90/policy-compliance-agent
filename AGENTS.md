# OpenCode Project Notes

This project is a personal demo for agentic policy review. The core workflow is deterministic Python, and OpenCode should wrap, inspect, and explain that workflow rather than reimplementing it.

## Useful Commands

- Sync dependencies: `uv sync --extra demo --extra dev`
- Run the local Agentic Loop: `uv run python -m sg_cases.cli.agentic_loop --config configs/demo.yaml`
- Require human review: `uv run python -m sg_cases.cli.agentic_loop --config configs/demo.yaml --require-human-review`
- Auto-approve LLM labels: `uv run python -m sg_cases.cli.agentic_loop --config configs/demo.yaml --auto-approve-llm`
- Run tests: `uv run pytest`
- Launch the Gradio app: `uv run python -m sg_cases.cli.demo_app --config configs/demo.yaml`

## Agentic Loop Contract

- The Python source of truth is `src/sg_cases/agentic/loop.py`.
- The app-facing Agentic Loop tab is in `src/sg_cases/demo/app.py`.
- Configuration lives in `configs/demo.yaml`.
- Loop artifacts are written under `data/results/demo/agentic`.
- Synthetic data lives at `data/results/demo/demo_synthetic_dataset.json`.
- Approved human/LLM additions live at `data/results/demo/approved_examples.json` and the augmented dataset path configured in `configs/demo.yaml`.

## Working Rules

- Do not edit the Transcript Review flow unless the user explicitly asks; it is currently the stable demo path.
- Keep OpenCode analysis auditable by citing the artifact files it inspected.
- Prefer read-only analysis unless the user asks to change code or data.
