# OpenCode Wrapper For The Agentic Loop

This project keeps the Agentic Loop deterministic in Python, then uses OpenCode as an operator-facing agent layer around it. That split is intentional:

- Python owns the auditable workflow: train gate, inference, Qwen review, coverage augmentation, retrain, holdout evaluation, and promotion.
- OpenCode owns interactive supervision: run the loop, inspect artifacts, explain results, ask focused subagents to review coverage or promotion risk.

## What Was Added

- `opencode.json` configures OpenCode to use local Ollama with `qwen3:4b`.
- `.opencode/agents/agentic-supervisor.md` explains and supervises the full loop.
- `.opencode/agents/coverage-analyst.md` reviews synthetic coverage gaps.
- `.opencode/agents/holdout-auditor.md` audits holdout metrics and promotion safety.
- `.opencode/commands/agentic-loop-run.md` runs the Python loop and summarizes artifacts.
- `.opencode/commands/agentic-loop-explain.md` explains the latest run without rerunning it.
- `.opencode/commands/agentic-loop-coverage.md` focuses on failed/corrected cases and dataset coverage.
- `.opencode/commands/agentic-loop-audit.md` checks whether promotion was safe.

## How To Use It

1. Install OpenCode if it is not available on the laptop.
2. Make sure Ollama is running and the local model exists:

```bash
ollama pull qwen3:4b
ollama serve
```

3. Start OpenCode from this project root:

```bash
opencode
```

4. Try these slash commands:

```text
/agentic-loop-run
/agentic-loop-run --transcripts data/results/demo/transcript_scripts --require-human-review
/agentic-loop-explain
/agentic-loop-coverage
/agentic-loop-audit
```

## Relationship To The Gradio App

The Gradio **Agentic Loop** tab calls the same Python orchestrator directly. OpenCode is not embedded inside the Gradio process. It is a separate local agent interface that wraps the same CLI and artifact files.

This means the demo has two views of the same loop:

- Gradio: friendly UI for running and reviewing the workflow.
- OpenCode: agent framework playground for running, inspecting, and explaining the workflow.

## Source Of Truth

- Orchestrator: `src/sg_cases/agentic/loop.py`
- CLI: `src/sg_cases/cli/agentic_loop.py`
- Config: `configs/demo.yaml`
- Artifacts: `data/results/demo/agentic`
- Synthetic dataset: `data/results/demo/demo_synthetic_dataset.json`

## Notes

- The local model is set to `qwen3:4b` to match the laptop-friendly demo setup.
- If tool-use quality is weak with `qwen3:4b`, keep the Python loop deterministic and use OpenCode mostly for explanations and artifact review.
- Keep OpenCode runs focused on the bundled synthetic examples and generated artifacts.
