"""Gradio app for the privacy-safe demo workflow."""

from __future__ import annotations

import base64
from copy import deepcopy
import html
import json
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from typing import Any

import pandas as pd

from ..agentic import answer_agentic_question
from ..agentic.loop import load_incoming_transcripts
from .services import (
    answer_demo_question,
    approve_demo_examples,
    complete_agentic_reinference_cycle,
    diagnose_score_regressions,
    label_borderline_items_with_ollama,
    load_demo_config,
    load_demo_samples,
    prepare_agentic_training_cycle,
    reset_demo_to_baseline,
    retrain_demo_verifier,
    run_agentic_review_cycle,
    run_demo_inference,
)


APP_CSS = """
#borderline-review-table table {
  table-layout: auto;
  width: 100%;
}

#borderline-review-table th,
#borderline-review-table td {
  white-space: normal !important;
  word-break: break-word;
  overflow-wrap: anywhere;
}

#borderline-review-table th:nth-child(1),
#borderline-review-table td:nth-child(1) {
  width: 84px;
  max-width: 84px;
}

#borderline-review-table th:nth-child(2),
#borderline-review-table td:nth-child(2) {
  width: 96px;
  max-width: 96px;
}

#borderline-review-table th:nth-child(3),
#borderline-review-table td:nth-child(3) {
  width: 136px;
  max-width: 136px;
}

#borderline-review-table th:nth-child(4),
#borderline-review-table td:nth-child(4) {
  min-width: 360px;
}

#borderline-review-table th:nth-child(5),
#borderline-review-table td:nth-child(5),
#borderline-review-table th:nth-child(6),
#borderline-review-table td:nth-child(6),
#borderline-review-table th:nth-child(7),
#borderline-review-table td:nth-child(7) {
  width: 172px;
  max-width: 172px;
}

#llm-review-results {
  line-height: 1.6;
}

#llm-review-shell {
  position: fixed;
  right: 20px;
  bottom: 20px;
  width: min(430px, calc(100vw - 32px));
  z-index: 1000;
}

#llm-review-panel {
  border-radius: 18px;
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
}

#llm-review-chatbot {
  height: 320px;
  max-height: 320px;
  overflow-y: auto;
}

#llm-prompt-buttons button {
  white-space: normal;
}

#agentic-analysis-shell {
  position: fixed;
  right: 20px;
  bottom: 20px;
  width: min(450px, calc(100vw - 32px));
  z-index: 999;
}

#agentic-analysis-panel {
  border-radius: 18px;
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
}

#agentic-analysis-chatbot {
  height: 340px;
  max-height: 340px;
  overflow-y: auto;
}

#agentic-prompt-buttons button {
  white-space: normal;
}

.app-hero {
  display: flex;
  align-items: stretch;
  justify-content: space-between;
  gap: 28px;
  padding: 22px 28px 22px 32px;
  margin-bottom: 14px;
  border: 1px solid #e8dcc8;
  border-radius: 28px;
  background:
    linear-gradient(120deg, rgba(255, 255, 255, 0.86), rgba(255, 251, 244, 0.92)),
    radial-gradient(circle at 100% 0%, rgba(36, 82, 78, 0.10), transparent 34%),
    radial-gradient(circle at 0% 100%, rgba(232, 119, 34, 0.10), transparent 32%);
  box-shadow: 0 18px 48px rgba(61, 44, 25, 0.07);
}

.app-hero-lab {
  color: #b45f24;
  font-size: clamp(1.05rem, 1.35vw, 1.45rem);
  font-weight: 900;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  margin-top: 10px;
}

.app-hero-title {
  color: #222;
  font-size: clamp(1.9rem, 3.1vw, 3.6rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 0.98;
  margin: 0;
}

.app-hero-subtitle {
  color: #645b52;
  font-size: 1rem;
  line-height: 1.52;
  max-width: 680px;
  margin-top: 16px;
}

.app-hero-subtitle strong {
  color: #173f42;
  font-weight: 900;
}

.app-hero-logo {
  flex: 0 0 auto;
  min-width: 320px;
  text-align: right;
  align-self: stretch;
}

.app-hero-logo-frame {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  min-height: 100%;
  padding: 18px 28px;
  border-radius: 24px;
  border: 1px solid rgba(232, 119, 34, 0.13);
  background: rgba(255, 255, 255, 0.76);
  box-shadow: 0 12px 28px rgba(31, 41, 55, 0.08);
}

.app-hero-logo img {
  width: min(100%, 300px);
  height: 100%;
  max-height: 172px;
  object-fit: contain;
}

#transcript-source-row {
  margin-top: 16px;
  margin-bottom: 14px;
}

.transcript-source-card {
  min-height: 210px;
  padding: 18px !important;
  border: 1px solid #e7dfd2 !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, #fffefd 0%, #fbf7f0 100%) !important;
  box-shadow: 0 14px 34px rgba(79, 55, 26, 0.08);
}

.transcript-source-card > .form,
.transcript-source-card .form {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
}

.source-kicker {
  color: #b45f24;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.source-title {
  color: #2f2a26;
  font-size: 1.05rem;
  font-weight: 800;
  margin-bottom: 4px;
}

.source-copy {
  color: #6f665d;
  font-size: 0.9rem;
  line-height: 1.35;
  margin-bottom: 14px;
}

.source-note {
  color: #7a7067;
  font-size: 0.82rem;
  line-height: 1.35;
  margin-top: 10px;
}

.transcript-source-card button {
  width: 100%;
}

#upload-transcript-btn {
  width: 100%;
}

#upload-transcript-btn,
#upload-transcript-btn button {
  min-height: 52px;
  border: 0 !important;
  border-radius: 14px !important;
  background: linear-gradient(135deg, #2c5f5d 0%, #153c3f 100%) !important;
  color: #fff !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 22px rgba(21, 60, 63, 0.18);
}

#upload-transcript-btn:hover,
#upload-transcript-btn button:hover {
  filter: brightness(1.05);
}

#agentic-continue-btn,
#agentic-continue-btn button {
  min-height: 44px;
  border: 0 !important;
  border-radius: 12px !important;
  background: linear-gradient(135deg, #1f8f69 0%, #0b4f45 100%) !important;
  color: #fff !important;
  font-weight: 900 !important;
  box-shadow: 0 10px 24px rgba(11, 79, 69, 0.22);
}

#agentic-continue-btn:hover,
#agentic-continue-btn button:hover {
  filter: brightness(1.04);
}

#agentic-continue-btn[disabled],
#agentic-continue-btn button:disabled {
  background: #e5e7eb !important;
  color: #8a8f98 !important;
  box-shadow: none;
}

.agentic-upload-row {
  margin: 14px 0 12px;
}

.agentic-upload-card {
  padding: 16px !important;
  border: 1px solid #e7dfd2 !important;
  border-radius: 18px !important;
  background: linear-gradient(180deg, #fffefd 0%, #fbf7f0 100%) !important;
  box-shadow: 0 12px 28px rgba(79, 55, 26, 0.07);
}

.agentic-upload-card > .form,
.agentic-upload-card .form {
  border: 0 !important;
  background: transparent !important;
  padding: 0 !important;
}

.agentic-upload-card input,
.agentic-upload-card button {
  min-height: 52px;
}

#agentic-progress {
  margin: 14px 0 16px;
}

.agentic-progress-shell {
  --progress: 0%;
  --accent: #2f7fa3;
  position: relative;
  padding: 12px;
  border: 1px solid #eadfce;
  border-radius: 20px;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(255, 248, 238, 0.84)),
    radial-gradient(circle at 20% 0%, rgba(232, 119, 34, 0.08), transparent 34%),
    radial-gradient(circle at 100% 100%, rgba(23, 63, 66, 0.08), transparent 30%);
  box-shadow: 0 12px 30px rgba(61, 44, 25, 0.06);
}

.agentic-progress-rail {
  position: absolute;
  left: 24px;
  right: 24px;
  top: 33px;
  height: 6px;
  border-radius: 999px;
  background: rgba(177, 163, 145, 0.22);
  overflow: hidden;
}

.agentic-progress-fill {
  display: block;
  width: var(--progress);
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #2f7fa3, #c98916, #7952b3, #2b8a5e);
  box-shadow: 0 0 20px color-mix(in srgb, var(--accent) 46%, transparent);
  transition: width 420ms ease;
  animation: agenticFlow 1.6s linear infinite;
  background-size: 220% 100%;
}

.agentic-progress-track {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.agentic-progress-step {
  position: relative;
  overflow: hidden;
  border: 1px solid #eadfce;
  border-radius: 16px;
  padding: 13px 12px 12px;
  background: rgba(248, 244, 237, 0.92);
  color: #756b60;
  font-size: 0.86rem;
  font-weight: 800;
  box-shadow: 0 8px 20px rgba(61, 44, 25, 0.04);
}

.agentic-progress-step::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg, transparent 0%, rgba(255, 255, 255, 0.52) 45%, transparent 70%);
  transform: translateX(-120%);
}

.agentic-progress-step .dot {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 999px;
  margin-right: 8px;
  background: #b9afa3;
  box-shadow: 0 0 0 4px rgba(185, 175, 163, 0.14);
  vertical-align: -1px;
}

.agentic-progress-step.done {
  border-color: #bad7ca;
  background: #eef8f2;
  color: #275640;
}

.agentic-progress-step.done .dot {
  background: #2f8c5b;
}

.agentic-progress-step.running {
  border-color: color-mix(in srgb, var(--accent) 56%, #ffffff);
  background: #fff5ec;
  color: #3b2a1f;
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(61, 44, 25, 0.10);
}

.agentic-progress-step.running::after {
  animation: agenticShimmer 1.35s ease-in-out infinite;
}

.agentic-progress-step.running .dot {
  background: var(--accent);
  box-shadow:
    0 0 0 4px color-mix(in srgb, var(--accent) 18%, transparent),
    0 0 18px color-mix(in srgb, var(--accent) 44%, transparent);
  animation: agenticPulse 1.05s ease-in-out infinite;
}

.agentic-progress-step.blocked {
  border-color: #e9b4b4;
  background: #fff1f1;
  color: #9b2c2c;
}

.agentic-progress-step.stage-review {
  border-color: #b7d9e8;
  background: #eff8fc;
}

.agentic-progress-step.stage-review .dot {
  background: #2f7fa3;
}

.agentic-progress-step.stage-approval {
  border-color: #ead49b;
  background: #fff8e6;
}

.agentic-progress-step.stage-approval .dot {
  background: #c98916;
}

.agentic-progress-step.stage-retrain {
  border-color: #d6c2ee;
  background: #f8f2ff;
}

.agentic-progress-step.stage-retrain .dot {
  background: #7952b3;
}

.agentic-progress-step.stage-reinference {
  border-color: #b9ded0;
  background: #effaf5;
}

.agentic-progress-step.stage-reinference .dot {
  background: #2b8a5e;
}

@keyframes agenticPulse {
  0%, 100% {
    transform: scale(0.82);
    opacity: 0.72;
  }
  50% {
    transform: scale(1.2);
    opacity: 1;
  }
}

@keyframes agenticFlow {
  0% {
    background-position: 0% 50%;
  }
  100% {
    background-position: 220% 50%;
  }
}

@keyframes agenticShimmer {
  0% {
    transform: translateX(-120%);
  }
  100% {
    transform: translateX(120%);
  }
}

@media (max-width: 900px) {
  #llm-review-shell {
    right: 12px;
    bottom: 12px;
    width: min(100vw - 24px, 430px);
  }

  #agentic-analysis-shell {
    right: 12px;
    bottom: 12px;
    width: min(100vw - 24px, 450px);
  }

  .app-hero {
    align-items: flex-start;
    flex-direction: column;
  }

  .app-hero-logo {
    text-align: left;
    min-width: 0;
  }
}
"""

QUICK_REVIEW_PROMPTS = [
    "Summarize the current inference.",
    "Explain the borderline phrases.",
    "Suggest final labels for borderline phrases.",
]

AGENTIC_QUICK_PROMPTS = [
    "Summarize the current agentic run.",
    "Explain which borderline scores should move up or down.",
    "Recommend whether the retraining improved the result.",
]

DEFAULT_LLM_PANEL_MESSAGE = ""

DEFAULT_AGENTIC_CHAT_MESSAGE = (
    "Ask about the latest agentic run."
)

DEMO_RULE_DISPLAY_NAMES = {
    "101": "Identity verification",
    "102": "Change of Booking",
}

DEMO_CLAIM_DISPLAY_NAMES = {
    ("102", "mandatory", 0): "Disclosure 1: Change fee",
    ("102", "mandatory", 1): "Disclosure 2: Fare difference or travel credit",
}

DEMO_TRANSCRIPT_SCRIPTS_DIR = Path("data/results/demo/transcript_scripts")
AGENTIC_PROGRESS_STEPS = [
    ("review", "Agentic Review"),
    ("approval", "Human Approval"),
    ("retrain", "Retrain"),
    ("reinference", "Re-inference"),
]
AGENTIC_PROGRESS_COLORS = {
    "review": "#2f7fa3",
    "approval": "#c98916",
    "retrain": "#7952b3",
    "reinference": "#2b8a5e",
}
AGENTIC_REINFERENCE_PROGRESS_DWELL_SEC = 1.2
GRADIO_UPLOAD_PREFIX_RE = re.compile(r"^gradio_[0-9a-f]{16,}[_-]+", flags=re.IGNORECASE)


def _logo_data_uri() -> str:
    logo_path = Path("resources/images/uob_logo.png")
    try:
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


def _hero_html() -> str:
    logo_uri = _logo_data_uri()
    logo_html = f"<div class='app-hero-logo-frame'><img src='{logo_uri}' alt='UOB logo'></div>" if logo_uri else ""
    return (
        "<div class='app-hero'>"
        "<div>"
        "<h1 class='app-hero-title'>Travel Agency Policy Studio</h1>"
        "<div class='app-hero-lab'>UOB AI Labs</div>"
        "<div class='app-hero-subtitle'>"
        "<strong>Agentic travel-disclosure review</strong> for transcript inference, human approval, dataset diagnosis, and model retraining."
        "</div>"
        "</div>"
        f"<div class='app-hero-logo'>{logo_html}</div>"
        "</div>"
    )


def _extend_gradio_server_start_timeout(timeout_sec: float = 30.0) -> None:
    import gradio.http_server as http_server
    from gradio.exceptions import ServerFailedToStartError

    original = getattr(http_server.Server.run_in_thread, "__name__", "")
    if original == "run_in_thread_with_extended_timeout":
        return

    def run_in_thread_with_extended_timeout(self) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        if getattr(self, "reloader", None):
            self.watch_thread = threading.Thread(target=self.watch, daemon=True)
            self.watch_thread.start()
        self.thread.start()
        start = time.time()
        while not self.started:
            time.sleep(1e-3)
            if time.time() - start > timeout_sec:
                raise ServerFailedToStartError("Server failed to start. Please check that the port is available.")

    http_server.Server.run_in_thread = run_in_thread_with_extended_timeout


def _rule_display_name(disclaimer_id: str, evidence: dict[str, Any]) -> str:
    rule_id = str(disclaimer_id).strip()
    return DEMO_RULE_DISPLAY_NAMES.get(rule_id) or str(evidence.get("description", "")).strip()


def _iter_claim_evidence(evidence: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    claims = evidence.get("claims", {})
    if not isinstance(claims, dict):
        return []

    output: list[tuple[str, int, dict[str, Any]]] = []
    for claim_type in ("single", "mandatory", "standard"):
        claim_items = claims.get(claim_type, [])
        if not isinstance(claim_items, list):
            continue
        for order, claim in enumerate(claim_items):
            if isinstance(claim, dict):
                output.append((claim_type, order, claim))
    return output


def _format_evidence_lines(disclaimer_id: str, evidence: dict[str, Any]) -> list[str]:
    claim_items = _iter_claim_evidence(evidence)
    if not claim_items:
        score = float(evidence.get("verification_score") or 0.0)
        best_text = str(evidence.get("match_text", "")).strip() or "No matched text available."
        return [
            "",
            "**Evidence**",
            f"- **Best text:** {best_text}",
            f"- **Score:** `{score:.3f}`",
        ]

    lines = ["", "**Evidence**"]
    if len(claim_items) > 1:
        headers: list[str] = []
        anchor_cells: list[str] = []
        text_cells: list[str] = []
        score_cells: list[str] = []
        for claim_type, order, claim in claim_items:
            display_name = DEMO_CLAIM_DISPLAY_NAMES.get((str(disclaimer_id).strip(), claim_type, order), f"Disclosure {order + 1}")
            anchor = str(claim.get("anchor", "")).strip() or "No anchor available."
            best_text = str(claim.get("match_text", "")).strip() or "No matched text available."
            score = float(claim.get("verification_score") or 0.0)
            headers.append(html.escape(display_name))
            anchor_cells.append(f"**Anchor:** {html.escape(anchor)}")
            text_cells.append(f"**Best text:** {html.escape(best_text)}")
            score_cells.append(f"**Score:** `{score:.3f}`")

        lines.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                "| " + " | ".join(anchor_cells) + " |",
                "| " + " | ".join(text_cells) + " |",
                "| " + " | ".join(score_cells) + " |",
            ]
        )
        return lines

    for claim_type, order, claim in claim_items:
        display_name = DEMO_CLAIM_DISPLAY_NAMES.get((str(disclaimer_id).strip(), claim_type, order))
        anchor = str(claim.get("anchor", "")).strip() or "No anchor available."
        best_text = str(claim.get("match_text", "")).strip() or "No matched text available."
        score = float(claim.get("verification_score") or 0.0)

        if display_name:
            lines.extend(["", f"**{display_name}**"])
        lines.extend(
            [
                f"- **Anchor:** {anchor}",
                f"- **Best text:** {best_text}",
                f"- **Score:** `{score:.3f}`",
            ]
        )
    return lines


def _format_results(payload: dict[str, Any]) -> str:
    results = payload.get("results", {})
    lines = ["## Inference Results"]
    for disclaimer_id, result in results.items():
        status = result.get("status", "FAIL")
        evidence = result.get("evidence", {})
        display_name = _rule_display_name(str(disclaimer_id), evidence)
        heading = f"### Rule {disclaimer_id}: {display_name}" if display_name else f"### Rule {disclaimer_id}"
        lines.extend(["", heading, f"**Status:** `{status}`"])
        lines.extend(_format_evidence_lines(str(disclaimer_id), evidence))
        lines.extend(["", "---"])
    if len(lines) == 1:
        lines.append("- No inference output available.")
    return "\n".join(lines)


def _verifier_label(score: float) -> str:
    return "Pass" if float(score) >= 0.5 else "Fail"


def _label_to_display(value: Any, *, default: str = "") -> str:
    label = str(value or "").strip().lower()
    if not label:
        return default
    normalized = label.replace("_", "-")
    if normalized in {"pass", "passed", "positive", "compliant", "yes", "y", "true", "1"}:
        return "Pass"
    if normalized in {"fail", "failed", "negative", "non-compliant", "non compliant", "noncompliant", "no", "n", "false", "0"}:
        return "Fail"
    if normalized in {"skip", "skipped", "ambiguous", "unclear"}:
        return "Skip"
    if "non-compliant" in normalized or "non compliant" in normalized or "noncompliant" in normalized:
        return "Fail"
    if "compliant" in normalized:
        return "Pass"
    return str(value or "").strip()


def _display_to_training_label(value: Any) -> str:
    display = _label_to_display(value)
    if display == "Pass":
        return "Compliant"
    if display == "Fail":
        return "Non-Compliant"
    return ""


def _text_preview(text: Any, max_words: int = 12) -> str:
    words = str(text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _display_final_label(item: dict[str, Any]) -> str:
    return _label_to_display(item.get("final_label") or item.get("llm_label"), default="")


def _borderline_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "add": bool(item.get("approved", False)),
            "rule": item["disclaimer_id"],
            "score": round(float(item["verification_score"]), 4),
            "text": _text_preview(item.get("text", "")),
            "model_label": _verifier_label(float(item["verification_score"])),
            "qwen_label": _label_to_display(item.get("llm_label")),
            "final_label": _display_final_label(item),
        }
        for item in items
    ]
    return pd.DataFrame(rows)


def _agentic_review_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "add": bool(item.get("approved", False)),
            "rule": item["disclaimer_id"],
            "score": round(float(item["verification_score"]), 4),
            "anchor": _text_preview(item.get("anchor", ""), max_words=8),
            "text": _text_preview(item.get("text", ""), max_words=8),
            "model": _verifier_label(float(item["verification_score"])),
            "qwen": _label_to_display(item.get("llm_label")),
            "human": _display_final_label(item),
        }
        for item in items
    ]
    return pd.DataFrame(rows)


def _agentic_comparison_dataframe(comparisons: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in comparisons:
        rows.append(
            {
                "transcript": str(item.get("transcript_id", "")).strip(),
                "rule": str(item.get("disclaimer_id", "")).strip(),
                "label": _label_to_display(item.get("final_label")),
                "before": round(float(item.get("before_score") or 0.0), 4),
                "after": round(float(item.get("after_score") or 0.0), 4),
                "target": str(item.get("target_direction", "")).strip(),
                "outcome": str(item.get("outcome", "")).strip(),
                "text": _text_preview(item.get("text", ""), max_words=14),
            }
        )
    return pd.DataFrame(rows)


def _format_agentic_progress(*, active: str | None = None, completed: set[str] | None = None, blocked: bool = False) -> str:
    completed = completed or set()
    step_keys = [key for key, _ in AGENTIC_PROGRESS_STEPS]
    if active in step_keys:
        active_index = step_keys.index(active)
        progress_units = active_index + 0.55
    elif completed:
        progress_units = max(step_keys.index(key) + 1 for key in completed if key in step_keys)
    else:
        progress_units = 0
    progress_percent = min(100, max(0, (progress_units / len(step_keys)) * 100))
    accent = AGENTIC_PROGRESS_COLORS.get(active or next((key for key in reversed(step_keys) if key in completed), "review"), "#2f7fa3")
    parts = [
        "<div class='agentic-progress-shell' "
        f"style='--progress:{progress_percent:.1f}%; --accent:{html.escape(accent)};'>"
        "<div class='agentic-progress-rail'><span class='agentic-progress-fill'></span></div>"
        "<div class='agentic-progress-track'>"
    ]
    for key, label in AGENTIC_PROGRESS_STEPS:
        css_class = "done" if key in completed else "running" if key == active else "pending"
        if blocked and key == active:
            css_class = "blocked"
        parts.append(
            "<div class='agentic-progress-step "
            f"stage-{html.escape(key)} {css_class}'><span class='dot'></span>{html.escape(label)}</div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


def _format_agentic_comparison_markdown(comparisons: list[dict[str, Any]] | None) -> str:
    comparisons = [item for item in comparisons or [] if isinstance(item, dict)]
    if not comparisons:
        return "### Before/After Score Comparison\nNo comparison available yet."

    lines = [
        "### Before/After Score Comparison",
        "| Rule | Final label | Before | After | Desired movement | Outcome | Phrase |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in comparisons:
        rule_id = str(item.get("disclaimer_id", "")).strip()
        label = _label_to_display(item.get("final_label")) or str(item.get("final_label", "")).strip()
        before_score = float(item.get("before_score") or 0.0)
        after_score = float(item.get("after_score") or 0.0)
        target = str(item.get("target_direction", "")).strip()
        outcome = str(item.get("outcome", "")).strip()
        phrase = _text_preview(item.get("text", ""), max_words=16)
        lines.append(
            f"| {html.escape(rule_id)} | {html.escape(label)} | `{before_score:.4f}` | `{after_score:.4f}` | "
            f"{html.escape(target)} | `{html.escape(outcome)}` | {html.escape(phrase)} |"
        )
    return "\n".join(lines)


def _row_checked(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "t", "yes", "y", "1", "checked"}


def _normalize_human_label(value: Any) -> str:
    return _display_to_training_label(value)


def _records_from_table_value(table_value: Any) -> list[dict[str, Any]]:
    if table_value is None:
        return []
    if isinstance(table_value, pd.DataFrame):
        if table_value.empty:
            return []
        return table_value.to_dict(orient="records")
    if isinstance(table_value, list):
        if not table_value:
            return []
        if all(isinstance(row, dict) for row in table_value):
            return [dict(row) for row in table_value]
        records: list[dict[str, Any]] = []
        for row in table_value:
            if not isinstance(row, (list, tuple)):
                continue
            headers = (
                ["add", "transcript", "rule", "score", "text", "model_label", "qwen_label", "final_label"]
                if len(row) >= 8
                else ["add", "rule", "score", "text", "model_label", "qwen_label", "final_label"]
            )
            if len(row) == 8:
                headers = ["add", "rule", "score", "anchor", "text", "model", "qwen", "human"]
            records.append(dict(zip(headers, row)))
        return records
    if isinstance(table_value, dict):
        data = table_value.get("data")
        headers = table_value.get("headers") or table_value.get("columns")
        if isinstance(data, list) and isinstance(headers, list):
            header_names = [
                str(header.get("name", header.get("id", ""))).strip() if isinstance(header, dict) else str(header).strip()
                for header in headers
            ]
            return [dict(zip(header_names, row)) for row in data if isinstance(row, (list, tuple))]
        if isinstance(data, list):
            return _records_from_table_value(data)
    return []


def _merge_review_items(table_value: Any, items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    base_items = [deepcopy(item) for item in (items or [])]
    records = _records_from_table_value(table_value)
    if not records:
        return []

    merged: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        base = deepcopy(base_items[index]) if index < len(base_items) else {}
        score_value = row.get("score", row.get("verification_score"))
        qwen_value = row.get("qwen", row.get("qwen_label"))
        model_value = row.get("model", row.get("model_label", row.get("verifier_label")))
        final_value = row.get("human", row.get("final_label", row.get("final")))
        merged.append(
            {
                **base,
                "approved": _row_checked(row.get("add", row.get("add_to_training"))),
                "disclaimer_id": str(row.get("rule") or row.get("rule_id") or base.get("disclaimer_id", "")).strip(),
                "text": str(base.get("text", "")).strip(),
                "verification_score": float(score_value or base.get("verification_score") or 0.0),
                "model_label": _normalize_human_label(model_value or base.get("model_label", "")),
                "llm_label": _normalize_human_label(qwen_value or base.get("llm_label", "")),
                "final_label": _normalize_human_label(
                    final_value
                    or model_value
                    or qwen_value
                    or base.get("final_label")
                    or base.get("llm_label")
                    or base.get("model_label")
                    or ""
                ),
            }
        )
    return merged


def _format_llm_review_summary(items: list[dict[str, Any]]) -> str:
    if not items:
        return DEFAULT_LLM_PANEL_MESSAGE

    lines = ["### LLM Review Suggestions"]
    for index, item in enumerate(items, start=1):
        label = _label_to_display(item.get("llm_label")) or "No suggestion"
        confidence = float(item.get("llm_confidence") or 0.0)
        rationale = str(item.get("llm_rationale", "")).strip() or "No rationale provided."
        phrase = str(item.get("text", "")).strip() or "No phrase available."
        anchor = str(item.get("anchor", "")).strip() or "No anchor available."
        lines.extend(
            [
                f"#### Review {index}",
                f"- Rule: `{item.get('disclaimer_id', '')}`",
                f"- Suggested label: `{label}`",
                f"- Confidence: `{confidence:.2f}`",
                f"- Anchor: {anchor}",
                f"- Phrase: {phrase}",
                f"- Rationale: {rationale}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _format_metric_snapshot(title: str, metrics: dict[str, Any] | None) -> list[str]:
    metrics = metrics if isinstance(metrics, dict) else {}
    if not metrics:
        return [f"### {title}", "- No metrics available."]
    return [
        f"### {title}",
        f"- Count: `{int(metrics.get('count', 0) or 0)}`",
        f"- Accuracy: `{float(metrics.get('accuracy', 0.0) or 0.0):.3f}`",
        f"- Precision: `{float(metrics.get('precision', 0.0) or 0.0):.3f}`",
        f"- Recall: `{float(metrics.get('recall', 0.0) or 0.0):.3f}`",
        f"- F1: `{float(metrics.get('f1', 0.0) or 0.0):.3f}`",
        f"- Macro F1: `{float(metrics.get('macro_f1', 0.0) or 0.0):.3f}`",
    ]


def _format_agentic_summary(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary:
        return "## Agentic Loop\n- No run output available yet."

    lines = [
        "## Agentic Loop",
        f"- Status: `{str(summary.get('status', 'unknown')).strip()}`",
    ]
    message = str(summary.get("message", "")).strip()
    if message:
        lines.append(f"- Message: {message}")

    bootstrap = summary.get("bootstrap", {})
    if isinstance(bootstrap, dict) and bootstrap:
        lines.extend(
            [
                "",
                "### Bootstrap Gate",
                f"- Status: `{str(bootstrap.get('status', 'unknown')).strip()}`",
            ]
        )
        bootstrap_message = str(bootstrap.get("message", "")).strip()
        if bootstrap_message:
            lines.append(f"- Message: {bootstrap_message}")
        gate_metrics = bootstrap.get("gate_metrics")
        if isinstance(gate_metrics, dict):
            lines.extend(["", *_format_metric_snapshot("Synthetic Gate Metrics", gate_metrics)])

    if "incoming_count" in summary or "holdout_count" in summary:
        lines.extend(
            [
                "",
                "### Data Scope",
                f"- Incoming transcripts: `{int(summary.get('incoming_count', 0) or 0)}`",
                f"- Holdout transcripts: `{int(summary.get('holdout_count', 0) or 0)}`",
                f"- Reviewed anchors: `{int(summary.get('reviewed_count', 0) or 0)}`",
            ]
        )

    lines.extend(["", *_format_metric_snapshot("Holdout Metrics Before", summary.get("holdout_metrics_before"))])
    if isinstance(summary.get("holdout_metrics_after"), dict):
        lines.extend(["", *_format_metric_snapshot("Holdout Metrics After", summary.get("holdout_metrics_after"))])

    coverage = summary.get("coverage", {})
    if isinstance(coverage, dict) and coverage:
        lines.extend(
            [
                "",
                "### Coverage Augmentation",
                f"- Candidate corrected cases: `{int(coverage.get('candidate_case_count', 0) or 0)}`",
                f"- Coverage analyses: `{int(coverage.get('analysis_count', 0) or 0)}`",
                f"- New synthetic rows added this run: `{int(coverage.get('new_rows_count', 0) or 0)}`",
            ]
        )

    if "promoted" in summary:
        lines.extend(
            [
                "",
                "### Promotion",
                f"- Promoted candidate model: `{'yes' if bool(summary.get('promoted', False)) else 'no'}`",
            ]
        )

    return "\n".join(lines)


def _format_app_agentic_summary(summary: dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary:
        return "## Agentic Loop\n- No run output available yet."

    lines = [
        "## Agentic Loop",
        f"- Status: `{str(summary.get('status', 'unknown')).strip()}`",
    ]
    message = str(summary.get("message", "")).strip()
    if message:
        lines.append(f"- Message: {message}")

    if "transcript_count" in summary:
        lines.append(f"- Uploaded transcripts: `{int(summary.get('transcript_count', 0) or 0)}`")
    if "borderline_count" in summary:
        lines.append(f"- Borderline phrases: `{int(summary.get('borderline_count', 0) or 0)}`")
    if "pass_review_count" in summary:
        lines.append(f"- Pass phrases for review: `{int(summary.get('pass_review_count', 0) or 0)}`")

    stage_status = summary.get("stage_status", [])
    if isinstance(stage_status, list) and stage_status:
        lines.append("")
        lines.append("### Agent Steps")
        for stage in stage_status:
            if not isinstance(stage, dict):
                continue
            agent = str(stage.get("agent", "Agent")).strip()
            status = str(stage.get("status", "unknown")).strip()
            stage_message = str(stage.get("message", "")).strip()
            lines.append(f"- **{agent}:** `{status}`{(' - ' + stage_message) if stage_message else ''}")

    supervisor_summary = str(summary.get("supervisor_summary", "")).strip()
    if supervisor_summary:
        summary_lines = supervisor_summary.splitlines()
        lines.extend(["", "### Supervisor Summary"])
        for line in summary_lines:
            stripped = line.strip()
            if not stripped:
                lines.append("")
            elif stripped.startswith("Transcript "):
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(f"**{stripped}**")
            elif stripped.startswith("SupervisorAgent prepared"):
                lines.append(f"**{stripped}**")
            elif stripped.endswith(":"):
                lines.append(f"**{stripped}**")
            elif line.startswith("  "):
                lines.append(f"  - {stripped}")
            else:
                lines.append(f"- {stripped}")

    diagnosis = summary.get("diagnosis", {})
    analyses = diagnosis.get("analyses", []) if isinstance(diagnosis, dict) else []
    if isinstance(analyses, list) and analyses:
        lines.extend(["", "### Dataset Diagnosis"])
        for index, item in enumerate(analyses[:5], start=1):
            if not isinstance(item, dict):
                continue
            tags = ", ".join(str(tag) for tag in item.get("cause_tags", []) if str(tag).strip()) or "n/a"
            issue = str(item.get("diagnosis_type") or item.get("label_change") or "dataset review").strip()
            lines.extend(
                [
                    f"#### Case {index}: Rule {str(item.get('disclaimer_id', '')).strip()}",
                    f"- Issue: `{issue}`",
                    f"- Cause: `{tags}`",
                    f"- Existing same-label samples: `{int(item.get('same_label_count', 0) or 0)}`",
                    f"- Existing opposite-label samples: `{int(item.get('opposite_label_count', 0) or 0)}`",
                    f"- Max same-label similarity: `{float(item.get('max_same_label_similarity', 0.0) or 0.0):.3f}`",
                    f"- Max opposite-label similarity: `{float(item.get('max_opposite_label_similarity', 0.0) or 0.0):.3f}`",
                    f"- Recommendation: {str(item.get('recommendation', '')).strip() or 'No recommendation.'}",
                ]
            )
            solution_steps = item.get("solution_steps", [])
            if isinstance(solution_steps, list) and solution_steps:
                lines.append(f"- Suggested action: {' '.join(str(step).strip() for step in solution_steps if str(step).strip())}")
    elif isinstance(diagnosis, dict) and "changed_case_count" in diagnosis:
        lines.extend(
            [
                "",
                "### Dataset Diagnosis",
                "- No human-overturned borderline cases were selected, so dataset diagnosis was not needed.",
            ]
        )

    recommendation = str(summary.get("recommendation", "")).strip()
    if recommendation:
        lines.extend(["", "### Recommendation", recommendation])

    return "\n".join(lines)


def _format_quality_report(report: dict[str, Any] | None) -> str:
    report = report if isinstance(report, dict) else {}
    audits = report.get("audits", [])
    if not report:
        return "### Synthetic Quality Report\n- No synthetic quality report available for this run."

    lines = [
        "### Synthetic Quality Report",
        f"- Status: `{str(report.get('status', 'unknown')).strip()}`",
        f"- Anchors below threshold: `{int(report.get('anchors_below_threshold', 0) or 0)}`",
    ]
    if "macro_f1" in report:
        lines.append(f"- Gate macro F1: `{float(report.get('macro_f1', 0.0) or 0.0):.3f}`")

    if isinstance(audits, list) and audits:
        for index, audit in enumerate(audits[:5], start=1):
            lines.extend(
                [
                    "",
                    f"#### Audit {index}",
                    f"- Rule: `{str(audit.get('disclaimer_id', '')).strip()}`",
                    f"- Anchor F1: `{float(audit.get('f1', 0.0) or 0.0):.3f}`",
                    f"- Coverage: `{str(audit.get('coverage_status', 'unknown')).strip()}`",
                    f"- Positive note: {str(audit.get('positive_quality_note', '')).strip() or 'No note.'}",
                    f"- Negative note: {str(audit.get('negative_quality_note', '')).strip() or 'No note.'}",
                    f"- Reason: {str(audit.get('reason', '')).strip() or 'No reason provided.'}",
                ]
            )
    return "\n".join(lines)


def _format_coverage_report(coverage: dict[str, Any] | None) -> str:
    coverage = coverage if isinstance(coverage, dict) else {}
    analyses = coverage.get("analyses", [])
    if not coverage:
        return "### Coverage Analysis\n- No coverage analysis available for this run."

    lines = [
        "### Coverage Analysis",
        f"- Candidate corrected cases: `{int(coverage.get('candidate_case_count', 0) or 0)}`",
        f"- New synthetic rows added: `{int(coverage.get('new_rows_count', 0) or 0)}`",
    ]
    if isinstance(analyses, list) and analyses:
        for index, item in enumerate(analyses[:6], start=1):
            lines.extend(
                [
                    "",
                    f"#### Case {index}",
                    f"- Rule: `{str(item.get('disclaimer_id', '')).strip()}`",
                    f"- Type: `{str(item.get('correction_type', '')).strip() or 'n/a'}`",
                    f"- Coverage: `{str(item.get('coverage_status', 'unknown')).strip()}`",
                    f"- Reviewed phrase: {str(item.get('reviewed_phrase', '')).strip() or 'No phrase available.'}",
                    f"- Reason: {str(item.get('reason', '')).strip() or 'No reason provided.'}",
                ]
            )
            variants = item.get("generated_variants", [])
            if isinstance(variants, list) and variants:
                lines.append(f"- Generated variants: {' | '.join(str(variant).strip() for variant in variants if str(variant).strip())}")
    return "\n".join(lines)


def _format_regression_investigation_report(summary: dict[str, Any] | None) -> str:
    summary = summary if isinstance(summary, dict) else {}
    comparisons = [
        item
        for item in summary.get("comparisons", [])
        if isinstance(item, dict) and str(item.get("outcome", "")).strip() == "regressed"
    ]
    diagnosis = summary.get("score_regression_diagnosis", {})
    analyses = diagnosis.get("analyses", []) if isinstance(diagnosis, dict) else []
    analyses = [item for item in analyses if isinstance(item, dict)]

    if not comparisons:
        return (
            "### InvestigatorAgent Diagnosis\n"
            "No approved phrase moved in the wrong direction after retraining, so no regression investigation is needed."
        )

    if not analyses:
        return (
            "### InvestigatorAgent Diagnosis\n"
            f"Found {len(comparisons)} regressed approved phrase(s), but no dataset diagnosis is available yet. "
            "Click again after re-inference completes, or rerun the retrain-and-compare step."
        )

    lines = [
        "### InvestigatorAgent Diagnosis",
        f"Found {len(comparisons)} approved phrase(s) that moved in the wrong direction after retraining.",
    ]
    for index, item in enumerate(analyses, start=1):
        rule_id = str(item.get("disclaimer_id", "")).strip() or "unknown"
        anchor = _text_preview(item.get("anchor", ""), max_words=32) or "No anchor available."
        phrase = _text_preview(item.get("phrase", ""), max_words=32) or "No phrase available."
        before_score = float(item.get("before_score") or item.get("score") or 0.0)
        after_score = float(item.get("after_score") or 0.0)
        target = str(item.get("target_direction", "")).strip() or "n/a"
        tags = [str(tag).strip() for tag in item.get("cause_tags", []) if str(tag).strip()]
        cause = ", ".join(tags) if tags else "needs dataset review"
        recommendation = str(item.get("recommendation", "")).strip() or "Review this anchor's synthetic coverage before another retrain."
        solution_steps = [str(step).strip() for step in item.get("solution_steps", []) if str(step).strip()]
        action = " ".join(solution_steps) if solution_steps else recommendation

        lines.extend(
            [
                "",
                f"#### Case {index}: Rule {rule_id}",
                f"- Score movement: `{before_score:.4f}` -> `{after_score:.4f}`; desired movement: `{target}`.",
                f"- Anchor: {anchor}",
                f"- Phrase: {phrase}",
                f"- Likely cause: `{cause}`.",
                f"- Dataset signal: `{int(item.get('same_label_count', 0) or 0)}` same-label sample(s), "
                f"`{int(item.get('opposite_label_count', 0) or 0)}` opposite-label sample(s), "
                f"max same-label similarity `{float(item.get('max_same_label_similarity', 0.0) or 0.0):.3f}`.",
                f"- Recommended action: {action}",
            ]
        )
    return "\n".join(lines)


def _artifact_json(summary: dict[str, Any], key: str, default: Any) -> Any:
    artifacts = summary.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return default
    path = artifacts.get(key)
    if not path:
        return default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _extract_uploaded_paths(upload_value: Any) -> list[Path]:
    if upload_value is None:
        return []
    values = upload_value if isinstance(upload_value, list) else [upload_value]
    paths: list[Path] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, str):
            paths.append(Path(item))
            continue
        if isinstance(item, dict):
            for key in ("path", "name"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    paths.append(Path(value))
                    break
            continue
        for attr in ("path", "name"):
            value = getattr(item, attr, None)
            if isinstance(value, str) and value:
                paths.append(Path(value))
                break
    return paths


def _clean_uploaded_filename(path: Path) -> str:
    name = path.name
    cleaned = GRADIO_UPLOAD_PREFIX_RE.sub("", name).strip()
    if cleaned and cleaned != name:
        return cleaned
    parent_match = re.match(r"^gradio_[0-9a-f]{16,}$", path.parent.name, flags=re.IGNORECASE)
    if parent_match:
        return name
    return cleaned or name


def _copy_uploaded_paths_to_temp(uploaded_paths: list[Path], temp_dir: Path) -> None:
    used_names: set[str] = set()
    for index, path in enumerate(uploaded_paths, start=1):
        filename = _clean_uploaded_filename(path)
        stem = Path(filename).stem or f"transcript_{index}"
        suffix = Path(filename).suffix or path.suffix
        candidate = f"{stem}{suffix}"
        counter = 2
        while candidate in used_names or (temp_dir / candidate).exists():
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate)
        shutil.copy2(path, temp_dir / candidate)


def _materialize_incoming_source(
    transcript_text: str,
    upload_value: Any,
    folder_upload_value: Any = None,
    active_source: str | None = None,
) -> tuple[str | None, list[Path], str | None]:
    cleanup_paths: list[Path] = []
    uploaded_paths = _extract_uploaded_paths(upload_value)
    folder_paths = _extract_uploaded_paths(folder_upload_value)
    if uploaded_paths and folder_paths:
        active_source = str(active_source or "").strip().lower()
        if active_source == "file":
            folder_paths = []
        elif active_source == "folder":
            uploaded_paths = []
        else:
            return None, cleanup_paths, "Use either file upload or folder upload for one run, not both."

    if folder_paths:
        txt_paths = [path for path in folder_paths if path.suffix.lower() == ".txt"]
        if not txt_paths:
            return None, cleanup_paths, "The uploaded folder must contain at least one `.txt` transcript file."
        temp_dir = Path(tempfile.mkdtemp(prefix="policy_compliance_agent_incoming_"))
        cleanup_paths.append(temp_dir)
        _copy_uploaded_paths_to_temp(txt_paths, temp_dir)
        return str(temp_dir), cleanup_paths, None

    if uploaded_paths:
        suffixes = {path.suffix.lower() for path in uploaded_paths}
        if len(uploaded_paths) == 1 and suffixes == {".json"}:
            return str(uploaded_paths[0]), cleanup_paths, None
        if suffixes != {".txt"}:
            return None, cleanup_paths, "Upload either a single `.txt`/`.json` file, multiple `.txt` files, or one folder of `.txt` files."
        temp_dir = Path(tempfile.mkdtemp(prefix="policy_compliance_agent_incoming_"))
        cleanup_paths.append(temp_dir)
        _copy_uploaded_paths_to_temp(uploaded_paths, temp_dir)
        return str(temp_dir), cleanup_paths, None

    transcript = str(transcript_text or "").strip()
    if transcript:
        temp_dir = Path(tempfile.mkdtemp(prefix="policy_compliance_agent_incoming_"))
        cleanup_paths.append(temp_dir)
        transcript_path = temp_dir / "interactive_transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        return str(transcript_path), cleanup_paths, None

    return None, cleanup_paths, "Upload transcript file(s) or a folder of `.txt` transcripts before running the agentic loop."


def _materialize_holdout_source(upload_value: Any) -> tuple[str | None, str | None]:
    uploaded_paths = _extract_uploaded_paths(upload_value)
    if not uploaded_paths:
        return None, None
    if len(uploaded_paths) != 1:
        return None, "Upload a single holdout JSON file when overriding the default holdout set."
    if uploaded_paths[0].suffix.lower() != ".json":
        return None, "The holdout override must be a `.json` file."
    return str(uploaded_paths[0]), None


def _load_uploaded_transcript(upload_value: Any) -> tuple[str, str]:
    uploaded_paths = _extract_uploaded_paths(upload_value)
    if not uploaded_paths:
        return "", "Choose a `.txt` transcript file first."
    if len(uploaded_paths) != 1:
        return "", "Upload one transcript `.txt` file at a time in Transcript Review."

    path = uploaded_paths[0]
    if path.suffix.lower() != ".txt":
        return "", "Transcript Review accepts `.txt` scripts only."
    try:
        transcript = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return "", f"Could not read uploaded transcript: {exc}"
    if not transcript:
        return "", f"Uploaded script `{path.name}` is empty."
    return transcript, f"Loaded uploaded script: {path.name}"


def _chat_context(payload: dict[str, Any], review_items: list[dict[str, Any]]) -> dict[str, Any]:
    guidance = (
        "This is a policy-review demo for two travel-agency/helpdesk-style controls. "
        "The inference payload contains rule-level pass/fail decisions, scores, anchors, and matched evidence text. "
        "The review_items list contains the current borderline rows; after Qwen labeling, it also contains qwen/llm labels, confidence, and rationale. "
        "Use only this context when answering, clearly distinguish the model score from Qwen or human review labels, and avoid inventing evidence."
    )
    return {
        "assistant_context": guidance,
        "inference": payload if isinstance(payload, dict) else {},
        "review_items": review_items if isinstance(review_items, list) else [],
    }


def _append_chat_messages(
    history: list[dict[str, Any]] | None,
    *,
    user_text: str,
    assistant_text: str,
) -> list[dict[str, str]]:
    messages = _normalized_chat_messages(history)
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _normalized_chat_messages(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str):
            messages.append({"role": item["role"], "content": item["content"]})
    return messages


def _append_user_message(history: list[dict[str, Any]] | None, user_text: str) -> list[dict[str, str]]:
    messages = _normalized_chat_messages(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def _preset_prompt(
    prompt: str,
    payload: dict[str, Any],
    review_items: list[dict[str, Any]],
    history: list[dict[str, Any]] | None,
    config: dict[str, Any] | None = None,
):
    try:
        answer = answer_demo_question(
            prompt,
            inference_payload=_chat_context(payload, review_items),
            chat_history=history,
            config=config,
        )
    except Exception as exc:
        answer = f"LLM request failed: {exc}"
    messages = _append_chat_messages(history, user_text=prompt, assistant_text=answer)
    return messages, messages, ""


def _preset_agentic_prompt(
    prompt: str,
    summary_payload: dict[str, Any],
    history: list[dict[str, Any]] | None,
    config: dict[str, Any] | None = None,
):
    try:
        answer = answer_agentic_question(prompt, summary_payload=summary_payload, chat_history=history, config=config)
    except Exception as exc:
        answer = f"LLM request failed: {exc}"
    return _append_chat_messages(history, user_text=prompt, assistant_text=answer), ""


def build_demo_app(config_path: str | None = None):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gradio is required for the demo app. Install with `pip install .[demo]`.") from exc

    config = load_demo_config(config_path)
    samples = load_demo_samples(config)
    sample_lookup = {item["transcript_id"]: item for item in samples if isinstance(item, dict) and item.get("transcript_id")}
    approved_summary = approve_demo_examples([], config=config)

    def load_sample(sample_id: str) -> tuple[str, str]:
        sample = sample_lookup.get(sample_id, {})
        transcript = str(sample.get("transcript", "")).strip()
        title = str(sample.get("title", sample_id)).strip()
        return transcript, f"Loaded sample: {title}" if transcript else "Sample not found."

    def load_uploaded_script(upload_value: Any) -> tuple[str, str]:
        return _load_uploaded_transcript(upload_value)

    def run_inference_ui(transcript: str) -> tuple[
        str,
        pd.DataFrame,
        dict[str, Any],
        str,
        list[dict[str, Any]],
        str,
        str,
        str,
        list[dict[str, str]],
        list[dict[str, str]],
    ]:
        payload = run_demo_inference(transcript, config=config)
        review_items = payload.get("borderline_items", [])
        dataframe = _borderline_dataframe(review_items)
        summary = f"Found {len(payload.get('borderline_items', []))} borderline phrase(s) in the `[0.30, 0.70)` band."
        return (
            _format_results(payload),
            dataframe,
            payload,
            summary,
            review_items,
            DEFAULT_LLM_PANEL_MESSAGE,
            DEFAULT_LLM_PANEL_MESSAGE,
            "",
            [],
            [],
        )

    def label_borderline_ui(items: list[dict[str, Any]] | None):
        items = items if isinstance(items, list) else []
        if not items:
            yield (
                _borderline_dataframe([]),
                [],
                DEFAULT_LLM_PANEL_MESSAGE,
                DEFAULT_LLM_PANEL_MESSAGE,
                "No borderline phrases to label.",
            )
            return

        yield (
            _borderline_dataframe(items),
            items,
            DEFAULT_LLM_PANEL_MESSAGE,
            DEFAULT_LLM_PANEL_MESSAGE,
            f"Labeling {len(items)} borderline phrase(s) with Qwen3-4B. This can take a few seconds the first time Ollama loads the model...",
        )
        try:
            labeled = label_borderline_items_with_ollama(items, config=config)
        except Exception as exc:
            error_message = f"### LLM Review Suggestions\nLLM request failed: {exc}"
            yield (
                _borderline_dataframe(items),
                items,
                error_message,
                error_message,
                f"LLM request failed: {exc}",
            )
            return
        summary = _format_llm_review_summary(labeled)
        yield (
            _borderline_dataframe(labeled),
            labeled,
            summary,
            DEFAULT_LLM_PANEL_MESSAGE,
            f"Labeled {len(labeled)} borderline phrase(s) with Ollama.",
        )

    def approve_ui(dataframe: pd.DataFrame, review_items: list[dict[str, Any]] | None) -> tuple[str, Any]:
        if not _records_from_table_value(dataframe):
            summary = approve_demo_examples([], config=config)
            return "No labeled rows available to save.", gr.update(interactive=summary["ready_to_retrain"])

        approved_items = _merge_review_items(dataframe, review_items)
        selected_count = sum(1 for item in approved_items if bool(item.get("approved")))
        trainable_count = sum(
            1
            for item in approved_items
            if bool(item.get("approved")) and str(item.get("final_label", "")).strip() in {"Compliant", "Non-Compliant"}
        )
        summary = approve_demo_examples(approved_items, config=config, replace_existing=True)
        if selected_count == 0:
            message = (
                "No rows were selected. Tick `add` for each sample you want to add, "
                "then click Save Selected Examples again. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        elif trainable_count == 0:
            message = (
                "No training examples were saved because the selected row(s) were marked `Skip` or did not have a valid `Pass`/`Fail` final label. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        elif summary["added_count"] == 0:
            message = (
                "No new labeled examples were added. "
                "The current rows may already exist in the latest-run training store. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        else:
            message = (
                f"Saved {summary['added_count']} labeled example(s) to the training store. "
                f"Latest-run stored examples: {summary['approved_count']}. "
                f"Retrain ready: {'yes' if summary['ready_to_retrain'] else 'no'}."
            )
        if not summary["ready_to_retrain"] and summary.get("readiness_note"):
            message = f"{message} {summary['readiness_note']}"
        return message, gr.update(interactive=summary["ready_to_retrain"])

    def retrain_ui() -> str:
        result = retrain_demo_verifier(config=config)
        return json.dumps(result, indent=2)

    def reset_base_models_ui():
        try:
            result = reset_demo_to_baseline(config=config)
            message = (
                "Synthetic-only base models are active again. "
                "Cleared approved examples, augmented data, and retrained model versions."
            )
            status = "reset"
        except Exception as exc:
            result = {"status": "error", "message": str(exc)}
            message = f"Base-model reset failed: {exc}"
            status = "error"
        summary = {
            "status": status,
            "message": message,
            "recommendation": "Upload transcript(s), then start agentic review from the base models.",
            "reset": result,
            "stage_status": [
                {
                    "agent": "SupervisorAgent",
                    "status": status,
                    "message": message,
                }
            ],
        }
        return (
            _format_agentic_progress(),
            _format_app_agentic_summary(summary),
            _agentic_review_dataframe([]),
            _format_agentic_comparison_markdown([]),
            json.dumps(result, indent=2, ensure_ascii=False),
            summary,
            [],
            "",
            gr.update(interactive=False),
        )

    def chat_ui(
        question: str,
        payload: dict[str, Any],
        review_items: list[dict[str, Any]] | None,
        history: list[dict[str, Any]] | None,
    ):
        question_text = str(question or "").strip()
        if not question_text:
            current_history = _normalized_chat_messages(history)
            yield current_history, current_history, ""
            return

        pending_history = _append_user_message(history, question_text)
        yield pending_history, pending_history, ""

        try:
            answer = answer_demo_question(
                question_text,
                inference_payload=_chat_context(payload, review_items or []),
                chat_history=history,
                config=config,
            )
        except Exception as exc:
            answer = f"LLM request failed: {exc}"
        final_history = [*pending_history, {"role": "assistant", "content": answer}]
        yield final_history, final_history, ""

    def start_agentic_review_ui(
        transcript: str,
        incoming_uploads: Any,
        incoming_folder_uploads: Any,
        active_upload_source: str,
    ):
        incoming_source, cleanup_paths, incoming_error = _materialize_incoming_source(
            transcript,
            incoming_uploads,
            incoming_folder_uploads,
            active_source=active_upload_source,
        )
        if incoming_error:
            failure = {"status": "blocked", "message": incoming_error}
            yield (
                _format_agentic_progress(active="review", blocked=True),
                _format_app_agentic_summary(failure),
                _agentic_review_dataframe([]),
                _format_agentic_comparison_markdown([]),
                "{}",
                failure,
                [],
                "",
                gr.update(interactive=False),
            )
            return

        yield (
            _format_agentic_progress(active="review"),
            "## Agentic Loop\n- Status: `running`\n- Message: InferenceAgent is processing transcript(s), then Qwen will classify any borderline phrases.",
            _agentic_review_dataframe([]),
            _format_agentic_comparison_markdown([]),
            "{}",
            {},
            [],
            "",
            gr.update(interactive=False),
        )

        try:
            transcripts = load_incoming_transcripts(
                config=config,
                incoming_source=incoming_source,
            )
            summary = run_agentic_review_cycle(transcripts, config=config)
        except Exception as exc:
            for path in cleanup_paths:
                shutil.rmtree(path, ignore_errors=True)
            failure = {
                "status": "error",
                "message": f"Agentic loop failed: {exc}",
            }
            yield (
                _format_agentic_progress(active="review", blocked=True),
                _format_app_agentic_summary(failure),
                _agentic_review_dataframe([]),
                _format_agentic_comparison_markdown([]),
                json.dumps(failure, indent=2),
                failure,
                [],
                "",
                gr.update(interactive=False),
            )
            return
        finally:
            for path in cleanup_paths:
                shutil.rmtree(path, ignore_errors=True)

        review_items = summary.get("review_items", []) if isinstance(summary.get("review_items"), list) else []
        completed_steps = {"review"} if review_items else {"review", "approval", "retrain", "reinference"}
        yield (
            _format_agentic_progress(active="approval" if review_items else None, completed=completed_steps),
            _format_app_agentic_summary(summary),
            _agentic_review_dataframe(review_items),
            _format_agentic_comparison_markdown([]),
            json.dumps(summary, indent=2, ensure_ascii=False),
            summary,
            [],
            "",
            gr.update(interactive=bool(review_items)),
        )

    def continue_agentic_ui(dataframe: pd.DataFrame, summary_payload: dict[str, Any]):
        summary_payload = summary_payload if isinstance(summary_payload, dict) else {}
        review_items = summary_payload.get("review_items", [])
        approved_items = _merge_review_items(dataframe, review_items if isinstance(review_items, list) else [])
        running = {
            **summary_payload,
            "status": "running",
            "message": "TrainerAgent is saving approved rows, retraining from base models, then rerunning inference with the candidate model.",
        }
        yield (
            _format_agentic_progress(active="retrain", completed={"review", "approval"}),
            _format_app_agentic_summary(running),
            _format_agentic_comparison_markdown([]),
            json.dumps(running, indent=2, ensure_ascii=False),
            running,
            [],
            "",
        )

        try:
            retrain_state = prepare_agentic_training_cycle(summary_payload, approved_items, config=config)
        except Exception as exc:
            retrain_state = {
                "status": "error",
                "message": f"Agentic retraining failed: {exc}",
                "comparisons": [],
            }

        if retrain_state.get("status") != "retrained":
            yield (
                _format_agentic_progress(active="retrain", completed={"review", "approval"}, blocked=True),
                _format_app_agentic_summary(retrain_state),
                _format_agentic_comparison_markdown([]),
                json.dumps(retrain_state, indent=2, ensure_ascii=False),
                retrain_state,
                [],
                "",
            )
            return

        reinference_running = {
            **retrain_state,
            "status": "running",
            "message": "InferenceAgent is rerunning the uploaded transcript(s) with the newly trained candidate model.",
        }
        yield (
            _format_agentic_progress(active="reinference", completed={"review", "approval", "retrain"}),
            _format_app_agentic_summary(reinference_running),
            _format_agentic_comparison_markdown([]),
            json.dumps(reinference_running, indent=2, ensure_ascii=False),
            reinference_running,
            [],
            "",
        )
        time.sleep(AGENTIC_REINFERENCE_PROGRESS_DWELL_SEC)

        try:
            result = complete_agentic_reinference_cycle(retrain_state, config=config)
        except Exception as exc:
            result = {
                **retrain_state,
                "status": "error",
                "message": f"Agentic re-inference failed: {exc}",
                "comparisons": [],
            }

        progress = (
            _format_agentic_progress(completed={"review", "approval", "retrain", "reinference"})
            if result.get("status") in {"completed", "needs_review"}
            else _format_agentic_progress(active="retrain", completed={"review", "approval"}, blocked=True)
        )
        yield (
            progress,
            _format_app_agentic_summary(result),
            _format_agentic_comparison_markdown(result.get("comparisons", []) if isinstance(result, dict) else []),
            json.dumps(result, indent=2, ensure_ascii=False),
            result,
            [],
            "",
        )

    def agentic_chat_ui(
        question: str,
        summary_payload: dict[str, Any],
        history: list[dict[str, Any]] | None,
    ):
        question_text = str(question or "").strip()
        if not question_text:
            yield _normalized_chat_messages(history), ""
            return

        pending_history = _append_user_message(history, question_text)
        yield pending_history, ""

        try:
            answer = answer_agentic_question(
                question_text,
                summary_payload=summary_payload,
                chat_history=history,
                config=config,
            )
        except Exception as exc:
            answer = f"LLM request failed: {exc}"
        yield [*pending_history, {"role": "assistant", "content": answer}], ""

    def append_agentic_user_message(
        question: str,
        history: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, str]], str, str]:
        question_text = str(question or "").strip()
        if not question_text:
            return _normalized_chat_messages(history), "", ""
        return _append_user_message(history, question_text), "", question_text

    def append_agentic_prompt_message(
        history: list[dict[str, Any]] | None,
        prompt: str,
    ) -> tuple[list[dict[str, str]], str, str]:
        return _append_user_message(history, prompt), "", prompt

    def answer_agentic_chat_message(
        question: str,
        summary_payload: dict[str, Any],
        history: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        question_text = str(question or "").strip()
        current_history = _normalized_chat_messages(history)
        if not question_text:
            return current_history
        try:
            answer = answer_agentic_question(
                question_text,
                summary_payload=summary_payload,
                chat_history=current_history[:-1],
                config=config,
            )
        except Exception as exc:
            answer = f"LLM request failed: {exc}"
        return [*current_history, {"role": "assistant", "content": answer}]

    def investigate_regressions_ui(summary_payload: dict[str, Any]) -> str:
        summary_payload = summary_payload if isinstance(summary_payload, dict) else {}
        comparisons = [
            item
            for item in summary_payload.get("comparisons", [])
            if isinstance(item, dict) and str(item.get("outcome", "")).strip() == "regressed"
        ]
        if comparisons:
            diagnosis = summary_payload.get("score_regression_diagnosis", {})
            analyses = diagnosis.get("analyses", []) if isinstance(diagnosis, dict) else []
            if not analyses:
                diagnosis = diagnose_score_regressions(comparisons, config=config)
                summary_payload = {
                    **summary_payload,
                    "score_regression_diagnosis": diagnosis,
                }
        return _format_regression_investigation_report(summary_payload)

    def choose_agentic_file_upload(upload_value: Any, current_source: str) -> tuple[Any, str]:
        if _extract_uploaded_paths(upload_value):
            return gr.update(value=None), "file"
        return gr.update(), current_source

    def choose_agentic_folder_upload(upload_value: Any, current_source: str) -> tuple[Any, str]:
        if _extract_uploaded_paths(upload_value):
            return gr.update(value=None), "folder"
        return gr.update(), current_source

    with gr.Blocks(title="UOB AI Labs") as app:
        gr.HTML(_hero_html())

        payload_state = gr.State({})
        review_items_state = gr.State([])
        chat_history_state = gr.State([])
        agentic_summary_state = gr.State({})
        agentic_empty_transcript_state = gr.State("")
        agentic_pending_question_state = gr.State("")
        agentic_upload_source_state = gr.State("")

        with gr.Tabs():
            with gr.Tab("Transcript Review"):
                with gr.Row(equal_height=True, elem_id="transcript-source-row"):
                    with gr.Column(scale=1, elem_classes=["transcript-source-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 1</div>"
                            "<div class='source-title'>Load a demo sample</div>"
                            "<div class='source-copy'>Start from a curated synthetic transcript for a quick walkthrough.</div>"
                        )
                        sample_dropdown = gr.Dropdown(
                            choices=sorted(sample_lookup.keys()),
                            label="Sample transcript",
                            value=next(iter(sample_lookup.keys()), None),
                        )
                        load_btn = gr.Button("Load selected sample")
                    with gr.Column(scale=1, elem_classes=["transcript-source-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 2</div>"
                            "<div class='source-title'>Upload your own script</div>"
                            "<div class='source-copy'>Use a local plain-text transcript and run it through the same review workflow.</div>"
                        )
                        transcript_upload = gr.UploadButton(
                            "Choose .txt transcript",
                            file_count="single",
                            file_types=[".txt"],
                            size="lg",
                            elem_id="upload-transcript-btn",
                        )
                        gr.HTML(
                            "<div class='source-note'>"
                            f"Tip: bundled scripts live in <code>{DEMO_TRANSCRIPT_SCRIPTS_DIR}</code>."
                            "</div>"
                        )

                transcript_box = gr.Textbox(label="Transcript", lines=10, placeholder="Paste a transcript here...")
                load_status = gr.Markdown()
                inference_btn = gr.Button("Run Inference", variant="primary")

                results_markdown = gr.Markdown()
                borderline_summary = gr.Markdown()
                borderline_table = gr.Dataframe(
                    headers=[
                        "add",
                        "rule",
                        "score",
                        "text",
                        "model_label",
                        "qwen_label",
                        "final_label",
                    ],
                    datatype=[
                        "bool",
                        "str",
                        "number",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    row_count=(0, "dynamic"),
                    column_count=(7, "fixed"),
                    interactive=True,
                    label="Borderline Review Queue",
                    elem_id="borderline-review-table",
                    wrap=True,
                    min_width=0,
                    column_widths=["84px", "96px", "136px", "420px", "172px", "172px", "172px"],
                )

                with gr.Row():
                    label_btn = gr.Button("Label Borderline Phrases with Qwen3-4B")
                    approve_btn = gr.Button("Save Selected Examples")
                    retrain_btn = gr.Button(
                        "Retrain Models",
                        interactive=approved_summary["ready_to_retrain"],
                        variant="secondary",
                    )

                approval_status = gr.Markdown()
                llm_review_results = gr.Markdown(DEFAULT_LLM_PANEL_MESSAGE, elem_id="llm-review-results")
                retrain_output = gr.Code(label="Retrain Output", language="json")

                with gr.Group(elem_id="llm-review-shell"):
                    with gr.Accordion("LLM Review Panel", open=False, elem_id="llm-review-panel"):
                        llm_review_summary = gr.Markdown(DEFAULT_LLM_PANEL_MESSAGE)
                        gr.Markdown("**Quick review prompts**")
                        with gr.Row(elem_id="llm-prompt-buttons"):
                            prompt_buttons = [gr.Button(prompt, size="sm") for prompt in QUICK_REVIEW_PROMPTS]
                        chatbot = gr.Chatbot(
                            label="Review Assistant",
                            elem_id="llm-review-chatbot",
                            height=320,
                            max_height=320,
                        )
                        chat_input = gr.Textbox(label="Ask a follow-up question")

                load_btn.click(load_sample, inputs=[sample_dropdown], outputs=[transcript_box, load_status])
                transcript_upload.upload(load_uploaded_script, inputs=[transcript_upload], outputs=[transcript_box, load_status])
                inference_btn.click(
                    run_inference_ui,
                    inputs=[transcript_box],
                    outputs=[
                        results_markdown,
                        borderline_table,
                        payload_state,
                        borderline_summary,
                        review_items_state,
                        llm_review_results,
                        llm_review_summary,
                        approval_status,
                        chatbot,
                        chat_history_state,
                    ],
                )
                label_btn.click(
                    label_borderline_ui,
                    inputs=[review_items_state],
                    outputs=[
                        borderline_table,
                        review_items_state,
                        llm_review_results,
                        llm_review_summary,
                        approval_status,
                    ],
                )
                approve_btn.click(
                    approve_ui,
                    inputs=[borderline_table, review_items_state],
                    outputs=[approval_status, retrain_btn],
                )
                retrain_btn.click(retrain_ui, outputs=[retrain_output])
                chat_input.submit(
                    chat_ui,
                    inputs=[chat_input, payload_state, review_items_state, chat_history_state],
                    outputs=[chatbot, chat_history_state, chat_input],
                )
                for button, prompt in zip(prompt_buttons, QUICK_REVIEW_PROMPTS):
                    button.click(
                        fn=lambda payload, review_items, history, prompt=prompt, config=config: _preset_prompt(prompt, payload, review_items, history, config=config),
                        inputs=[payload_state, review_items_state, chat_history_state],
                        outputs=[chatbot, chat_history_state, chat_input],
                    )

            with gr.Tab("Agentic Loop"):
                gr.Markdown(
                    "Upload transcript(s), let the agents find and classify borderline evidence, approve training rows, "
                    "then retrain and compare the same transcript scores before and after."
                )
                with gr.Row(equal_height=True, elem_classes=["agentic-upload-row"]):
                    with gr.Column(scale=1, elem_classes=["agentic-upload-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 1</div>"
                            "<div class='source-title'>Upload file(s)</div>"
                            "<div class='source-copy'>Choose one <code>.txt</code>/<code>.json</code> file or several <code>.txt</code> transcripts.</div>"
                        )
                        incoming_upload = gr.File(
                            label="Transcript file(s)",
                            file_count="multiple",
                            file_types=[".txt", ".json"],
                            height=110,
                        )
                    with gr.Column(scale=1, elem_classes=["agentic-upload-card"]):
                        gr.HTML(
                            "<div class='source-kicker'>Option 2</div>"
                            "<div class='source-title'>Upload folder</div>"
                            "<div class='source-copy'>Choose a local folder. The loop will use every <code>.txt</code> transcript inside it.</div>"
                        )
                        incoming_folder_upload = gr.File(
                            label="Folder of .txt transcripts",
                            file_count="directory",
                            file_types=[".txt"],
                            height=110,
                        )
                with gr.Row():
                    agentic_reset_base_btn = gr.Button("Reset to Base Models", variant="secondary")
                    agentic_run_btn = gr.Button("Start Agentic Review", variant="primary")
                    agentic_continue_btn = gr.Button(
                        "Approve, Retrain, Compare",
                        interactive=False,
                        variant="secondary",
                        elem_id="agentic-continue-btn",
                    )
                agentic_progress = gr.HTML(_format_agentic_progress(), elem_id="agentic-progress")
                agentic_summary_markdown = gr.Markdown("## Agentic Loop\n- No run output available yet.")
                agentic_review_table = gr.Dataframe(
                    headers=[
                        "add",
                        "rule",
                        "score",
                        "anchor",
                        "text",
                        "model",
                        "qwen",
                        "human",
                    ],
                    datatype=[
                        "bool",
                        "str",
                        "number",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    row_count=(0, "dynamic"),
                    column_count=(8, "fixed"),
                    interactive=True,
                    label="Human Approval Queue",
                    wrap=True,
                    min_width=0,
                )
                agentic_comparison_markdown = gr.Markdown(_format_agentic_comparison_markdown([]))
                with gr.Row():
                    agentic_investigate_btn = gr.Button("Investigate Score Regressions", variant="secondary")
                agentic_investigation_markdown = gr.Markdown(
                    "### InvestigatorAgent Diagnosis\nRun retrain-and-compare, then click this if any approved phrase regresses."
                )
                with gr.Accordion("Raw Agentic Output", open=False):
                    agentic_raw_output = gr.Code(label="Loop Output", language="json")

                with gr.Group(elem_id="agentic-analysis-shell"):
                    with gr.Accordion("Agentic Analysis Assistant", open=False, elem_id="agentic-analysis-panel"):
                        agentic_chat_summary = gr.Markdown(DEFAULT_AGENTIC_CHAT_MESSAGE)
                        gr.Markdown("**Quick prompts**")
                        with gr.Row(elem_id="agentic-prompt-buttons"):
                            agentic_prompt_buttons = [gr.Button(prompt, size="sm") for prompt in AGENTIC_QUICK_PROMPTS]
                        agentic_chatbot = gr.Chatbot(
                            label="Agentic Analysis Assistant",
                            elem_id="agentic-analysis-chatbot",
                            height=340,
                            max_height=340,
                        )
                        agentic_chat_input = gr.Textbox(label="Ask about the agentic run")

                incoming_upload.change(
                    choose_agentic_file_upload,
                    inputs=[incoming_upload, agentic_upload_source_state],
                    outputs=[incoming_folder_upload, agentic_upload_source_state],
                    queue=False,
                )
                incoming_folder_upload.change(
                    choose_agentic_folder_upload,
                    inputs=[incoming_folder_upload, agentic_upload_source_state],
                    outputs=[incoming_upload, agentic_upload_source_state],
                    queue=False,
                )

                agentic_run_btn.click(
                    start_agentic_review_ui,
                    inputs=[
                        agentic_empty_transcript_state,
                        incoming_upload,
                        incoming_folder_upload,
                        agentic_upload_source_state,
                    ],
                    outputs=[
                        agentic_progress,
                        agentic_summary_markdown,
                        agentic_review_table,
                        agentic_comparison_markdown,
                        agentic_raw_output,
                        agentic_summary_state,
                        agentic_chatbot,
                        agentic_chat_input,
                        agentic_continue_btn,
                    ],
                )

                agentic_reset_base_btn.click(
                    reset_base_models_ui,
                    outputs=[
                        agentic_progress,
                        agentic_summary_markdown,
                        agentic_review_table,
                        agentic_comparison_markdown,
                        agentic_raw_output,
                        agentic_summary_state,
                        agentic_chatbot,
                        agentic_chat_input,
                        agentic_continue_btn,
                    ],
                )

                agentic_continue_btn.click(
                    continue_agentic_ui,
                    inputs=[agentic_review_table, agentic_summary_state],
                    outputs=[
                        agentic_progress,
                        agentic_summary_markdown,
                        agentic_comparison_markdown,
                        agentic_raw_output,
                        agentic_summary_state,
                        agentic_chatbot,
                        agentic_chat_input,
                    ],
                )

                agentic_investigate_btn.click(
                    investigate_regressions_ui,
                    inputs=[agentic_summary_state],
                    outputs=[agentic_investigation_markdown],
                )

                agentic_chat_input.submit(
                    append_agentic_user_message,
                    inputs=[agentic_chat_input, agentic_chatbot],
                    outputs=[agentic_chatbot, agentic_chat_input, agentic_pending_question_state],
                    queue=False,
                ).then(
                    answer_agentic_chat_message,
                    inputs=[agentic_pending_question_state, agentic_summary_state, agentic_chatbot],
                    outputs=[agentic_chatbot],
                )
                for button, prompt in zip(agentic_prompt_buttons, AGENTIC_QUICK_PROMPTS):
                    button.click(
                        fn=lambda history, prompt=prompt: append_agentic_prompt_message(history, prompt),
                        inputs=[agentic_chatbot],
                        outputs=[agentic_chatbot, agentic_chat_input, agentic_pending_question_state],
                        queue=False,
                    ).then(
                        answer_agentic_chat_message,
                        inputs=[agentic_pending_question_state, agentic_summary_state, agentic_chatbot],
                        outputs=[agentic_chatbot],
                    )

    return app.queue(default_concurrency_limit=1)


def launch_demo_app(
    *,
    config_path: str | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    _extend_gradio_server_start_timeout()
    app = build_demo_app(config_path)
    app.launch(server_name=server_name, server_port=server_port, share=share, css=APP_CSS)
