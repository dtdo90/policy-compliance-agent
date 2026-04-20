"""Services for the privacy-safe demo workflow."""

from __future__ import annotations

import json
import math
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.config import load_config
from ..core.disclosures import filter_disclaimers, load_disclaimers
from ..core.paths import ensure_parent_dir, resolve_project_path
from ..inference.semantic import DEFAULT_EXCLUDED_RULE_IDS, SemanticComplianceAnalyzer
from ..training.cross_encoder import train_cross_encoder_with_overrides
from ..training.sentence_transformer import train_sentence_transformer_with_overrides


DEFAULT_DEMO_CONFIG_PATH = "configs/demo.yaml"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", flags=re.IGNORECASE | re.DOTALL)
REASONING_LEAK_RE = re.compile(
    r"^\s*(okay|alright|let me|i need to|the user|looking at|first,|wait,|we need to|we are given|rules to follow|from the context)\b",
    flags=re.IGNORECASE,
)
MAX_CHAT_HISTORY_MESSAGES = 6
MAX_CHAT_HISTORY_CHARS = 500
MAX_CHAT_CONTEXT_TEXT_CHARS = 240
MAX_CHAT_REVIEW_ITEMS = 5
CHAT_NUM_PREDICT = 768
LABEL_NUM_PREDICT = 128


@dataclass
class OllamaChatClient:
    base_url: str
    model: str
    timeout_sec: int = 90
    think: bool | None = None
    num_predict: int | None = None

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        json_mode: bool = False,
        num_predict: int | None = None,
    ) -> str:
        options = {
            "temperature": temperature,
            "top_p": 1.0,
        }
        predict_limit = self.num_predict if num_predict is None else num_predict
        if predict_limit is not None:
            options["num_predict"] = int(predict_limit)

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        }
        if json_mode:
            body["format"] = "json"
        if self.think is not None:
            body["think"] = bool(self.think)
        request = Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        message = payload.get("message", {})
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError(f"Unexpected Ollama response payload: {payload}")
        return message["content"].strip()


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def load_demo_config(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return load_config(config_path or DEFAULT_DEMO_CONFIG_PATH, overrides=overrides)


def _demo_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("demo", {})


def _load_demo_analyzer(config: dict[str, Any]) -> SemanticComplianceAnalyzer:
    data_settings = config.get("data", {})
    semantic_settings = config.get("semantic_inference", {})
    include_rule_ids = semantic_settings.get("include_rule_ids")
    disclosures = load_disclaimers(data_settings["disclosures_file"])
    disclosures = filter_disclaimers(
        disclosures,
        include_rule_ids=include_rule_ids,
        exclude_rule_ids=DEFAULT_EXCLUDED_RULE_IDS,
    )
    return SemanticComplianceAnalyzer(disclosures, config)


def _results_only(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "results" in payload and isinstance(payload["results"], dict):
        return payload["results"]
    return payload


def _json_load(path: str | Path, default: Any) -> Any:
    resolved = resolve_project_path(path)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _json_save(path: str | Path, data: Any) -> Path:
    resolved = ensure_parent_dir(path)
    resolved.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


def _normalize_training_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("disclaimer_id", "")).strip().lower(),
        " ".join(str(item.get("anchor", "")).strip().lower().split()),
        " ".join(str(item.get("dialogue", "")).strip().lower().split()),
        str(item.get("type", "")).strip().lower(),
    )


def _review_label_to_model_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if text in {"pass", "passed", "positive", "compliant", "yes", "true", "1"}:
        return "Compliant"
    if text in {"fail", "failed", "negative", "non-compliant", "noncompliant", "no", "false", "0"}:
        return "Non-Compliant"
    if "non-compliant" in text or "noncompliant" in text:
        return "Non-Compliant"
    if "compliant" in text:
        return "Compliant"
    return ""


def _load_disclosure_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    disclosures_path = resolve_project_path(config.get("data", {}).get("disclosures_file", ""))
    data = json.loads(disclosures_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _build_rule_rubric(disclosure: dict[str, Any]) -> str:
    description = str(disclosure.get("Description", "")).strip()
    purpose = str(disclosure.get("Purpose_of_control", "")).strip()
    criteria = str(disclosure.get("Criteria", "")).strip()
    anchor = disclosure.get("anchor", "")

    if isinstance(anchor, dict):
        mandatory = [
            str(item).strip()
            for item in anchor.get("mandatory", [])
            if isinstance(item, str) and item.strip()
        ]
        anchor_summary = "\n".join(f"- {item}" for item in mandatory)
    else:
        anchor_summary = f"- {str(anchor).strip()}"

    return (
        f"Description: {description}\n"
        f"Purpose: {purpose}\n"
        f"Criteria: {criteria}\n"
        "Anchor meanings that count as compliant:\n"
        f"{anchor_summary}"
    )


def run_demo_inference(
    transcript: str,
    *,
    transcript_id: str = "interactive_demo",
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    analyzer: SemanticComplianceAnalyzer | None = None,
) -> dict[str, Any]:
    config = config or load_demo_config(config_path)
    analyzer = analyzer or _load_demo_analyzer(config)
    results = analyzer.analyze_transcript(transcript, transcript_id)
    payload = {
        "transcript_id": transcript_id,
        "transcript": transcript,
        "results": results,
    }
    payload["borderline_items"] = get_borderline_items(payload, config=config)
    return payload


def get_borderline_items(
    result: dict[str, Any],
    *,
    low: float = 0.30,
    high: float = 0.70,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    config = config or load_demo_config(config_path)
    disclosures = _load_disclosure_lookup(config)
    results = _results_only(result)

    items: list[dict[str, Any]] = []
    for disclaimer_id, disclaimer_result in results.items():
        if not isinstance(disclaimer_result, dict):
            continue
        disclosure = disclosures.get(str(disclaimer_id), {})
        evidence = disclaimer_result.get("evidence", {})
        claims = evidence.get("claims", {}) if isinstance(evidence, dict) else {}
        claim_review_items: list[dict[str, Any]] = []
        if isinstance(claims, dict):
            for claim_group in ("single", "mandatory", "standard"):
                claim_rows = claims.get(claim_group, [])
                if not isinstance(claim_rows, list):
                    continue
                for claim_order, claim in enumerate(claim_rows):
                    if not isinstance(claim, dict):
                        continue
                    claim_review_items.append(
                        {
                            "text": str(claim.get("match_text", "")).strip(),
                            "retrieval_score": float(claim.get("retrieval_score") or 0.0),
                            "verification_score": float(claim.get("verification_score") or 0.0),
                            "claim_type": str(claim.get("claim_type") or claim_group).strip(),
                            "claim_text": str(claim.get("anchor", "")).strip(),
                            "claim_order": int(claim.get("claim_idx", claim_order) or claim_order),
                        }
                    )

        # Prefer the single best phrase per anchor from structured claim evidence.
        # Older/minimal test payloads may only expose the legacy for_review list.
        review_items = claim_review_items or [
            item for item in disclaimer_result.get("for_review", []) if isinstance(item, dict)
        ]
        for review_item in review_items:
            if not isinstance(review_item, dict):
                continue
            score = float(review_item.get("verification_score") or 0.0)
            if not (low <= score < high):
                continue
            items.append(
                {
                    "transcript_id": str(result.get("transcript_id", "interactive_demo")),
                    "disclaimer_id": str(disclaimer_id),
                    "description": str(disclosure.get("Description", "")).strip(),
                    "rubric": _build_rule_rubric(disclosure) if isinstance(disclosure, dict) else "",
                    "claim_type": str(review_item.get("claim_type", "")).strip(),
                    "anchor": str(review_item.get("claim_text", "")).strip(),
                    "text": str(review_item.get("text", "")).strip(),
                    "retrieval_score": float(review_item.get("retrieval_score") or 0.0),
                    "verification_score": score,
                    "claim_order": int(review_item.get("claim_order", 0) or 0),
                }
            )

    items.sort(key=lambda item: (item["disclaimer_id"], item["claim_order"], item["verification_score"], item["text"]))
    return items


def get_agentic_review_items(
    result: dict[str, Any],
    *,
    low: float = 0.30,
    high: float = 0.70,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return pass and borderline claim-level phrases for human review."""
    config = config or load_demo_config(config_path)
    disclosures = _load_disclosure_lookup(config)
    results = _results_only(result)

    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for disclaimer_id, disclaimer_result in results.items():
        if not isinstance(disclaimer_result, dict):
            continue
        disclosure = disclosures.get(str(disclaimer_id), {})
        evidence = disclaimer_result.get("evidence", {})
        claims = evidence.get("claims", {}) if isinstance(evidence, dict) else {}
        if not isinstance(claims, dict):
            continue

        for claim_group in ("single", "mandatory", "standard"):
            claim_rows = claims.get(claim_group, [])
            if not isinstance(claim_rows, list):
                continue
            for claim_order, claim in enumerate(claim_rows):
                if not isinstance(claim, dict):
                    continue
                score = float(claim.get("verification_score") or 0.0)
                passed = bool(claim.get("passed", False))
                is_borderline = low <= score < high
                if not passed and not is_borderline:
                    continue
                anchor = str(claim.get("anchor", "")).strip()
                text = str(claim.get("match_text", "")).strip()
                key = (str(disclaimer_id), str(claim.get("claim_type") or claim_group), anchor, text)
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "transcript_id": str(result.get("transcript_id", "interactive_demo")),
                        "disclaimer_id": str(disclaimer_id),
                        "description": str(disclosure.get("Description", "")).strip(),
                        "rubric": _build_rule_rubric(disclosure) if isinstance(disclosure, dict) else "",
                        "claim_type": str(claim.get("claim_type") or claim_group).strip(),
                        "anchor": anchor,
                        "text": text,
                        "retrieval_score": float(claim.get("retrieval_score") or 0.0),
                        "verification_score": score,
                        "claim_order": int(claim.get("claim_idx", claim_order) or claim_order),
                        "review_type": "borderline" if is_borderline else "pass",
                        "model_label": "Compliant" if passed else "Non-Compliant",
                    }
                )

    items.sort(
        key=lambda item: (
            item["transcript_id"],
            item["disclaimer_id"],
            item["claim_order"],
            0 if item.get("review_type") == "borderline" else 1,
            item["text"],
        )
    )
    return items


def _parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "").strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start < 0:
        return {}
    candidate = text[start:]
    for end in range(len(candidate), 0, -1):
        snippet = candidate[:end]
        try:
            value = json.loads(snippet)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return {}


def strip_think_blocks(text: str) -> str:
    stripped = THINK_BLOCK_RE.sub("", str(text or "")).strip()
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].strip()
    if stripped.lower().startswith("<think>"):
        return ""
    return stripped


def _compact_text(value: Any, max_chars: int = MAX_CHAT_CONTEXT_TEXT_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _compact_chat_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = strip_think_blocks(str(item.get("content", ""))).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append(
            {
                "role": role,
                "content": _compact_text(content, MAX_CHAT_HISTORY_CHARS),
            }
        )
    return messages[-MAX_CHAT_HISTORY_MESSAGES:]


def _compact_demo_chat_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {"summary": "No inference result is available yet."}

    payload = context.get("inference") if isinstance(context.get("inference"), dict) else context
    results = payload.get("results") if isinstance(payload, dict) else {}
    compact_rules: list[dict[str, Any]] = []

    if isinstance(results, dict):
        for rule_id, result in results.items():
            if not isinstance(result, dict):
                continue
            evidence = result.get("evidence", {})
            evidence = evidence if isinstance(evidence, dict) else {}
            try:
                rule_score = round(float(evidence.get("verification_score") or 0.0), 4)
            except (TypeError, ValueError):
                rule_score = 0.0

            rule_summary: dict[str, Any] = {
                "rule_id": str(rule_id),
                "status": str(result.get("status", "UNKNOWN")).strip() or "UNKNOWN",
                "score": rule_score,
            }
            description = _compact_text(evidence.get("description", ""), 160)
            if description:
                rule_summary["description"] = description

            claims = evidence.get("claims", {})
            claim_summaries: list[dict[str, Any]] = []
            if isinstance(claims, dict):
                for claim_group in ("single", "mandatory", "standard"):
                    claim_rows = claims.get(claim_group, [])
                    if not isinstance(claim_rows, list):
                        continue
                    for index, claim in enumerate(claim_rows):
                        if not isinstance(claim, dict):
                            continue
                        try:
                            claim_score = round(float(claim.get("verification_score") or 0.0), 4)
                        except (TypeError, ValueError):
                            claim_score = 0.0
                        claim_summaries.append(
                            {
                                "claim_type": str(claim.get("claim_type") or claim_group),
                                "claim_index": int(claim.get("claim_idx", index) or index),
                                "passed": bool(claim.get("passed", False)),
                                "score": claim_score,
                                "anchor": _compact_text(claim.get("anchor", "")),
                                "best_text": _compact_text(claim.get("match_text", "")),
                            }
                        )
            if claim_summaries:
                rule_summary["claims"] = claim_summaries
            else:
                rule_summary["best_text"] = _compact_text(evidence.get("match_text", ""))
            compact_rules.append(rule_summary)

    compact_review_items: list[dict[str, Any]] = []
    review_items = context.get("review_items")
    if isinstance(review_items, list):
        for item in review_items[:MAX_CHAT_REVIEW_ITEMS]:
            if not isinstance(item, dict):
                continue
            try:
                score = round(float(item.get("verification_score") or 0.0), 4)
            except (TypeError, ValueError):
                score = 0.0
            compact_review_items.append(
                {
                    "rule_id": str(item.get("disclaimer_id", "")).strip(),
                    "score": score,
                    "anchor": _compact_text(item.get("anchor", "")),
                    "phrase": _compact_text(item.get("text", "")),
                    "qwen_label": _compact_text(item.get("llm_label", ""), 80),
                    "final_label": _compact_text(item.get("final_label", ""), 80),
                    "rationale": _compact_text(item.get("llm_rationale", ""), 180),
                }
            )

    return {
        "assistant_context": _compact_text(context.get("assistant_context", ""), 500),
        "rules": compact_rules,
        "review_items": compact_review_items,
        "omitted": "Full transcript and raw top-k payload are intentionally omitted to keep local chat responsive.",
    }


def _display_review_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("_", "-")
    if not label:
        return ""
    if "non-compliant" in label or "non compliant" in label or "noncompliant" in label:
        return "Fail"
    if "compliant" in label:
        return "Pass"
    if label in {"pass", "passed", "positive", "yes", "y", "true", "1"}:
        return "Pass"
    if label in {"fail", "failed", "negative", "no", "n", "false", "0"}:
        return "Fail"
    if label in {"skip", "skipped", "ambiguous", "unclear"}:
        return "Skip"
    return str(value or "").strip()


def _context_review_items(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    review_items = context.get("review_items")
    return [item for item in review_items if isinstance(item, dict)] if isinstance(review_items, list) else []


def _suggest_final_labels(context: dict[str, Any] | None) -> str:
    review_items = _context_review_items(context)
    if not review_items:
        return f"There are no borderline phrases in the current review queue.\n{_short_inference_summary(context)}"

    lines = ["Suggested final labels:"]
    for index, item in enumerate(review_items, start=1):
        score = float(item.get("verification_score") or 0.0)
        rule_id = str(item.get("disclaimer_id", "")).strip() or "unknown"
        qwen_label = _display_review_label(item.get("llm_label"))
        final_label = _display_review_label(item.get("final_label"))
        suggested_label = final_label or qwen_label or ("Pass" if score >= 0.5 else "Fail")
        phrase = _compact_text(item.get("text", ""), 180) or "No phrase available."
        rationale = _compact_text(item.get("llm_rationale", ""), 180)
        note = f" Qwen label: {qwen_label}." if qwen_label else ""
        if rationale:
            note += f" Reason: {rationale}"
        lines.append(f"{index}. Rule {rule_id}: {suggested_label}, score {score:.3f}. Phrase: \"{phrase}\".{note}")
    return "\n".join(lines)


def _explain_borderline_items(context: dict[str, Any] | None) -> str:
    review_items = _context_review_items(context)
    if not review_items:
        return f"There are no borderline phrases in the current review queue.\n{_short_inference_summary(context)}"

    lines = ["Borderline phrases:"]
    for index, item in enumerate(review_items, start=1):
        score = float(item.get("verification_score") or 0.0)
        rule_id = str(item.get("disclaimer_id", "")).strip() or "unknown"
        anchor = _compact_text(item.get("anchor", ""), 180) or "No anchor available."
        phrase = _compact_text(item.get("text", ""), 180) or "No phrase available."
        qwen_label = _display_review_label(item.get("llm_label")) or "not labeled yet"
        lines.append(
            f"{index}. Rule {rule_id}, score {score:.3f}, Qwen label: {qwen_label}.\n"
            f"   Anchor: {anchor}\n"
            f"   Phrase: {phrase}"
        )
    return "\n".join(lines)


def _direct_demo_answer(question_text: str, context: dict[str, Any] | None) -> str:
    normalized = " ".join(question_text.lower().rstrip(".?!").split())
    if "borderline" in normalized and "final label" in normalized and "suggest" in normalized:
        return _suggest_final_labels(context)
    if "borderline" in normalized and ("explain" in normalized or "phrase" in normalized):
        return _explain_borderline_items(context)
    return ""


def _looks_like_reasoning_leak(text: str) -> bool:
    answer = str(text or "").strip()
    if not answer:
        return False
    if REASONING_LEAK_RE.search(answer):
        return True
    lower = answer[:500].lower()
    reasoning_markers = (
        "we are given",
        "rules to follow",
        "the user is asking",
        "let me check",
        "looking at the",
        "i need to",
        "the current context",
        "from the context",
        "let me draft",
        "we can say",
        "wait,",
    )
    return sum(1 for marker in reasoning_markers if marker in lower) >= 2


def _coerce_llm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    normalized = text.replace("_", "-")
    if "non-compliant" in normalized or "non compliant" in normalized or "noncompliant" in normalized:
        return "Non-Compliant"
    if "compliant" in normalized:
        return "Compliant"
    return ""


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(confidence, 1.0))


def _ollama_client_from_config(config: dict[str, Any]) -> OllamaChatClient:
    demo_settings = _demo_settings(config)
    return OllamaChatClient(
        base_url=str(demo_settings.get("ollama_url", "http://127.0.0.1:11434")).strip(),
        model=str(demo_settings.get("ollama_model", "qwen3:4b")).strip(),
        timeout_sec=int(demo_settings.get("ollama_timeout_sec", 90)),
        think=demo_settings.get("ollama_think"),
        num_predict=int(demo_settings["ollama_num_predict"]) if "ollama_num_predict" in demo_settings else None,
    )


def _claim_review_hint(item: dict[str, Any]) -> str:
    anchor = str(item.get("anchor", "")).strip().lower()
    if "change fee" in anchor:
        return (
            "Treat this anchor as satisfied when the phrase clearly says a fee or charge applies for changing the booking. "
            "Do not require the exact wording about confirmation timing."
        )
    if "fare difference" in anchor or "travel credit" in anchor:
        return (
            "Treat this anchor as satisfied when the phrase clearly says the new itinerary changes the fare, "
            "for example paying extra, receiving credit, or the new option being higher or lower priced. "
            "Do not require the exact wording about confirmation timing."
        )
    if "verify your identity first" in anchor or "reset or unlock" in anchor:
        return (
            "Treat this anchor as satisfied when the phrase clearly says verification or confirming details must happen "
            "before reset, unlock, or account access changes."
        )
    return "Judge only whether the phrase conveys the target anchor meaning in natural language."


def _semantic_anchor_override(item: dict[str, Any]) -> tuple[str, str] | None:
    anchor = str(item.get("anchor", "")).strip().lower()
    text = " ".join(str(item.get("text", "")).strip().lower().split())

    if "change fee" in anchor:
        fee_terms = ("change fee", "fee to make the change", "fee for the change", "change charge", "rebooking fee")
        if any(term in text for term in fee_terms) or ("fee" in text and "change" in text):
            return (
                "Compliant",
                "The phrase clearly states that a fee applies for making the booking change, which satisfies the target change-fee anchor.",
            )

    if "fare difference" in anchor or "travel credit" in anchor:
        pricing_terms = ("higher", "lower", "more expensive", "cheaper", "extra amount", "travel credit", "credit", "balance")
        if any(term in text for term in pricing_terms):
            return (
                "Compliant",
                "The phrase clearly indicates that the new itinerary changes the price outcome, which satisfies the target fare-difference or credit anchor.",
            )

    if "verify your identity first" in anchor or "reset or unlock" in anchor:
        if (
            any(term in text for term in ("verify", "identity", "confirm a couple of details", "confirm some information"))
            and any(term in text for term in ("before", "first"))
        ):
            return (
                "Compliant",
                "The phrase clearly states that verification or confirming details happens before the account change, which satisfies the target verification anchor.",
            )

    return None


def label_review_items_with_ollama(
    items: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    client: OllamaChatClient | None = None,
) -> list[dict[str, Any]]:
    config = config or load_demo_config(config_path)
    client = client or _ollama_client_from_config(config)

    suggestions: list[dict[str, Any]] = []
    system_prompt = (
        "You are helping review borderline compliance matches for a privacy-safe demo app.\n"
        "Use non-thinking mode. Do not reveal reasoning or chain-of-thought.\n"
        "Judge the candidate phrase against the specific target anchor only.\n"
        "Do not require the phrase to satisfy any other anchors from the same rule.\n"
        "Treat the target anchor semantically, not as an exact quote.\n"
        "Equivalent paraphrases count as Compliant when the core meaning is clearly present.\n"
        "Do not over-penalize missing framing words such as 'before I confirm' or 'before I reset' if the core disclosure or requirement is conveyed.\n"
        "These are borderline candidates, so keep confidence moderate rather than extreme.\n"
        "Do not include chain-of-thought, markdown, or explanatory prose outside JSON.\n"
        "Return exactly one valid JSON object with keys: label, confidence, rationale.\n"
        'label must be exactly "Compliant" or "Non-Compliant".'
    )

    for item in items:
        payload = {
            "task": "Evaluate whether the candidate phrase satisfies the target anchor only.",
            "claim_type": item.get("claim_type", ""),
            "claim_review_hint": _claim_review_hint(item),
            "target_anchor": item.get("anchor", ""),
            "candidate_phrase": item.get("text", ""),
            "current_verification_score": item.get("verification_score", 0.0),
        }
        raw_response = client.chat(
            system_prompt=system_prompt,
            user_prompt=(
                "/no_think\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
                "/no_think\n"
                "Return final JSON only."
            ),
            temperature=0.0,
            json_mode=True,
            num_predict=LABEL_NUM_PREDICT,
        )
        parsed = _parse_llm_json(raw_response)
        override = _semantic_anchor_override(item)
        if override is not None:
            label, rationale = override
            confidence = _coerce_confidence(parsed.get("confidence"), default=0.70)
        else:
            label = _coerce_llm_label(parsed.get("label")) or _coerce_llm_label(raw_response)
            rationale = str(parsed.get("rationale", "")).strip()
            confidence = _coerce_confidence(parsed.get("confidence"), default=0.55 if label else 0.0)
            if not label:
                rationale = (
                    "Ollama did not return a usable Compliant/Non-Compliant label. "
                    "Please label this row manually before approving it."
                )
        suggestions.append(
            {
                **deepcopy(item),
                "llm_label": label,
                "llm_confidence": min(confidence, 0.75),
                "llm_rationale": rationale,
            }
        )
    return suggestions


def label_borderline_items_with_ollama(
    items: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    client: OllamaChatClient | None = None,
) -> list[dict[str, Any]]:
    return label_review_items_with_ollama(
        items,
        config=config,
        config_path=config_path,
        client=client,
    )


def _approved_examples_path(config: dict[str, Any]) -> Path:
    return resolve_project_path(config.get("outputs", {}).get("approved_examples_json_path", "data/results/demo/approved_examples.json"))


def _load_approved_examples(config: dict[str, Any]) -> list[dict[str, Any]]:
    value = _json_load(_approved_examples_path(config), [])
    return value if isinstance(value, list) else []


def _approved_summary(config: dict[str, Any], approved_examples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = {str(item.get("type", "")).strip().lower() for item in approved_examples}
    min_required = int(_demo_settings(config).get("min_approved_examples_for_retrain", 10))
    require_both_labels = bool(_demo_settings(config).get("require_both_labels_for_retrain", True))
    has_minimum = len(approved_examples) >= min_required
    has_required_labels = {"compliant", "non-compliant"}.issubset(labels) if require_both_labels else bool(labels)
    ready = has_minimum and has_required_labels
    missing_labels = sorted({"compliant", "non-compliant"} - labels) if require_both_labels else []

    notes: list[str] = []
    if not has_minimum:
        notes.append(f"Need at least {min_required} stored labeled example(s).")
    if require_both_labels and missing_labels:
        notes.append(f"Still need label(s): {', '.join(missing_labels)}.")
    if not require_both_labels and not labels:
        notes.append("Save at least one labeled example to enable retraining.")
    return {
        "approved_count": len(approved_examples),
        "min_required": min_required,
        "labels_present": sorted(labels),
        "require_both_labels": require_both_labels,
        "missing_labels": missing_labels,
        "readiness_note": " ".join(notes).strip(),
        "ready_to_retrain": ready,
    }


def approve_demo_examples(
    items: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    replace_existing: bool = False,
) -> dict[str, Any]:
    config = config or load_demo_config(config_path)
    approved_examples = [] if replace_existing else _load_approved_examples(config)
    existing_keys = {_normalize_training_key(item) for item in approved_examples}
    added = 0

    for item in items:
        decision = bool(item.get("approved", True))
        label = _review_label_to_model_label(
            item.get("final_label")
            or item.get("llm_label")
            or item.get("model_label")
            or ""
        )
        if not decision or label not in {"Compliant", "Non-Compliant"}:
            continue

        phrase = str(item.get("text", "")).strip()
        display_label = "Pass" if label == "Compliant" else "Fail"
        model_label = (
            _review_label_to_model_label(item.get("model_label"))
            or _model_label_from_score(float(item.get("verification_score") or 0.0))
        )
        qwen_label = _review_label_to_model_label(item.get("llm_label"))
        training_row = {
            "disclaimer_id": str(item.get("disclaimer_id", "")).strip(),
            "anchor": str(item.get("anchor", "")).strip(),
            "dialogue": phrase,
            "phrase": phrase,
            "type": "compliant" if label == "Compliant" else "non-compliant",
            "label": display_label,
            "training_label": label,
            "source": "approved_human_review",
            "review_type": str(item.get("review_type", "")).strip(),
            "transcript_id": str(item.get("transcript_id", "interactive_demo")).strip(),
            "verification_score": float(item.get("verification_score") or 0.0),
            "retrieval_score": float(item.get("retrieval_score") or 0.0),
            "model_label": model_label,
            "qwen_label": qwen_label,
            "final_label": label,
            "llm_label": qwen_label,
            "llm_confidence": float(item.get("llm_confidence") or 0.0),
            "llm_rationale": str(item.get("llm_rationale", "")).strip(),
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        key = _normalize_training_key(training_row)
        if key in existing_keys:
            continue
        approved_examples.append(training_row)
        existing_keys.add(key)
        added += 1

    _json_save(_approved_examples_path(config), approved_examples)
    return {
        "added_count": added,
        "replace_existing": replace_existing,
        "approved_examples_path": str(_approved_examples_path(config)),
        **_approved_summary(config, approved_examples),
    }


def _dedupe_dataset_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = _normalize_training_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _load_dataset_rows(path: str | Path) -> list[dict[str, Any]]:
    value = _json_load(path, [])
    return value if isinstance(value, list) else []


def _evaluate_cross_encoder(model_path: Path, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder
    from sklearn.metrics import accuracy_score, average_precision_score

    model = CrossEncoder(str(model_path))
    raw_scores = model.predict([(row["sentence1"], row["sentence2"]) for row in eval_rows], show_progress_bar=False)
    probabilities = [_sigmoid(score) for score in raw_scores]
    labels = [int(float(row["label"]) >= 0.5) for row in eval_rows]
    predictions = [1 if score >= 0.5 else 0 for score in probabilities]

    average_precision = average_precision_score(labels, probabilities) if len(set(labels)) > 1 else 0.0
    accuracy = accuracy_score(labels, predictions) if labels else 0.0
    return {
        "average_precision": float(average_precision),
        "accuracy": float(accuracy),
        "count": len(eval_rows),
    }


def _promote_model(candidate_path: Path, active_path: Path) -> None:
    if active_path.exists():
        shutil.rmtree(active_path, ignore_errors=True)
    shutil.copytree(candidate_path, active_path)


def _retrain_base_model(model_settings: dict[str, Any], *, baseline_key: str, raw_base_key: str) -> str:
    """Prefer the synthetic-trained baseline snapshot, with raw HF base as fallback."""
    baseline_value = str(model_settings.get(baseline_key, "")).strip()
    if baseline_value:
        baseline_path = resolve_project_path(baseline_value)
        if baseline_path.exists():
            return str(baseline_path)
    return str(model_settings.get(raw_base_key, "")).strip()


def freeze_current_demo_baseline(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    config = config or load_demo_config(config_path)
    model_settings = config.get("models", {})
    active_retriever_path = resolve_project_path(model_settings["bi_encoder_path"])
    active_verifier_path = resolve_project_path(model_settings["cross_encoder_path"])
    baseline_retriever_path = resolve_project_path(
        model_settings.get("baseline_bi_encoder_path", "models/demo/baseline/sentence-transformer")
    )
    baseline_verifier_path = resolve_project_path(
        model_settings.get("baseline_cross_encoder_path", "models/demo/baseline/cross-encoder")
    )

    if not active_retriever_path.exists():
        raise FileNotFoundError(f"Active retriever model does not exist: {active_retriever_path}")
    if not active_verifier_path.exists():
        raise FileNotFoundError(f"Active verifier model does not exist: {active_verifier_path}")
    if not overwrite and (baseline_retriever_path.exists() or baseline_verifier_path.exists()):
        raise FileExistsError("Baseline model snapshot already exists.")

    _promote_model(active_retriever_path, baseline_retriever_path)
    _promote_model(active_verifier_path, baseline_verifier_path)
    return {
        "status": "frozen",
        "baseline_retriever_path": str(baseline_retriever_path),
        "baseline_verifier_path": str(baseline_verifier_path),
    }


def reset_demo_to_baseline(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    clear_review_artifacts: bool = True,
) -> dict[str, Any]:
    config = config or load_demo_config(config_path)
    model_settings = config.get("models", {})
    output_settings = config.get("outputs", {})

    baseline_retriever_path = resolve_project_path(
        model_settings.get("baseline_bi_encoder_path", "models/demo/baseline/sentence-transformer")
    )
    baseline_verifier_path = resolve_project_path(
        model_settings.get("baseline_cross_encoder_path", "models/demo/baseline/cross-encoder")
    )
    active_retriever_path = resolve_project_path(model_settings["bi_encoder_path"])
    active_verifier_path = resolve_project_path(model_settings["cross_encoder_path"])

    if not baseline_retriever_path.exists():
        raise FileNotFoundError(f"Baseline retriever model does not exist: {baseline_retriever_path}")
    if not baseline_verifier_path.exists():
        raise FileNotFoundError(f"Baseline verifier model does not exist: {baseline_verifier_path}")

    _promote_model(baseline_retriever_path, active_retriever_path)
    _promote_model(baseline_verifier_path, active_verifier_path)

    cleared_paths: list[str] = []
    for key in ("retriever_versions_dir", "verifier_versions_dir"):
        versions_dir = resolve_project_path(output_settings.get(key, ""))
        if not str(versions_dir).strip():
            continue
        versions_dir.mkdir(parents=True, exist_ok=True)
        for child in versions_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        cleared_paths.append(str(versions_dir))

    if clear_review_artifacts:
        approved_path = _approved_examples_path(config)
        approved_path.parent.mkdir(parents=True, exist_ok=True)
        approved_path.write_text("[]", encoding="utf-8")
        cleared_paths.append(str(approved_path))

        augmented_path = resolve_project_path(output_settings.get("augmented_dataset_path", ""))
        if augmented_path.exists():
            augmented_path.unlink()
        cleared_paths.append(str(augmented_path))

    return {
        "status": "reset",
        "active_retriever_path": str(active_retriever_path),
        "active_verifier_path": str(active_verifier_path),
        "baseline_retriever_path": str(baseline_retriever_path),
        "baseline_verifier_path": str(baseline_verifier_path),
        "cleared_paths": cleared_paths,
    }


def retrain_demo_verifier(
    approved_examples: Iterable[dict[str, Any]] | None = None,
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    promote_candidate: bool = True,
) -> dict[str, Any]:
    config = config or load_demo_config(config_path)
    approved_rows = list(approved_examples) if approved_examples is not None else _load_approved_examples(config)
    summary = _approved_summary(config, approved_rows)
    if not summary["ready_to_retrain"]:
        return {
            "status": "blocked",
            "message": "Not enough approved examples to retrain yet.",
            **summary,
        }

    data_settings = config.get("data", {})
    output_settings = config.get("outputs", {})
    model_settings = config.get("models", {})

    base_dataset_rows = _load_dataset_rows(data_settings["synthetic_dataset_path"])
    combined_rows = _dedupe_dataset_rows([*base_dataset_rows, *approved_rows])
    augmented_dataset_path = _json_save(output_settings["augmented_dataset_path"], combined_rows)

    active_retriever_path = resolve_project_path(model_settings["bi_encoder_path"])
    active_verifier_path = resolve_project_path(model_settings["cross_encoder_path"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    retriever_versions_dir = resolve_project_path(output_settings.get("retriever_versions_dir", "models/demo/retriever_versions"))
    verifier_versions_dir = resolve_project_path(output_settings["verifier_versions_dir"])
    retriever_versions_dir.mkdir(parents=True, exist_ok=True)
    verifier_versions_dir.mkdir(parents=True, exist_ok=True)

    candidate_retriever_path = retriever_versions_dir / f"retriever_{timestamp}"
    candidate_verifier_path = verifier_versions_dir / f"verifier_{timestamp}"
    retriever_monitor_output_dir = resolve_project_path(
        output_settings.get("retriever_retrain_monitor_output_dir", "data/results/demo/checkpoints/st_retrain")
    )
    verifier_monitor_output_dir = resolve_project_path(output_settings["retrain_monitor_output_dir"])

    eval_rows = _load_dataset_rows(data_settings["eval_dataset_path"])
    if not eval_rows:
        raise ValueError("Demo eval dataset is empty.")

    from ..training.cross_encoder import prepare_training_rows

    prepared_eval_rows = prepare_training_rows(
        {
            "data": {
                "disclosures_file": data_settings["disclosures_file"],
                "synthetic_dataset_path": str(resolve_project_path(data_settings["eval_dataset_path"])),
            },
            "training": {"use_extra_sampling": False},
        },
        dataset_path=str(resolve_project_path(data_settings["eval_dataset_path"])),
    )

    metrics_before = None
    if active_verifier_path.exists():
        metrics_before = _evaluate_cross_encoder(active_verifier_path, prepared_eval_rows)

    retriever_training_base = _retrain_base_model(
        model_settings,
        baseline_key="baseline_bi_encoder_path",
        raw_base_key="sentence_transformer_base",
    )
    verifier_training_base = _retrain_base_model(
        model_settings,
        baseline_key="baseline_cross_encoder_path",
        raw_base_key="cross_encoder_base",
    )

    train_sentence_transformer_with_overrides(
        config=config,
        dataset_path=str(augmented_dataset_path),
        output_dir=str(candidate_retriever_path),
        monitor_output_dir=str(retriever_monitor_output_dir),
        base_model_name_or_path=retriever_training_base,
    )
    train_cross_encoder_with_overrides(
        config=config,
        dataset_path=str(augmented_dataset_path),
        output_dir=str(candidate_verifier_path),
        monitor_output_dir=str(verifier_monitor_output_dir),
        base_model_name_or_path=verifier_training_base,
    )

    metrics_after = _evaluate_cross_encoder(candidate_verifier_path, prepared_eval_rows)
    promotion_recommended = metrics_before is None or metrics_after["average_precision"] >= metrics_before["average_precision"] - 1e-9
    promoted = bool(promote_candidate and promotion_recommended)
    if promoted:
        _promote_model(candidate_retriever_path, active_retriever_path)
        _promote_model(candidate_verifier_path, active_verifier_path)

    return {
        "status": "trained",
        "message": "Retriever and verifier retraining completed on the full augmented dataset.",
        "dataset_policy": "original_synthetic_plus_latest_approved_phrases",
        "approved_count": summary["approved_count"],
        "promotion_recommended": promotion_recommended,
        "promoted": promoted,
        "candidate_retriever_path": str(candidate_retriever_path),
        "candidate_verifier_path": str(candidate_verifier_path),
        "active_retriever_path": str(active_retriever_path),
        "active_verifier_path": str(active_verifier_path),
        "candidate_model_path": str(candidate_verifier_path),
        "active_model_path": str(active_verifier_path),
        "retriever_training_base": retriever_training_base,
        "verifier_training_base": verifier_training_base,
        "augmented_dataset_path": str(augmented_dataset_path),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }


def _claim_item_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("transcript_id", "")).strip(),
        str(item.get("disclaimer_id", "")).strip(),
        " ".join(str(item.get("anchor", "")).strip().lower().split()),
    )


def _model_label_from_score(score: float) -> str:
    return "Compliant" if float(score) >= 0.5 else "Non-Compliant"


def _dataset_label_to_model_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"compliant", "positive", "pass"}:
        return "Compliant"
    if text in {"non-compliant", "non compliant", "noncompliant", "negative", "fail"}:
        return "Non-Compliant"
    return ""


def _text_similarity(left: Any, right: Any) -> float:
    left_text = " ".join(str(left or "").strip().lower().split())
    right_text = " ".join(str(right or "").strip().lower().split())
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _anchor_matches(row: dict[str, Any], *, disclaimer_id: str, anchor: str) -> bool:
    row_disclaimer = str(row.get("disclaimer_id", "")).strip()
    row_anchor = " ".join(str(row.get("anchor", "")).strip().lower().split())
    target_anchor = " ".join(str(anchor or "").strip().lower().split())
    return row_disclaimer == str(disclaimer_id).strip() and row_anchor == target_anchor


def diagnose_label_changed_cases(
    items: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Inspect synthetic coverage when human review overturns a borderline verifier label."""
    config = config or load_demo_config(config_path)
    settings = config.get("agentic", {})
    similarity_threshold = float(settings.get("coverage_similarity_threshold", 0.74))
    weak_count_threshold = int(settings.get("coverage_weak_count_threshold", 2))

    base_rows = _load_dataset_rows(config.get("data", {}).get("synthetic_dataset_path", ""))
    prior_approved_rows = _load_approved_examples(config)
    dataset_rows = [*base_rows, *prior_approved_rows]

    analyses: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        final_label = str(item.get("final_label") or item.get("llm_label") or "").strip()
        if final_label not in {"Compliant", "Non-Compliant"}:
            continue
        score = float(item.get("verification_score") or 0.0)
        model_label = _model_label_from_score(score)
        if model_label == final_label:
            continue

        disclaimer_id = str(item.get("disclaimer_id", "")).strip()
        anchor = str(item.get("anchor", "")).strip()
        phrase = str(item.get("text", "")).strip()
        anchor_rows = [
            row
            for row in dataset_rows
            if isinstance(row, dict) and _anchor_matches(row, disclaimer_id=disclaimer_id, anchor=anchor)
        ]
        same_label_rows = [
            row for row in anchor_rows if _dataset_label_to_model_label(row.get("type")) == final_label
        ]
        opposite_label_rows = [
            row
            for row in anchor_rows
            if _dataset_label_to_model_label(row.get("type")) and _dataset_label_to_model_label(row.get("type")) != final_label
        ]

        same_examples = sorted(
            [
                {
                    "text": str(row.get("dialogue", "")).strip(),
                    "label": _dataset_label_to_model_label(row.get("type")),
                    "similarity": _text_similarity(phrase, row.get("dialogue", "")),
                    "source": str(row.get("source", "synthetic")).strip() or "synthetic",
                }
                for row in same_label_rows
            ],
            key=lambda row: row["similarity"],
            reverse=True,
        )
        opposite_examples = sorted(
            [
                {
                    "text": str(row.get("dialogue", "")).strip(),
                    "label": _dataset_label_to_model_label(row.get("type")),
                    "similarity": _text_similarity(phrase, row.get("dialogue", "")),
                    "source": str(row.get("source", "synthetic")).strip() or "synthetic",
                }
                for row in opposite_label_rows
            ],
            key=lambda row: row["similarity"],
            reverse=True,
        )

        max_same_similarity = same_examples[0]["similarity"] if same_examples else 0.0
        max_opposite_similarity = opposite_examples[0]["similarity"] if opposite_examples else 0.0
        possible_label_noise = [
            example for example in opposite_examples if example["similarity"] >= similarity_threshold
        ][:3]

        cause_tags: list[str] = []
        solution_steps: list[str] = []
        if not same_examples or max_same_similarity < similarity_threshold:
            cause_tags.append("missing_coverage")
            solution_steps.append(
                "Ask a DataGeneratorAgent to create 2-3 additional synthetic phrases for this anchor and human label."
            )
        elif len(same_examples) <= weak_count_threshold:
            cause_tags.append("thin_coverage")
            solution_steps.append(
                "Add this reviewed phrase to training because the dataset covers the pattern only weakly."
            )
        else:
            cause_tags.append("underweighted_pattern")
            solution_steps.append(
                "Add this reviewed phrase as an emphasis sample; similar data exists, but the model still underweighted the pattern."
            )

        if possible_label_noise:
            cause_tags.append("possible_label_noise")
            solution_steps.append(
                "Review the highly similar opposite-label samples; they may be mislabeled or too close to the positive boundary."
            )

        correction_type = "false_positive" if model_label == "Compliant" else "false_negative"
        if correction_type == "false_positive":
            human_readable_change = "model Pass -> human Fail"
        else:
            human_readable_change = "model Fail -> human Pass"

        if "missing_coverage" in cause_tags:
            recommendation = (
                "This situation does not appear to be covered strongly enough in the current synthetic data. "
                "Generate targeted synthetic examples, then add the reviewed phrase."
            )
        elif "thin_coverage" in cause_tags:
            recommendation = (
                "This situation appears only lightly represented. Add the reviewed phrase to strengthen this anchor pattern."
            )
        else:
            recommendation = (
                "Similar coverage exists, so this is likely an emphasis or boundary-learning issue. "
                "Add the reviewed phrase and inspect any opposite-label near matches."
            )

        analyses.append(
            {
                "transcript_id": str(item.get("transcript_id", "")).strip(),
                "disclaimer_id": disclaimer_id,
                "anchor": anchor,
                "phrase": phrase,
                "score": score,
                "model_label": model_label,
                "human_label": final_label,
                "label_change": human_readable_change,
                "correction_type": correction_type,
                "anchor_sample_count": len(anchor_rows),
                "same_label_count": len(same_examples),
                "opposite_label_count": len(opposite_examples),
                "max_same_label_similarity": round(max_same_similarity, 6),
                "max_opposite_label_similarity": round(max_opposite_similarity, 6),
                "cause_tags": cause_tags,
                "same_label_examples": same_examples[:3],
                "opposite_label_examples": opposite_examples[:3],
                "possible_label_noise": possible_label_noise,
                "recommendation": recommendation,
                "solution_steps": solution_steps,
            }
        )

    report = {
        "status": "completed",
        "changed_case_count": len(analyses),
        "analyses": analyses,
    }
    _json_save("data/results/demo/agentic/app_loop_dataset_diagnosis.json", report)
    return report


def diagnose_score_regressions(
    comparisons: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Inspect anchor data when retraining moves an approved phrase in the wrong direction."""
    config = config or load_demo_config(config_path)
    settings = config.get("agentic", {})
    similarity_threshold = float(settings.get("coverage_similarity_threshold", 0.74))
    weak_count_threshold = int(settings.get("coverage_weak_count_threshold", 2))

    base_rows = _load_dataset_rows(config.get("data", {}).get("synthetic_dataset_path", ""))
    prior_approved_rows = _load_approved_examples(config)
    dataset_rows = [*base_rows, *prior_approved_rows]

    analyses: list[dict[str, Any]] = []
    for item in comparisons:
        if not isinstance(item, dict) or item.get("outcome") != "regressed":
            continue
        final_label = str(item.get("final_label", "")).strip()
        if final_label not in {"Compliant", "Non-Compliant"}:
            continue

        disclaimer_id = str(item.get("disclaimer_id", "")).strip()
        anchor = str(item.get("anchor", "")).strip()
        phrase = str(item.get("text", "")).strip()
        anchor_rows = [
            row
            for row in dataset_rows
            if isinstance(row, dict) and _anchor_matches(row, disclaimer_id=disclaimer_id, anchor=anchor)
        ]
        same_label_rows = [
            row for row in anchor_rows if _dataset_label_to_model_label(row.get("type")) == final_label
        ]
        opposite_label_rows = [
            row
            for row in anchor_rows
            if _dataset_label_to_model_label(row.get("type")) and _dataset_label_to_model_label(row.get("type")) != final_label
        ]

        same_examples = sorted(
            [
                {
                    "text": str(row.get("dialogue", "")).strip(),
                    "label": _dataset_label_to_model_label(row.get("type")),
                    "similarity": _text_similarity(phrase, row.get("dialogue", "")),
                    "source": str(row.get("source", "synthetic")).strip() or "synthetic",
                }
                for row in same_label_rows
            ],
            key=lambda row: row["similarity"],
            reverse=True,
        )
        opposite_examples = sorted(
            [
                {
                    "text": str(row.get("dialogue", "")).strip(),
                    "label": _dataset_label_to_model_label(row.get("type")),
                    "similarity": _text_similarity(phrase, row.get("dialogue", "")),
                    "source": str(row.get("source", "synthetic")).strip() or "synthetic",
                }
                for row in opposite_label_rows
            ],
            key=lambda row: row["similarity"],
            reverse=True,
        )

        max_same_similarity = same_examples[0]["similarity"] if same_examples else 0.0
        max_opposite_similarity = opposite_examples[0]["similarity"] if opposite_examples else 0.0
        possible_label_noise = [
            example for example in opposite_examples if example["similarity"] >= similarity_threshold
        ][:3]

        cause_tags: list[str] = []
        solution_steps: list[str] = []
        if not same_examples or max_same_similarity < similarity_threshold:
            cause_tags.append("missing_coverage")
            solution_steps.append(
                "Ask a DataGeneratorAgent to generate 2-3 same-label synthetic phrases for this anchor pattern."
            )
        elif len(same_examples) <= weak_count_threshold:
            cause_tags.append("thin_coverage")
            solution_steps.append(
                "Add more same-label examples because the current anchor coverage is too sparse to stabilize retraining."
            )
        else:
            cause_tags.append("post_retrain_boundary_regression")
            solution_steps.append(
                "Keep this phrase in the training set and inspect nearby opposite-label examples before another retrain."
            )

        if possible_label_noise:
            cause_tags.append("possible_label_noise")
            solution_steps.append(
                "Review highly similar opposite-label samples; they may be mislabeled or too close to the decision boundary."
            )

        if "missing_coverage" in cause_tags:
            recommendation = (
                "The approved phrase regressed after retraining and this pattern is not covered strongly in the synthetic data. "
                "Generate targeted same-label examples before trusting another retrain."
            )
        elif "thin_coverage" in cause_tags:
            recommendation = (
                "The approved phrase regressed after retraining and same-label coverage is thin. "
                "Add this phrase plus a few variants to make the pattern less fragile."
            )
        elif possible_label_noise:
            recommendation = (
                "The approved phrase regressed after retraining despite existing coverage. "
                "Inspect similar opposite-label samples for label noise or overly hard negatives."
            )
        else:
            recommendation = (
                "The approved phrase regressed after retraining even though similar same-label data exists. "
                "Treat this as a boundary-learning issue and add targeted emphasis samples."
            )

        analyses.append(
            {
                "transcript_id": str(item.get("transcript_id", "")).strip(),
                "disclaimer_id": disclaimer_id,
                "anchor": anchor,
                "phrase": phrase,
                "score": float(item.get("before_score") or 0.0),
                "before_score": float(item.get("before_score") or 0.0),
                "after_score": float(item.get("after_score") or 0.0),
                "target_direction": str(item.get("target_direction", "")).strip(),
                "model_label": "Compliant" if float(item.get("before_score") or 0.0) >= 0.5 else "Non-Compliant",
                "human_label": final_label,
                "label_change": "post-retrain score regression",
                "diagnosis_type": "Approved phrase moved in the wrong direction after retraining",
                "correction_type": "score_regression",
                "anchor_sample_count": len(anchor_rows),
                "same_label_count": len(same_examples),
                "opposite_label_count": len(opposite_examples),
                "max_same_label_similarity": round(max_same_similarity, 6),
                "max_opposite_label_similarity": round(max_opposite_similarity, 6),
                "cause_tags": cause_tags,
                "same_label_examples": same_examples[:3],
                "opposite_label_examples": opposite_examples[:3],
                "possible_label_noise": possible_label_noise,
                "recommendation": recommendation,
                "solution_steps": solution_steps,
            }
        )

    report = {
        "status": "completed",
        "regressed_case_count": len(analyses),
        "analyses": analyses,
    }
    _json_save("data/results/demo/agentic/app_loop_score_regression_diagnosis.json", report)
    return report


def _merge_diagnosis_reports(
    primary: dict[str, Any] | None,
    secondary: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = deepcopy(primary) if isinstance(primary, dict) else {"status": "completed", "analyses": []}
    primary_analyses = merged.get("analyses", [])
    if not isinstance(primary_analyses, list):
        primary_analyses = []
    secondary_analyses = secondary.get("analyses", []) if isinstance(secondary, dict) else []
    if not isinstance(secondary_analyses, list):
        secondary_analyses = []
    merged["analyses"] = [*primary_analyses, *secondary_analyses]
    if isinstance(secondary, dict):
        merged["regressed_case_count"] = int(secondary.get("regressed_case_count", 0) or 0)
    return merged


def _collect_claim_items(payloads: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        transcript_id = str(payload.get("transcript_id", "")).strip()
        results = payload.get("results", {})
        if not isinstance(results, dict):
            continue
        for disclaimer_id, result in results.items():
            if not isinstance(result, dict):
                continue
            evidence = result.get("evidence", {})
            claims = evidence.get("claims", {}) if isinstance(evidence, dict) else {}
            if not isinstance(claims, dict):
                continue
            for claim_group in ("single", "mandatory", "standard"):
                claim_rows = claims.get(claim_group, [])
                if not isinstance(claim_rows, list):
                    continue
                for order, claim in enumerate(claim_rows):
                    if not isinstance(claim, dict):
                        continue
                    score = float(claim.get("verification_score") or 0.0)
                    items.append(
                        {
                            "transcript_id": transcript_id,
                            "disclaimer_id": str(disclaimer_id),
                            "claim_type": str(claim.get("claim_type") or claim_group).strip(),
                            "claim_order": int(claim.get("claim_idx", order) or order),
                            "anchor": str(claim.get("anchor", "")).strip(),
                            "text": str(claim.get("match_text", "")).strip(),
                            "verification_score": score,
                            "model_label": _model_label_from_score(score),
                            "passed": bool(claim.get("passed", False)),
                        }
                    )
    return items


def _summarize_rule_counts(payloads: Iterable[dict[str, Any]]) -> list[str]:
    rule_counts: dict[str, dict[str, int]] = {}
    for payload in payloads:
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        if not isinstance(results, dict):
            continue
        for rule_id, result in results.items():
            bucket = rule_counts.setdefault(str(rule_id), {"PASS": 0, "FAIL": 0})
            status = str(result.get("status", "FAIL")).strip().upper() if isinstance(result, dict) else "FAIL"
            bucket["PASS" if status == "PASS" else "FAIL"] += 1

    lines = []
    for rule_id in sorted(rule_counts):
        counts = rule_counts[rule_id]
        lines.append(f"Rule {rule_id}: {counts['PASS']} pass, {counts['FAIL']} fail")
    return lines


def _summarize_rule_evidence(payloads: Iterable[dict[str, Any]], *, max_chars: int = 180) -> list[str]:
    lines: list[str] = []
    for transcript_index, payload in enumerate(payloads, start=1):
        transcript_id = str(payload.get("transcript_id", "")).strip()
        results = payload.get("results", {}) if isinstance(payload, dict) else {}
        if not isinstance(results, dict):
            continue
        if lines:
            lines.append("")
        if transcript_id:
            lines.append(f"Transcript {transcript_index}: {transcript_id}")
        for rule_id, result in results.items():
            if not isinstance(result, dict):
                continue
            status = str(result.get("status", "UNKNOWN")).strip() or "UNKNOWN"
            evidence = result.get("evidence", {})
            claims = evidence.get("claims", {}) if isinstance(evidence, dict) else {}
            lines.append(f"Rule {rule_id}: {status}")
            if not isinstance(claims, dict):
                continue
            for claim_group in ("single", "mandatory", "standard"):
                claim_rows = claims.get(claim_group, [])
                if not isinstance(claim_rows, list):
                    continue
                for claim_order, claim in enumerate(claim_rows, start=1):
                    if not isinstance(claim, dict):
                        continue
                    anchor = _compact_text(claim.get("anchor", ""), max_chars)
                    best_text = _compact_text(claim.get("match_text", ""), max_chars)
                    score = float(claim.get("verification_score") or 0.0)
                    anchor_label = f"Anchor {claim_order}" if str(rule_id).strip() == "102" and len(claim_rows) > 1 else "Anchor"
                    lines.append(f"  {anchor_label}: {anchor or 'No anchor available.'}")
                    lines.append(f"  Best text: {best_text or 'No matched text available.'}")
                    lines.append(f"  Score: {score:.3f}")
    return lines


def _summarize_review_items(items: Iterable[dict[str, Any]], *, max_chars: int = 220) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("disclaimer_id", "")).strip()
        anchor = _compact_text(item.get("anchor", ""), max_chars)
        best_text = _compact_text(item.get("text", ""), max_chars)
        score = float(item.get("verification_score") or 0.0)
        label = _model_label_from_score(score)
        lines.append(f"Review item {index}: Rule {rule_id}")
        lines.append(f"  Anchor: {anchor or 'No anchor available.'}")
        lines.append(f"  Best text: {best_text or 'No matched text available.'}")
        lines.append(f"  Score: {score:.3f}")
        lines.append(f"  Model label: {label}")
        qwen_label = str(item.get("llm_label", "")).strip()
        if qwen_label:
            lines.append(f"  Qwen label: {qwen_label}")
    return lines


def run_agentic_review_cycle(
    transcripts: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    client: OllamaChatClient | None = None,
) -> dict[str, Any]:
    """Run inference and classifier review, stopping before human approval/retraining."""
    config = config or load_demo_config(config_path)
    transcript_rows = [dict(item) for item in transcripts if isinstance(item, dict) and str(item.get("transcript", "")).strip()]
    if not transcript_rows:
        return {
            "status": "blocked",
            "message": "No usable transcript text was provided.",
            "transcript_count": 0,
            "before_payloads": [],
            "review_items": [],
            "stage_status": [],
        }

    analyzer = _load_demo_analyzer(config)
    before_payloads: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    borderline_count = 0
    pass_count = 0
    for index, transcript in enumerate(transcript_rows, start=1):
        transcript_id = str(transcript.get("transcript_id") or f"transcript_{index}").strip()
        payload = run_demo_inference(
            str(transcript.get("transcript", "")).strip(),
            transcript_id=transcript_id,
            config=config,
            analyzer=analyzer,
        )
        before_payloads.append(payload)
        payload_review_items = get_agentic_review_items(payload, config=config)
        review_items.extend(payload_review_items)
        borderline_count += sum(1 for item in payload_review_items if item.get("review_type") == "borderline")
        pass_count += sum(1 for item in payload_review_items if item.get("review_type") == "pass")

    stage_status = [
        {
            "agent": "InferenceAgent",
            "status": "completed",
            "message": f"Processed {len(before_payloads)} transcript(s) through retriever + verifier inference.",
        }
    ]

    if not review_items:
        summary_lines = [
            *_summarize_rule_evidence(before_payloads),
            "SupervisorAgent found no borderline phrases in the uploaded transcript set.",
            "No retraining is recommended from this run because there are no uncertain samples for human approval.",
        ]
        result = {
            "status": "completed",
            "message": "No borderline phrases were found.",
            "transcript_count": len(before_payloads),
            "borderline_count": 0,
            "pass_review_count": 0,
            "before_payloads": before_payloads,
            "review_items": [],
            "stage_status": [
                *stage_status,
                {
                    "agent": "SupervisorAgent",
                    "status": "completed",
                    "message": "No borderline phrases found; loop stopped before classification and retraining.",
                },
            ],
            "supervisor_summary": "\n".join(summary_lines),
            "recommendation": "No training update is needed for this upload.",
        }
        _json_save("data/results/demo/agentic/app_loop_review.json", result)
        return result

    borderline_items = [item for item in review_items if item.get("review_type") == "borderline"]
    pass_items = [item for item in review_items if item.get("review_type") == "pass"]

    try:
        labeled_borderline_items = label_review_items_with_ollama(borderline_items, config=config, client=client) if borderline_items else []
        labeled_items = [*labeled_borderline_items, *pass_items]
        labeled_items.sort(
            key=lambda item: (
                str(item.get("transcript_id", "")),
                str(item.get("disclaimer_id", "")),
                int(item.get("claim_order", 0) or 0),
                0 if item.get("review_type") == "borderline" else 1,
            )
        )
        classifier_status = {
            "agent": "ClassifierAgent",
            "status": "completed",
            "message": f"Classified {len(labeled_borderline_items)} borderline phrase(s) with Qwen; pass phrases are included for human approval.",
        }
    except Exception as exc:
        labeled_items = review_items
        classifier_status = {
            "agent": "ClassifierAgent",
            "status": "needs_manual_review",
            "message": f"Qwen labeling failed: {exc}. Human can still type final labels manually.",
        }

    prepared_message = (
        f"SupervisorAgent prepared {len(labeled_items)} phrase(s) for human review: "
        f"{pass_count} pass phrase(s) and {borderline_count} borderline phrase(s)."
    )
    summary_lines = [*_summarize_rule_evidence(before_payloads)]
    result = {
        "status": "awaiting_human_approval",
        "message": "Pass and borderline phrases are ready for human approval.",
        "transcript_count": len(before_payloads),
        "borderline_count": borderline_count,
        "pass_review_count": pass_count,
        "before_payloads": before_payloads,
        "review_items": labeled_items,
        "stage_status": [
            *stage_status,
            {
                "agent": "SupervisorAgent",
                "status": "needs_human_approval",
                "message": f"Identified {pass_count} pass phrase(s) and {borderline_count} borderline phrase(s).",
            },
            classifier_status,
        ],
        "supervisor_summary": "\n".join(summary_lines),
        "recommendation": f"Human approval is required before retraining.\n\n**{prepared_message}**",
    }
    _json_save("data/results/demo/agentic/app_loop_review.json", result)
    return result


def compare_agentic_score_changes(
    before_review_items: Iterable[dict[str, Any]],
    after_payloads: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    after_claim_items = _collect_claim_items(after_payloads)
    after_lookup = {_claim_item_key(item): item for item in after_claim_items}
    fallback_lookup = {
        (
            str(item.get("disclaimer_id", "")).strip(),
            " ".join(str(item.get("anchor", "")).strip().lower().split()),
        ): item
        for item in after_claim_items
    }
    comparisons: list[dict[str, Any]] = []
    for item in before_review_items:
        if not isinstance(item, dict):
            continue
        final_label = str(item.get("final_label") or item.get("llm_label") or "").strip()
        if final_label not in {"Compliant", "Non-Compliant"}:
            continue

        before_score = float(item.get("verification_score") or 0.0)
        after_item = after_lookup.get(_claim_item_key(item))
        if after_item is None:
            after_item = fallback_lookup.get(
                (
                    str(item.get("disclaimer_id", "")).strip(),
                    " ".join(str(item.get("anchor", "")).strip().lower().split()),
                ),
                {},
            )
        after_score = float(after_item.get("verification_score") or 0.0)
        direction = "higher" if final_label == "Compliant" else "lower"
        delta = after_score - before_score
        improved = delta > 1e-6 if direction == "higher" else delta < -1e-6
        unchanged = abs(delta) <= 1e-6
        if improved:
            outcome = "improved"
        elif unchanged:
            outcome = "unchanged"
        else:
            outcome = "regressed"

        comparisons.append(
            {
                "transcript_id": str(item.get("transcript_id", "")).strip(),
                "disclaimer_id": str(item.get("disclaimer_id", "")).strip(),
                "anchor": str(item.get("anchor", "")).strip(),
                "text": str(item.get("text", "")).strip(),
                "final_label": final_label,
                "target_direction": direction,
                "before_score": before_score,
                "after_score": after_score,
                "delta": delta,
                "outcome": outcome,
                "after_text": str(after_item.get("text", "")).strip(),
            }
        )
    return comparisons


def prepare_agentic_training_cycle(
    review_state: dict[str, Any],
    approved_items: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Save human-approved rows and retrain, stopping before re-inference."""
    config = config or load_demo_config(config_path)
    state = review_state if isinstance(review_state, dict) else {}
    before_payloads = state.get("before_payloads", [])
    if not isinstance(before_payloads, list) or not before_payloads:
        return {
            "status": "blocked",
            "message": "Run the agentic review step before retraining.",
            "comparisons": [],
        }

    approved_rows = [dict(item) for item in approved_items if isinstance(item, dict)]
    selected_trainable = [
        item
        for item in approved_rows
        if bool(item.get("approved")) and str(item.get("final_label", "")).strip() in {"Compliant", "Non-Compliant"}
    ]
    if not selected_trainable:
        return {
            "status": "blocked",
            "message": "No approved Pass/Fail rows were selected for retraining.",
            "comparisons": [],
        }

    diagnosis = diagnose_label_changed_cases(selected_trainable, config=config)
    approval_summary = approve_demo_examples(selected_trainable, config=config, replace_existing=True)
    if approval_summary.get("added_count", 0) == 0 and approval_summary.get("approved_count", 0) == 0:
        return {
            "status": "blocked",
            "message": "No training rows were saved. Check that rows are ticked and final_label is Pass or Fail.",
            "approval": approval_summary,
            "diagnosis": diagnosis,
            "comparisons": [],
        }

    retrain_result = retrain_demo_verifier(config=config)
    if retrain_result.get("status") != "trained":
        result = {
            "status": str(retrain_result.get("status", "blocked")),
            "message": str(retrain_result.get("message", "Retraining did not complete.")),
            "approval": approval_summary,
            "diagnosis": diagnosis,
            "retrain": retrain_result,
            "comparisons": [],
            "before_payloads": before_payloads,
            "selected_trainable_review_items": selected_trainable,
            "stage_status": [
                {
                    "agent": "SupervisorAgent",
                    "status": "completed",
                    "message": f"Diagnosed {diagnosis['changed_case_count']} human-overturned borderline case(s).",
                },
                {
                    "agent": "TrainerAgent",
                    "status": str(retrain_result.get("status", "blocked")),
                    "message": str(retrain_result.get("message", "Retraining did not complete.")),
                }
            ],
        }
        _json_save("data/results/demo/agentic/app_loop_training.json", result)
        return result

    result = {
        "status": "retrained",
        "message": "Retraining completed. Ready to rerun inference with the candidate model.",
        "approval": approval_summary,
        "diagnosis": diagnosis,
        "retrain": retrain_result,
        "before_payloads": before_payloads,
        "selected_trainable_review_items": selected_trainable,
        "comparisons": [],
        "stage_status": [
            {
                "agent": "SupervisorAgent",
                "status": "completed",
                "message": f"Diagnosed {diagnosis['changed_case_count']} human-overturned borderline case(s) before retraining.",
            },
            {
                "agent": "TrainerAgent",
                "status": "completed",
                "message": "Trained retriever and verifier from the base models on the full augmented dataset.",
            },
        ],
    }
    _json_save("data/results/demo/agentic/app_loop_retrain.json", result)
    return result


def complete_agentic_reinference_cycle(
    retrain_state: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Rerun inference with the candidate model and compare before/after scores."""
    config = config or load_demo_config(config_path)
    state = retrain_state if isinstance(retrain_state, dict) else {}
    before_payloads = state.get("before_payloads", [])
    selected_trainable = state.get("selected_trainable_review_items", [])
    retrain_result = state.get("retrain", {})
    if not isinstance(before_payloads, list) or not before_payloads:
        return {
            **state,
            "status": "blocked",
            "message": "No original transcript payloads were available for re-inference.",
            "comparisons": [],
        }
    if not isinstance(selected_trainable, list) or not selected_trainable:
        return {
            **state,
            "status": "blocked",
            "message": "No selected training rows were available for before/after comparison.",
            "comparisons": [],
        }
    if not isinstance(retrain_result, dict) or retrain_result.get("status") != "trained":
        return {
            **state,
            "status": "blocked",
            "message": "Retraining did not complete, so re-inference cannot run.",
            "comparisons": [],
        }

    candidate_config = deepcopy(config)
    candidate_config.setdefault("models", {})
    if retrain_result.get("candidate_retriever_path"):
        candidate_config["models"]["bi_encoder_path"] = str(retrain_result["candidate_retriever_path"])
    if retrain_result.get("candidate_verifier_path"):
        candidate_config["models"]["cross_encoder_path"] = str(retrain_result["candidate_verifier_path"])

    analyzer = _load_demo_analyzer(candidate_config)
    after_payloads: list[dict[str, Any]] = []
    for payload in before_payloads:
        transcript_id = str(payload.get("transcript_id", "")).strip()
        transcript = str(payload.get("transcript", "")).strip()
        if not transcript:
            continue
        after_payloads.append(
            run_demo_inference(
                transcript,
                transcript_id=transcript_id,
                config=candidate_config,
                analyzer=analyzer,
            )
        )

    comparisons = compare_agentic_score_changes(selected_trainable, after_payloads)
    improved_count = sum(1 for item in comparisons if item["outcome"] == "improved")
    regressed_count = sum(1 for item in comparisons if item["outcome"] == "regressed")
    unchanged_count = sum(1 for item in comparisons if item["outcome"] == "unchanged")
    score_regression_diagnosis = diagnose_score_regressions(comparisons, config=config) if regressed_count else {
        "status": "completed",
        "regressed_case_count": 0,
        "analyses": [],
    }
    diagnosis = _merge_diagnosis_reports(state.get("diagnosis", {}), score_regression_diagnosis)
    success = bool(comparisons) and regressed_count == 0 and improved_count > 0
    if success:
        recommendation = "The retrained model moved the approved score(s) in the desired direction."
    elif regressed_count:
        recommendation = (
            f"SupervisorAgent found {regressed_count} approved phrase score regression(s) after retraining. "
            "Dataset diagnosis has been triggered; review anchor coverage and possible label noise before trusting this retrain."
        )
    else:
        recommendation = "Review the comparison before trusting this retrain; approved score(s) did not clearly improve."
    result = {
        "status": "completed" if success else "needs_review",
        "message": recommendation,
        "approval": state.get("approval", {}),
        "diagnosis": diagnosis,
        "score_regression_diagnosis": score_regression_diagnosis,
        "retrain": retrain_result,
        "before_payloads": before_payloads,
        "after_payloads": after_payloads,
        "selected_trainable_review_items": selected_trainable,
        "comparisons": comparisons,
        "improved_count": improved_count,
        "unchanged_count": unchanged_count,
        "regressed_count": regressed_count,
        "recommendation": recommendation,
        "stage_status": [
            *[item for item in state.get("stage_status", []) if isinstance(item, dict)],
            {
                "agent": "InferenceAgent",
                "status": "completed",
                "message": "Reran inference on the uploaded transcript(s) with the newly trained candidate model.",
            },
            {
                "agent": "SupervisorAgent",
                "status": "completed" if success else "needs_review",
                "message": recommendation,
            },
            *(
                [
                    {
                        "agent": "CoverageAugmentationAgent",
                        "status": "needs_review",
                        "message": f"Inspected synthetic coverage for {regressed_count} post-retrain score regression(s).",
                    }
                ]
                if regressed_count
                else []
            ),
        ],
    }
    _json_save("data/results/demo/agentic/app_loop_training.json", result)
    return result


def continue_agentic_training_cycle(
    review_state: dict[str, Any],
    approved_items: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Save rows, retrain, rerun inference, and compare scores."""
    config = config or load_demo_config(config_path)
    retrain_state = prepare_agentic_training_cycle(review_state, approved_items, config=config)
    if retrain_state.get("status") != "retrained":
        return retrain_state
    return complete_agentic_reinference_cycle(retrain_state, config=config)


def load_demo_samples(config: dict[str, Any] | None = None, config_path: str | None = None) -> list[dict[str, Any]]:
    config = config or load_demo_config(config_path)
    samples = _json_load(config.get("data", {}).get("sample_transcripts_path", ""), [])
    return samples if isinstance(samples, list) else []


def _short_inference_summary(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return "No inference result is available yet."

    payload = context.get("inference") if isinstance(context.get("inference"), dict) else context
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict) or not results:
        return "No inference result is available yet."

    parts = []
    for rule_id, result in results.items():
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "UNKNOWN")).strip() or "UNKNOWN"
        evidence = result.get("evidence", {})
        score = ""
        if isinstance(evidence, dict):
            try:
                score = f" score {float(evidence.get('verification_score') or 0.0):.3f}"
            except (TypeError, ValueError):
                score = ""
        parts.append(f"Rule {rule_id}: {status},{score}" if score else f"Rule {rule_id}: {status}")
    return "\n".join(parts) + "." if parts else "No inference result is available yet."


def _current_inference_summary(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return "No inference result is available yet."

    payload = context.get("inference") if isinstance(context.get("inference"), dict) else context
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict) or not results:
        return "No inference result is available yet."

    lines = ["Current inference summary:"]
    for rule_id, result in results.items():
        if not isinstance(result, dict):
            continue

        status = str(result.get("status", "UNKNOWN")).strip() or "UNKNOWN"
        evidence = result.get("evidence", {})
        score = 0.0
        if isinstance(evidence, dict):
            try:
                score = float(evidence.get("verification_score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0

        detail = ""
        claims = evidence.get("claims", {}) if isinstance(evidence, dict) else {}
        if isinstance(claims, dict):
            claim_rows = [
                claim
                for claim_group in ("single", "mandatory", "standard")
                for claim in claims.get(claim_group, [])
                if isinstance(claim, dict)
            ]
            if claim_rows:
                passed_count = sum(1 for claim in claim_rows if bool(claim.get("passed")))
                total_count = len(claim_rows)
                if total_count == 1:
                    detail = "required evidence was found" if passed_count else "required evidence was not found"
                else:
                    detail = f"{passed_count}/{total_count} required disclosures passed"

        suffix = f" - {detail}" if detail else ""
        lines.append(f"Rule {rule_id}: {status}, score {score:.3f}{suffix}.")

    return "\n".join(lines)


def answer_demo_question(
    question: str,
    *,
    inference_payload: dict[str, Any] | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    client: OllamaChatClient | None = None,
) -> str:
    question_text = str(question or "").strip()
    normalized_question = " ".join(question_text.lower().rstrip(".?!").split())
    if question_text.lower() in {"hi", "hello", "hey", "yo"}:
        return "Hi. I can help explain the current inference result, borderline phrases, or Qwen labels."
    review_items = []
    if isinstance(inference_payload, dict) and isinstance(inference_payload.get("review_items"), list):
        review_items = inference_payload["review_items"]
    if "borderline" in question_text.lower() and not review_items:
        return f"There are no borderline phrases in the current review queue.\n{_short_inference_summary(inference_payload)}"
    direct_answer = _direct_demo_answer(question_text, inference_payload)
    if direct_answer:
        return direct_answer

    config = config or load_demo_config(config_path)
    client = client or _ollama_client_from_config(config)
    context = json.dumps(_compact_demo_chat_context(inference_payload), ensure_ascii=False, indent=2)
    history_context = json.dumps(_compact_chat_history(chat_history), ensure_ascii=False, indent=2)
    system_prompt = (
        "You are the assistant inside a privacy-safe compliance demo app. "
        "Use non-thinking mode. Answer directly and do not narrate your analysis. "
        "Answer in 1-4 concise sentences or short bullets unless the user asks for detail. "
        "If the user greets you, greet them briefly and offer help. "
        "Do not include hidden reasoning, chain-of-thought, scratchpad notes, or phrases like 'let me process this'. "
        "If a thinking trace is unavoidable, wrap it in <think>...</think> so the app can hide it. "
        "Explain results clearly, avoid inventing evidence, and stay grounded in the compact inference payload."
    )
    answer = client.chat(
        system_prompt=system_prompt,
        user_prompt=(
            "/no_think\n"
            "Recent chat history, oldest first:\n"
            f"{history_context}\n\n"
            "Current inference context:\n"
            f"{context}\n\n"
            f"User question:\n{question_text}\n\n"
            "/no_think\n"
            "Answer directly. Do not show reasoning."
        ),
        temperature=0.2,
        num_predict=CHAT_NUM_PREDICT,
    )
    cleaned_answer = strip_think_blocks(answer)
    if _looks_like_reasoning_leak(cleaned_answer):
        fallback_answer = _direct_demo_answer(question_text, inference_payload)
        if fallback_answer:
            return fallback_answer
        return _short_inference_summary(inference_payload)
    return cleaned_answer or "I can help explain the current inference result or review labels."
