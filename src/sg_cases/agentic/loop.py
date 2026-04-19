"""Local-LLM-assisted loop for synthetic gating, incoming review, augmentation, and fixed holdout evaluation."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from ..core.paths import ensure_parent_dir, resolve_project_path
from ..core.transcripts import load_transcripts_from_folder
from ..demo import services as demo_services
from ..training import train_cross_encoder, train_sentence_transformer
from ..training.data_utils import DEFAULT_CROSS_ENCODER_BASE, resolve_training_anchor_text

LABELS = {"Compliant", "Non-Compliant"}
AGENTIC_CHAT_NUM_PREDICT = 1280


def _agentic_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("agentic", {})


def _json_save(path: str | Path, payload: Any) -> Path:
    resolved = ensure_parent_dir(path)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


def _json_load(path: str | Path, default: Any) -> Any:
    resolved = resolve_project_path(path)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_dataset_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"compliant", "positive", "pass"}:
        return "compliant"
    if text in {"non-compliant", "noncompliant", "negative", "fail"}:
        return "non-compliant"
    return ""


def _binary_to_label(value: int | float) -> str:
    return "Compliant" if int(value) == 1 else "Non-Compliant"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _training_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("disclaimer_id", "")).strip().lower(),
        _normalize_text(item.get("anchor", "")).lower(),
        _normalize_text(item.get("dialogue", "")).lower(),
        _normalize_dataset_label(item.get("type", "")),
    )


def _load_disclosures_lookup(config: dict[str, Any]) -> dict[str, Any]:
    disclosures_path = resolve_project_path(config.get("data", {}).get("disclosures_file", ""))
    value = json.loads(disclosures_path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _resolve_entry_anchor(entry: dict[str, Any], config: dict[str, Any], disclosures: dict[str, Any] | None = None) -> str:
    lookup = disclosures if disclosures is not None else _load_disclosures_lookup(config)
    anchor = resolve_training_anchor_text(entry, lookup)
    if isinstance(anchor, str) and anchor.strip():
        return anchor.strip()
    explicit_anchor = str(entry.get("anchor", "")).strip()
    return explicit_anchor


def _prepare_synthetic_rows(raw_rows: Iterable[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    disclosures = _load_disclosures_lookup(config)
    prepared: list[dict[str, Any]] = []
    for entry in raw_rows:
        if not isinstance(entry, dict):
            continue
        label_text = _normalize_dataset_label(entry.get("type", ""))
        if label_text not in {"compliant", "non-compliant"}:
            continue
        anchor = _resolve_entry_anchor(entry, config, disclosures)
        dialogue = str(entry.get("dialogue", "")).strip()
        disclaimer_id = str(entry.get("disclaimer_id", "")).strip()
        if not disclaimer_id or not isinstance(anchor, str) or not anchor.strip() or not dialogue:
            continue
        prepared.append(
            {
                "sentence1": anchor.strip(),
                "sentence2": dialogue,
                "label": 1 if label_text == "compliant" else 0,
                "anchor": anchor.strip(),
                "disclaimer_id": disclaimer_id,
                "type": label_text,
            }
        )
    return prepared


def _split_synthetic_rows(
    raw_rows: Iterable[dict[str, Any]],
    *,
    eval_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(row) for row in raw_rows if _normalize_dataset_label(row.get("type", ""))]
    if len(rows) < 2:
        return rows, []

    from sklearn.model_selection import train_test_split

    labels = [_normalize_dataset_label(row.get("type", "")) for row in rows]
    label_counts: dict[str, int] = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    stratify = labels if len(set(labels)) > 1 and min(label_counts.values()) >= 2 else None
    train_rows, eval_rows = train_test_split(
        rows,
        test_size=eval_size,
        random_state=seed,
        stratify=stratify,
    )
    return list(train_rows), list(eval_rows)


def _fit_cross_encoder_rows(
    train_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    base_model_name_or_path: str | None = None,
) -> Path:
    if not train_rows:
        raise ValueError("No rows were provided for gate training.")

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    training_settings = config.get("training", {})
    model_settings = config.get("models", {})
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = base_model_name_or_path or model_settings.get("cross_encoder_base", DEFAULT_CROSS_ENCODER_BASE)
    max_length = int(training_settings.get("max_length", 256))
    learning_rate = float(training_settings.get("learning_rate", 2e-5))
    num_epochs = int(training_settings.get("num_epochs", 1))
    train_batch_size = int(training_settings.get("train_batch_size", 16))
    warmup_ratio = float(training_settings.get("warmup_ratio", 0.1))
    force_cpu = bool(training_settings.get("force_cpu", True))
    device = "cpu" if force_cpu else None

    model_kwargs = {"device": device} if device is not None else {}
    model = CrossEncoder(model_name, num_labels=1, max_length=max_length, **model_kwargs)
    train_dataloader = DataLoader(
        [InputExample(texts=[row["sentence1"], row["sentence2"]], label=float(row["label"])) for row in train_rows],
        shuffle=True,
        batch_size=train_batch_size,
    )
    warmup_steps = max(1, int(len(train_dataloader) * num_epochs * warmup_ratio))
    model.old_fit(
        train_dataloader=train_dataloader,
        evaluator=None,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        weight_decay=0.01,
        output_path=None,
        save_best_model=False,
        show_progress_bar=False,
    )
    model.save(str(output_path))
    return output_path


def _evaluate_cross_encoder_rows(model_path: str | Path, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not eval_rows:
        return {
            "count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "per_anchor": {},
        }

    from sentence_transformers import CrossEncoder
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model = CrossEncoder(str(resolve_project_path(model_path)))
    logits = model.predict([(row["sentence1"], row["sentence2"]) for row in eval_rows], show_progress_bar=False)
    probabilities = [_sigmoid(score) for score in logits]
    predictions = [1 if score >= 0.5 else 0 for score in probabilities]
    gold = [int(row["label"]) for row in eval_rows]

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predictions,
        average="binary",
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        gold,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(gold, predictions)

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    grouped_predictions: dict[tuple[str, str], list[int]] = defaultdict(list)
    grouped_examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(eval_rows):
        key = (str(row.get("disclaimer_id", "")).strip(), str(row.get("anchor", "")).strip())
        grouped[key].append(gold[index])
        grouped_predictions[key].append(predictions[index])
        grouped_examples[key].append(
            {
                "phrase": row["sentence2"],
                "label": _binary_to_label(gold[index]),
                "predicted_label": _binary_to_label(predictions[index]),
                "score": round(float(probabilities[index]), 6),
            }
        )

    per_anchor: dict[str, Any] = {}
    for (disclaimer_id, anchor), anchor_gold in grouped.items():
        anchor_predictions = grouped_predictions[(disclaimer_id, anchor)]
        anchor_precision, anchor_recall, anchor_f1, _ = precision_recall_fscore_support(
            anchor_gold,
            anchor_predictions,
            average="binary",
            zero_division=0,
        )
        per_anchor[f"{disclaimer_id}|{anchor}"] = {
            "disclaimer_id": disclaimer_id,
            "anchor": anchor,
            "count": len(anchor_gold),
            "precision": float(anchor_precision),
            "recall": float(anchor_recall),
            "f1": float(anchor_f1),
            "examples": grouped_examples[(disclaimer_id, anchor)][:4],
        }

    return {
        "count": len(eval_rows),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "per_anchor": per_anchor,
    }


def _agentic_paths(config: dict[str, Any]) -> dict[str, Path]:
    settings = _agentic_settings(config)
    output_dir = resolve_project_path(settings.get("outputs_dir", "data/results/demo/agentic"))
    return {
        "output_dir": output_dir,
        "gate_train_split": output_dir / "synthetic_gate_train.json",
        "gate_eval_split": output_dir / "synthetic_gate_eval.json",
        "gate_model_dir": output_dir / "synthetic_gate_model",
        "gate_metrics": output_dir / "synthetic_gate_metrics.json",
        "synthetic_quality_report": output_dir / "synthetic_quality_report.json",
        "incoming_inference": output_dir / "incoming_inference.json",
        "review_items": output_dir / "review_items.json",
        "coverage_analysis": output_dir / "coverage_analysis.json",
        "synthetic_extensions": output_dir / "synthetic_extensions.json",
        "holdout_inference_before": output_dir / "holdout_inference_before.json",
        "holdout_predictions_before": output_dir / "holdout_predictions_before.json",
        "holdout_metrics_before": output_dir / "holdout_metrics_before.json",
        "holdout_inference_after": output_dir / "holdout_inference_after.json",
        "holdout_predictions_after": output_dir / "holdout_predictions_after.json",
        "holdout_metrics_after": output_dir / "holdout_metrics_after.json",
        "summary": output_dir / "loop_summary.json",
    }


def _load_transcripts_from_json(path: Path, *, dataset_role: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of transcripts: {path}")

    transcripts: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        transcript = str(item.get("transcript", "")).strip()
        if not transcript:
            continue
        transcript_id = str(item.get("transcript_id") or f"transcript_{index}").strip()
        transcripts.append(
            {
                "transcript_id": transcript_id,
                "title": str(item.get("title", "")).strip(),
                "transcript": transcript,
                "dataset_role": dataset_role,
            }
        )
    if not transcripts:
        raise ValueError(f"No usable transcripts were found in {path}")
    return transcripts


def load_incoming_transcripts(
    *,
    config: dict[str, Any],
    incoming_source: str | None = None,
) -> list[dict[str, Any]]:
    settings = _agentic_settings(config)
    source = incoming_source or settings.get("incoming_source") or config.get("data", {}).get("sample_transcripts_path")
    if not source:
        raise ValueError("No incoming transcript source is configured.")

    resolved = resolve_project_path(source)
    if not resolved.exists():
        raise FileNotFoundError(f"Incoming transcript source not found: {resolved}")

    if resolved.is_dir():
        return [
            {
                **item,
                "title": str(item.get("transcript_id", "")).strip(),
                "dataset_role": "incoming",
            }
            for item in load_transcripts_from_folder(resolved)
        ]

    if resolved.suffix.lower() == ".json":
        return _load_transcripts_from_json(resolved, dataset_role="incoming")

    if resolved.suffix.lower() == ".txt":
        transcript = resolved.read_text(encoding="utf-8", errors="ignore").strip()
        if not transcript:
            raise ValueError(f"Incoming transcript file is empty: {resolved}")
        return [
            {
                "transcript_id": resolved.stem,
                "title": resolved.stem,
                "transcript": transcript,
                "dataset_role": "incoming",
            }
        ]

    raise ValueError(f"Unsupported incoming transcript source: {resolved}")


def _parse_expected_labels(raw_labels: Any) -> list[dict[str, str]]:
    if not isinstance(raw_labels, list):
        return []

    parsed: list[dict[str, str]] = []
    for item in raw_labels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        disclaimer_id = str(item.get("disclaimer_id", "")).strip()
        anchor = str(item.get("anchor", "")).strip()
        if label not in LABELS or not disclaimer_id or not anchor:
            continue
        parsed.append(
            {
                "disclaimer_id": disclaimer_id,
                "claim_type": str(item.get("claim_type", "")).strip(),
                "anchor": anchor,
                "label": label,
            }
        )
    return parsed


def load_holdout_dataset(
    *,
    config: dict[str, Any],
    holdout_source: str | None = None,
) -> list[dict[str, Any]]:
    settings = _agentic_settings(config)
    source = holdout_source or settings.get("holdout_source")
    if not source:
        raise ValueError("No holdout transcript source is configured.")

    resolved = resolve_project_path(source)
    if not resolved.exists():
        raise FileNotFoundError(f"Holdout transcript source not found: {resolved}")
    if resolved.suffix.lower() != ".json":
        raise ValueError(f"Holdout transcript source must be a JSON file: {resolved}")

    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of holdout transcripts: {resolved}")

    holdout: list[dict[str, Any]] = []
    expected_count = 0
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        transcript = str(item.get("transcript", "")).strip()
        if not transcript:
            continue
        transcript_id = str(item.get("transcript_id") or f"holdout_{index}").strip()
        expected_labels = _parse_expected_labels(item.get("expected_labels", []))
        expected_count += len(expected_labels)
        holdout.append(
            {
                "transcript_id": transcript_id,
                "title": str(item.get("title", "")).strip(),
                "transcript": transcript,
                "dataset_role": "holdout",
                "expected_labels": expected_labels,
            }
        )

    if not holdout:
        raise ValueError(f"No usable holdout transcripts were found in {resolved}")
    if expected_count == 0:
        raise ValueError(f"Holdout transcript set must include expected_labels: {resolved}")
    return holdout


def collect_anchor_review_units(
    inference_records: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any],
    review_categories: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    semantic_settings = config.get("semantic_inference", {})
    low = float(semantic_settings.get("borderline_low", 0.30))
    high = float(semantic_settings.get("borderline_high", 0.70))
    categories = review_categories or _agentic_settings(config).get("review_categories", ["pass", "borderline", "fail"])
    allowed_categories = {
        str(value).strip().lower()
        for value in categories
        if str(value).strip()
    }
    model_version = str(config.get("models", {}).get("cross_encoder_path", "")).strip()

    units: list[dict[str, Any]] = []
    for record in inference_records:
        transcript_id = str(record.get("transcript_id", "")).strip()
        dataset_role = str(record.get("dataset_role", "")).strip()
        results = record.get("results", {})
        if not isinstance(results, dict):
            continue

        for disclaimer_id, disclaimer_result in results.items():
            if not isinstance(disclaimer_result, dict):
                continue
            evidence = disclaimer_result.get("evidence", {})
            if not isinstance(evidence, dict):
                continue
            rule_description = str(evidence.get("description", "")).strip()
            claims = evidence.get("claims", {})
            if not isinstance(claims, dict):
                continue

            for claim_group in ("single", "mandatory", "standard"):
                claim_items = claims.get(claim_group, [])
                if not isinstance(claim_items, list):
                    continue

                for order, claim in enumerate(claim_items):
                    if not isinstance(claim, dict):
                        continue
                    score = float(claim.get("verification_score") or 0.0)
                    model_label = "Compliant" if bool(claim.get("passed")) else "Non-Compliant"
                    if low <= score < high:
                        review_bucket = "borderline"
                    elif model_label == "Compliant":
                        review_bucket = "pass"
                    else:
                        review_bucket = "fail"

                    if review_bucket not in allowed_categories:
                        continue

                    units.append(
                        {
                            "review_id": f"{dataset_role}:{transcript_id}:{disclaimer_id}:{claim_group}:{order}",
                            "dataset_role": dataset_role,
                            "transcript_id": transcript_id,
                            "disclaimer_id": str(disclaimer_id).strip(),
                            "description": rule_description,
                            "claim_type": str(claim.get("claim_type", claim_group)).strip() or claim_group,
                            "claim_order": int(claim.get("claim_idx", order)),
                            "anchor": str(claim.get("anchor", "")).strip(),
                            "text": str(claim.get("match_text", "")).strip(),
                            "retrieval_score": float(claim.get("retrieval_score") or 0.0),
                            "verification_score": score,
                            "model_label": model_label,
                            "model_rule_status": str(disclaimer_result.get("status", "")).strip() or ("PASS" if model_label == "Compliant" else "FAIL"),
                            "review_bucket": review_bucket,
                            "model_version": model_version,
                        }
                    )

    units.sort(key=lambda item: (item["dataset_role"], item["transcript_id"], item["disclaimer_id"], item["claim_type"], item["claim_order"]))
    return units


def annotate_expected_labels(
    prediction_units: Iterable[dict[str, Any]],
    *,
    holdout_dataset: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    exact_lookup: dict[tuple[str, str, str, str], str] = {}
    fallback_lookup: dict[tuple[str, str, str], str] = {}

    for transcript in holdout_dataset:
        transcript_id = str(transcript.get("transcript_id", "")).strip()
        for label_item in transcript.get("expected_labels", []):
            if not isinstance(label_item, dict):
                continue
            disclaimer_id = str(label_item.get("disclaimer_id", "")).strip()
            claim_type = str(label_item.get("claim_type", "")).strip()
            anchor = _normalize_text(label_item.get("anchor", "")).lower()
            label = str(label_item.get("label", "")).strip()
            if not transcript_id or not disclaimer_id or not anchor or label not in LABELS:
                continue
            exact_lookup[(transcript_id, disclaimer_id, claim_type, anchor)] = label
            fallback_lookup[(transcript_id, disclaimer_id, anchor)] = label

    annotated: list[dict[str, Any]] = []
    for item in prediction_units:
        transcript_id = str(item.get("transcript_id", "")).strip()
        disclaimer_id = str(item.get("disclaimer_id", "")).strip()
        claim_type = str(item.get("claim_type", "")).strip()
        anchor = _normalize_text(item.get("anchor", "")).lower()
        expected = exact_lookup.get((transcript_id, disclaimer_id, claim_type, anchor))
        if expected is None:
            expected = fallback_lookup.get((transcript_id, disclaimer_id, anchor), "")
        annotated.append({**item, "expected_label": expected})
    return annotated


def build_label_metrics(
    items: Iterable[dict[str, Any]],
    *,
    expected_field: str = "expected_label",
    predicted_field: str = "model_label",
) -> dict[str, Any]:
    comparable = [
        item
        for item in items
        if str(item.get(expected_field, "")).strip() in LABELS and str(item.get(predicted_field, "")).strip() in LABELS
    ]
    if not comparable:
        return {
            "count": 0,
            "correct": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "per_rule": {},
        }

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    gold = [1 if str(item[expected_field]).strip() == "Compliant" else 0 for item in comparable]
    predictions = [1 if str(item[predicted_field]).strip() == "Compliant" else 0 for item in comparable]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predictions,
        average="binary",
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        gold,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(gold, predictions)

    per_rule: dict[str, dict[str, int]] = {}
    for item in comparable:
        bucket = per_rule.setdefault(str(item.get("disclaimer_id", "")).strip(), {"count": 0, "correct": 0})
        bucket["count"] += 1
        if str(item[expected_field]).strip() == str(item[predicted_field]).strip():
            bucket["correct"] += 1

    return {
        "count": len(comparable),
        "correct": sum(1 for item in comparable if str(item[expected_field]).strip() == str(item[predicted_field]).strip()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "per_rule": {
            rule_id: {
                "count": values["count"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["count"] if values["count"] else 0.0,
            }
            for rule_id, values in per_rule.items()
        },
    }


def _expected_label(item: dict[str, Any]) -> str:
    final_label = str(item.get("final_label", "")).strip()
    if final_label in LABELS:
        return final_label
    llm_label = str(item.get("llm_label", "")).strip()
    if llm_label in LABELS:
        return llm_label
    return ""


def _load_extension_rows(path: str | Path) -> list[dict[str, Any]]:
    value = _json_load(path, [])
    return value if isinstance(value, list) else []


def _merge_extension_rows(existing_rows: Iterable[dict[str, Any]], new_rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for row in [*list(existing_rows), *list(new_rows)]:
        key = _training_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(row))

    existing_keys = {_training_key(row) for row in existing_rows}
    added_count = sum(1 for row in new_rows if _training_key(row) not in existing_keys)
    return merged, added_count


def _promote_candidate_model(candidate_model_path: str | Path, active_model_path: str | Path) -> None:
    if not str(candidate_model_path).strip():
        raise ValueError("Candidate model path is required for promotion.")
    if not str(active_model_path).strip():
        raise ValueError("Active model path is required for promotion.")
    candidate_path = resolve_project_path(candidate_model_path)
    active_path = resolve_project_path(active_model_path)
    if active_path.exists():
        shutil.rmtree(active_path, ignore_errors=True)
    shutil.copytree(candidate_path, active_path)


@dataclass
class InferenceAgent:
    config: dict[str, Any]

    def run_inference_records(
        self,
        transcripts: Iterable[dict[str, Any]],
        *,
        cross_encoder_path: str | None = None,
    ) -> list[dict[str, Any]]:
        config = _deep_merge(self.config, {"models": {"cross_encoder_path": cross_encoder_path}}) if cross_encoder_path else self.config
        analyzer = demo_services._load_demo_analyzer(config)

        records: list[dict[str, Any]] = []
        for transcript in transcripts:
            transcript_id = str(transcript.get("transcript_id", "")).strip()
            text = str(transcript.get("transcript", "")).strip()
            payload = demo_services.run_demo_inference(
                text,
                transcript_id=transcript_id,
                config=config,
                analyzer=analyzer,
            )
            records.append(
                {
                    "dataset_role": str(transcript.get("dataset_role", "")).strip(),
                    "transcript_id": transcript_id,
                    "title": str(transcript.get("title", "")).strip(),
                    "transcript": text,
                    "results": payload.get("results", {}),
                }
            )
        return records


@dataclass
class ReviewAgent:
    config: dict[str, Any]
    client: Any | None = None

    def review(self, items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        auto_approve = bool(_agentic_settings(self.config).get("auto_approve_llm", True))
        ready_for_llm: list[dict[str, Any]] = []
        reviewed: list[dict[str, Any]] = []

        for item in items:
            if str(item.get("text", "")).strip():
                ready_for_llm.append(item)
            else:
                reviewed.append(
                    {
                        **dict(item),
                        "llm_label": "Non-Compliant",
                        "llm_confidence": 0.75,
                        "llm_rationale": "No candidate phrase was retrieved for this anchor, so it cannot satisfy the target claim.",
                    }
                )

        if ready_for_llm:
            reviewed.extend(
                demo_services.label_review_items_with_ollama(
                    ready_for_llm,
                    config=self.config,
                    client=self.client,
                )
            )

        finalized: list[dict[str, Any]] = []
        for item in reviewed:
            final_label = str(item.get("llm_label", "")).strip() if auto_approve else ""
            finalized.append(
                {
                    **item,
                    "prediction_matches_review": str(item.get("model_label", "")).strip() == str(item.get("llm_label", "")).strip(),
                    "auto_approved": auto_approve,
                    "final_label": final_label,
                }
            )
        finalized.sort(key=lambda item: (item["dataset_role"], item["transcript_id"], item["disclaimer_id"], item["claim_type"], item["claim_order"]))
        return finalized


@dataclass
class CoverageAugmentationAgent:
    config: dict[str, Any]
    client: Any | None = None

    def _client(self):
        return self.client or demo_services._ollama_client_from_config(self.config)

    def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, _normalize_text(left).lower(), _normalize_text(right).lower()).ratio()

    def _candidate_cases(self, reviewed_items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        semantic_settings = self.config.get("semantic_inference", {})
        low = float(semantic_settings.get("borderline_low", 0.30))
        threshold = float(semantic_settings.get("verification_threshold", 0.5))

        candidates: list[dict[str, Any]] = []
        for item in reviewed_items:
            label = _expected_label(item)
            score = float(item.get("verification_score") or 0.0)
            model_label = str(item.get("model_label", "")).strip()
            if label not in LABELS or not str(item.get("text", "")).strip():
                continue

            correction_type = ""
            if score >= threshold and model_label == "Compliant" and label == "Non-Compliant":
                correction_type = "false_positive"
            elif low <= score < threshold and model_label == "Non-Compliant" and label == "Compliant":
                correction_type = "rescued_positive"

            if correction_type:
                candidates.append({**item, "correction_type": correction_type})
        return candidates

    def _audit_quality_with_llm(
        self,
        *,
        anchor: str,
        compliant_examples: list[str],
        non_compliant_examples: list[str],
        reviewed_phrase: str | None = None,
        corrected_label: str | None = None,
        heuristic_status: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "task": "Assess synthetic data quality for this anchor. Positive samples must truly satisfy the anchor. Negative samples should be hard negatives, but not mislabeled positives.",
            "anchor": anchor,
            "reviewed_phrase": reviewed_phrase or "",
            "corrected_label": corrected_label or "",
            "heuristic_status": heuristic_status or "",
            "positive_examples": compliant_examples,
            "negative_examples": non_compliant_examples,
            "instructions": {
                "coverage_status": "Return covered if the reviewed phrase pattern is already represented for the corrected label. Return gap if it is not represented well enough.",
                "positive_quality": "Comment on whether positives truly express the anchor meaning and whether coverage is diverse enough.",
                "negative_quality": "Comment on whether negatives are meaningful hard negatives without crossing into mislabeled positives.",
                "generated_variants": "If coverage_status is gap, provide up to 2 additional phrases for the corrected label and anchor.",
            },
        }
        system_prompt = (
            "You are a synthetic data quality analyst for an anchor-based compliance dataset.\n"
            "Be strict about label correctness.\n"
            "Positive samples must semantically satisfy the anchor.\n"
            "Negative samples should stay close in topic but remain clearly non-compliant.\n"
            "Return only valid JSON with keys: coverage_status, positive_quality_note, negative_quality_note, reason, generated_variants.\n"
            'coverage_status must be either "covered" or "gap".'
        )
        raw = self._client().chat(
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
            temperature=0.1,
        )
        parsed = demo_services._parse_llm_json(raw)
        generated = parsed.get("generated_variants", [])
        if not isinstance(generated, list):
            generated = []
        return {
            "coverage_status": str(parsed.get("coverage_status", "")).strip().lower(),
            "positive_quality_note": str(parsed.get("positive_quality_note", "")).strip(),
            "negative_quality_note": str(parsed.get("negative_quality_note", "")).strip(),
            "reason": str(parsed.get("reason", "")).strip(),
            "generated_variants": [str(item).strip() for item in generated if str(item).strip()],
        }

    def audit_synthetic_quality(
        self,
        *,
        raw_synthetic_rows: list[dict[str, Any]],
        gate_metrics: dict[str, Any],
        output_path: str | Path,
    ) -> dict[str, Any]:
        threshold = float(_agentic_settings(self.config).get("training_gate_threshold", 0.9))
        example_limit = int(_agentic_settings(self.config).get("coverage_example_limit", 4))
        disclosures = _load_disclosures_lookup(self.config)
        grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
        for entry in raw_synthetic_rows:
            if not isinstance(entry, dict):
                continue
            anchor = _resolve_entry_anchor(entry, self.config, disclosures)
            disclaimer_id = str(entry.get("disclaimer_id", "")).strip()
            label = _normalize_dataset_label(entry.get("type", ""))
            dialogue = str(entry.get("dialogue", "")).strip()
            if not anchor or not disclaimer_id or label not in {"compliant", "non-compliant"} or not dialogue:
                continue
            grouped.setdefault((disclaimer_id, anchor), {"compliant": [], "non-compliant": []})
            grouped[(disclaimer_id, anchor)][label].append(dialogue)

        low_anchors = [
            value
            for value in gate_metrics.get("per_anchor", {}).values()
            if float(value.get("f1", 0.0) or 0.0) < threshold
        ]
        audits: list[dict[str, Any]] = []
        for anchor_metrics in low_anchors:
            key = (str(anchor_metrics.get("disclaimer_id", "")).strip(), str(anchor_metrics.get("anchor", "")).strip())
            pool = grouped.get(key, {"compliant": [], "non-compliant": []})
            heuristic = {
                "coverage_status": "gap" if len(pool["compliant"]) < 2 or len(pool["non-compliant"]) < 2 else "covered",
                "positive_quality_note": "Positive coverage looks thin." if len(pool["compliant"]) < 2 else "Positive coverage count looks acceptable.",
                "negative_quality_note": "Negative coverage looks thin." if len(pool["non-compliant"]) < 2 else "Negative coverage count looks acceptable.",
                "reason": "Heuristic fallback without LLM review.",
                "generated_variants": [],
            }
            try:
                llm_result = self._audit_quality_with_llm(
                    anchor=key[1],
                    compliant_examples=pool["compliant"][:example_limit],
                    non_compliant_examples=pool["non-compliant"][:example_limit],
                )
                if llm_result["coverage_status"] not in {"covered", "gap"}:
                    llm_result["coverage_status"] = heuristic["coverage_status"]
                quality = {**heuristic, **llm_result}
            except Exception as exc:
                quality = {**heuristic, "reason": f"{heuristic['reason']} LLM audit failed: {exc}"}

            audits.append(
                {
                    "disclaimer_id": key[0],
                    "anchor": key[1],
                    "count": int(anchor_metrics.get("count", 0)),
                    "f1": float(anchor_metrics.get("f1", 0.0) or 0.0),
                    "examples": anchor_metrics.get("examples", []),
                    **quality,
                }
            )

        report = {
            "status": "quality_audited",
            "macro_f1": float(gate_metrics.get("macro_f1", 0.0) or 0.0),
            "anchors_below_threshold": len(audits),
            "audits": audits,
        }
        _json_save(output_path, report)
        return report

    def analyze_failures(
        self,
        *,
        reviewed_items: Iterable[dict[str, Any]],
        raw_synthetic_rows: list[dict[str, Any]],
        existing_extension_rows: list[dict[str, Any]],
        output_analysis_path: str | Path,
        output_extensions_path: str | Path,
    ) -> dict[str, Any]:
        candidates = self._candidate_cases(reviewed_items)
        example_limit = int(_agentic_settings(self.config).get("coverage_example_limit", 4))
        similarity_threshold = float(_agentic_settings(self.config).get("coverage_similarity_threshold", 0.74))
        variants_per_gap = int(_agentic_settings(self.config).get("augmentation_variants_per_gap", 2))

        pool_rows = [*raw_synthetic_rows, *existing_extension_rows]
        disclosures = _load_disclosures_lookup(self.config)
        grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
        for entry in pool_rows:
            if not isinstance(entry, dict):
                continue
            anchor = _resolve_entry_anchor(entry, self.config, disclosures)
            disclaimer_id = str(entry.get("disclaimer_id", "")).strip()
            label = _normalize_dataset_label(entry.get("type", ""))
            dialogue = str(entry.get("dialogue", "")).strip()
            if not anchor or not disclaimer_id or label not in {"compliant", "non-compliant"} or not dialogue:
                continue
            grouped.setdefault((disclaimer_id, anchor), {"compliant": [], "non-compliant": []})
            grouped[(disclaimer_id, anchor)][label].append(dialogue)

        analyses: list[dict[str, Any]] = []
        new_rows: list[dict[str, Any]] = []
        for item in candidates:
            disclaimer_id = str(item.get("disclaimer_id", "")).strip()
            anchor = str(item.get("anchor", "")).strip()
            corrected_label = _expected_label(item)
            normalized_label = _normalize_dataset_label(corrected_label)
            pool = grouped.get((disclaimer_id, anchor), {"compliant": [], "non-compliant": []})
            same_label_examples = pool.get(normalized_label, [])
            opposite_label = "non-compliant" if normalized_label == "compliant" else "compliant"
            opposite_examples = pool.get(opposite_label, [])
            reviewed_phrase = str(item.get("text", "")).strip()
            max_similarity = max((self._similarity(reviewed_phrase, example) for example in same_label_examples), default=0.0)
            heuristic_status = "covered" if same_label_examples and max_similarity >= similarity_threshold else "gap"
            heuristic = {
                "coverage_status": heuristic_status,
                "positive_quality_note": "Positive pool size is acceptable." if len(pool["compliant"]) >= 2 else "Positive pool is still thin.",
                "negative_quality_note": "Negative pool size is acceptable." if len(pool["non-compliant"]) >= 2 else "Negative pool is still thin.",
                "reason": "Heuristic coverage decision.",
                "generated_variants": [],
            }
            try:
                llm_result = self._audit_quality_with_llm(
                    anchor=anchor,
                    compliant_examples=pool["compliant"][:example_limit],
                    non_compliant_examples=pool["non-compliant"][:example_limit],
                    reviewed_phrase=reviewed_phrase,
                    corrected_label=corrected_label,
                    heuristic_status=heuristic_status,
                )
                if llm_result["coverage_status"] not in {"covered", "gap"}:
                    llm_result["coverage_status"] = heuristic_status
                analysis = {**heuristic, **llm_result}
            except Exception as exc:
                analysis = {**heuristic, "reason": f"{heuristic['reason']} LLM coverage review failed: {exc}"}

            analyses.append(
                {
                    "review_id": item.get("review_id", ""),
                    "disclaimer_id": disclaimer_id,
                    "anchor": anchor,
                    "correction_type": str(item.get("correction_type", "")).strip(),
                    "reviewed_phrase": reviewed_phrase,
                    "corrected_label": corrected_label,
                    "same_label_example_count": len(same_label_examples),
                    "opposite_label_example_count": len(opposite_examples),
                    "max_same_label_similarity": round(float(max_similarity), 6),
                    **analysis,
                }
            )

            reviewed_row = {
                "disclaimer_id": disclaimer_id,
                "anchor": anchor,
                "dialogue": reviewed_phrase,
                "type": normalized_label,
                "source": "reviewed_phrase",
                "transcript_id": str(item.get("transcript_id", "")).strip(),
                "review_id": str(item.get("review_id", "")).strip(),
            }
            new_rows.append(reviewed_row)

            if analysis["coverage_status"] == "gap":
                for variant in analysis.get("generated_variants", [])[:variants_per_gap]:
                    new_rows.append(
                        {
                            "disclaimer_id": disclaimer_id,
                            "anchor": anchor,
                            "dialogue": variant,
                            "type": normalized_label,
                            "source": "coverage_gap_generated",
                            "transcript_id": str(item.get("transcript_id", "")).strip(),
                            "review_id": str(item.get("review_id", "")).strip(),
                        }
                    )

        merged_rows, added_count = _merge_extension_rows(existing_extension_rows, new_rows)
        _json_save(output_analysis_path, analyses)
        _json_save(output_extensions_path, merged_rows)
        return {
            "candidate_case_count": len(candidates),
            "analysis_count": len(analyses),
            "new_rows_count": added_count,
            "all_extension_rows": merged_rows,
            "analyses": analyses,
        }


@dataclass
class TrainingGateAgent:
    config: dict[str, Any]

    def _bootstrap_mode(self) -> str:
        return str(_agentic_settings(self.config).get("bootstrap_mode", "if_missing")).strip().lower()

    def _models_exist(self) -> bool:
        model_settings = self.config.get("models", {})
        retriever_path = resolve_project_path(model_settings.get("bi_encoder_path", ""))
        verifier_path = resolve_project_path(model_settings.get("cross_encoder_path", ""))
        return retriever_path.exists() and verifier_path.exists()

    def _should_run_bootstrap(self) -> bool:
        settings = _agentic_settings(self.config)
        mode = str(settings.get("bootstrap_mode", "if_missing")).strip().lower()
        if mode == "always":
            return True
        if mode == "never":
            return False

        return not self._models_exist()

    def run(self, *, coverage_agent: CoverageAugmentationAgent, paths: dict[str, Path]) -> dict[str, Any]:
        if self._bootstrap_mode() == "never" and not self._models_exist():
            return {
                "status": "blocked",
                "message": "Bootstrap was disabled, but the active retriever/verifier models do not exist yet.",
            }
        if not self._should_run_bootstrap():
            return {
                "status": "skipped",
                "message": "Bootstrap training gate skipped because active models already exist.",
            }

        settings = _agentic_settings(self.config)
        raw_rows = _json_load(self.config.get("data", {}).get("synthetic_dataset_path", ""), [])
        if not isinstance(raw_rows, list) or not raw_rows:
            return {
                "status": "blocked",
                "message": "Synthetic dataset is empty, so bootstrap training cannot start.",
            }

        eval_size = float(settings.get("training_gate_eval_size", 0.2))
        seed = int(settings.get("training_gate_seed", self.config.get("training", {}).get("seed", 42)))
        metric_name = str(settings.get("training_gate_metric", "macro_f1")).strip() or "macro_f1"
        threshold = float(settings.get("training_gate_threshold", 0.9))

        gate_train_raw, gate_eval_raw = _split_synthetic_rows(raw_rows, eval_size=eval_size, seed=seed)
        _json_save(paths["gate_train_split"], gate_train_raw)
        _json_save(paths["gate_eval_split"], gate_eval_raw)

        gate_train_rows = _prepare_synthetic_rows(gate_train_raw, self.config)
        gate_eval_rows = _prepare_synthetic_rows(gate_eval_raw, self.config)
        if not gate_train_rows or not gate_eval_rows:
            return {
                "status": "blocked",
                "message": "Synthetic dataset is too small to run the 80/20 gate split.",
            }
        gate_model_path = _fit_cross_encoder_rows(
            gate_train_rows,
            config=self.config,
            output_dir=paths["gate_model_dir"],
            base_model_name_or_path=self.config.get("models", {}).get("cross_encoder_base", DEFAULT_CROSS_ENCODER_BASE),
        )
        metrics = _evaluate_cross_encoder_rows(gate_model_path, gate_eval_rows)
        _json_save(paths["gate_metrics"], metrics)

        metric_value = float(metrics.get(metric_name, 0.0) or 0.0)
        if metric_value < threshold:
            quality_report = coverage_agent.audit_synthetic_quality(
                raw_synthetic_rows=raw_rows,
                gate_metrics=metrics,
                output_path=paths["synthetic_quality_report"],
            )
            return {
                "status": "blocked",
                "message": f"Synthetic gate failed: {metric_name}={metric_value:.4f} is below {threshold:.4f}.",
                "gate_metrics": metrics,
                "quality_report": quality_report,
            }

        retriever_path = train_sentence_transformer(config=self.config)
        verifier_path = train_cross_encoder(config=self.config)
        return {
            "status": "ready",
            "message": f"Synthetic gate passed: {metric_name}={metric_value:.4f}. Full synthetic training completed.",
            "gate_metrics": metrics,
            "retriever_path": str(retriever_path),
            "verifier_path": str(verifier_path),
        }


@dataclass
class TrainerAgent:
    config: dict[str, Any]

    def retrain(self, extension_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rows = list(extension_rows)
        if not rows:
            return {
                "status": "skipped",
                "message": "No synthetic extension rows were available for retraining.",
            }
        return demo_services.retrain_demo_verifier(
            rows,
            config=self.config,
            promote_candidate=False,
        )


@dataclass
class SupervisorAgent:
    config: dict[str, Any]
    client: Any | None = None

    def __post_init__(self) -> None:
        self.inference_agent = InferenceAgent(self.config)
        self.review_agent = ReviewAgent(self.config, client=self.client)
        self.coverage_agent = CoverageAugmentationAgent(self.config, client=self.client)
        self.training_gate_agent = TrainingGateAgent(self.config)
        self.trainer_agent = TrainerAgent(self.config)

    def _should_promote(self, before_metrics: dict[str, Any], after_metrics: dict[str, Any], retrain_result: dict[str, Any]) -> bool:
        metric_name = str(_agentic_settings(self.config).get("promotion_metric", "macro_f1")).strip() or "macro_f1"
        before_value = float(before_metrics.get(metric_name, 0.0) or 0.0)
        after_value = float(after_metrics.get(metric_name, 0.0) or 0.0)
        return bool(retrain_result.get("promotion_recommended", False)) and after_value >= before_value - 1e-9

    def _evaluate_holdout(
        self,
        holdout_dataset: list[dict[str, Any]],
        *,
        cross_encoder_path: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        records = self.inference_agent.run_inference_records(holdout_dataset, cross_encoder_path=cross_encoder_path)
        predictions = collect_anchor_review_units(
            records,
            config=self.config,
            review_categories=("pass", "borderline", "fail"),
        )
        annotated = annotate_expected_labels(predictions, holdout_dataset=holdout_dataset)
        metrics = build_label_metrics(annotated)
        return records, annotated, metrics

    def run(
        self,
        *,
        incoming_source: str | None = None,
        holdout_source: str | None = None,
    ) -> dict[str, Any]:
        paths = _agentic_paths(self.config)
        gate_result = self.training_gate_agent.run(coverage_agent=self.coverage_agent, paths=paths)
        if gate_result.get("status") == "blocked":
            summary = {
                "status": "blocked",
                "message": gate_result.get("message", "Synthetic gate failed."),
                "bootstrap": gate_result,
                "artifacts": {name: str(path) for name, path in paths.items()},
            }
            _json_save(paths["summary"], summary)
            return summary

        incoming_transcripts = load_incoming_transcripts(config=self.config, incoming_source=incoming_source)
        holdout_dataset = load_holdout_dataset(config=self.config, holdout_source=holdout_source)

        holdout_records_before, holdout_predictions_before, holdout_metrics_before = self._evaluate_holdout(holdout_dataset)
        _json_save(paths["holdout_inference_before"], holdout_records_before)
        _json_save(paths["holdout_predictions_before"], holdout_predictions_before)
        _json_save(paths["holdout_metrics_before"], holdout_metrics_before)

        incoming_records = self.inference_agent.run_inference_records(incoming_transcripts)
        _json_save(paths["incoming_inference"], incoming_records)

        review_items = collect_anchor_review_units(incoming_records, config=self.config)
        reviewed_items = self.review_agent.review(review_items)
        _json_save(paths["review_items"], reviewed_items)

        auto_approve = bool(_agentic_settings(self.config).get("auto_approve_llm", True))
        if not auto_approve:
            summary = {
                "status": "awaiting_review",
                "message": "LLM review completed. Final labels need human approval before coverage augmentation and retraining.",
                "bootstrap": gate_result,
                "incoming_count": len(incoming_transcripts),
                "holdout_count": len(holdout_dataset),
                "reviewed_count": len(reviewed_items),
                "holdout_metrics_before": holdout_metrics_before,
                "artifacts": {name: str(path) for name, path in paths.items()},
            }
            _json_save(paths["summary"], summary)
            return summary

        raw_synthetic_rows = _json_load(self.config.get("data", {}).get("synthetic_dataset_path", ""), [])
        existing_extensions = _load_extension_rows(paths["synthetic_extensions"])
        coverage_result = self.coverage_agent.analyze_failures(
            reviewed_items=reviewed_items,
            raw_synthetic_rows=raw_synthetic_rows if isinstance(raw_synthetic_rows, list) else [],
            existing_extension_rows=existing_extensions,
            output_analysis_path=paths["coverage_analysis"],
            output_extensions_path=paths["synthetic_extensions"],
        )

        if coverage_result["new_rows_count"] == 0:
            summary = {
                "status": "no_training_examples",
                "message": "No corrected failure patterns required synthetic extensions in this run.",
                "bootstrap": gate_result,
                "incoming_count": len(incoming_transcripts),
                "holdout_count": len(holdout_dataset),
                "reviewed_count": len(reviewed_items),
                "coverage": coverage_result,
                "holdout_metrics_before": holdout_metrics_before,
                "artifacts": {name: str(path) for name, path in paths.items()},
            }
            _json_save(paths["summary"], summary)
            return summary

        retrain_result = self.trainer_agent.retrain(coverage_result["all_extension_rows"])
        if retrain_result.get("status") != "trained":
            summary = {
                "status": str(retrain_result.get("status", "blocked")),
                "message": str(retrain_result.get("message", "Retraining did not complete.")),
                "bootstrap": gate_result,
                "incoming_count": len(incoming_transcripts),
                "holdout_count": len(holdout_dataset),
                "reviewed_count": len(reviewed_items),
                "coverage": coverage_result,
                "holdout_metrics_before": holdout_metrics_before,
                "retrain": retrain_result,
                "artifacts": {name: str(path) for name, path in paths.items()},
            }
            _json_save(paths["summary"], summary)
            return summary

        candidate_model_path = str(retrain_result.get("candidate_model_path", "")).strip()
        holdout_records_after, holdout_predictions_after, holdout_metrics_after = self._evaluate_holdout(
            holdout_dataset,
            cross_encoder_path=candidate_model_path,
        )
        _json_save(paths["holdout_inference_after"], holdout_records_after)
        _json_save(paths["holdout_predictions_after"], holdout_predictions_after)
        _json_save(paths["holdout_metrics_after"], holdout_metrics_after)

        promoted = self._should_promote(holdout_metrics_before, holdout_metrics_after, retrain_result)
        if promoted:
            _promote_candidate_model(candidate_model_path, retrain_result.get("active_model_path", ""))

        summary = {
            "status": "completed",
            "message": "The local-LLM-assisted loop completed.",
            "bootstrap": gate_result,
            "incoming_count": len(incoming_transcripts),
            "holdout_count": len(holdout_dataset),
            "reviewed_count": len(reviewed_items),
            "coverage": coverage_result,
            "holdout_metrics_before": holdout_metrics_before,
            "holdout_metrics_after": holdout_metrics_after,
            "promoted": promoted,
            "retrain": {**retrain_result, "promoted": promoted},
            "artifacts": {name: str(path) for name, path in paths.items()},
        }
        _json_save(paths["summary"], summary)
        return summary


def answer_agentic_question(
    question: str,
    *,
    summary_payload: dict[str, Any] | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    client: Any | None = None,
) -> str:
    question_text = str(question or "").strip()
    if question_text.lower() in {"hi", "hello", "hey", "yo"}:
        return "Hi. I can help explain the latest agentic run, dataset diagnosis, or before/after score movement."

    config = config or demo_services.load_demo_config(config_path)
    client = client or demo_services._ollama_client_from_config(config)
    context = json.dumps(_compact_agentic_chat_context(summary_payload), ensure_ascii=False, indent=2)
    history_context = json.dumps(demo_services._compact_chat_history(chat_history), ensure_ascii=False, indent=2)
    system_prompt = (
        "You are a local analysis copilot for an anchor-based compliance workflow. "
        "Use non-thinking mode. Answer directly and do not narrate your analysis. "
        "Help a human review transcript inference, borderline phrases, synthetic data diagnosis, retraining impact, and before/after score movement. "
        "When summarizing a run, always start with rule results: Rule 101 pass/fail, Rule 102 pass/fail, "
        "and for Rule 102 include each disclosure's pass/fail status before discussing borderline phrases. "
        "Do not include hidden reasoning, chain-of-thought, scratchpad notes, or phrases like 'let me check'. "
        "Answer in concise paragraphs or bullets grounded only in the compact context."
    )
    answer = client.chat(
        system_prompt=system_prompt,
        user_prompt=(
            "/no_think\n"
            "Recent chat history, oldest first:\n"
            f"{history_context}\n\n"
            "Current compact agentic-run context:\n"
            f"{context}\n\n"
            f"User question:\n{question_text}\n\n"
            "/no_think\n"
            "Answer directly. Do not show reasoning."
        ),
        temperature=0.2,
        num_predict=AGENTIC_CHAT_NUM_PREDICT,
    )
    cleaned = demo_services.strip_think_blocks(answer)
    if demo_services._looks_like_reasoning_leak(cleaned):
        return _fallback_agentic_answer(summary_payload)
    return cleaned or "I can help explain the latest agentic run."


def _compact_rule_results(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    payloads = summary_payload.get("after_payloads") or summary_payload.get("before_payloads") or []
    if not isinstance(payloads, list):
        return []

    rule_results: list[dict[str, Any]] = []
    for payload in payloads[:3]:
        if not isinstance(payload, dict):
            continue
        transcript_id = str(payload.get("transcript_id", "")).strip()
        results = payload.get("results", {})
        if not isinstance(results, dict):
            continue
        for rule_id, result in results.items():
            if not isinstance(result, dict):
                continue
            evidence = result.get("evidence", {})
            evidence = evidence if isinstance(evidence, dict) else {}
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
                        claim_summaries.append(
                            {
                                "claim_type": str(claim.get("claim_type") or claim_group).strip(),
                                "claim_index": int(claim.get("claim_idx", index) or index),
                                "passed": bool(claim.get("passed", False)),
                                "score": round(float(claim.get("verification_score") or 0.0), 4),
                                "anchor": demo_services._compact_text(claim.get("anchor", ""), 220),
                                "best_text": demo_services._compact_text(claim.get("match_text", ""), 220),
                            }
                        )
            rule_results.append(
                {
                    "transcript_id": transcript_id,
                    "rule_id": str(rule_id),
                    "status": str(result.get("status", "UNKNOWN")).strip() or "UNKNOWN",
                    "score": round(float(evidence.get("verification_score") or 0.0), 4),
                    "claims": claim_summaries,
                }
            )
    return rule_results


def _compact_agentic_chat_context(summary_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary_payload, dict):
        return {"summary": "No agentic run is available yet."}

    def compact(value: Any, max_chars: int = 260) -> str:
        return demo_services._compact_text(value, max_chars)

    context: dict[str, Any] = {
        "status": str(summary_payload.get("status", "")).strip(),
        "message": compact(summary_payload.get("message", ""), 360),
        "recommendation": compact(summary_payload.get("recommendation", ""), 420),
        "transcript_count": summary_payload.get("transcript_count"),
        "borderline_count": summary_payload.get("borderline_count"),
        "supervisor_summary": compact(summary_payload.get("supervisor_summary", ""), 1000),
        "rule_results": _compact_rule_results(summary_payload),
    }

    stage_status = summary_payload.get("stage_status", [])
    if isinstance(stage_status, list):
        context["stage_status"] = [
            {
                "agent": str(item.get("agent", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "message": compact(item.get("message", ""), 260),
            }
            for item in stage_status
            if isinstance(item, dict)
        ][:8]

    review_items = summary_payload.get("review_items", [])
    if isinstance(review_items, list):
        context["review_items"] = [
            {
                "rule_id": str(item.get("disclaimer_id", "")).strip(),
                "score": round(float(item.get("verification_score") or 0.0), 4),
                "anchor": compact(item.get("anchor", ""), 220),
                "best_text": compact(item.get("text", ""), 220),
                "qwen_label": compact(item.get("llm_label", ""), 80),
                "final_label": compact(item.get("final_label", ""), 80),
                "rationale": compact(item.get("llm_rationale", ""), 220),
            }
            for item in review_items[:8]
            if isinstance(item, dict)
        ]

    diagnosis = summary_payload.get("diagnosis", {})
    if isinstance(diagnosis, dict):
        analyses = diagnosis.get("analyses", [])
        context["diagnosis"] = {
            "changed_case_count": diagnosis.get("changed_case_count", 0),
            "analyses": [
                {
                    "rule_id": str(item.get("disclaimer_id", "")).strip(),
                    "label_change": compact(item.get("label_change", ""), 120),
                    "cause_tags": item.get("cause_tags", []),
                    "same_label_count": item.get("same_label_count", 0),
                    "opposite_label_count": item.get("opposite_label_count", 0),
                    "recommendation": compact(item.get("recommendation", ""), 360),
                }
                for item in analyses[:5]
                if isinstance(item, dict)
            ]
            if isinstance(analyses, list)
            else [],
        }

    comparisons = summary_payload.get("comparisons", [])
    if isinstance(comparisons, list):
        context["comparisons"] = [
            {
                "rule_id": str(item.get("disclaimer_id", "")).strip(),
                "final_label": compact(item.get("final_label", ""), 80),
                "before_score": round(float(item.get("before_score") or 0.0), 4),
                "after_score": round(float(item.get("after_score") or 0.0), 4),
                "target_direction": compact(item.get("target_direction", ""), 80),
                "outcome": compact(item.get("outcome", ""), 80),
                "text": compact(item.get("text", ""), 220),
            }
            for item in comparisons[:8]
            if isinstance(item, dict)
        ]

    retrain = summary_payload.get("retrain", {})
    if isinstance(retrain, dict):
        context["retrain"] = {
            "status": str(retrain.get("status", "")).strip(),
            "promoted": retrain.get("promoted"),
            "metrics_before": retrain.get("metrics_before"),
            "metrics_after": retrain.get("metrics_after"),
        }

    context["omitted"] = "Full transcripts, raw model payloads, and model paths are omitted to keep local chat fast."
    return context


def _fallback_agentic_answer(summary_payload: dict[str, Any] | None) -> str:
    if not isinstance(summary_payload, dict) or not summary_payload:
        return "No agentic run is available yet."
    status = str(summary_payload.get("status", "unknown")).strip()
    recommendation = str(summary_payload.get("recommendation", "")).strip()
    rule_results = _compact_rule_results(summary_payload)
    lines: list[str] = []
    if rule_results:
        for item in rule_results[:4]:
            rule_id = str(item.get("rule_id", "")).strip()
            status_text = str(item.get("status", "UNKNOWN")).strip() or "UNKNOWN"
            line = f"Rule {rule_id}: {status_text}"
            claims = item.get("claims", [])
            if rule_id == "102" and isinstance(claims, list) and claims:
                disclosures = []
                for fallback_index, claim in enumerate(claims, start=1):
                    if not isinstance(claim, dict):
                        continue
                    claim_index = int(claim.get("claim_index", 0) or 0)
                    display_index = claim_index if claim_index > 0 else fallback_index
                    claim_status = "PASS" if bool(claim.get("passed", False)) else "FAIL"
                    disclosures.append(f"Disclosure {display_index}: {claim_status}")
                if disclosures:
                    line = f"{line} ({'; '.join(disclosures)})"
            lines.append(f"- {line}")
    else:
        lines.append(f"- Agentic run status: {status}")

    borderline_count = summary_payload.get("borderline_count")
    if borderline_count is None:
        review_items = summary_payload.get("review_items", [])
        borderline_count = len(review_items) if isinstance(review_items, list) else 0
    if int(borderline_count or 0) > 0:
        lines.append(f"- Borderline phrases: {int(borderline_count)} requiring human review.")
    else:
        lines.append("- Borderline phrases: none found.")

    comparisons = summary_payload.get("comparisons", [])
    if isinstance(comparisons, list) and comparisons:
        improved = sum(1 for item in comparisons if isinstance(item, dict) and item.get("outcome") == "improved")
        regressed = sum(1 for item in comparisons if isinstance(item, dict) and item.get("outcome") == "regressed")
        lines.append(f"- Score movement: {improved} improved, {regressed} regressed.")
    if recommendation:
        lines.append(f"- Recommendation: {recommendation}")
    return "\n".join(lines)


def run_local_agentic_loop(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    transcripts_source: str | None = None,
    holdout_source: str | None = None,
    auto_approve_llm: bool | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if auto_approve_llm is not None:
        overrides["agentic"] = {"auto_approve_llm": bool(auto_approve_llm)}

    if config is None:
        config = demo_services.load_demo_config(config_path, overrides=overrides or None)
    elif overrides:
        config = _deep_merge(config, overrides)

    supervisor = SupervisorAgent(config, client=client)
    return supervisor.run(incoming_source=transcripts_source, holdout_source=holdout_source)
