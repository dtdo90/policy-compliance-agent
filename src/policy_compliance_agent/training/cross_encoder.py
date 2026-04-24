"""Cross-encoder training for semantic compliance verification."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from ..core.config import load_config
from ..core.paths import resolve_project_path
from ..core.runtime import configure_cpu_runtime
from .data_utils import DEFAULT_CROSS_ENCODER_BASE, resolve_training_anchor_text

CONFUSION_PAIRS = [
    (("3", "single", 0), ("4", "single", 0)),
    (("2", "single", 0), ("10", "mandatory", 0)),
    (("2", "single", 0), ("10", "mandatory", 1)),
    (("8", "single", 0), ("10", "mandatory", 0)),
    (("6", "single", 0), ("2", "single", 0)),
]


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _resolve_anchor_text(disclosures: dict[str, Any], anchor_ref) -> str | None:
    scenario_id, anchor_type, anchor_idx = anchor_ref
    item = disclosures.get(str(scenario_id), {})
    anchor_data = item.get("anchor", "")

    if isinstance(anchor_data, str):
        if anchor_type != "single":
            return None
        text = anchor_data.strip()
        return text if text else None

    if not isinstance(anchor_data, dict) or anchor_type == "single":
        return None

    anchor_values = anchor_data.get(anchor_type) or []
    if not isinstance(anchor_values, list) or not (0 <= anchor_idx < len(anchor_values)):
        return None

    text = anchor_values[anchor_idx]
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text if text else None


def _sample_count(total: int, fraction: float) -> int:
    if total <= 0 or fraction <= 0:
        return 0
    return max(1, int(total * fraction))


def _add_confusion_pair_negatives(
    rows: list[dict[str, Any]],
    disclosures: dict[str, Any],
    seed: int,
    confusion_neg_fraction: float,
) -> None:
    rng = random.Random(seed)
    seen = {
        (_normalize_text(row["sentence1"]), _normalize_text(row["sentence2"]), float(row["label"]))
        for row in rows
    }

    compliant_by_anchor: dict[str, list[str]] = {}
    for row in rows:
        if float(row["label"]) != 1.0:
            continue
        compliant_by_anchor.setdefault(_normalize_text(row["sentence1"]), []).append(row["sentence2"])

    added = 0
    for ref_a, ref_b in CONFUSION_PAIRS:
        anchor_a = _resolve_anchor_text(disclosures, ref_a)
        anchor_b = _resolve_anchor_text(disclosures, ref_b)
        if not anchor_a or not anchor_b:
            continue

        anchor_a_key = _normalize_text(anchor_a)
        anchor_b_key = _normalize_text(anchor_b)
        anchor_a_rows = compliant_by_anchor.get(anchor_a_key, [])
        anchor_b_rows = compliant_by_anchor.get(anchor_b_key, [])
        if not anchor_a_rows or not anchor_b_rows:
            continue

        sample_ab = min(_sample_count(len(anchor_b_rows), confusion_neg_fraction), len(anchor_b_rows))
        for dialogue in rng.sample(anchor_b_rows, sample_ab):
            triple = (anchor_a_key, _normalize_text(dialogue), 0.0)
            if triple in seen:
                continue
            rows.append({"sentence1": anchor_a, "sentence2": dialogue.strip(), "label": 0.0})
            seen.add(triple)
            added += 1

        sample_ba = min(_sample_count(len(anchor_a_rows), confusion_neg_fraction), len(anchor_a_rows))
        for dialogue in rng.sample(anchor_a_rows, sample_ba):
            triple = (anchor_b_key, _normalize_text(dialogue), 0.0)
            if triple in seen:
                continue
            rows.append({"sentence1": anchor_b, "sentence2": dialogue.strip(), "label": 0.0})
            seen.add(triple)
            added += 1

    print(f"Added confusion-pair negatives: {added}")


def prepare_training_rows(config: dict[str, Any], dataset_path: str | None = None) -> list[dict[str, Any]]:
    data_settings = config.get("data", {})
    disclosures_path = resolve_project_path(data_settings["disclosures_file"])
    dataset_file = resolve_project_path(dataset_path or data_settings["synthetic_dataset_path"])
    training_settings = config.get("training", {})

    disclosures = json.loads(disclosures_path.read_text(encoding="utf-8"))
    raw_rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    skipped_unknown = 0
    skipped_missing_anchor = 0
    for entry in raw_rows:
        disclaimer_id = str(entry.get("disclaimer_id", "")).strip()
        if not disclaimer_id:
            continue

        anchor_text = resolve_training_anchor_text(entry, disclosures)
        dialogue = entry.get("dialogue", "")
        label_text = str(entry.get("type", "")).strip().lower()
        if not isinstance(anchor_text, str) or not anchor_text.strip():
            skipped_missing_anchor += 1
            continue
        if not isinstance(dialogue, str) or not dialogue.strip():
            continue

        if label_text == "compliant":
            label = 1.0
        elif label_text in {"non-compliant", "non_compliant", "noncompliant"}:
            label = 0.0
        else:
            skipped_unknown += 1
            continue

        try:
            sample_weight = int(entry.get("sample_weight", 1) or 1)
        except (TypeError, ValueError):
            sample_weight = 1
        sample_weight = max(1, sample_weight)
        for _ in range(sample_weight):
            rows.append({"sentence1": anchor_text, "sentence2": dialogue.strip(), "label": label})

    if training_settings.get("use_extra_sampling", True):
        _add_confusion_pair_negatives(
            rows,
            disclosures=disclosures,
            seed=int(training_settings.get("seed", 42)),
            confusion_neg_fraction=float(training_settings.get("confusion_neg_fraction", 0.1)),
        )

    if skipped_unknown:
        print(f"Skipped {skipped_unknown} entries with unknown type")
    if skipped_missing_anchor:
        print(f"Skipped {skipped_missing_anchor} entries without a resolvable anchor")
    return rows


def _cleanup_checkpoints(output_dir: Path) -> int:
    removed = 0
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            removed += 1
    return removed


def train_cross_encoder(config: dict[str, Any] | None = None, config_path: str | None = None) -> Path:
    return train_cross_encoder_with_overrides(config=config, config_path=config_path)


def train_cross_encoder_with_overrides(
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
    *,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    monitor_output_dir: str | None = None,
    base_model_name_or_path: str | None = None,
) -> Path:
    config = config or load_config(config_path)
    configure_cpu_runtime(int(config.get("runtime", {}).get("cpu_threads", 8)))

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    training_settings = config.get("training", {})
    model_settings = config.get("models", {})

    rows = prepare_training_rows(config, dataset_path=dataset_path)
    if not rows:
        raise ValueError("No training rows were produced from the configured dataset.")

    model_name = base_model_name_or_path or model_settings.get("cross_encoder_base", DEFAULT_CROSS_ENCODER_BASE)
    output_dir_final = resolve_project_path(output_dir or model_settings["cross_encoder_output_dir"])
    output_dir_final.mkdir(parents=True, exist_ok=True)

    max_length = int(training_settings.get("max_length", 256))
    learning_rate = float(training_settings.get("learning_rate", 2e-5))
    num_epochs = int(training_settings.get("cross_encoder_num_epochs", training_settings.get("num_epochs", 1)))
    train_batch_size = int(training_settings.get("train_batch_size", 16))
    eval_batch_size = int(training_settings.get("eval_batch_size", 16))
    warmup_ratio = float(training_settings.get("warmup_ratio", 0.1))
    force_cpu = bool(training_settings.get("force_cpu", True))
    seed = int(training_settings.get("seed", 42))
    device = "cpu" if force_cpu else None

    model_kwargs = {"device": device} if device is not None else {}
    model = CrossEncoder(model_name, num_labels=1, max_length=max_length, **model_kwargs)
    train_dataloader = DataLoader(
        [InputExample(texts=[row["sentence1"], row["sentence2"]], label=float(row["label"])) for row in rows],
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
        show_progress_bar=True,
    )
    _cleanup_checkpoints(output_dir_final)
    model.save(str(output_dir_final))
    return output_dir_final


def main(config_path: str | None = None) -> Path:
    return train_cross_encoder(config_path=config_path)
