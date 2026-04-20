"""Sentence-transformer training for semantic retrieval."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from ..core.config import load_config
from ..core.paths import resolve_project_path
from ..core.runtime import configure_cpu_runtime
from .data_utils import DEFAULT_SENTENCE_TRANSFORMER_BASE, resolve_training_anchor_text


def generate_triplet_rows(
    dataset_rows: list[dict[str, Any]],
    seed: int,
    disclosures: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    random.seed(seed)

    def normalize_type(value: str) -> str:
        text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
        if text in {"compliant", "positive", "pass"}:
            return "compliant"
        if text in {"non-compliant", "noncompliant", "negative", "fail"}:
            return "non-compliant"
        return ""

    data_pool: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in dataset_rows:
        if not isinstance(entry, dict):
            continue

        disclaimer_id = str(entry.get("disclaimer_id", "")).strip()
        anchor_text = ""
        if disclosures is not None:
            resolved_anchor = resolve_training_anchor_text(entry, disclosures)
            anchor_text = resolved_anchor.strip() if isinstance(resolved_anchor, str) else ""
        if not anchor_text:
            anchor_text = str(entry.get("anchor", "")).strip()
        label_type = normalize_type(entry.get("type", ""))
        dialogue = str(entry.get("dialogue", "")).strip()
        if not disclaimer_id or not anchor_text or label_type not in {"compliant", "non-compliant"} or not dialogue:
            continue

        key = (disclaimer_id, anchor_text)
        data_pool.setdefault(key, {"anchor": anchor_text, "compliant": [], "non-compliant": []})
        data_pool[key][label_type].append(dialogue)

    anchors: list[str] = []
    positives: list[str] = []
    negatives: list[str] = []
    keys = list(data_pool.keys())

    for key in keys:
        group = data_pool[key]
        anchor_text = group["anchor"]
        compliant_rows = group["compliant"]
        hard_negatives = group["non-compliant"]
        if not anchor_text or not compliant_rows or not hard_negatives:
            continue

        other_keys = [other_key for other_key in keys if other_key != key]
        shuffled_compliant = compliant_rows.copy()
        shuffled_hard_negatives = hard_negatives.copy()
        random.shuffle(shuffled_compliant)
        random.shuffle(shuffled_hard_negatives)

        paired_negative_by_positive: dict[str, str] = {}
        for index, positive_text in enumerate(shuffled_compliant):
            negative_text = shuffled_hard_negatives[index % len(shuffled_hard_negatives)]
            paired_negative_by_positive[positive_text] = negative_text
            anchors.append(anchor_text)
            positives.append(positive_text)
            negatives.append(negative_text)

        for positive_text in compliant_rows:
            banned = paired_negative_by_positive.get(positive_text)
            pool = [negative for negative in hard_negatives if negative != banned] or hard_negatives
            for negative_text in random.sample(pool, min(2, len(pool))):
                anchors.append(anchor_text)
                positives.append(positive_text)
                negatives.append(negative_text)

        if other_keys:
            for positive_text in compliant_rows:
                selected_other = random.sample(other_keys, min(2, len(other_keys)))
                for other_key in selected_other:
                    other_positive_rows = data_pool[other_key]["compliant"]
                    if not other_positive_rows:
                        continue
                    anchors.append(anchor_text)
                    positives.append(positive_text)
                    negatives.append(random.choice(other_positive_rows))

    return {"anchor": anchors, "positive": positives, "negative": negatives}


def _cleanup_checkpoints(output_dir: Path) -> None:
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def train_sentence_transformer(config: dict[str, Any] | None = None, config_path: str | None = None) -> Path:
    return train_sentence_transformer_with_overrides(config=config, config_path=config_path)


def train_sentence_transformer_with_overrides(
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

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.evaluation import TripletEvaluator
    from sentence_transformers.losses import TripletDistanceMetric
    import torch

    training_settings = config.get("training", {})
    model_settings = config.get("models", {})
    data_settings = config.get("data", {})

    dataset_file = resolve_project_path(dataset_path or data_settings["synthetic_dataset_path"])
    dataset_rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    disclosures_path = resolve_project_path(data_settings["disclosures_file"])
    disclosures = json.loads(disclosures_path.read_text(encoding="utf-8"))
    seed = int(training_settings.get("seed", 42))
    triplets = generate_triplet_rows(dataset_rows, seed=seed, disclosures=disclosures)
    if not triplets["anchor"]:
        raise ValueError("No triplets were produced from the configured dataset.")

    dataset = Dataset.from_dict(triplets)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    base_model = base_model_name_or_path or model_settings.get("sentence_transformer_base", DEFAULT_SENTENCE_TRANSFORMER_BASE)
    eval_output_dir = resolve_project_path(monitor_output_dir or "data/results/checkpoints/st_eval")
    final_output_dir = resolve_project_path(output_dir or model_settings["sentence_transformer_output_dir"])
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(training_settings.get("sentence_transformer_batch_size", 32))
    eval_batch_size = int(training_settings.get("sentence_transformer_eval_batch_size", 64))
    learning_rate = float(training_settings.get("learning_rate", 2e-5))
    num_epochs = int(training_settings.get("num_epochs", 1))
    logging_steps = int(training_settings.get("logging_steps", 50))
    dataloader_workers = int(training_settings.get("dataloader_workers", 4))
    triplet_margin = float(training_settings.get("triplet_margin", 0.5))
    force_cpu = bool(training_settings.get("force_cpu", False))
    use_cuda = torch.cuda.is_available() and not force_cpu

    def run_training_step(step_name: str, train_set: Dataset, eval_set: Dataset | None, output_dir: Path) -> None:
        print(f"=== {step_name} ===")
        model = SentenceTransformer(base_model)
        loss = losses.TripletLoss(
            model=model,
            distance_metric=TripletDistanceMetric.COSINE,
            triplet_margin=triplet_margin,
        )

        evaluator = None
        if eval_set is not None:
            evaluator = TripletEvaluator(
                anchors=eval_set["anchor"],
                positives=eval_set["positive"],
                negatives=eval_set["negative"],
                name="triplet_eval",
            )

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            bf16=bool(use_cuda and torch.cuda.is_bf16_supported()),
            dataloader_num_workers=dataloader_workers,
            dataloader_pin_memory=False,
            use_cpu=not use_cuda,
            logging_steps=logging_steps,
            save_strategy="epoch" if eval_set is not None else "no",
            eval_strategy="epoch" if eval_set is not None else "no",
            save_total_limit=1,
            load_best_model_at_end=True if eval_set is not None else False,
            metric_for_best_model="triplet_eval_cosine_accuracy" if eval_set is not None else None,
            greater_is_better=True if eval_set is not None else None,
            report_to="none",
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            loss=loss,
            evaluator=evaluator,
        )
        trainer.train()
        trainer.save_model(str(output_dir))
        _cleanup_checkpoints(output_dir)

    run_training_step("Validation Run", train_dataset, eval_dataset, eval_output_dir)
    run_training_step("Full Production Run", dataset, None, final_output_dir)
    return final_output_dir


def main(config_path: str | None = None) -> Path:
    return train_sentence_transformer(config_path=config_path)
