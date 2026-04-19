"""Evaluation helpers for compliance reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.config import load_config
from ..core.json_utils import save_json
from ..core.paths import resolve_project_path


def clean_nullable(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def normalize_ground_truth_label(value: Any, valid_labels: set[str]) -> str | None:
    text = clean_nullable(value)
    if text is None:
        return None
    label = text.upper()
    return label if label in valid_labels else None


def normalize_ground_truth_dict(raw_ground_truth: Any, valid_labels: set[str]) -> dict[str, str | None]:
    normalized = {str(scenario_id): None for scenario_id in range(1, 15)}
    if not isinstance(raw_ground_truth, dict):
        return normalized
    for scenario_id in range(1, 15):
        normalized[str(scenario_id)] = normalize_ground_truth_label(raw_ground_truth.get(str(scenario_id)), valid_labels)
    return normalized


def load_report(report_path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(resolve_project_path(report_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected report JSON object: {report_path}")
    return data


def load_truth_lookup(truth_path: str | Path, valid_labels: set[str]) -> dict[str, dict[str, Any]]:
    data = json.loads(resolve_project_path(truth_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected truth JSON object: {truth_path}")

    lookup: dict[str, dict[str, Any]] = {}
    for transcript_id, metadata in data.items():
        if not isinstance(metadata, dict):
            continue
        clean_transcript_id = clean_nullable(transcript_id)
        if clean_transcript_id is None:
            continue
        clean_transcript_id = clean_transcript_id.rsplit(".", 1)[0]

        ground_truth = normalize_ground_truth_dict(metadata.get("ground_truth"), valid_labels)
        if all(value is None for value in ground_truth.values()):
            ground_truth = {
                str(scenario_id): normalize_ground_truth_label(metadata.get(str(scenario_id)), valid_labels)
                for scenario_id in range(1, 15)
            }

        lookup[clean_transcript_id] = {
            "op_code": clean_nullable(metadata.get("op_code")),
            "client_name": clean_nullable(metadata.get("client_name") or metadata.get("order_person_spoken_to")),
            "trade_date": clean_nullable(metadata.get("trade_date") or metadata.get("trade_date_raw")),
            "client_phone_number": clean_nullable(metadata.get("client_phone_number")),
            "ground_truth": ground_truth,
        }
    return lookup


def prediction_status(report: dict[str, dict[str, Any]], transcript_id: str, scenario_id: str) -> str | None:
    transcript_result = report.get(transcript_id)
    if not isinstance(transcript_result, dict):
        return None
    scenario_result = transcript_result.get(scenario_id)
    if not isinstance(scenario_result, dict):
        return None
    status = clean_nullable(scenario_result.get("status"))
    return status.upper() if status else None


def prediction_label(status: str | None) -> str:
    return "pass" if status == "PASS" else "fail"


def scenario_is_available(scenario_id: int, metadata_entry: dict[str, Any]) -> bool:
    if scenario_id == 1:
        return metadata_entry.get("client_name") is not None
    if 2 <= scenario_id <= 12:
        return True
    if scenario_id == 13:
        return metadata_entry.get("trade_date") is not None
    if scenario_id == 14:
        return metadata_entry.get("client_phone_number") is not None
    return False


def evaluate(
    report_path: str | Path,
    truth_path: str | Path,
    config: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    config = config or load_config(config_path)
    valid_labels = set(config.get("evaluation", {}).get("valid_ground_truth", ["T", "F", "B", "N"]))

    report = load_report(report_path)
    truth_lookup = load_truth_lookup(truth_path, valid_labels)

    results: dict[str, dict[str, str]] = {}
    missed_cases: dict[str, dict[str, Any]] = {}

    for scenario_id in range(1, 15):
        scenario_key = str(scenario_id)
        total_compliant = 0
        passed_compliant = 0
        total_non_compliant = 0
        passed_non_compliant = 0

        for transcript_id in report:
            truth_entry = truth_lookup.get(transcript_id)
            if truth_entry is None or not scenario_is_available(scenario_id, truth_entry):
                continue

            ground_truth = truth_entry.get("ground_truth", {})
            truth_label = normalize_ground_truth_label(ground_truth.get(scenario_key), valid_labels)
            if truth_label is None:
                continue

            status = prediction_status(report, transcript_id, scenario_key)
            predicted = prediction_label(status)

            if truth_label == "T":
                total_compliant += 1
                if predicted == "pass":
                    passed_compliant += 1
                else:
                    missed_cases.setdefault(transcript_id, {"op_code": truth_entry.get("op_code")})[scenario_key] = f"{predicted} ({truth_label})"
            elif truth_label == "F":
                total_non_compliant += 1
                if predicted == "fail":
                    passed_non_compliant += 1
                else:
                    missed_cases.setdefault(transcript_id, {"op_code": truth_entry.get("op_code")})[scenario_key] = f"{predicted} ({truth_label})"

        compliant_score = passed_compliant / total_compliant if total_compliant else 0.0
        non_compliant_score = passed_non_compliant / total_non_compliant if total_non_compliant else 0.0
        results[scenario_key] = {
            "compliant": f"{passed_compliant}/{total_compliant} ({compliant_score * 100:.2f}%)",
            "non-compliant": f"{passed_non_compliant}/{total_non_compliant} ({non_compliant_score * 100:.2f}%)",
        }

    output_settings = config.get("outputs", {})
    if output_settings.get("evaluation_output_path"):
        save_json(output_settings["evaluation_output_path"], results)
    if output_settings.get("evaluation_missed_cases_path"):
        save_json(output_settings["evaluation_missed_cases_path"], missed_cases)

    return results, missed_cases


def main(report_path: str, truth_path: str, config_path: str | None = None) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    return evaluate(report_path=report_path, truth_path=truth_path, config_path=config_path)
