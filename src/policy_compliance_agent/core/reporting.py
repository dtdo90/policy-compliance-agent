"""Reporting helpers for inference outputs."""

from __future__ import annotations

from typing import Any

from .paths import ensure_parent_dir


def _sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def generate_csv_report(final_report: dict[str, dict[str, Any]], csv_path: str) -> None:
    import pandas as pd

    all_disclaimer_ids = set()
    for script_results in final_report.values():
        if isinstance(script_results, dict):
            all_disclaimer_ids.update(str(key) for key in script_results.keys())

    sorted_disclaimer_ids = sorted(
        all_disclaimer_ids,
        key=_sort_key,
    )
    columns = ["transcript_id"] + [str(disclaimer_id) for disclaimer_id in sorted_disclaimer_ids] + ["COMPLIANT"]

    rows = []
    for transcript_id in sorted(final_report.keys(), key=_sort_key):
        transcript_results = final_report[transcript_id]
        row = [transcript_id]
        passed_ids = []

        for disclaimer_id in sorted_disclaimer_ids:
            result = transcript_results.get(disclaimer_id, {})
            status = result.get("status", "FAIL")
            evidence = result.get("evidence", {})
            score = float(evidence.get("verification_score") or 0.0)
            row.append(round(score, 4))
            if status == "PASS":
                passed_ids.append(str(disclaimer_id))

        row.append(", ".join(passed_ids) if passed_ids else "NONE")
        rows.append(row)

    output_path = ensure_parent_dir(csv_path)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def build_annotation_output(final_report: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    annotation_output: dict[str, dict[str, Any]] = {}

    for transcript_id, transcript_result in final_report.items():
        annotation_output[transcript_id] = {}
        for scenario_id, scenario_result in transcript_result.items():
            evidence = scenario_result.get("evidence", {})
            description = evidence.get("description", "")
            purpose_of_control = evidence.get("purpose_of_control", "")
            criteria = evidence.get("criteria", "")
            keywords = evidence.get("keywords", [])
            verification_score = evidence.get("verification_score", 0.0)
            claims = evidence.get("claims", {})

            if scenario_id in {"2", "3", "4", "5", "6", "7", "8", "12"}:
                single_claims = claims.get("single", [])
                claim = single_claims[0] if single_claims else {}
                payload = {
                    "description": description,
                    "purpose_of_control": purpose_of_control,
                    "criteria": criteria,
                    "keywords": keywords,
                    "claims": {
                        "claim_type": "mandatory",
                        "anchor": claim.get("anchor"),
                        "best_text": claim.get("match_text"),
                        "verification_score": claim.get("verification_score"),
                    },
                }
            elif scenario_id in {"9", "10"}:
                payload = {
                    "description": description,
                    "purpose_of_control": purpose_of_control,
                    "criteria": criteria,
                    "keywords": keywords,
                    "claims": [
                        {
                            "claim_type": "mandatory",
                            "anchor": claim.get("anchor"),
                            "best_text": claim.get("match_text"),
                            "verification_score": claim.get("verification_score"),
                        }
                        for claim in claims.get("mandatory", [])
                    ],
                }
            elif scenario_id == "11":
                payload = {
                    "description": description,
                    "purpose_of_control": purpose_of_control,
                    "criteria": criteria,
                    "keywords": keywords,
                    "claims": [
                        {
                            "claim_type": claim_type,
                            "anchor": claim.get("anchor"),
                            "best_text": claim.get("match_text"),
                            "verification_score": claim.get("verification_score"),
                        }
                        for claim_type in ("mandatory", "standard")
                        for claim in claims.get(claim_type, [])
                    ],
                }
            else:
                payload = {
                    "description": description,
                    "purpose_of_control": purpose_of_control,
                    "criteria": criteria,
                    "keywords": keywords,
                    "verification_score": verification_score,
                }

            annotation_output[transcript_id][scenario_id] = payload

    return annotation_output
