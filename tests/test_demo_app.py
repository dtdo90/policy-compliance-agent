from pathlib import Path
import shutil

import pandas as pd
import pytest

from policy_compliance_agent.demo import app as demo_app


def test_borderline_dataframe_defaults_to_unselected_blank_human_labels():
    dataframe = demo_app._borderline_dataframe(
        [
            {
                "disclaimer_id": "102",
                "verification_score": 0.3872,
                "text": "There will be a fee to make the change and the new flight is higher.",
            }
        ]
    )

    assert dataframe.to_dict(orient="records") == [
        {
            "add": False,
            "rule": "102",
            "score": 0.3872,
            "text": "There will be a fee to make the change and the new ...",
            "model_label": "Fail",
            "qwen_label": "",
            "final_label": "",
        }
    ]


def test_merge_review_items_saves_only_checked_rows_and_human_final_label():
    review_items = [
        {
            "disclaimer_id": "102",
            "text": "There will be a fee to make the change.",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "verification_score": 0.3872,
            "llm_label": "Compliant",
        },
        {
            "disclaimer_id": "101",
            "text": "I can reset it now.",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "verification_score": 0.42,
            "llm_label": "Non-Compliant",
        },
    ]
    dataframe = pd.DataFrame(
        [
            {
                "add": True,
                "rule": "102",
                "score": 0.3872,
                "text": "There will be a fee to make the change.",
                "model_label": "Fail",
                "qwen_label": "Pass",
                "final_label": "fail",
            },
            {
                "add": False,
                "rule": "101",
                "score": 0.42,
                "text": "I can reset it now.",
                "model_label": "Fail",
                "qwen_label": "Fail",
                "final_label": "Fail",
            },
        ]
    )

    merged = demo_app._merge_review_items(dataframe, review_items)

    assert merged[0]["approved"] is True
    assert merged[0]["llm_label"] == "Compliant"
    assert merged[0]["final_label"] == "Non-Compliant"
    assert merged[1]["approved"] is False


def test_merge_review_items_accepts_gradio_table_payloads():
    review_items = [
        {
            "disclaimer_id": "102",
            "text": "There will be a fee to make the change.",
            "verification_score": 0.6889,
            "llm_label": "Compliant",
        }
    ]
    table_payload = {
        "headers": ["add", "transcript", "rule", "score", "text", "model_label", "qwen_label", "final_label"],
        "data": [[True, "tx1", "102", 0.6889, "There will be a fee...", "Pass", "Pass", "Pass"]],
    }

    merged = demo_app._merge_review_items(table_payload, review_items)

    assert merged[0]["approved"] is True
    assert merged[0]["disclaimer_id"] == "102"
    assert merged[0]["verification_score"] == 0.6889
    assert merged[0]["final_label"] == "Compliant"


def test_merge_review_items_treats_skip_as_not_trainable():
    dataframe = pd.DataFrame(
        [
            {
                "add": True,
                "rule": "102",
                "score": 0.38,
                "text": "ambiguous phrase",
                "model_label": "Fail",
                "qwen_label": "Pass",
                "final_label": "Skip",
            }
        ]
    )

    merged = demo_app._merge_review_items(dataframe, [{"disclaimer_id": "102", "text": "ambiguous phrase"}])

    assert merged[0]["approved"] is True
    assert merged[0]["final_label"] == ""


def test_merge_review_items_uses_edited_model_label_when_final_label_blank():
    dataframe = pd.DataFrame(
        [
            {
                "add": True,
                "rule": "102",
                "score": 0.91,
                "text": "clear pass phrase",
                "model_label": "Fail",
                "qwen_label": "",
                "final_label": "",
            }
        ]
    )

    merged = demo_app._merge_review_items(dataframe, [{"disclaimer_id": "102", "text": "clear pass phrase"}])

    assert merged[0]["approved"] is True
    assert merged[0]["llm_label"] == ""
    assert merged[0]["model_label"] == "Non-Compliant"
    assert merged[0]["final_label"] == "Non-Compliant"


def test_agentic_review_dataframe_hides_transcript_and_shows_full_anchor_text():
    dataframe = demo_app._agentic_review_dataframe(
        [
            {
                "transcript_id": "tx1",
                "disclaimer_id": "102",
                "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                "text": "There will be a fee to make the change and the new flight is higher.",
                "verification_score": 0.6889,
                "llm_label": "Compliant",
            }
        ]
    )

    assert list(dataframe.columns) == ["add", "rule", "score", "anchor", "text", "model", "qwen", "human"]
    row = dataframe.to_dict(orient="records")[0]
    assert row["anchor"] == "Before I confirm this booking change, there is a change fee that will apply."
    assert row["text"] == "There will be a fee to make the change and the new flight is higher."
    assert row["model"] == "Pass"
    assert row["qwen"] == "Pass"
    assert "transcript" not in row


def test_refresh_investigator_generated_queue_replaces_stale_generated_rows():
    review_items = [
        {
            "transcript_id": "tx1",
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "text": "now. After I unlock your account, I will verify your identity for our records.",
            "final_label": "Non-Compliant",
        },
        {
            "transcript_id": "investigator_generated_101_1",
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "text": "old generated sample",
            "review_type": "generated_synthetic",
            "final_label": "Non-Compliant",
        },
    ]
    generated_items = [
        {
            "transcript_id": "investigator_generated_101_1",
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "text": "new generated sample a",
            "review_type": "generated_synthetic",
            "final_label": "Non-Compliant",
        },
        {
            "transcript_id": "investigator_generated_101_2",
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "text": "new generated sample b",
            "review_type": "generated_synthetic",
            "final_label": "Non-Compliant",
        },
    ]

    refreshed = demo_app._refresh_investigator_generated_queue(review_items, generated_items)

    assert [item["text"] for item in refreshed] == [
        "now. After I unlock your account, I will verify your identity for our records.",
        "new generated sample a",
        "new generated sample b",
    ]


def test_agentic_review_dataframe_blanks_generated_score_and_model():
    dataframe = demo_app._agentic_review_dataframe(
        [
            {
                "disclaimer_id": "101",
                "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                "text": "I will unlock your account first, then verify identity afterward.",
                "verification_score": 0.0,
                "review_type": "generated_synthetic",
                "llm_label": "Non-Compliant",
                "final_label": "Non-Compliant",
            }
        ]
    )

    row = dataframe.to_dict(orient="records")[0]
    assert row["score"] == ""
    assert row["model"] == ""
    assert row["qwen"] == "Fail"
    assert row["human"] == "Fail"


def test_format_agentic_comparison_markdown_shows_case_outcomes():
    text = demo_app._format_agentic_comparison_markdown(
        [
            {
                "disclaimer_id": "101",
                "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                "text": "now. After I unlock your account, I will verify your identity for our records.",
                "verification_score": 0.9986,
                "before_score": 0.9986,
                "after_score": 0.4817,
                "final_label": "Non-Compliant",
                "outcome": "improved",
            }
        ]
    )

    assert "Case 1: Rule 101" in text
    assert "Status:** `Success`" in text
    assert "model `Pass` -> human `Fail` -> model `Fail` after retrain" in text
    assert "`0.9986` -> `0.4817`" in text
    assert "| Rule |" not in text


def test_format_results_uses_short_demo_names_and_per_anchor_evidence():
    payload = {
        "results": {
            "101": {
                "status": "PASS",
                "evidence": {
                    "description": "Support agent requires identity verification before resetting or unlocking the account.",
                    "verification_score": 0.99,
                    "claims": {
                        "single": [
                            {
                                "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                                "match_text": "Before I reset your account, I need to verify your identity first.",
                                "verification_score": 0.99,
                            }
                        ],
                        "mandatory": [],
                        "standard": [],
                    },
                },
            },
            "102": {
                "status": "FAIL",
                "evidence": {
                    "description": "Travel agent discloses the change fee and fare difference or credit before confirming the booking change.",
                    "verification_score": 0.41,
                    "claims": {
                        "single": [],
                        "mandatory": [
                            {
                                "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                "match_text": "There is a fee to make this change.",
                                "verification_score": 0.64,
                            },
                            {
                                "anchor": "There is also a fare difference on the new itinerary, so you will either pay the extra amount or receive the balance as travel credit.",
                                "match_text": "The new flight is a bit higher.",
                                "verification_score": 0.41,
                            },
                        ],
                        "standard": [],
                    },
                },
            },
        }
    }

    text = demo_app._format_results(payload)

    assert "Rule 101: Identity verification" in text
    assert "Rule 102: Change of Booking" in text
    assert "| Disclosure 1: Change fee | Disclosure 2: Fare difference or travel credit |" in text
    assert "**Anchor:** Before I reset or unlock your account" in text
    assert "**Best text:** The new flight is a bit higher." in text


def test_format_agentic_summary_includes_bootstrap_and_holdout_metrics():
    summary = {
        "status": "completed",
        "message": "Loop finished.",
        "bootstrap": {
            "status": "ready",
            "message": "Gate passed.",
            "gate_metrics": {
                "count": 24,
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1": 0.95,
                "macro_f1": 0.95,
            },
        },
        "incoming_count": 2,
        "holdout_count": 3,
        "reviewed_count": 4,
        "coverage": {
            "candidate_case_count": 1,
            "analysis_count": 1,
            "new_rows_count": 3,
        },
        "holdout_metrics_before": {
            "count": 10,
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.9,
            "f1": 0.82,
            "macro_f1": 0.81,
        },
        "holdout_metrics_after": {
            "count": 10,
            "accuracy": 0.9,
            "precision": 0.88,
            "recall": 0.92,
            "f1": 0.90,
            "macro_f1": 0.90,
        },
        "promoted": True,
    }

    text = demo_app._format_agentic_summary(summary)

    assert "Bootstrap Gate" in text
    assert "Holdout Metrics After" in text
    assert "Coverage Augmentation" in text
    assert "`yes`" in text


def test_format_app_agentic_summary_renders_supervisor_table_with_collapsible_details():
    text = demo_app._format_app_agentic_summary(
        {
            "status": "awaiting_human_approval",
            "message": "Review ready.",
            "supervisor_summary": "Transcript 1: sample_helpdesk_bad",
            "before_payloads": [
                {
                    "transcript_id": "sample_helpdesk_bad",
                    "transcript": "Agent: I will unlock first.",
                    "results": {
                        "101": {
                            "status": "FAIL",
                            "evidence": {
                                "verification_score": 0.102,
                                "claims": {
                                    "single": [
                                        {
                                            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                                            "match_text": "I will unlock first.",
                                            "verification_score": 0.102,
                                        }
                                    ]
                                },
                            },
                        },
                        "102": {
                            "status": "PASS",
                            "evidence": {"verification_score": 0.901, "claims": {"mandatory": []}},
                        },
                    },
                }
            ],
        }
    )

    assert "<table class='supervisor-table'>" in text
    assert "<th>Transcript</th><th>Rule 101</th><th>Rule 102</th><th>PASS</th>" in text
    assert "<summary>Transcript 1: sample_helpdesk_bad</summary>" in text
    assert "<td><code>0.102</code>" in text
    assert "<td><code>0.901</code>" in text
    assert "<td>102</td>" in text
    assert "Show transcript text" in text


def test_format_regression_investigation_report_summarizes_root_cause():
    text = demo_app._format_regression_investigation_report(
        {
            "comparisons": [
                {
                    "disclaimer_id": "102",
                    "outcome": "regressed",
                    "before_score": 0.754,
                    "after_score": 0.7233,
                }
            ],
            "score_regression_diagnosis": {
                "analyses": [
                    {
                        "disclaimer_id": "102",
                        "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                        "phrase": "There is a seventy-five dollar change fee.",
                        "before_score": 0.754,
                        "after_score": 0.7233,
                        "target_direction": "higher",
                        "investigator_outcome": "under_represented_pattern",
                        "recommendation": "Generate two nearby pass variants for this anchor.",
                    }
                ]
            },
        }
    )

    assert "InvestigatorAgent Diagnosis" in text
    assert "0.7540" in text
    assert "0.7233" in text
    assert "under_represented_pattern" in text
    assert "Generate two nearby pass variants" in text
    assert "Dataset signal" not in text
    assert "Likely cause" not in text


def test_format_review_case_investigation_report_summarizes_dataset_coverage():
    text = demo_app._format_review_case_investigation_report(
        {
            "analyses": [
                {
                    "disclaimer_id": "101",
                    "label_change": "model Pass -> human Fail",
                    "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                    "phrase": "After I reset or unlock your account, I need to verify your identity first.",
                    "cause_tags": ["missing_coverage"],
                    "same_label_count": 0,
                    "opposite_label_count": 4,
                    "max_same_label_similarity": 0.11,
                    "investigator_rationale": "This internal rationale should stay out of the visible report.",
                    "solution_steps": ["Ask a DataGeneratorAgent to create 2-3 additional synthetic phrases for this anchor and human label."],
                    "generated_samples": [
                        "I unlock first, then verify identity afterward.",
                        "Access is restored before identity is checked.",
                    ],
                }
            ]
        }
    )

    assert "model Pass -> human Fail" in text
    assert "missing_coverage" in text
    assert "Synthetic coverage" not in text
    assert "same-label" not in text
    assert "DataGeneratorAgent" in text
    assert "Likely cause" not in text
    assert "LLM rationale" not in text
    assert "Generated samples appended to the approval queue" not in text
    assert "I unlock first" in text


def test_reset_base_models_button_wires_existing_service(monkeypatch):
    gr = pytest.importorskip("gradio")
    app = demo_app.build_demo_app("configs/demo.yaml")
    buttons = [component for component in app.blocks.values() if isinstance(component, gr.Button)]

    assert any(getattr(button, "value", None) == "Reset to Base Models" for button in buttons)
    assert any(getattr(button, "value", None) == "Investigate Failed Cases" for button in buttons)


def test_materialize_incoming_source_creates_temp_file_for_text():
    source, cleanup_paths, error = demo_app._materialize_incoming_source("Agent: hello world", None)

    try:
        assert error is None
        assert source is not None
        assert Path(source).exists()
    finally:
        for path in cleanup_paths:
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def test_materialize_incoming_source_cleans_gradio_folder_prefix(tmp_path):
    gradio_folder = tmp_path / "gradio_2afe1e4aa48ae66e487eba76139b9ba3c70003305b5f8bf48b4e9b6b88f88344"
    gradio_folder.mkdir()
    transcript_path = gradio_folder / "sample_travel_borderline_review.txt"
    transcript_path.write_text("Agent: There will be a fee.\nCustomer: Okay.", encoding="utf-8")

    source, cleanup_paths, error = demo_app._materialize_incoming_source("", None, [{"path": str(transcript_path)}])

    try:
        assert error is None
        assert source is not None
        copied_files = list(Path(source).glob("*.txt"))
        assert [path.stem for path in copied_files] == ["sample_travel_borderline_review"]
    finally:
        for path in cleanup_paths:
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def test_materialize_incoming_source_prefers_active_upload_when_stale_folder_exists(tmp_path):
    file_path = tmp_path / "fresh_file.txt"
    folder_path = tmp_path / "stale_folder_file.txt"
    file_path.write_text("Agent: fresh", encoding="utf-8")
    folder_path.write_text("Agent: stale", encoding="utf-8")

    source, cleanup_paths, error = demo_app._materialize_incoming_source(
        "",
        [{"path": str(file_path)}],
        [{"path": str(folder_path)}],
        active_source="file",
    )

    try:
        assert error is None
        copied_files = list(Path(source).glob("*.txt")) if source else []
        assert [path.name for path in copied_files] == ["fresh_file.txt"]
    finally:
        for path in cleanup_paths:
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def test_materialize_incoming_source_prefers_active_folder_when_stale_file_exists(tmp_path):
    file_path = tmp_path / "stale_file.txt"
    folder_path = tmp_path / "fresh_folder_file.txt"
    file_path.write_text("Agent: stale", encoding="utf-8")
    folder_path.write_text("Agent: fresh", encoding="utf-8")

    source, cleanup_paths, error = demo_app._materialize_incoming_source(
        "",
        [{"path": str(file_path)}],
        [{"path": str(folder_path)}],
        active_source="folder",
    )

    try:
        assert error is None
        copied_files = list(Path(source).glob("*.txt")) if source else []
        assert [path.name for path in copied_files] == ["fresh_folder_file.txt"]
    finally:
        for path in cleanup_paths:
            if path.exists() and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)


def test_materialize_holdout_source_rejects_non_json():
    path, error = demo_app._materialize_holdout_source([{"name": "/tmp/holdout.txt"}])

    assert path is None
    assert "`.json`" in error


def test_load_uploaded_transcript_accepts_single_txt(tmp_path):
    transcript_path = tmp_path / "script.txt"
    transcript_path.write_text("Agent: Hello\nCustomer: Hi", encoding="utf-8")

    transcript, status = demo_app._load_uploaded_transcript({"path": str(transcript_path)})

    assert transcript == "Agent: Hello\nCustomer: Hi"
    assert "Loaded uploaded script" in status


def test_load_uploaded_transcript_rejects_non_txt(tmp_path):
    transcript_path = tmp_path / "script.json"
    transcript_path.write_text("{}", encoding="utf-8")

    transcript, status = demo_app._load_uploaded_transcript({"path": str(transcript_path)})

    assert transcript == ""
    assert "`.txt`" in status
