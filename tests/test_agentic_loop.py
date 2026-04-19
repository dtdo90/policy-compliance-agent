import json

from sg_cases.agentic import loop as agentic_loop
from sg_cases.demo import services as demo_services


class GapAuditClient:
    def chat(self, *, system_prompt, user_prompt, temperature=0.0):
        return json.dumps(
            {
                "coverage_status": "gap",
                "positive_quality_note": "Positive examples are narrow.",
                "negative_quality_note": "Negative examples are useful but sparse.",
                "reason": "The reviewed phrase introduces a new case pattern.",
                "generated_variants": [
                    "A change fee will apply before I finalize this booking update.",
                    "You will need to pay the booking change fee before I confirm the new itinerary.",
                ],
            }
        )


def test_answer_agentic_question_uses_compact_no_think_prompt():
    class CaptureClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "<think>hidden</think>The retrain improved the reviewed score."

    client = CaptureClient()
    long_transcript = "Agent: " + "very long transcript text " * 300
    answer = agentic_loop.answer_agentic_question(
        "Summarize the run.",
        summary_payload={
            "status": "completed",
            "before_payloads": [{"transcript": long_transcript}],
            "review_items": [
                {
                    "disclaimer_id": "102",
                    "verification_score": 0.6889,
                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                    "text": "There will be a fee to make the change.",
                    "llm_label": "Compliant",
                }
            ],
            "comparisons": [
                {
                    "disclaimer_id": "102",
                    "final_label": "Compliant",
                    "before_score": 0.6889,
                    "after_score": 0.8503,
                    "target_direction": "higher",
                    "outcome": "improved",
                }
            ],
        },
        chat_history=[{"role": "user", "content": "Previous question."}],
        config=demo_services.load_demo_config(),
        client=client,
    )

    assert answer == "The retrain improved the reviewed score."
    assert client.calls[0]["num_predict"] == agentic_loop.AGENTIC_CHAT_NUM_PREDICT
    assert client.calls[0]["user_prompt"].startswith("/no_think")
    assert "Recent chat history" in client.calls[0]["user_prompt"]
    assert "Previous question." in client.calls[0]["user_prompt"]
    assert "very long transcript text" not in client.calls[0]["user_prompt"]
    assert "Full transcripts" in client.calls[0]["user_prompt"]


def test_answer_agentic_question_falls_back_when_reasoning_leaks():
    class ReasoningClient:
        def chat(self, **kwargs):
            return (
                "We are given a compact context and need to summarize the current agentic run.\n"
                "Rules to follow: start with the rule results.\n"
                "Let me draft the final answer."
            )

    answer = agentic_loop.answer_agentic_question(
        "Summarize the current agentic run.",
        summary_payload={
            "status": "completed",
            "message": "No borderline phrases were found.",
            "recommendation": "No training update is needed for this upload.",
            "borderline_count": 0,
            "before_payloads": [
                {
                    "transcript_id": "tx1",
                    "results": {
                        "101": {
                            "status": "FAIL",
                            "evidence": {
                                "claims": {
                                    "single": [
                                        {
                                            "claim_type": "single",
                                            "claim_idx": 0,
                                            "passed": False,
                                            "verification_score": 0.0,
                                        }
                                    ]
                                }
                            },
                        },
                        "102": {
                            "status": "FAIL",
                            "evidence": {
                                "claims": {
                                    "mandatory": [
                                        {
                                            "claim_type": "mandatory",
                                            "claim_idx": 1,
                                            "passed": True,
                                            "verification_score": 0.85,
                                        },
                                        {
                                            "claim_type": "mandatory",
                                            "claim_idx": 2,
                                            "passed": False,
                                            "verification_score": 0.14,
                                        },
                                    ]
                                }
                            },
                        },
                    },
                }
            ],
        },
        config=demo_services.load_demo_config(),
        client=ReasoningClient(),
    )

    assert "We are given" not in answer
    assert "Rule 101: FAIL" in answer
    assert "Rule 102: FAIL (Disclosure 1: PASS; Disclosure 2: FAIL)" in answer
    assert "Borderline phrases: none found." in answer


def test_collect_anchor_review_units_extracts_per_anchor():
    config = demo_services.load_demo_config(
        overrides={
            "agentic": {
                "review_categories": ["pass", "borderline"],
            }
        }
    )
    records = [
        {
            "dataset_role": "incoming",
            "transcript_id": "tx1",
            "results": {
                "102": {
                    "status": "FAIL",
                    "evidence": {
                        "description": "Travel rule",
                        "claims": {
                            "mandatory": [
                                {
                                    "claim_idx": 0,
                                    "claim_type": "mandatory",
                                    "passed": True,
                                    "anchor": "Fee applies.",
                                    "match_text": "There is a fee for the change.",
                                    "retrieval_score": 0.71,
                                    "verification_score": 0.92,
                                },
                                {
                                    "claim_idx": 1,
                                    "claim_type": "mandatory",
                                    "passed": False,
                                    "anchor": "Fare difference or credit applies.",
                                    "match_text": "The new flight is a bit higher.",
                                    "retrieval_score": 0.62,
                                    "verification_score": 0.41,
                                },
                                {
                                    "claim_idx": 2,
                                    "claim_type": "mandatory",
                                    "passed": False,
                                    "anchor": "Ignored fail anchor.",
                                    "match_text": "",
                                    "retrieval_score": 0.11,
                                    "verification_score": 0.01,
                                },
                            ]
                        },
                    },
                }
            },
        }
    ]

    units = agentic_loop.collect_anchor_review_units(records, config=config)

    assert len(units) == 2
    assert units[0]["review_bucket"] == "pass"
    assert units[1]["review_bucket"] == "borderline"


def test_coverage_augmentation_agent_adds_reviewed_phrase_and_generated_variants(tmp_path):
    config = demo_services.load_demo_config(
        overrides={
            "agentic": {
                "outputs_dir": str(tmp_path / "agentic"),
                "coverage_similarity_threshold": 0.99,
                "augmentation_variants_per_gap": 2,
            }
        }
    )
    agent = agentic_loop.CoverageAugmentationAgent(config=config, client=GapAuditClient())
    base_rows = [
        {
            "disclaimer_id": "102",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "dialogue": "There is an eighty dollar fee for making this change.",
            "type": "non-compliant",
        },
        {
            "disclaimer_id": "102",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "dialogue": "A change fee applies before I confirm the booking.",
            "type": "compliant",
        },
    ]
    reviewed_items = [
        {
            "review_id": "incoming:tx1:102:mandatory:0",
            "dataset_role": "incoming",
            "transcript_id": "tx1",
            "disclaimer_id": "102",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "text": "There will be a fee to make the change before I lock it in.",
            "verification_score": 0.34,
            "model_label": "Non-Compliant",
            "llm_label": "Compliant",
            "final_label": "Compliant",
        }
    ]

    result = agent.analyze_failures(
        reviewed_items=reviewed_items,
        raw_synthetic_rows=base_rows,
        existing_extension_rows=[],
        output_analysis_path=tmp_path / "analysis.json",
        output_extensions_path=tmp_path / "extensions.json",
    )

    assert result["candidate_case_count"] == 1
    assert result["new_rows_count"] == 3
    assert len(result["all_extension_rows"]) == 3
    assert result["analyses"][0]["coverage_status"] == "gap"


def test_training_gate_agent_blocks_and_runs_quality_audit(tmp_path, monkeypatch):
    synthetic_path = tmp_path / "synthetic.json"
    synthetic_path.write_text(
        json.dumps(
            [
                {
                    "disclaimer_id": "101",
                    "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                    "dialogue": "I need to verify you first before I unlock it.",
                    "type": "compliant",
                },
                {
                    "disclaimer_id": "101",
                    "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                    "dialogue": "I'll unlock it first and verify later.",
                    "type": "non-compliant",
                },
            ]
        ),
        encoding="utf-8",
    )
    config = demo_services.load_demo_config(
        overrides={
            "data": {
                "synthetic_dataset_path": str(synthetic_path),
            },
            "agentic": {
                "outputs_dir": str(tmp_path / "agentic"),
                "bootstrap_mode": "always",
                "training_gate_threshold": 0.9,
            },
        }
    )
    paths = agentic_loop._agentic_paths(config)
    gate_agent = agentic_loop.TrainingGateAgent(config=config)
    coverage_agent = agentic_loop.CoverageAugmentationAgent(config=config, client=GapAuditClient())
    audit_calls = {}

    monkeypatch.setattr(agentic_loop, "_fit_cross_encoder_rows", lambda train_rows, config, output_dir, base_model_name_or_path=None: tmp_path / "gate-model")
    monkeypatch.setattr(
        agentic_loop,
        "_evaluate_cross_encoder_rows",
        lambda model_path, eval_rows: {
            "macro_f1": 0.42,
            "f1": 0.40,
            "per_anchor": {
                "101|Before I reset or unlock your account, I need to verify your identity first.": {
                    "disclaimer_id": "101",
                    "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                    "count": 1,
                    "f1": 0.2,
                    "examples": [],
                }
            },
        },
    )
    monkeypatch.setattr(agentic_loop, "train_sentence_transformer", lambda config=None: (_ for _ in ()).throw(AssertionError("should not train")))
    monkeypatch.setattr(agentic_loop, "train_cross_encoder", lambda config=None: (_ for _ in ()).throw(AssertionError("should not train")))

    def fake_audit(**kwargs):
        audit_calls["called"] = True
        return {"status": "quality_audited", "anchors_below_threshold": 1}

    monkeypatch.setattr(coverage_agent, "audit_synthetic_quality", fake_audit)

    result = gate_agent.run(coverage_agent=coverage_agent, paths=paths)

    assert result["status"] == "blocked"
    assert audit_calls["called"] is True


def test_run_local_agentic_loop_with_gate_skip_and_coverage_retrain(tmp_path, monkeypatch):
    incoming_path = tmp_path / "incoming.json"
    incoming_path.write_text(
        json.dumps(
            [
                {"transcript_id": "new1", "transcript": "Agent: uploaded text one"},
                {"transcript_id": "new2", "transcript": "Agent: uploaded text two"},
            ]
        ),
        encoding="utf-8",
    )
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(
        json.dumps(
            [
                {
                    "transcript_id": "holdout1",
                    "transcript": "Agent: holdout text one",
                    "expected_labels": [
                        {
                            "disclaimer_id": "102",
                            "claim_type": "mandatory",
                            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                            "label": "Compliant",
                        }
                    ],
                },
                {
                    "transcript_id": "holdout2",
                    "transcript": "Agent: holdout text two",
                    "expected_labels": [
                        {
                            "disclaimer_id": "102",
                            "claim_type": "mandatory",
                            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                            "label": "Non-Compliant",
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    config = demo_services.load_demo_config(
        overrides={
            "agentic": {
                "incoming_source": str(incoming_path),
                "holdout_source": str(holdout_path),
                "outputs_dir": str(tmp_path / "agentic"),
                "bootstrap_mode": "never",
                "auto_approve_llm": True,
                "review_categories": ["pass", "borderline", "fail"],
            }
        }
    )

    def fake_run_demo_inference(transcript, *, transcript_id="interactive_demo", config=None, config_path=None, analyzer=None):
        if transcript_id.startswith("holdout"):
            score = 0.20
            if config and str(config.get("models", {}).get("cross_encoder_path", "")).endswith("candidate-model"):
                score = 0.82 if transcript_id == "holdout1" else 0.18
        else:
            score = 0.88 if transcript_id == "new1" else 0.41
        passed = score >= 0.5
        return {
            "transcript_id": transcript_id,
            "results": {
                "102": {
                    "status": "PASS" if passed else "FAIL",
                    "evidence": {
                        "description": "Travel rule",
                        "claims": {
                            "mandatory": [
                                {
                                    "claim_idx": 0,
                                    "claim_type": "mandatory",
                                    "passed": passed,
                                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                    "match_text": "There will be a fee to make the change.",
                                    "retrieval_score": 0.6,
                                    "verification_score": score,
                                }
                            ]
                        },
                    },
                }
            },
            "borderline_items": [],
        }

    def fake_load_demo_analyzer(config):
        return object()

    def fake_label_review_items(items, *, config=None, config_path=None, client=None):
        labeled = []
        for item in items:
            label = "Compliant" if item["transcript_id"] == "new2" else item["model_label"]
            labeled.append(
                {
                    **item,
                    "llm_label": label,
                    "llm_confidence": 0.66,
                    "llm_rationale": "Reviewed against the anchor.",
                }
            )
        return labeled

    candidate_path = tmp_path / "candidate-model"
    candidate_path.mkdir()
    (candidate_path / "config.json").write_text("{}", encoding="utf-8")
    active_path = tmp_path / "active-model"
    active_path.mkdir()
    (active_path / "config.json").write_text("{}", encoding="utf-8")

    def fake_retrain(extension_rows):
        return {
            "status": "trained",
            "message": "ok",
            "approved_count": len(list(extension_rows or [])),
            "promotion_recommended": True,
            "promoted": False,
            "candidate_model_path": str(candidate_path),
            "active_model_path": str(active_path),
            "augmented_dataset_path": str(tmp_path / "augmented.json"),
            "metrics_before": {"macro_f1": 0.5},
            "metrics_after": {"macro_f1": 0.8},
        }

    def fake_coverage(self, **kwargs):
        all_rows = [
            {
                "disclaimer_id": "102",
                "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                "dialogue": "There will be a fee to make the change.",
                "type": "compliant",
                "source": "reviewed_phrase",
            }
        ]
        agentic_loop._json_save(kwargs["output_analysis_path"], [{"coverage_status": "covered"}])
        agentic_loop._json_save(kwargs["output_extensions_path"], all_rows)
        return {
            "candidate_case_count": 1,
            "analysis_count": 1,
            "new_rows_count": 1,
            "all_extension_rows": all_rows,
            "analyses": [{"coverage_status": "covered"}],
        }

    monkeypatch.setattr(agentic_loop.demo_services, "_load_demo_analyzer", fake_load_demo_analyzer)
    monkeypatch.setattr(agentic_loop.demo_services, "run_demo_inference", fake_run_demo_inference)
    monkeypatch.setattr(agentic_loop.demo_services, "label_review_items_with_ollama", fake_label_review_items)
    monkeypatch.setattr(agentic_loop.TrainingGateAgent, "run", lambda self, coverage_agent, paths: {"status": "skipped", "message": "skip"})
    monkeypatch.setattr(agentic_loop.CoverageAugmentationAgent, "analyze_failures", fake_coverage)
    monkeypatch.setattr(agentic_loop.TrainerAgent, "retrain", lambda self, extension_rows: fake_retrain(extension_rows))

    summary = agentic_loop.run_local_agentic_loop(config=config)

    assert summary["status"] == "completed"
    assert summary["incoming_count"] == 2
    assert summary["holdout_count"] == 2
    assert summary["coverage"]["new_rows_count"] == 1
    assert summary["promoted"] is True
    assert summary["holdout_metrics_after"]["macro_f1"] >= summary["holdout_metrics_before"]["macro_f1"]
    assert (tmp_path / "agentic" / "loop_summary.json").exists()
