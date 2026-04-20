import json
from pathlib import Path

from policy_compliance_agent.demo import services as demo_services


class FakeOllamaClient:
    def __init__(self):
        self.calls = []

    def chat(self, *, system_prompt, user_prompt, temperature=0.0, json_mode=False, num_predict=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "json_mode": json_mode,
                "num_predict": num_predict,
            }
        )
        return json.dumps(
            {
                "label": "Compliant",
                "confidence": 0.82,
                "rationale": "The phrase clearly states the required sequence.",
            }
        )


def test_get_borderline_items_uses_exact_band():
    result = {
        "transcript_id": "tx1",
        "results": {
            "101": {
                "status": "FAIL",
                "for_review": [
                    {
                        "text": "verify before reset",
                        "retrieval_score": 0.5,
                        "verification_score": 0.30,
                        "claim_type": "single",
                        "claim_text": "Before I reset or unlock your account, I need to verify your identity first.",
                    },
                    {
                        "text": "too high",
                        "retrieval_score": 0.5,
                        "verification_score": 0.70,
                        "claim_type": "single",
                        "claim_text": "Before I reset or unlock your account, I need to verify your identity first.",
                    },
                ],
            }
        },
    }

    items = demo_services.get_borderline_items(result, config=demo_services.load_demo_config())

    assert len(items) == 1
    assert items[0]["text"] == "verify before reset"


def test_get_borderline_items_prefers_best_claim_evidence_over_extra_top_k_candidates():
    result = {
        "transcript_id": "tx1",
        "results": {
            "102": {
                "status": "FAIL",
                "evidence": {
                    "claims": {
                        "single": [],
                        "mandatory": [
                            {
                                "claim_type": "mandatory",
                                "claim_idx": 0,
                                "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                "match_text": "There is a change fee.",
                                "retrieval_score": 0.8,
                                "verification_score": 0.483,
                            }
                        ],
                        "standard": [],
                    }
                },
                "for_review": [
                    {
                        "text": "older top-k candidate",
                        "retrieval_score": 0.5,
                        "verification_score": 0.3504,
                        "claim_type": "mandatory",
                        "claim_text": "Before I confirm this booking change, there is a change fee that will apply.",
                    }
                ],
            }
        },
    }

    items = demo_services.get_borderline_items(result, config=demo_services.load_demo_config())

    assert len(items) == 1
    assert items[0]["text"] == "There is a change fee."
    assert items[0]["verification_score"] == 0.483


def test_run_demo_inference_preserves_transcript_id_on_borderline_items():
    class FakeAnalyzer:
        def analyze_transcript(self, transcript, transcript_id):
            return {
                "102": {
                    "status": "FAIL",
                    "evidence": {
                        "claims": {
                            "mandatory": [
                                {
                                    "claim_type": "mandatory",
                                    "claim_idx": 0,
                                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                    "match_text": "There will be a fee to make the change.",
                                    "retrieval_score": 0.7,
                                    "verification_score": 0.48,
                                }
                            ]
                        }
                    },
                }
            }

    payload = demo_services.run_demo_inference(
        "Agent: There will be a fee to make the change.",
        transcript_id="uploaded_script_1",
        config=demo_services.load_demo_config(),
        analyzer=FakeAnalyzer(),
    )

    assert payload["borderline_items"][0]["transcript_id"] == "uploaded_script_1"


def test_agentic_review_summary_lists_rules_before_borderline(monkeypatch):
    class FakeAnalyzer:
        def analyze_transcript(self, transcript, transcript_id):
            return {
                "101": {
                    "status": "FAIL",
                    "evidence": {
                        "claims": {
                            "single": [
                                {
                                    "claim_type": "single",
                                    "claim_idx": 0,
                                    "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                                    "match_text": "I can unlock it now.",
                                    "verification_score": 0.1,
                                }
                            ],
                            "mandatory": [],
                            "standard": [],
                        }
                    },
                },
                "102": {
                    "status": "FAIL",
                    "evidence": {
                        "claims": {
                            "single": [],
                            "mandatory": [
                                {
                                    "claim_type": "mandatory",
                                    "claim_idx": 0,
                                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                    "match_text": "There will be a fee to make the change.",
                                    "verification_score": 0.62,
                                }
                            ],
                            "standard": [],
                        }
                    },
                },
            }

    def fake_label(items, **kwargs):
        labeled = [dict(item) for item in items]
        labeled[0]["llm_label"] = "Compliant"
        return labeled

    monkeypatch.setattr(demo_services, "_load_demo_analyzer", lambda config: FakeAnalyzer())
    monkeypatch.setattr(demo_services, "label_review_items_with_ollama", fake_label)

    result = demo_services.run_agentic_review_cycle(
        [{"transcript_id": "tx1", "transcript": "Agent: There will be a fee to make the change."}],
        config=demo_services.load_demo_config(),
    )

    summary = result["supervisor_summary"]
    assert summary.index("Transcript 1: tx1") < summary.index("Rule 101: FAIL")
    assert "Rule 101: FAIL" in summary
    assert "Rule 102: FAIL" in summary
    assert "SupervisorAgent prepared" not in summary
    assert "SupervisorAgent prepared 1 phrase" in result["recommendation"]
    assert "Review item 1" not in summary
    assert "Review the labels" not in summary
    assert "Anchor 1:" not in summary
    assert "Borderline evidence:" not in summary
    assert "Evidence reviewed:" not in summary


def test_agentic_review_includes_pass_and_borderline_phrases(monkeypatch):
    class FakeAnalyzer:
        def analyze_transcript(self, transcript, transcript_id):
            return {
                "102": {
                    "status": "FAIL",
                    "evidence": {
                        "claims": {
                            "single": [],
                            "mandatory": [
                                {
                                    "claim_type": "mandatory",
                                    "claim_idx": 0,
                                    "passed": True,
                                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                    "match_text": "There will be a fee to make the change.",
                                    "verification_score": 0.92,
                                },
                                {
                                    "claim_type": "mandatory",
                                    "claim_idx": 1,
                                    "passed": True,
                                    "anchor": "There is also a fare difference on the new itinerary, so you will either pay the extra amount or receive the balance as travel credit.",
                                    "match_text": "The new flight is a bit higher.",
                                    "verification_score": 0.62,
                                },
                            ],
                            "standard": [],
                        }
                    },
                }
            }

    def fake_label(items, **kwargs):
        labeled = [dict(item) for item in items]
        for item in labeled:
            item["llm_label"] = "Compliant"
        return labeled

    monkeypatch.setattr(demo_services, "_load_demo_analyzer", lambda config: FakeAnalyzer())
    monkeypatch.setattr(demo_services, "label_review_items_with_ollama", fake_label)

    result = demo_services.run_agentic_review_cycle(
        [{"transcript_id": "tx1", "transcript": "Agent: There will be a fee."}],
        config=demo_services.load_demo_config(),
    )

    assert result["pass_review_count"] == 1
    assert result["borderline_count"] == 1
    assert [item["review_type"] for item in result["review_items"]] == ["pass", "borderline"]
    assert result["review_items"][0].get("llm_label", "") == ""
    assert result["review_items"][1]["llm_label"] == "Compliant"
    assert "Anchor 1:" in result["supervisor_summary"]
    assert "Anchor 2:" in result["supervisor_summary"]


def test_label_borderline_items_with_ollama_uses_client():
    items = [
        {
            "disclaimer_id": "101",
            "description": "Support agent requires identity verification before resetting or unlocking the account.",
            "claim_type": "single",
            "rubric": "Verification before reset",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "text": "I need to verify you before I unlock it.",
            "verification_score": 0.52,
        }
    ]
    client = FakeOllamaClient()

    labeled = demo_services.label_borderline_items_with_ollama(items, config=demo_services.load_demo_config(), client=client)

    assert len(labeled) == 1
    assert labeled[0]["llm_label"] == "Compliant"
    assert client.calls
    assert "specific target anchor only" in client.calls[0]["system_prompt"]
    assert "target_anchor" in client.calls[0]["user_prompt"]
    assert client.calls[0]["json_mode"] is True
    assert client.calls[0]["num_predict"] == demo_services.LABEL_NUM_PREDICT


def test_ollama_chat_client_sends_think_false_and_returns_content_only(monkeypatch):
    sent_payloads = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "thinking": "hidden chain of thought",
                        "content": "Final answer only.",
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        sent_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(demo_services, "urlopen", fake_urlopen)

    client = demo_services.OllamaChatClient(
        base_url="http://127.0.0.1:11434",
        model="qwen3:4b",
        think=False,
    )
    answer = client.chat(system_prompt="system", user_prompt="user", num_predict=64)

    assert answer == "Final answer only."
    assert sent_payloads[0]["think"] is False
    assert sent_payloads[0]["stream"] is False
    assert sent_payloads[0]["options"]["num_predict"] == 64
    assert "hidden chain of thought" not in answer


def test_label_borderline_items_with_ollama_can_override_to_compliant_for_clear_anchor_match():
    class NonCompliantClient:
        def chat(self, *, system_prompt, user_prompt, temperature=0.0, json_mode=False, num_predict=None):
            return json.dumps(
                {
                    "label": "Non-Compliant",
                    "confidence": 0.95,
                    "rationale": "Too strict.",
                }
            )

    items = [
        {
            "disclaimer_id": "102",
            "claim_type": "mandatory",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "text": "There will be a fee to make the change, and the new flight is a bit higher.",
            "verification_score": 0.38,
        }
    ]

    labeled = demo_services.label_borderline_items_with_ollama(
        items,
        config=demo_services.load_demo_config(),
        client=NonCompliantClient(),
    )

    assert labeled[0]["llm_label"] == "Compliant"
    assert "fee applies for making the booking change" in labeled[0]["llm_rationale"]


def test_label_borderline_items_with_ollama_falls_back_when_qwen_returns_reasoning():
    class ReasoningClient:
        def chat(self, *, system_prompt, user_prompt, temperature=0.0, json_mode=False, num_predict=None):
            return (
                'We are given target_anchor: "Before I confirm this booking change, there is a change fee that will apply." '
                'candidate_phrase: "There will be a fee to make the change, and the new flight is a bit higher." '
                "The task is to judge the candidate phrase against the anchor only."
            )

    items = [
        {
            "disclaimer_id": "102",
            "claim_type": "mandatory",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "text": "There will be a fee to make the change, and the new flight is a bit higher. The airline is",
            "verification_score": 0.3872,
        }
    ]

    labeled = demo_services.label_borderline_items_with_ollama(
        items,
        config=demo_services.load_demo_config(),
        client=ReasoningClient(),
    )

    assert labeled[0]["llm_label"] == "Compliant"
    assert labeled[0]["llm_confidence"] == 0.70
    assert "fee applies for making the booking change" in labeled[0]["llm_rationale"]


def test_answer_demo_question_handles_greeting_without_llm_call():
    class FailingClient:
        def chat(self, **kwargs):
            raise AssertionError("Greeting should not call Ollama")

    answer = demo_services.answer_demo_question("hi", config=demo_services.load_demo_config(), client=FailingClient())

    assert "Hi." in answer


def test_answer_demo_question_strips_think_blocks_and_caps_tokens():
    class ThinkingClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "<think>long hidden reasoning</think>Short answer."

    client = ThinkingClient()
    answer = demo_services.answer_demo_question(
        "summarize",
        inference_payload={"results": {}},
        config=demo_services.load_demo_config(),
        client=client,
    )

    assert answer == "Short answer."
    assert client.calls[0]["num_predict"] == demo_services.CHAT_NUM_PREDICT
    assert client.calls[0]["user_prompt"].startswith("/no_think")
    assert client.calls[0]["user_prompt"].rstrip().endswith("Answer directly. Do not show reasoning.")


def test_answer_demo_question_handles_no_borderline_without_llm_call():
    class FailingClient:
        def chat(self, **kwargs):
            raise AssertionError("No-borderline summary should not call Ollama")

    answer = demo_services.answer_demo_question(
        "Explain the current borderline phrases.",
        inference_payload={
            "inference": {
                "results": {
                    "101": {"status": "PASS", "evidence": {"verification_score": 0.91}},
                    "102": {"status": "FAIL", "evidence": {"verification_score": 0.20}},
                }
            },
            "review_items": [],
        },
        config=demo_services.load_demo_config(),
        client=FailingClient(),
    )

    assert "There are no borderline phrases" in answer
    assert "Rule 101: PASS, score 0.910" in answer
    assert "Rule 102: FAIL, score 0.200" in answer
    assert "\nRule 101" in answer


def test_answer_demo_question_summarizes_current_inference_with_llm_call():
    class SummaryClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "Both rules failed, with Rule 102 partially satisfying one disclosure."

    client = SummaryClient()
    answer = demo_services.answer_demo_question(
        "Summarize the current inference.",
        inference_payload={
            "inference": {
                "results": {
                    "101": {
                        "status": "FAIL",
                        "evidence": {
                            "verification_score": 0.0,
                            "claims": {
                                "single": [{"passed": False}],
                                "mandatory": [],
                                "standard": [],
                            },
                        },
                    },
                    "102": {
                        "status": "FAIL",
                        "evidence": {
                            "verification_score": 0.245,
                            "claims": {
                                "single": [],
                                "mandatory": [{"passed": True}, {"passed": False}],
                                "standard": [],
                            },
                        },
                    },
                }
            },
            "review_items": [],
        },
        config=demo_services.load_demo_config(),
        client=client,
    )

    assert answer == "Both rules failed, with Rule 102 partially satisfying one disclosure."
    assert client.calls
    assert client.calls[0]["num_predict"] == demo_services.CHAT_NUM_PREDICT
    assert client.calls[0]["user_prompt"].startswith("/no_think")


def test_answer_demo_question_suggests_final_labels_without_llm_call():
    class FailingClient:
        def chat(self, **kwargs):
            raise AssertionError("Final-label quick prompt should not call Ollama")

    answer = demo_services.answer_demo_question(
        "Suggest final labels for borderline phrases.",
        inference_payload={
            "inference": {"results": {}},
            "review_items": [
                {
                    "disclaimer_id": "102",
                    "verification_score": 0.6889,
                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                    "text": "There will be a fee to make the change, and the new flight is a bit higher.",
                    "llm_label": "Compliant",
                    "llm_rationale": "The phrase states the change fee applies.",
                }
            ],
        },
        config=demo_services.load_demo_config(),
        client=FailingClient(),
    )

    assert "Suggested final labels" in answer
    assert "Rule 102: Pass" in answer
    assert "Qwen label: Pass" in answer
    assert "Compliant" not in answer


def test_answer_demo_question_falls_back_when_qwen_leaks_reasoning():
    class ReasoningClient:
        def chat(self, **kwargs):
            return "Okay, the user is asking for a summary. Let me check the current context first."

    answer = demo_services.answer_demo_question(
        "Summarize the current inference.",
        inference_payload={
            "inference": {
                "results": {
                    "101": {"status": "FAIL", "evidence": {"verification_score": 0.0}},
                    "102": {"status": "PASS", "evidence": {"verification_score": 0.91}},
                }
            },
            "review_items": [],
        },
        config=demo_services.load_demo_config(),
        client=ReasoningClient(),
    )

    assert answer == "Rule 101: FAIL, score 0.000\nRule 102: PASS, score 0.910."
    assert "Okay" not in answer
    assert "Let me" not in answer


def test_answer_demo_question_includes_recent_chat_history():
    class HistoryClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "Rule 102 has the lower score."

    client = HistoryClient()
    answer = demo_services.answer_demo_question(
        "Which anchor receives a lower verification_score?",
        inference_payload={"inference": {"results": {}}},
        chat_history=[
            {"role": "user", "content": "Summarize the current inference."},
            {"role": "assistant", "content": "<think>hidden</think>Rule 101 passed and Rule 102 failed."},
        ],
        config=demo_services.load_demo_config(),
        client=client,
    )

    user_prompt = client.calls[0]["user_prompt"]
    assert answer == "Rule 102 has the lower score."
    assert "Recent chat history" in user_prompt
    assert "Summarize the current inference." in user_prompt
    assert "Rule 101 passed and Rule 102 failed." in user_prompt
    assert "hidden" not in user_prompt
    assert "Which anchor receives a lower verification_score?" in user_prompt


def test_answer_demo_question_uses_compact_context_without_full_transcript():
    class CaptureClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "Rule 102 has one borderline claim."

    long_transcript = "Agent: " + "very long transcript text " * 200
    long_match = "This best evidence sentence should be compacted " * 30
    client = CaptureClient()

    demo_services.answer_demo_question(
        "What is the result?",
        inference_payload={
            "assistant_context": "Use the current result only.",
            "inference": {
                "transcript": long_transcript,
                "results": {
                    "102": {
                        "status": "FAIL",
                        "evidence": {
                            "description": "Travel change disclosure.",
                            "verification_score": 0.245,
                            "claims": {
                                "single": [],
                                "mandatory": [
                                    {
                                        "claim_type": "mandatory",
                                        "claim_idx": 0,
                                        "passed": True,
                                        "verification_score": 0.6889,
                                        "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                        "match_text": long_match,
                                    }
                                ],
                                "standard": [],
                            },
                        },
                    }
                },
            },
            "review_items": [
                {
                    "disclaimer_id": "102",
                    "verification_score": 0.6889,
                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                    "text": long_match,
                    "llm_label": "Compliant",
                }
            ],
        },
        config=demo_services.load_demo_config(),
        client=client,
    )

    user_prompt = client.calls[0]["user_prompt"]
    assert "very long transcript text" not in user_prompt
    assert "Full transcript and raw top-k payload are intentionally omitted" in user_prompt
    assert len(user_prompt) < 3500
    assert "Rule 102" not in user_prompt
    assert '"rule_id": "102"' in user_prompt
    assert long_match not in user_prompt


def test_approve_demo_examples_dedupes_and_tracks_readiness(tmp_path):
    config = demo_services.load_demo_config(
        overrides={
            "outputs": {
                "approved_examples_json_path": str(tmp_path / "approved.json"),
            },
            "demo": {
                "min_approved_examples_for_retrain": 2,
            },
        }
    )

    summary = demo_services.approve_demo_examples(
        [
            {
                "approved": True,
                "disclaimer_id": "101",
                "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                "text": "I need to verify your identity before I unlock it.",
                "llm_label": "Compliant",
                "verification_score": 0.44,
                "retrieval_score": 0.55,
            },
            {
                "approved": True,
                "disclaimer_id": "101",
                "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                "text": "I need to verify your identity before I unlock it.",
                "llm_label": "Compliant",
                "verification_score": 0.44,
                "retrieval_score": 0.55,
            },
            {
                "approved": True,
                "disclaimer_id": "102",
                "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                "text": "There is a change fee before I confirm the switch.",
                "llm_label": "Non-Compliant",
                "verification_score": 0.49,
                "retrieval_score": 0.51,
            },
        ],
        config=config,
    )

    assert summary["added_count"] == 2
    assert summary["approved_count"] == 2
    assert summary["ready_to_retrain"] is True


def test_approve_demo_examples_saves_anchor_phrase_and_model_pass_fallback(tmp_path):
    approved_path = tmp_path / "approved.json"
    config = demo_services.load_demo_config(
        overrides={
            "outputs": {
                "approved_examples_json_path": str(approved_path),
            },
            "demo": {
                "min_approved_examples_for_retrain": 1,
                "require_both_labels_for_retrain": False,
            },
        }
    )

    anchor = "Before I confirm this booking change, there is a change fee that will apply."
    phrase = "There will be a fee to make the change, and the new flight is higher."
    summary = demo_services.approve_demo_examples(
        [
            {
                "approved": True,
                "disclaimer_id": "102",
                "anchor": anchor,
                "text": phrase,
                "model_label": "Pass",
                "final_label": "",
                "verification_score": 0.754,
                "retrieval_score": 0.82,
            }
        ],
        config=config,
    )

    stored = json.loads(approved_path.read_text(encoding="utf-8"))
    assert summary["added_count"] == 1
    assert stored[0]["anchor"] == anchor
    assert stored[0]["dialogue"] == phrase
    assert stored[0]["phrase"] == phrase
    assert stored[0]["type"] == "compliant"
    assert stored[0]["label"] == "Pass"
    assert stored[0]["training_label"] == "Compliant"
    assert stored[0]["model_label"] == "Compliant"


def test_approve_demo_examples_replace_existing_keeps_latest_run_only(tmp_path):
    approved_path = tmp_path / "approved.json"
    old_anchor = "Before I reset or unlock your account, I need to verify your identity first."
    approved_path.write_text(
        json.dumps(
            [
                {
                    "disclaimer_id": "101",
                    "anchor": old_anchor,
                    "dialogue": "Old phrase from an earlier run.",
                    "type": "compliant",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = demo_services.load_demo_config(
        overrides={
            "outputs": {
                "approved_examples_json_path": str(approved_path),
            },
            "demo": {
                "min_approved_examples_for_retrain": 1,
                "require_both_labels_for_retrain": False,
            },
        }
    )

    new_anchor = "Before I confirm this booking change, there is a change fee that will apply."
    demo_services.approve_demo_examples(
        [
            {
                "approved": True,
                "disclaimer_id": "102",
                "anchor": new_anchor,
                "text": "There is a change fee for this booking update.",
                "final_label": "Pass",
            }
        ],
        config=config,
        replace_existing=True,
    )

    stored = json.loads(approved_path.read_text(encoding="utf-8"))
    assert len(stored) == 1
    assert stored[0]["disclaimer_id"] == "102"
    assert stored[0]["anchor"] == new_anchor
    assert stored[0]["dialogue"] == "There is a change fee for this booking update."
    assert "Old phrase" not in json.dumps(stored)


def test_retrain_demo_verifier_blocks_until_threshold(tmp_path):
    approved_path = tmp_path / "approved.json"
    approved_path.write_text(
        json.dumps(
            [
                {
                    "disclaimer_id": "101",
                    "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                    "dialogue": "I need to verify you first.",
                    "type": "compliant",
                }
            ]
        ),
        encoding="utf-8",
    )
    config = demo_services.load_demo_config(
        overrides={
            "outputs": {
                "approved_examples_json_path": str(approved_path),
            },
            "demo": {
                "min_approved_examples_for_retrain": 2,
            },
        }
    )

    result = demo_services.retrain_demo_verifier(config=config)

    assert result["status"] == "blocked"


def test_retrain_demo_verifier_promotes_better_model(tmp_path, monkeypatch):
    synthetic_path = tmp_path / "synthetic.json"
    eval_path = tmp_path / "eval.json"
    approved_path = tmp_path / "approved.json"
    active_retriever_path = tmp_path / "models" / "retriever-active"
    active_model_path = tmp_path / "models" / "verifier-active"
    baseline_retriever_path = tmp_path / "models" / "baseline-retriever"
    baseline_verifier_path = tmp_path / "models" / "baseline-verifier"
    retriever_versions_dir = tmp_path / "models" / "retriever-versions"
    versions_dir = tmp_path / "models" / "verifier-versions"
    retriever_monitor_dir = tmp_path / "retriever-monitor"
    monitor_dir = tmp_path / "verifier-monitor"

    base_rows = [
        {
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "dialogue": "I need to verify your identity before I reset it.",
            "type": "compliant",
        }
    ]
    eval_rows = [
        {
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "dialogue": "I need to verify your identity before I reset it.",
            "type": "compliant",
        },
        {
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "dialogue": "I'll reset it now and verify you later.",
            "type": "non-compliant",
        }
    ]
    approved_rows = [
        {
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "dialogue": "I must verify you first before I unlock it.",
            "type": "compliant",
        },
        {
            "disclaimer_id": "102",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "dialogue": "There is a fee before I confirm it.",
            "type": "non-compliant",
        }
    ]
    synthetic_path.write_text(json.dumps(base_rows), encoding="utf-8")
    eval_path.write_text(json.dumps(eval_rows), encoding="utf-8")
    approved_path.write_text(json.dumps(approved_rows), encoding="utf-8")
    active_retriever_path.mkdir(parents=True)
    (active_retriever_path / "config.json").write_text("{}", encoding="utf-8")
    active_model_path.mkdir(parents=True)
    (active_model_path / "config.json").write_text("{}", encoding="utf-8")
    baseline_retriever_path.mkdir(parents=True)
    (baseline_retriever_path / "config.json").write_text("{}", encoding="utf-8")
    baseline_verifier_path.mkdir(parents=True)
    (baseline_verifier_path / "config.json").write_text("{}", encoding="utf-8")

    config = demo_services.load_demo_config(
        overrides={
            "data": {
                "synthetic_dataset_path": str(synthetic_path),
                "eval_dataset_path": str(eval_path),
            },
            "outputs": {
                "approved_examples_json_path": str(approved_path),
                "augmented_dataset_path": str(tmp_path / "augmented.json"),
                "retriever_versions_dir": str(retriever_versions_dir),
                "verifier_versions_dir": str(versions_dir),
                "retriever_retrain_monitor_output_dir": str(retriever_monitor_dir),
                "retrain_monitor_output_dir": str(monitor_dir),
            },
            "models": {
                    "bi_encoder_path": str(active_retriever_path),
                    "baseline_bi_encoder_path": str(baseline_retriever_path),
                    "sentence_transformer_base": "sentence-transformers/all-MiniLM-L6-v2",
                    "cross_encoder_path": str(active_model_path),
                    "baseline_cross_encoder_path": str(baseline_verifier_path),
                    "cross_encoder_base": "cross-encoder/ms-marco-MiniLM-L12-v2",
                },
            "demo": {
                "min_approved_examples_for_retrain": 2,
            },
            "training": {
                "use_extra_sampling": False,
            },
        }
    )

    train_calls = []

    def fake_train(**kwargs):
        train_calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text("{}", encoding="utf-8")
        return output_dir

    def fake_eval(model_path, eval_rows):
        if Path(model_path) == active_model_path:
            return {"average_precision": 0.40, "accuracy": 0.50, "count": len(eval_rows)}
        return {"average_precision": 0.65, "accuracy": 0.75, "count": len(eval_rows)}

    monkeypatch.setattr(demo_services, "train_sentence_transformer_with_overrides", fake_train)
    monkeypatch.setattr(demo_services, "train_cross_encoder_with_overrides", fake_train)
    monkeypatch.setattr(demo_services, "_evaluate_cross_encoder", fake_eval)

    result = demo_services.retrain_demo_verifier(config=config)

    assert result["status"] == "trained"
    assert result["dataset_policy"] == "original_synthetic_plus_latest_approved_phrases"
    assert result["promoted"] is True
    assert active_retriever_path.exists()
    assert (active_retriever_path / "config.json").exists()
    assert active_model_path.exists()
    assert (active_model_path / "config.json").exists()
    assert len(train_calls) == 2
    assert all(Path(call["dataset_path"]) == tmp_path / "augmented.json" for call in train_calls)
    augmented_rows = json.loads((tmp_path / "augmented.json").read_text(encoding="utf-8"))
    assert any(row["dialogue"] == "I must verify you first before I unlock it." for row in augmented_rows)
    assert any(row["dialogue"] == "There is a fee before I confirm it." for row in augmented_rows)
    assert train_calls[0]["base_model_name_or_path"] == str(baseline_retriever_path)
    assert train_calls[1]["base_model_name_or_path"] == str(baseline_verifier_path)
    assert result["retriever_training_base"] == str(baseline_retriever_path)
    assert result["verifier_training_base"] == str(baseline_verifier_path)


def test_retrain_base_model_prefers_synthetic_baseline_snapshot(tmp_path):
    baseline_path = tmp_path / "baseline-model"
    baseline_path.mkdir()

    selected = demo_services._retrain_base_model(
        {
            "baseline_cross_encoder_path": str(baseline_path),
            "cross_encoder_base": "cross-encoder/raw-base",
        },
        baseline_key="baseline_cross_encoder_path",
        raw_base_key="cross_encoder_base",
    )

    assert selected == str(baseline_path)


def test_freeze_and_reset_demo_baseline_copies_snapshot_and_clears_artifacts(tmp_path):
    active_retriever_path = tmp_path / "models" / "active-retriever"
    active_verifier_path = tmp_path / "models" / "active-verifier"
    baseline_retriever_path = tmp_path / "models" / "baseline" / "retriever"
    baseline_verifier_path = tmp_path / "models" / "baseline" / "verifier"
    retriever_versions_dir = tmp_path / "models" / "retriever-versions"
    verifier_versions_dir = tmp_path / "models" / "verifier-versions"
    approved_path = tmp_path / "approved.json"
    augmented_path = tmp_path / "augmented.json"

    active_retriever_path.mkdir(parents=True)
    active_verifier_path.mkdir(parents=True)
    (active_retriever_path / "model.txt").write_text("retriever-v1", encoding="utf-8")
    (active_verifier_path / "model.txt").write_text("verifier-v1", encoding="utf-8")
    (retriever_versions_dir / "old-retriever").mkdir(parents=True)
    (verifier_versions_dir / "old-verifier").mkdir(parents=True)
    approved_path.write_text(json.dumps([{"row": 1}]), encoding="utf-8")
    augmented_path.write_text(json.dumps([{"row": 1}]), encoding="utf-8")

    config = demo_services.load_demo_config(
        overrides={
            "models": {
                "bi_encoder_path": str(active_retriever_path),
                "cross_encoder_path": str(active_verifier_path),
                "baseline_bi_encoder_path": str(baseline_retriever_path),
                "baseline_cross_encoder_path": str(baseline_verifier_path),
            },
            "outputs": {
                "retriever_versions_dir": str(retriever_versions_dir),
                "verifier_versions_dir": str(verifier_versions_dir),
                "approved_examples_json_path": str(approved_path),
                "augmented_dataset_path": str(augmented_path),
            },
        }
    )

    frozen = demo_services.freeze_current_demo_baseline(config=config)
    assert frozen["status"] == "frozen"
    assert (baseline_retriever_path / "model.txt").read_text(encoding="utf-8") == "retriever-v1"
    assert (baseline_verifier_path / "model.txt").read_text(encoding="utf-8") == "verifier-v1"

    (active_retriever_path / "model.txt").write_text("retriever-mutated", encoding="utf-8")
    (active_verifier_path / "model.txt").write_text("verifier-mutated", encoding="utf-8")

    reset = demo_services.reset_demo_to_baseline(config=config)

    assert reset["status"] == "reset"
    assert (active_retriever_path / "model.txt").read_text(encoding="utf-8") == "retriever-v1"
    assert (active_verifier_path / "model.txt").read_text(encoding="utf-8") == "verifier-v1"
    assert list(retriever_versions_dir.iterdir()) == []
    assert list(verifier_versions_dir.iterdir()) == []
    assert json.loads(approved_path.read_text(encoding="utf-8")) == []
    assert not augmented_path.exists()


def test_compare_agentic_score_changes_uses_human_label_direction():
    before_items = [
        {
            "transcript_id": "tx1",
            "disclaimer_id": "102",
            "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
            "text": "There will be a fee to make the change.",
            "verification_score": 0.38,
            "final_label": "Compliant",
        },
        {
            "transcript_id": "tx1",
            "disclaimer_id": "101",
            "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
            "text": "I can unlock it now.",
            "verification_score": 0.62,
            "final_label": "Non-Compliant",
        },
    ]
    after_payloads = [
        {
            "transcript_id": "tx1",
            "results": {
                "102": {
                    "evidence": {
                        "claims": {
                            "mandatory": [
                                {
                                    "claim_type": "mandatory",
                                    "claim_idx": 0,
                                    "anchor": "Before I confirm this booking change, there is a change fee that will apply.",
                                    "match_text": "There will be a fee to make the change.",
                                    "verification_score": 0.71,
                                    "passed": True,
                                }
                            ]
                        }
                    }
                },
                "101": {
                    "evidence": {
                        "claims": {
                            "single": [
                                {
                                    "claim_type": "single",
                                    "claim_idx": 0,
                                    "anchor": "Before I reset or unlock your account, I need to verify your identity first.",
                                    "match_text": "I can unlock it now.",
                                    "verification_score": 0.24,
                                    "passed": False,
                                }
                            ]
                        }
                    }
                },
            },
        }
    ]

    comparisons = demo_services.compare_agentic_score_changes(before_items, after_payloads)

    assert comparisons[0]["target_direction"] == "higher"
    assert comparisons[0]["outcome"] == "improved"
    assert comparisons[1]["target_direction"] == "lower"
    assert comparisons[1]["outcome"] == "improved"


def test_diagnose_label_changed_cases_recommends_missing_coverage_and_flags_label_noise(tmp_path):
    synthetic_path = tmp_path / "synthetic.json"
    approved_path = tmp_path / "approved.json"
    anchor = "Before I confirm this booking change, there is a change fee that will apply."
    synthetic_path.write_text(
        json.dumps(
            [
                {
                    "disclaimer_id": "102",
                    "anchor": anchor,
                    "dialogue": "There will be a fee to make the change.",
                    "type": "compliant",
                },
            ]
        ),
        encoding="utf-8",
    )
    approved_path.write_text("[]", encoding="utf-8")
    config = demo_services.load_demo_config(
        overrides={
            "data": {
                "synthetic_dataset_path": str(synthetic_path),
            },
            "outputs": {
                "approved_examples_json_path": str(approved_path),
            },
            "agentic": {
                "coverage_similarity_threshold": 0.74,
            },
        }
    )

    report = demo_services.diagnose_label_changed_cases(
        [
            {
                "transcript_id": "tx1",
                "disclaimer_id": "102",
                "anchor": anchor,
                "text": "There will be a fee to make the change.",
                "verification_score": 0.62,
                "final_label": "Non-Compliant",
            }
        ],
        config=config,
    )

    assert report["changed_case_count"] == 1
    analysis = report["analyses"][0]
    assert analysis["label_change"] == "model Pass -> human Fail"
    assert "missing_coverage" in analysis["cause_tags"]
    assert "possible_label_noise" in analysis["cause_tags"]
    assert "DataGeneratorAgent" in analysis["solution_steps"][0]


def test_complete_agentic_reinference_triggers_dataset_diagnosis_for_regression(tmp_path, monkeypatch):
    synthetic_path = tmp_path / "synthetic.json"
    approved_path = tmp_path / "approved.json"
    anchor = "Before I confirm this booking change, there is a change fee that will apply."
    synthetic_path.write_text(
        json.dumps(
            [
                {
                    "disclaimer_id": "102",
                    "anchor": anchor,
                    "dialogue": "A change fee applies before I confirm this booking change.",
                    "type": "compliant",
                }
            ]
        ),
        encoding="utf-8",
    )
    approved_path.write_text("[]", encoding="utf-8")
    config = demo_services.load_demo_config(
        overrides={
            "data": {"synthetic_dataset_path": str(synthetic_path)},
            "outputs": {"approved_examples_json_path": str(approved_path)},
        }
    )

    monkeypatch.setattr(demo_services, "_load_demo_analyzer", lambda config: object())

    def fake_run_demo_inference(transcript, *, transcript_id, config, analyzer):
        return {
            "transcript_id": transcript_id,
            "transcript": transcript,
            "results": {
                "102": {
                    "status": "PASS",
                    "evidence": {
                        "claims": {
                            "mandatory": [
                                {
                                    "claim_type": "mandatory",
                                    "claim_idx": 0,
                                    "passed": True,
                                    "anchor": anchor,
                                    "match_text": "There will be a fee to make the change.",
                                    "verification_score": 0.61,
                                }
                            ]
                        }
                    },
                }
            },
        }

    monkeypatch.setattr(demo_services, "run_demo_inference", fake_run_demo_inference)

    result = demo_services.complete_agentic_reinference_cycle(
        {
            "before_payloads": [{"transcript_id": "tx1", "transcript": "Agent: There will be a fee."}],
            "selected_trainable_review_items": [
                {
                    "transcript_id": "tx1",
                    "disclaimer_id": "102",
                    "anchor": anchor,
                    "text": "There will be a fee to make the change.",
                    "verification_score": 0.69,
                    "final_label": "Compliant",
                }
            ],
            "retrain": {"status": "trained"},
            "diagnosis": {"status": "completed", "changed_case_count": 0, "analyses": []},
            "stage_status": [],
        },
        config=config,
    )

    assert result["status"] == "needs_review"
    assert result["regressed_count"] == 1
    assert result["score_regression_diagnosis"]["regressed_case_count"] == 1
    analysis = result["diagnosis"]["analyses"][0]
    assert analysis["diagnosis_type"] == "Approved phrase moved in the wrong direction after retraining"
    assert "score regression" in result["recommendation"]
