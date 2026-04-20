from policy_compliance_agent.core.disclosures import filter_disclaimers
from policy_compliance_agent.core.models import Disclaimer
from policy_compliance_agent.core.config import load_config
from policy_compliance_agent.inference import SemanticComplianceAnalyzer, aggregate_rule_result


class FakeRetriever:
    def encode(self, texts, **kwargs):
        return list(texts)


class FakeVerifier:
    def predict(self, pairs, **kwargs):
        scores = []
        for anchor, chunk in pairs:
            scores.append(4.0 if anchor.lower().split()[0] in chunk.lower() else -4.0)
        return scores


class NeutralLogitVerifier:
    def predict(self, pairs, **kwargs):
        return [0.0 for _ in pairs]


def fake_semantic_search(query_embeddings, corpus_embeddings, top_k=5):
    results = []
    for anchor in query_embeddings:
        hits = []
        for index, chunk in enumerate(corpus_embeddings):
            if anchor.lower().split()[0] in chunk.lower():
                hits.append({"corpus_id": index, "score": 0.9})
        results.append(hits[:top_k])
    return results


def test_rules_can_be_excluded_by_id():
    disclaimers = [
        Disclaimer("1", "", "", "", "", "", []),
        Disclaimer("2", "", "", "", "", "", []),
        Disclaimer("13", "", "", "", "", "", []),
        Disclaimer("14", "", "", "", "", "", []),
    ]
    filtered = filter_disclaimers(disclaimers, include_rule_ids=[1, 2, 13, 14], exclude_rule_ids=[1, 13, 14])
    assert [item.id for item in filtered] == ["2"]


def test_aggregate_rule_result_requires_all_mandatory_claims():
    disclaimer = Disclaimer("9", "Suitability", "desc", "purpose", {"mandatory": ["a", "b"]}, "", [])
    groups = {"single": [], "mandatory": [0, 1], "standard": []}
    claim_results = [
        {"passed": True, "best_ver": 0.9, "best_retr": 0.8, "best_text": "match a", "for_review": []},
        {"passed": False, "best_ver": 0.3, "best_retr": 0.4, "best_text": "weak b", "for_review": [{"text": "weak b"}]},
    ]
    claim_texts = ["anchor a", "anchor b"]
    claim_meta = [
        {"claim_type": "mandatory", "claim_idx": 0},
        {"claim_type": "mandatory", "claim_idx": 1},
    ]

    result = aggregate_rule_result(disclaimer, groups, claim_results, claim_texts, claim_meta)

    assert result["status"] == "FAIL"
    assert result["evidence"]["verification_score"] == 0.3
    assert result["for_review"] == [{"text": "weak b"}]


def test_semantic_analyzer_returns_structured_output():
    config = load_config(
        overrides={
            "semantic_inference": {
                "verification_threshold": 0.5,
                "borderline_low": 0.2,
                "borderline_high": 0.5,
                "speaker_labels": ["Agent"],
            }
        }
    )
    disclaimer = Disclaimer(
        id="2",
        theme="Suitability",
        description="Risk mismatch",
        purpose_of_control="Warn on mismatch",
        anchor="risk mismatch",
        criteria="",
        keywords=[],
    )

    analyzer = SemanticComplianceAnalyzer(
        [disclaimer],
        config,
        retriever=FakeRetriever(),
        verifier=FakeVerifier(),
        semantic_search_fn=fake_semantic_search,
    )
    result = analyzer.analyze_transcript(
        "User: I need to let you know there is a risk mismatch for this transaction.\n"
        "Agent: I need to let you know there is a risk mismatch for this transaction.",
        "tx001",
    )

    assert result["2"]["status"] == "PASS"
    assert result["2"]["evidence"]["claims"]["single"][0]["anchor"] == "risk mismatch"
    assert "risk mismatch" in result["2"]["evidence"]["match_text"].lower()


def test_semantic_analyzer_ignores_non_agent_lines_when_speaker_filter_is_enabled():
    config = load_config(
        overrides={
            "semantic_inference": {
                "verification_threshold": 0.5,
                "borderline_low": 0.2,
                "borderline_high": 0.5,
                "speaker_labels": ["Agent"],
                "max_chunk_words": 20,
                "chunk_overlap": 15,
                "retrieval_top_k": 5,
            }
        }
    )
    disclaimer = Disclaimer(
        id="2",
        theme="Suitability",
        description="Risk mismatch",
        purpose_of_control="Warn on mismatch",
        anchor="risk mismatch",
        criteria="",
        keywords=[],
    )

    analyzer = SemanticComplianceAnalyzer(
        [disclaimer],
        config,
        retriever=FakeRetriever(),
        verifier=FakeVerifier(),
        semantic_search_fn=fake_semantic_search,
    )
    result = analyzer.analyze_transcript(
        "User: There is a risk mismatch for this transaction and I understand it.\n"
        "Agent: I can walk you through the account issue, but let me confirm the next step.",
        "tx002",
    )

    assert result["2"]["status"] == "FAIL"
    assert result["2"]["evidence"]["match_text"] == ""
    assert result["2"]["evidence"]["verification_score"] == 0.0


def test_semantic_analyzer_applies_single_sigmoid_to_cross_encoder_logits():
    config = load_config(
        overrides={
            "semantic_inference": {
                "verification_threshold": 0.55,
                "speaker_labels": ["Agent"],
            }
        }
    )
    disclaimer = Disclaimer(
        id="2",
        theme="Suitability",
        description="Risk mismatch",
        purpose_of_control="Warn on mismatch",
        anchor="risk mismatch",
        criteria="",
        keywords=[],
    )

    analyzer = SemanticComplianceAnalyzer(
        [disclaimer],
        config,
        retriever=FakeRetriever(),
        verifier=NeutralLogitVerifier(),
        semantic_search_fn=fake_semantic_search,
    )
    result = analyzer.analyze_transcript(
        "Agent: I need to let you know there is a risk mismatch for this transaction.",
        "tx003",
    )

    assert result["2"]["status"] == "FAIL"
    assert result["2"]["evidence"]["verification_score"] == 0.5
