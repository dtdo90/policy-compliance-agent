import json
from pathlib import Path

from policy_compliance_agent.core.config import load_config


def test_demo_disclosures_use_expected_anchor_shapes():
    disclosures_path = Path(__file__).resolve().parents[1] / "resources" / "demo_disclosures.json"
    disclosures = json.loads(disclosures_path.read_text(encoding="utf-8"))

    assert disclosures["101"]["anchor"] == "Before I reset or unlock your account, I need to verify your identity first."
    assert disclosures["102"]["anchor"]["mandatory"] == [
        "Before I confirm this booking change, there is a change fee that will apply.",
        "There is also a fare difference on the new itinerary, so you will either pay the extra amount or receive the balance as travel credit.",
    ]
    assert disclosures["102"]["anchor"]["standard"] == []


def test_demo_config_points_to_demo_assets():
    config = load_config("configs/demo.yaml")

    assert config["data"]["disclosures_file"] == "resources/demo_disclosures.json"
    assert config["synthetic"]["backend"] == "external_api"
    assert config["semantic_inference"]["include_rule_ids"] == [101, 102]
    assert config["demo"]["ollama_model"] == "qwen3:4b"
