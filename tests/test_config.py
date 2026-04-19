from pathlib import Path

from sg_cases.core.config import load_config
from sg_cases.core.paths import DEFAULT_CONFIG_PATH


def test_load_default_config_from_any_working_directory(tmp_path, monkeypatch):
    nested = tmp_path / "nested" / "cwd"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    config = load_config()

    assert DEFAULT_CONFIG_PATH.exists()
    assert config["data"]["disclosures_file"] == "resources/demo_disclosures.json"
    assert config["semantic_inference"]["include_rule_ids"] == [101, 102]
