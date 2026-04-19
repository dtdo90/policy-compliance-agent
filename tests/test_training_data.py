import json
from pathlib import Path

from sg_cases.training.cross_encoder import prepare_training_rows
from sg_cases.training.data_utils import resolve_training_anchor_text
from sg_cases.training.sentence_transformer import generate_triplet_rows


def _load_disclosures():
    path = Path(__file__).resolve().parents[1] / "resources" / "disclosures_with_anchors_sg.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_resolve_training_anchor_text_uses_requested_cross_encoder_model_mapping():
    disclosures = _load_disclosures()
    row = {
        "disclaimer_id": "10",
        "prompt_index": 1,
        "dialogue": "Please confirm you understand the higher risk and accept it.",
        "type": "compliant",
    }

    anchor = resolve_training_anchor_text(row, disclosures)

    assert anchor == disclosures["10"]["anchor"]["mandatory"][1]


def test_resolve_training_anchor_text_maps_rule_9_prompt_zero_to_first_mandatory_anchor():
    disclosures = _load_disclosures()
    row = {
        "disclaimer_id": "9",
        "prompt_index": 0,
        "dialogue": "Because it's your first time, our specialist is on the line to explain the product.",
        "type": "compliant",
    }

    assert resolve_training_anchor_text(row, disclosures) == disclosures["9"]["anchor"]["mandatory"][0]


def test_generate_triplet_rows_can_resolve_anchor_from_generated_dataset_shape():
    disclosures = _load_disclosures()
    dataset_rows = [
        {
            "disclaimer_id": "2",
            "prompt_index": 0,
            "dialogue": "This product is higher risk than your profile, so there is a mismatch.",
            "type": "compliant",
        },
        {
            "disclaimer_id": "2",
            "prompt_index": 0,
            "dialogue": "This product has some risk, but let's proceed.",
            "type": "non-compliant",
        },
    ]

    triplets = generate_triplet_rows(dataset_rows, seed=42, disclosures=disclosures)

    assert triplets["anchor"]
    assert triplets["anchor"][0] == disclosures["2"]["anchor"][0]


def test_prepare_training_rows_resolves_multi_anchor_prompt_index():
    disclosures = _load_disclosures()
    dataset_path = Path(__file__).resolve().parent / "tmp_training_rows.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "disclaimer_id": "11",
                    "prompt_index": 4,
                    "dialogue": "The new fund may give you worse value at a higher cost.",
                    "type": "compliant",
                }
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "data": {
            "disclosures_file": "resources/disclosures_with_anchors_sg.json",
            "synthetic_dataset_path": str(dataset_path),
        },
        "training": {
            "use_extra_sampling": False,
        },
    }

    try:
        rows = prepare_training_rows(config, dataset_path=str(dataset_path))
    finally:
        dataset_path.unlink(missing_ok=True)

    assert len(rows) == 1
    assert rows[0]["sentence1"] == disclosures["11"]["anchor"]["standard"][1]
