import json
from pathlib import Path

from sg_cases.training.cross_encoder import prepare_training_rows
from sg_cases.training.data_utils import resolve_training_anchor_text
from sg_cases.training.sentence_transformer import generate_triplet_rows


def _load_disclosures():
    path = Path(__file__).resolve().parents[1] / "resources" / "demo_disclosures.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_resolve_training_anchor_text_uses_plain_string_anchor():
    disclosures = _load_disclosures()
    row = {
        "disclaimer_id": "101",
        "dialogue": "Before I unlock your account, I need to verify your identity first.",
        "type": "compliant",
    }

    anchor = resolve_training_anchor_text(row, disclosures)

    assert anchor == disclosures["101"]["anchor"]


def test_resolve_training_anchor_text_maps_prompt_index_to_mandatory_anchor():
    disclosures = _load_disclosures()
    row = {
        "disclaimer_id": "102",
        "prompt_index": 1,
        "dialogue": "The new itinerary costs fifty dollars more, so you need to pay the difference.",
        "type": "compliant",
    }

    assert resolve_training_anchor_text(row, disclosures) == disclosures["102"]["anchor"]["mandatory"][1]


def test_generate_triplet_rows_can_resolve_anchor_from_generated_dataset_shape():
    disclosures = _load_disclosures()
    dataset_rows = [
        {
            "disclaimer_id": "102",
            "prompt_index": 0,
            "dialogue": "Before I confirm this booking change, there is a change fee that will apply.",
            "type": "compliant",
        },
        {
            "disclaimer_id": "102",
            "prompt_index": 0,
            "dialogue": "I can confirm the change now and we can discuss any fee later.",
            "type": "non-compliant",
        },
    ]

    triplets = generate_triplet_rows(dataset_rows, seed=42, disclosures=disclosures)

    assert triplets["anchor"]
    assert triplets["anchor"][0] == disclosures["102"]["anchor"]["mandatory"][0]


def test_prepare_training_rows_resolves_multi_anchor_prompt_index():
    disclosures = _load_disclosures()
    dataset_path = Path(__file__).resolve().parent / "tmp_training_rows.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "disclaimer_id": "102",
                    "prompt_index": 1,
                    "dialogue": "The new itinerary is fifty dollars higher, so you would need to pay that extra amount.",
                    "type": "compliant",
                }
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "data": {
            "disclosures_file": "resources/demo_disclosures.json",
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
    assert rows[0]["sentence1"] == disclosures["102"]["anchor"]["mandatory"][1]
