import json

from sg_cases.core.config import load_config
from sg_cases.evaluation import evaluate


def test_evaluate_uses_passed_paths_without_hard_coded_machine_paths(tmp_path):
    report_path = tmp_path / "report.json"
    truth_path = tmp_path / "truth.json"
    output_path = tmp_path / "evaluation.json"
    missed_path = tmp_path / "missed.json"

    report_path.write_text(
        json.dumps({"tx1": {"2": {"status": "PASS", "evidence": {"verification_score": 0.9}}}}),
        encoding="utf-8",
    )
    truth_path.write_text(
        json.dumps({"tx1": {"ground_truth": {"2": "T"}, "op_code": "OP1"}}),
        encoding="utf-8",
    )

    config = load_config(
        overrides={
            "outputs": {
                "evaluation_output_path": str(output_path),
                "evaluation_missed_cases_path": str(missed_path),
            }
        }
    )

    results, missed_cases = evaluate(report_path=report_path, truth_path=truth_path, config=config)

    assert results["2"]["compliant"] == "1/1 (100.00%)"
    assert missed_cases == {}
    assert output_path.exists()
    assert missed_path.exists()
