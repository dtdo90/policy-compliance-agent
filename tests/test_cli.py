import sys

from sg_cases.cli import demo_app as demo_app_cli
from sg_cases.cli import evaluate as evaluate_cli
from sg_cases.cli import generate_synthetic as generate_cli
from sg_cases.cli import run_inference as inference_cli
from sg_cases.cli import train_cross_encoder as ce_cli
from sg_cases.cli import train_sentence_transformer as st_cli
from sg_cases.cli import agentic_loop as agentic_loop_cli


def test_generate_cli_passes_config(monkeypatch):
    calls = {}
    monkeypatch.setattr(generate_cli, "generate", lambda config_path=None: calls.setdefault("config", config_path) or "ok")
    monkeypatch.setattr(sys, "argv", ["sg-generate-synthetic", "--config", "cfg.yaml"])
    generate_cli.main()
    assert calls["config"] == "cfg.yaml"


def test_train_ce_cli_passes_config(monkeypatch):
    calls = {}
    monkeypatch.setattr(ce_cli, "train_cross_encoder", lambda config_path=None: calls.setdefault("config", config_path) or "ok")
    monkeypatch.setattr(sys, "argv", ["sg-train-ce", "--config", "cfg.yaml"])
    ce_cli.main()
    assert calls["config"] == "cfg.yaml"


def test_train_st_cli_passes_config(monkeypatch):
    calls = {}
    monkeypatch.setattr(st_cli, "train_sentence_transformer", lambda config_path=None: calls.setdefault("config", config_path) or "ok")
    monkeypatch.setattr(sys, "argv", ["sg-train-st", "--config", "cfg.yaml"])
    st_cli.main()
    assert calls["config"] == "cfg.yaml"


def test_inference_cli_passes_arguments(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        inference_cli,
        "run_semantic_inference",
        lambda input_path=None, config_path=None: calls.update({"input": input_path, "config": config_path}),
    )
    monkeypatch.setattr(sys, "argv", ["sg-run-inference", "--config", "cfg.yaml", "--input", "data/voice_logs"])
    inference_cli.main()
    assert calls == {"input": "data/voice_logs", "config": "cfg.yaml"}


def test_evaluate_cli_passes_arguments(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        evaluate_cli,
        "evaluate",
        lambda report_path=None, truth_path=None, config_path=None: calls.update(
            {"report": report_path, "truth": truth_path, "config": config_path}
        )
        or ({}, {}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["sg-evaluate", "--config", "cfg.yaml", "--report", "report.json", "--truth", "truth.json"],
    )
    evaluate_cli.main()
    assert calls == {"report": "report.json", "truth": "truth.json", "config": "cfg.yaml"}


def test_demo_app_cli_passes_arguments(monkeypatch):
    calls = {}

    from sg_cases.demo import app as demo_app_module

    monkeypatch.setattr(
        demo_app_module,
        "launch_demo_app",
        lambda config_path=None, server_name=None, server_port=None, share=None: calls.update(
            {
                "config": config_path,
                "server_name": server_name,
                "server_port": server_port,
                "share": share,
            }
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["sg-demo-app", "--config", "configs/demo.yaml", "--server-name", "0.0.0.0", "--server-port", "9000", "--share"],
    )

    demo_app_cli.main()

    assert calls == {
        "config": "configs/demo.yaml",
        "server_name": "0.0.0.0",
        "server_port": 9000,
        "share": True,
    }


def test_agentic_loop_cli_passes_arguments(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        agentic_loop_cli,
        "run_local_agentic_loop",
        lambda config_path=None, transcripts_source=None, holdout_source=None, auto_approve_llm=None: calls.update(
            {
                "config": config_path,
                "transcripts": transcripts_source,
                "holdout": holdout_source,
                "auto_approve_llm": auto_approve_llm,
            }
        )
        or {"status": "ok"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["sg-agentic-loop", "--config", "configs/demo.yaml", "--transcripts", "demo.json", "--holdout", "holdout.json", "--require-human-review"],
    )

    agentic_loop_cli.main()

    assert calls == {
        "config": "configs/demo.yaml",
        "transcripts": "demo.json",
        "holdout": "holdout.json",
        "auto_approve_llm": False,
    }
