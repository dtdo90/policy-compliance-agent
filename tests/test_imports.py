import importlib


def test_public_modules_import():
    modules = [
        "sg_cases",
        "sg_cases.core",
        "sg_cases.demo",
        "sg_cases.synthetic",
        "sg_cases.training",
        "sg_cases.inference",
        "sg_cases.cli.demo_app",
        "sg_cases.cli.generate_synthetic",
        "sg_cases.cli.train_cross_encoder",
        "sg_cases.cli.train_sentence_transformer",
        "sg_cases.cli.run_inference",
        "sg_cases.cli.agentic_loop",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name) is not None
