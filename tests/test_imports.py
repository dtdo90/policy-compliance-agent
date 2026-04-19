import importlib
import importlib.util


def test_public_modules_import():
    modules = [
        "sg_cases",
        "sg_cases.core",
        "sg_cases.demo",
        "sg_cases.synthetic",
        "sg_cases.training",
        "sg_cases.inference",
        "sg_cases.evaluation",
        "sg_cases.cli.demo_app",
        "sg_cases.cli.generate_synthetic",
        "sg_cases.cli.train_cross_encoder",
        "sg_cases.cli.train_sentence_transformer",
        "sg_cases.cli.run_inference",
        "sg_cases.cli.evaluate",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_metadata_package_imports_when_optional_dependencies_exist():
    if importlib.util.find_spec("pyspark") is None:
        return
    assert importlib.import_module("sg_cases.metadata.joiner_hive") is not None
    assert importlib.import_module("sg_cases.metadata.vl_excel_hive") is not None
    assert importlib.import_module("sg_cases.metadata.enrichment") is not None
