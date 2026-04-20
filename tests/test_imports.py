import importlib


def test_public_modules_import():
    modules = [
        "policy_compliance_agent",
        "policy_compliance_agent.core",
        "policy_compliance_agent.demo",
        "policy_compliance_agent.synthetic",
        "policy_compliance_agent.training",
        "policy_compliance_agent.inference",
        "policy_compliance_agent.cli.demo_app",
        "policy_compliance_agent.cli.generate_synthetic",
        "policy_compliance_agent.cli.train_cross_encoder",
        "policy_compliance_agent.cli.train_sentence_transformer",
        "policy_compliance_agent.cli.run_inference",
        "policy_compliance_agent.cli.agentic_loop",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name) is not None
