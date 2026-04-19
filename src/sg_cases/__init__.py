"""SG banking compliance pipeline package."""

from .demo import (
    approve_demo_examples,
    get_borderline_items,
    label_borderline_items_with_ollama,
    retrain_demo_verifier,
    run_demo_inference,
)
from .evaluation import evaluate
from .inference import run_semantic_inference
from .synthetic import generate
from .training import train_cross_encoder, train_sentence_transformer

__all__ = [
    "approve_demo_examples",
    "evaluate",
    "generate",
    "get_borderline_items",
    "label_borderline_items_with_ollama",
    "retrain_demo_verifier",
    "run_semantic_inference",
    "run_demo_inference",
    "train_cross_encoder",
    "train_sentence_transformer",
]
