"""Tests unitaires des métriques (ne nécessitent ni Ollama ni modèles)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import retrieval_metrics, rouge_l, token_f1  # noqa: E402


def test_hit_rate_and_mrr():
    m = retrieval_metrics(["a.txt", "b.txt", "c.txt"], gold=["b.txt"], k=3)
    assert m["hit_rate"] == 1.0
    assert m["mrr"] == 0.5  # b.txt en 2e position -> 1/2


def test_no_hit():
    m = retrieval_metrics(["x.txt", "y.txt"], gold=["z.txt"], k=2)
    assert m["hit_rate"] == 0.0
    assert m["mrr"] == 0.0


def test_token_f1_identical():
    assert token_f1("the cat sat", "the cat sat") == 1.0


def test_rouge_l_partial():
    score = rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.0 < score <= 1.0


if __name__ == "__main__":
    test_hit_rate_and_mrr()
    test_no_hit()
    test_token_f1_identical()
    test_rouge_l_partial()
    print("Tous les tests passent.")
