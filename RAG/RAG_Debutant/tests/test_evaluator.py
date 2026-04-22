"""
tests/test_evaluator.py — Tests du module d'évaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Lancer : pytest tests/test_evaluator.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.evaluator import RAGEvaluator, EvalResult


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def mock_rag_pipeline():
    """Pipeline RAG mocké pour les tests."""
    from src.rag_pipeline import RAGResult

    pipeline = MagicMock()
    pipeline.embeddings = MagicMock()
    pipeline.llm = MagicMock()

    def fake_query(question):
        return RAGResult(
            question=question,
            answer=f"Réponse simulée pour : {question}",
            sources=[
                {"content": "Le Transformer utilise l'attention multi-têtes.", "metadata": {"origin": "wikipedia"}, "score": 0.88},
                {"content": "BERT est un encodeur bidirectionnel.", "metadata": {"origin": "arxiv"}, "score": 0.75},
            ],
            retrieval_time=0.1,
            generation_time=0.2,
        )

    pipeline.query.side_effect = fake_query
    return pipeline


@pytest.fixture
def evaluator(mock_rag_pipeline):
    return RAGEvaluator(mock_rag_pipeline)


@pytest.fixture
def sample_questions():
    return [
        "Qu'est-ce que le Transformer ?",
        "Comment fonctionne BERT ?",
    ]


@pytest.fixture
def sample_ground_truths():
    return [
        "Le Transformer est une architecture basée sur l'attention introduite en 2017.",
        "BERT est un modèle bidirectionnel pré-entraîné par Google pour la compréhension du langage.",
    ]


# ── Tests EvalResult ────────────────────────────────────────

class TestEvalResult:

    def test_ragas_score_moyenne_harmonique(self):
        result = EvalResult(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.85,
        )
        assert 0.0 < result.ragas_score < 1.0

    def test_ragas_score_zero_si_tout_zero(self):
        result = EvalResult()
        assert result.ragas_score == 0.0

    def test_ragas_score_parfait(self):
        result = EvalResult(
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_precision=1.0,
            context_recall=1.0,
        )
        assert result.ragas_score == 1.0

    def test_to_dict_contient_les_cles(self):
        result = EvalResult(faithfulness=0.8, answer_relevancy=0.7,
                            context_precision=0.9, context_recall=0.6, n_questions=3)
        d = result.to_dict()
        assert "ragas_score" in d
        assert "faithfulness" in d
        assert "answer_relevancy" in d
        assert "context_precision" in d
        assert "context_recall" in d
        assert d["n_questions"] == 3

    def test_quality_label_excellent(self):
        result = EvalResult(faithfulness=0.9, answer_relevancy=0.9,
                            context_precision=0.9, context_recall=0.9)
        assert "Excellent" in result._quality_label()

    def test_quality_label_faible(self):
        result = EvalResult(faithfulness=0.2, answer_relevancy=0.3,
                            context_precision=0.2, context_recall=0.3)
        assert "Faible" in result._quality_label()

    def test_display_no_error(self, capsys):
        result = EvalResult(faithfulness=0.8, answer_relevancy=0.75,
                            context_precision=0.85, context_recall=0.7, n_questions=2)
        result.display()
        captured = capsys.readouterr()
        assert "RAGAS" in captured.out
        assert "0.8" in captured.out


# ── Tests RAGEvaluator (métriques simples) ───────────────────

class TestRAGEvaluatorSimple:

    def test_evaluate_simple_retourne_dict(self, evaluator, sample_questions, sample_ground_truths):
        result = evaluator.evaluate_simple(sample_questions, sample_ground_truths)
        assert isinstance(result, dict)
        assert result["n_questions"] == 2

    def test_evaluate_simple_cles_presentes(self, evaluator, sample_questions, sample_ground_truths):
        result = evaluator.evaluate_simple(sample_questions, sample_ground_truths)
        required_keys = [
            "n_questions", "avg_retrieval_score", "avg_latency_ms",
            "avg_answer_words", "avg_source_diversity", "per_question"
        ]
        for key in required_keys:
            assert key in result, f"Clé manquante : {key}"

    def test_evaluate_simple_avec_ground_truth(self, evaluator, sample_questions, sample_ground_truths):
        result = evaluator.evaluate_simple(sample_questions, sample_ground_truths)
        assert "avg_token_overlap" in result
        assert 0.0 <= result["avg_token_overlap"] <= 1.0

    def test_evaluate_simple_sans_ground_truth(self, evaluator, sample_questions):
        result = evaluator.evaluate_simple(sample_questions)
        assert "avg_token_overlap" not in result

    def test_per_question_detail(self, evaluator, sample_questions, sample_ground_truths):
        result = evaluator.evaluate_simple(sample_questions, sample_ground_truths)
        assert len(result["per_question"]) == len(sample_questions)
        for pq in result["per_question"]:
            assert "question" in pq
            assert "answer" in pq
            assert "avg_retrieval_score" in pq
            assert "latency_ms" in pq

    def test_retrieval_score_dans_intervalle(self, evaluator, sample_questions):
        result = evaluator.evaluate_simple(sample_questions)
        assert 0.0 <= result["avg_retrieval_score"] <= 1.0


# ── Tests token_overlap ──────────────────────────────────────

class TestTokenOverlap:

    def test_overlap_identique(self):
        text = "Le Transformer utilise l'attention"
        score = RAGEvaluator._token_overlap(text, text)
        assert score == 1.0

    def test_overlap_aucun(self):
        score = RAGEvaluator._token_overlap("bonjour monde", "hello world")
        assert score == 0.0

    def test_overlap_partiel(self):
        score = RAGEvaluator._token_overlap("le chat mange", "le chat dort")
        assert 0.0 < score < 1.0

    def test_overlap_vide(self):
        score = RAGEvaluator._token_overlap("", "référence")
        assert score == 0.0

    def test_overlap_symetrique(self):
        a = "Le Transformer est basé sur l'attention"
        b = "L'attention est le cœur du Transformer"
        s1 = RAGEvaluator._token_overlap(a, b)
        s2 = RAGEvaluator._token_overlap(b, a)
        assert abs(s1 - s2) < 0.01   # Quasi-symétrique (F1)


# ── Tests collect_pipeline_outputs ───────────────────────────

class TestCollectOutputs:

    def test_collecte_les_reponses(self, evaluator, sample_questions, sample_ground_truths):
        data = evaluator._collect_pipeline_outputs(
            sample_questions, sample_ground_truths, verbose=False
        )
        assert len(data["answers"]) == len(sample_questions)
        assert len(data["contexts"]) == len(sample_questions)
        assert all(isinstance(c, list) for c in data["contexts"])

    def test_contextes_sont_des_strings(self, evaluator, sample_questions):
        data = evaluator._collect_pipeline_outputs(sample_questions, verbose=False)
        for ctx_list in data["contexts"]:
            for ctx in ctx_list:
                assert isinstance(ctx, str)

    def test_ground_truths_vides_si_absents(self, evaluator, sample_questions):
        data = evaluator._collect_pipeline_outputs(sample_questions, verbose=False)
        assert all(gt == "" for gt in data["ground_truths"])