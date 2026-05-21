"""
Tests unitaires — src/models/evaluate.py

Vérifie les métriques d'évaluation produites par evaluate()
en utilisant le `trained_model` et `features_df` des fixtures session.
"""
import pytest
from tests.conftest import FEATURE_COLS

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def eval_metrics(features_df, trained_model):
    """
    Calcule les métriques une seule fois pour tous les tests de ce module.
    Utilise le jeu de test (20 % de features_df avec seed=42).
    """
    from src.models.evaluate import evaluate
    from src.models.train import split_data
    _, test_df = split_data(features_df, train_ratio=0.8, seed=42)
    return evaluate(trained_model, test_df, FEATURE_COLS)


class TestMetricsKeys:

    def test_all_expected_keys_present(self, eval_metrics):
        """Le dict de métriques doit contenir toutes les clés attendues."""
        expected_keys = {
            "auc", "accuracy", "f1", "precision",
            "recall", "confusion_matrix", "feature_importances",
        }
        assert expected_keys <= set(eval_metrics.keys())

    def test_confusion_matrix_is_dict(self, eval_metrics):
        """confusion_matrix doit être un dict keyed par (label, prediction)."""
        cm = eval_metrics["confusion_matrix"]
        assert isinstance(cm, dict)

    def test_feature_importances_is_list(self, eval_metrics):
        """feature_importances doit être une liste de tuples (name, score)."""
        fi = eval_metrics["feature_importances"]
        assert isinstance(fi, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in fi)


class TestMetricRanges:

    def test_auc_in_zero_one(self, eval_metrics):
        """AUC ∈ [0, 1]."""
        assert 0.0 <= eval_metrics["auc"] <= 1.0

    def test_accuracy_in_zero_one(self, eval_metrics):
        """Accuracy ∈ [0, 1]."""
        assert 0.0 <= eval_metrics["accuracy"] <= 1.0

    def test_f1_in_zero_one(self, eval_metrics):
        """F1 ∈ [0, 1]."""
        assert 0.0 <= eval_metrics["f1"] <= 1.0

    def test_precision_in_zero_one(self, eval_metrics):
        """Precision ∈ [0, 1]."""
        assert 0.0 <= eval_metrics["precision"] <= 1.0

    def test_recall_in_zero_one(self, eval_metrics):
        """Recall ∈ [0, 1]."""
        assert 0.0 <= eval_metrics["recall"] <= 1.0

    def test_auc_better_than_random(self, eval_metrics):
        """
        Le signal dans features_df est fort (churned vs retained bien séparés),
        l'AUC doit être nettement supérieure à 0.5.
        """
        assert eval_metrics["auc"] >= 0.7


class TestFeatureImportances:

    def test_importances_cover_all_features(self, eval_metrics):
        """L'importance doit être rapportée pour chacune des 7 features."""
        names = {name for name, _ in eval_metrics["feature_importances"]}
        assert names == set(FEATURE_COLS)

    def test_importances_sum_to_one(self, eval_metrics):
        """La somme des importances RF doit valoir ≈ 1.0."""
        total = sum(score for _, score in eval_metrics["feature_importances"])
        assert abs(total - 1.0) < 0.01

    def test_importances_non_negative(self, eval_metrics):
        """Toutes les importances doivent être ≥ 0."""
        for name, score in eval_metrics["feature_importances"]:
            assert score >= 0.0, f"Score négatif pour '{name}' : {score}"
