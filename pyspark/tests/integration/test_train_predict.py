"""
Tests d'intégration — Pipeline ML (cli/train.py + cli/predict.py)

Vérifie l'enchaînement complet :
  split_data → train → evaluate → save → load → predict_batch → top_n_at_risk

Chaque test est indépendant grâce à `tmp_path` (dossier temporaire unique).
Le `trained_model` et `features_df` de session sont réutilisés pour la vitesse.

Marqueur : @pytest.mark.integration
Lance séparément avec : pytest -m integration
"""
import json
import pytest
from pathlib import Path
from pyspark.sql import functions as F
from tests.conftest import FEATURE_COLS

pytestmark = pytest.mark.integration


# ──────────────────────────────────────────────────────────────────────────────
# Tests d'entraînement complet
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainPipeline:
    """Teste split_data → train → evaluate end-to-end."""

    @pytest.fixture(scope="class")
    def train_test_split(self, features_df):
        from src.models.train import split_data
        return split_data(features_df, train_ratio=0.8, seed=42)

    def test_split_sizes(self, features_df, train_test_split):
        """Train + test doivent reconstituer le total (sans garantie exacte due au split)."""
        train_df, test_df = train_test_split
        n_total = features_df.count()
        n_train = train_df.count()
        n_test  = test_df.count()
        assert n_train + n_test == n_total
        assert n_train > n_test   # train_ratio=0.8 → train > test

    def test_train_produces_pipeline_model(self, train_test_split):
        """train() doit retourner un PipelineModel."""
        from src.config import ModelConfig
        from src.models.train import train
        from pyspark.ml import PipelineModel
        train_df, _ = train_test_split
        cfg   = ModelConfig(num_trees=3, max_depth=2, use_cv=False, seed=42)
        model = train(train_df, cfg, FEATURE_COLS)
        assert isinstance(model, PipelineModel)

    def test_trained_model_has_correct_stages(self, train_test_split):
        """Le modèle entraîné doit avoir 3 stages (Assembler, Scaler, Classifier)."""
        from src.config import ModelConfig
        from src.models.train import train
        train_df, _ = train_test_split
        cfg   = ModelConfig(num_trees=3, max_depth=2, use_cv=False, seed=42)
        model = train(train_df, cfg, FEATURE_COLS)
        assert len(model.stages) == 3

    def test_evaluate_after_train(self, train_test_split):
        """evaluate() doit retourner des métriques valides après entraînement."""
        from src.config import ModelConfig
        from src.models.train import train
        from src.models.evaluate import evaluate
        train_df, test_df = train_test_split
        cfg     = ModelConfig(num_trees=3, max_depth=2, use_cv=False, seed=42)
        model   = train(train_df, cfg, FEATURE_COLS)
        metrics = evaluate(model, test_df, FEATURE_COLS)
        assert 0.0 <= metrics["auc"]      <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"]       <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests sauvegarde / rechargement du modèle
# ──────────────────────────────────────────────────────────────────────────────

class TestModelSaveLoad:
    """Teste la persistence du modèle : save → load → predict."""

    def test_model_saves_to_disk(self, trained_model, tmp_path):
        """Le modèle doit créer les fichiers sur disque (dossier non vide)."""
        model_path = tmp_path / "model"
        trained_model.write().overwrite().save(str(model_path))
        assert model_path.exists()
        assert any(model_path.iterdir())   # dossier non vide

    def test_reloaded_model_is_pipeline_model(self, trained_model, tmp_path):
        """Un modèle rechargé doit être une instance de PipelineModel."""
        from pyspark.ml import PipelineModel
        from src.models.predict import load_model
        model_path = str(tmp_path / "model_reload")
        trained_model.write().overwrite().save(model_path)
        reloaded = load_model(model_path)
        assert isinstance(reloaded, PipelineModel)

    def test_reloaded_model_same_predictions(self, features_df, trained_model, tmp_path):
        """Les prédictions du modèle rechargé doivent être identiques à l'original."""
        from src.models.predict import load_model, predict_batch
        model_path = str(tmp_path / "model_preds")
        trained_model.write().overwrite().save(model_path)
        reloaded = load_model(model_path)

        preds_orig   = predict_batch(trained_model, features_df).orderBy("CustomerID").collect()
        preds_reload = predict_batch(reloaded,      features_df).orderBy("CustomerID").collect()

        assert len(preds_orig) == len(preds_reload)
        for orig, rel in zip(preds_orig, preds_reload):
            assert orig["CustomerID"] == rel["CustomerID"]
            assert abs(orig["churn_proba"] - rel["churn_proba"]) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Test pipeline complet : train → predict → top_n (end-to-end)
# ──────────────────────────────────────────────────────────────────────────────

class TestFullMLPipeline:
    """
    Pipeline complet : features → train → evaluate → save → load → predict → top_n.
    Simule exactement ce que font cli/train.py + cli/predict.py.
    """

    @pytest.fixture(scope="class")
    def pipeline_artifacts(self, features_df, tmp_path_factory):
        """
        Exécute le pipeline complet et retourne les artefacts pour les tests.
        (scope=class pour ne l'exécuter qu'une fois pour toute la classe)
        """
        from src.config import ModelConfig
        from src.models.train import split_data, train
        from src.models.evaluate import evaluate
        from src.models.predict import load_model, predict_batch, top_n_at_risk

        tmp_path   = tmp_path_factory.mktemp("full_pipeline")
        model_path = str(tmp_path / "model")
        metrics_path = tmp_path / "metrics.json"

        # 1. Split
        train_df, test_df = split_data(features_df, train_ratio=0.8, seed=42)

        # 2. Train
        cfg   = ModelConfig(num_trees=5, max_depth=3, use_cv=False, seed=42)
        model = train(train_df, cfg, FEATURE_COLS)

        # 3. Evaluate
        metrics = evaluate(model, test_df, FEATURE_COLS)

        # 4. Save métriques JSON
        safe_metrics = {
            k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in metrics.items()
        }
        with open(metrics_path, "w") as f:
            json.dump(safe_metrics, f)

        # 5. Save modèle
        model.write().overwrite().save(model_path)

        # 6. Load modèle
        reloaded = load_model(model_path)

        # 7. Predict
        scored = predict_batch(reloaded, features_df).cache()

        # 8. Top-10
        top10 = top_n_at_risk(scored, n=10)

        return {
            "metrics":      metrics,
            "metrics_path": metrics_path,
            "scored":       scored,
            "top10":        top10,
        }

    def test_metrics_json_written(self, pipeline_artifacts):
        """Le fichier metrics.json doit exister sur disque."""
        assert pipeline_artifacts["metrics_path"].exists()

    def test_metrics_json_valid(self, pipeline_artifacts):
        """Le JSON doit être parsable et contenir la clé 'auc'."""
        with open(pipeline_artifacts["metrics_path"]) as f:
            data = json.load(f)
        assert "auc" in data

    def test_scored_all_customers(self, features_df, pipeline_artifacts):
        """Tous les clients doivent être scorés (aucun perdu)."""
        n_input  = features_df.count()
        n_scored = pipeline_artifacts["scored"].count()
        assert n_scored == n_input

    def test_top10_count(self, pipeline_artifacts):
        """top_n_at_risk(n=10) → exactement 10 clients."""
        assert pipeline_artifacts["top10"].count() == 10

    def test_top10_highest_probas(self, pipeline_artifacts):
        """Le top-10 doit contenir les probabilités les plus élevées."""
        scored = pipeline_artifacts["scored"]
        top10  = pipeline_artifacts["top10"]

        min_top10 = top10.agg(F.min("churn_proba")).collect()[0][0]
        max_rest  = (
            scored
            .join(top10.select("CustomerID"), on="CustomerID", how="left_anti")
            .agg(F.max("churn_proba"))
            .collect()[0][0]
        )
        if max_rest is not None:
            assert max_rest <= min_top10 + 1e-9
