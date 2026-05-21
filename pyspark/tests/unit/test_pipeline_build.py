"""
Tests unitaires — src/models/pipeline.py

Vérifie la construction du Pipeline MLlib :
  VectorAssembler → StandardScaler → Classifier

Ces tests instancient le Pipeline sans l'entraîner.
"""
import pytest
from src.config import ModelConfig

pytestmark = pytest.mark.unit


class TestPipelineStages:

    def test_pipeline_has_three_stages(self):
        """Le pipeline doit avoir exactement 3 stages."""
        from src.models.pipeline import build_pipeline
        p = build_pipeline()
        assert len(p.getStages()) == 3

    def test_first_stage_is_vector_assembler(self):
        """Stage 0 = VectorAssembler."""
        from src.models.pipeline import build_pipeline
        from pyspark.ml.feature import VectorAssembler
        stages = build_pipeline().getStages()
        assert isinstance(stages[0], VectorAssembler)

    def test_second_stage_is_standard_scaler(self):
        """Stage 1 = StandardScaler."""
        from src.models.pipeline import build_pipeline
        from pyspark.ml.feature import StandardScaler
        stages = build_pipeline().getStages()
        assert isinstance(stages[1], StandardScaler)

    def test_assembler_uses_features_raw(self):
        """VectorAssembler doit produire la colonne 'features_raw'."""
        from src.models.pipeline import build_pipeline
        assembler = build_pipeline().getStages()[0]
        assert assembler.getOutputCol() == "features_raw"

    def test_scaler_input_from_assembler(self):
        """StandardScaler doit lire 'features_raw' (sortie de l'assembler)."""
        from src.models.pipeline import build_pipeline
        scaler = build_pipeline().getStages()[1]
        assert scaler.getInputCol() == "features_raw"

    def test_scaler_output_is_features(self):
        """StandardScaler doit produire la colonne 'features'."""
        from src.models.pipeline import build_pipeline
        scaler = build_pipeline().getStages()[1]
        assert scaler.getOutputCol() == "features"


class TestClassifierSelection:

    def test_random_forest_by_default(self):
        """Algorithme par défaut = Random Forest."""
        from src.models.pipeline import build_pipeline
        from pyspark.ml.classification import RandomForestClassifier
        cfg = ModelConfig(algorithm="random_forest")
        clf = build_pipeline(cfg).getStages()[-1]
        assert isinstance(clf, RandomForestClassifier)

    def test_gbt_classifier(self):
        """algo='gbt' → GBTClassifier."""
        from src.models.pipeline import build_pipeline
        from pyspark.ml.classification import GBTClassifier
        cfg = ModelConfig(algorithm="gbt")
        clf = build_pipeline(cfg).getStages()[-1]
        assert isinstance(clf, GBTClassifier)

    def test_logistic_regression(self):
        """algo='logistic' → LogisticRegression."""
        from src.models.pipeline import build_pipeline
        from pyspark.ml.classification import LogisticRegression
        cfg = ModelConfig(algorithm="logistic")
        clf = build_pipeline(cfg).getStages()[-1]
        assert isinstance(clf, LogisticRegression)

    def test_unknown_algo_raises_value_error(self):
        """Un algo inconnu doit lever ValueError, pas crasher silencieusement."""
        from src.models.pipeline import build_pipeline
        cfg = ModelConfig(algorithm="xgboost_unknown")
        with pytest.raises(ValueError, match="inconnu"):
            build_pipeline(cfg)

    def test_rf_num_trees_propagated(self):
        """num_trees dans ModelConfig doit être transmis au RandomForestClassifier."""
        from src.models.pipeline import build_pipeline
        cfg = ModelConfig(algorithm="random_forest", num_trees=123)
        rf = build_pipeline(cfg).getStages()[-1]
        assert rf.getNumTrees() == 123

    def test_rf_max_depth_propagated(self):
        """max_depth dans ModelConfig doit être transmis au RandomForestClassifier."""
        from src.models.pipeline import build_pipeline
        cfg = ModelConfig(algorithm="random_forest", max_depth=7)
        rf = build_pipeline(cfg).getStages()[-1]
        assert rf.getMaxDepth() == 7


class TestCustomFeatureCols:

    def test_custom_feature_cols_set_on_assembler(self):
        """feature_cols personnalisées doivent être transmises à VectorAssembler."""
        from src.models.pipeline import build_pipeline
        custom = ["recency", "monetary"]
        assembler = build_pipeline(feature_cols=custom).getStages()[0]
        assert list(assembler.getInputCols()) == custom
