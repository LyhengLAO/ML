"""
Évaluation d'un modèle entraîné.
"""
from typing import Dict, Any

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame


def evaluate(model: PipelineModel,
             test_df: DataFrame,
             feature_cols: list) -> Dict[str, Any]:
    """
    Calcule les métriques d'évaluation et retourne un dict
    sérialisable en JSON.
    """
    preds = model.transform(test_df)

    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(preds)

    eval_mc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )
    accuracy = eval_mc.evaluate(preds, {eval_mc.metricName: "accuracy"})
    f1       = eval_mc.evaluate(preds, {eval_mc.metricName: "f1"})
    prec     = eval_mc.evaluate(preds, {eval_mc.metricName: "weightedPrecision"})
    rec      = eval_mc.evaluate(preds, {eval_mc.metricName: "weightedRecall"})

    cm = (preds.groupBy("label", "prediction")
                .count()
                .orderBy("label", "prediction")
                .collect())
    cm_dict = {(int(r["label"]), int(r["prediction"])): r["count"] for r in cm}

    classifier = model.stages[-1]
    if hasattr(classifier, "featureImportances"):
        importances = list(zip(feature_cols, classifier.featureImportances.toArray()))
        importances.sort(key=lambda x: x[1], reverse=True)
        importances = [(f, round(float(i), 4)) for f, i in importances]
    else:
        importances = []

    return {
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm_dict,
        "feature_importances": importances,
    }


def print_evaluation(metrics: Dict[str, Any]) -> None:
    """Affichage humain des métriques."""
    print("=" * 70)
    print(" RÉSULTATS DU MODÈLE")
    print("=" * 70)
    print(f"AUC         : {metrics['auc']}")
    print(f"Accuracy    : {metrics['accuracy']}")
    print(f"F1 score    : {metrics['f1']}")
    print(f"Precision   : {metrics['precision']}")
    print(f"Recall      : {metrics['recall']}")

    cm = metrics["confusion_matrix"]
    tn = cm.get((0, 0), 0); fp = cm.get((0, 1), 0)
    fn = cm.get((1, 0), 0); tp = cm.get((1, 1), 0)
    print("\nMatrice de confusion :")
    print(f"                 pred=0    pred=1")
    print(f"  actual=0   {tn:>8}  {fp:>8}")
    print(f"  actual=1   {fn:>8}  {tp:>8}")

    if metrics["feature_importances"]:
        print("\nImportance des features :")
        for name, imp in metrics["feature_importances"]:
            bar = "#" * int(imp * 50)
            print(f"  {name:<14} {imp:>6.4f}  {bar}")
