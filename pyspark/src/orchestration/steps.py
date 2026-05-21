"""
Étapes atomiques du pipeline.

Chaque fonction run_*_step() :
  - Prend les dépendances nécessaires en argument (pas de global)
  - Encapsule ses erreurs → StepResult.success = False + StepResult.error
  - Retourne un tuple (StepResult, artefact_produit)
  - Logue ses timings et métriques clés

Elles sont utilisées par src/orchestration/pipeline.py
et peuvent être appelées indépendamment dans les tests ou les notebooks.
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession

from src.config import Config


# ──────────────────────────────────────────────────────────────────────────────
# StepResult — résultat d'une étape
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """
    Résultat d'une étape du pipeline.

    Attributes
    ----------
    name        : nom de l'étape
    success     : True si l'étape s'est terminée sans erreur
    elapsed_s   : temps d'exécution en secondes
    metrics     : dict de métriques clés (sérialisable JSON)
    error       : message d'erreur si success=False
    output_path : chemin de l'artefact produit (parquet, modèle, CSV…)
    started_at  : horodatage UTC du démarrage
    """
    name:        str
    success:     bool
    elapsed_s:   float
    metrics:     Dict[str, Any]   = field(default_factory=dict)
    error:       Optional[str]    = None
    output_path: Optional[str]    = None
    started_at:  Optional[str]    = None   # ISO-8601 UTC

    def __str__(self) -> str:
        icon = "✅" if self.success else "❌"
        err  = f"  ERROR: {self.error}" if self.error else ""
        return f"{icon} {self.name:<22}  {self.elapsed_s:>6.1f}s{err}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "success":     self.success,
            "elapsed_s":   round(self.elapsed_s, 2),
            "metrics":     self.metrics,
            "error":       self.error,
            "output_path": self.output_path,
            "started_at":  self.started_at,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — Data Pipeline (download + clean + features)
# ──────────────────────────────────────────────────────────────────────────────

def run_data_step(
    spark: SparkSession,
    cfg: Config,
    csv_path: Optional[str]     = None,
    output_parquet: Optional[str] = None,
    skip_download: bool          = False,
    force_download: bool         = False,
) -> Tuple[StepResult, Optional[DataFrame]]:
    """
    Télécharge les données brutes, nettoie et construit les features RFM.

    Étapes internes :
      1. Téléchargement UCI (si nécessaire)
      2. Chargement CSV → Spark DataFrame
      3. Nettoyage (filtres qualité + Revenue)
      4. Feature engineering RFM + label churn
      5. Sauvegarde Parquet

    Returns
    -------
    (StepResult, features_df)
    """
    from datetime import datetime as dt

    from src.data.cleaning import clean_transactions, quality_report
    from src.data.download import extract, load_csv, zip_file_download
    from src.feature.builder import build_features_and_label
    from pyspark.sql import functions as F

    t0         = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        raw_dir  = Path(cfg.data.raw_path)
        csv_file = Path(csv_path) if csv_path else raw_dir / "online_retail_II.csv"
        out_dir  = (
            Path(output_parquet)
            if output_parquet
            else Path(cfg.data.processed_path) / "features"
        )

        # ── Téléchargement ────────────────────────────────────────────────────
        if not csv_file.exists() or force_download:
            if skip_download:
                raise FileNotFoundError(
                    f"CSV introuvable ({csv_file}) et --skip-download actif."
                )
            zip_file_download(dest=raw_dir)
            extract()

        # ── Chargement ────────────────────────────────────────────────────────
        raw_df = load_csv(spark, str(csv_file))
        n_raw  = raw_df.cache().count()

        # ── Nettoyage ─────────────────────────────────────────────────────────
        clean_df = clean_transactions(raw_df)
        n_clean  = clean_df.cache().count()

        # ── Features ──────────────────────────────────────────────────────────
        cutoff_dt   = dt.strptime(cfg.features.cutoff_date, "%Y-%m-%d")
        features_df = build_features_and_label(
            clean_df, cutoff_dt, cfg.features.horizon_days
        )
        n_clients = features_df.cache().count()

        label_counts = {
            r["label"]: r["count"]
            for r in features_df.groupBy("label").count().collect()
        }
        n_churned  = label_counts.get(1, 0)
        n_retained = label_counts.get(0, 0)

        # ── Sauvegarde Parquet ────────────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        features_df.write.mode("overwrite").parquet(str(out_dir))

        return StepResult(
            name="DataPipeline",
            success=True,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            output_path=str(out_dir),
            metrics={
                "n_raw_rows":    n_raw,
                "n_clean_rows":  n_clean,
                "n_dropped":     n_raw - n_clean,
                "n_clients":     n_clients,
                "n_churned":     n_churned,
                "n_retained":    n_retained,
                "churn_rate":    round(n_churned / n_clients, 4) if n_clients > 0 else 0.0,
                "csv_source":    str(csv_file),
                "features_path": str(out_dir),
            },
        ), features_df

    except Exception as exc:
        return StepResult(
            name="DataPipeline",
            success=False,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            error=str(exc),
        ), None


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Training (split + train + evaluate + save)
# ──────────────────────────────────────────────────────────────────────────────

def run_train_step(
    spark: SparkSession,
    cfg: Config,
    features_df: DataFrame,
    output_dir: str = "output",
) -> Tuple[StepResult, Optional[Any], Optional[DataFrame], Optional[DataFrame]]:
    """
    Entraîne le modèle ML et le sauvegarde avec ses métriques.

    Étapes internes :
      1. Split train / test (ratio depuis cfg.model.train_ratio)
      2. Entraînement Pipeline MLlib (RF / GBT / Logistic)
      3. Évaluation AUC, F1, accuracy, confusion matrix
      4. Sauvegarde PipelineModel → output/model/
      5. Sauvegarde métriques JSON → output/metrics.json

    Returns
    -------
    (StepResult, model, train_df, test_df)
    """
    from src.models.evaluate import evaluate
    from src.models.train import split_data, train

    t0         = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        out          = Path(output_dir)
        model_dir    = out / "model"
        metrics_path = out / "metrics.json"

        # ── Split ─────────────────────────────────────────────────────────────
        train_df, test_df = split_data(
            features_df, cfg.model.train_ratio, cfg.model.seed
        )
        n_train = train_df.cache().count()
        n_test  = test_df.cache().count()

        # ── Entraînement ──────────────────────────────────────────────────────
        model = train(train_df, cfg.model, cfg.features.feature_cols)

        # ── Évaluation ────────────────────────────────────────────────────────
        metrics = evaluate(model, test_df, cfg.features.feature_cols)

        # ── Sauvegarde modèle ─────────────────────────────────────────────────
        model_dir.mkdir(parents=True, exist_ok=True)
        model.write().overwrite().save(str(model_dir))

        # ── Sauvegarde métriques JSON ─────────────────────────────────────────
        safe_metrics = {
            k: (
                {f"{ki[0]},{ki[1]}": vi for ki, vi in v.items()}
                if isinstance(v, dict) else v
            )
            for k, v in metrics.items()
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(safe_metrics, f, indent=2, ensure_ascii=False)

        return StepResult(
            name="Training",
            success=True,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            output_path=str(model_dir),
            metrics={
                "algorithm": cfg.model.algorithm,
                "use_cv":    cfg.model.use_cv,
                "n_train":   n_train,
                "n_test":    n_test,
                "auc":       metrics["auc"],
                "f1":        metrics["f1"],
                "accuracy":  metrics["accuracy"],
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
            },
        ), model, train_df, test_df

    except Exception as exc:
        return StepResult(
            name="Training",
            success=False,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            error=str(exc),
        ), None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Scoring (predict_batch + sauvegarde)
# ──────────────────────────────────────────────────────────────────────────────

def run_predict_step(
    spark: SparkSession,
    cfg: Config,
    model: Any,
    features_df: DataFrame,
    output_dir: str  = "output",
    threshold: float = 0.5,
    top_n: int       = 20,
) -> Tuple[StepResult, Optional[DataFrame]]:
    """
    Score tous les clients et sauvegarde les prédictions.

    Étapes internes :
      1. predict_batch → scored_df (CustomerID, churn_proba, prediction)
      2. Application du seuil personnalisé
      3. Sauvegarde CSV complet → output/predictions.csv
      4. Sauvegarde Top-N à risque → output/top_at_risk.csv

    Returns
    -------
    (StepResult, scored_df)
    """
    from pyspark.sql import functions as F

    from src.models.predict import predict_batch, top_n_at_risk

    t0         = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        out       = Path(output_dir)
        pred_csv  = out / "predictions.csv"
        top_csv   = out / "top_at_risk.csv"

        # ── Scoring ───────────────────────────────────────────────────────────
        scored_df = predict_batch(model, features_df)

        # Appliquer le seuil
        scored_df = scored_df.withColumn(
            "prediction",
            F.when(F.col("churn_proba") >= threshold, 1).otherwise(0).cast("double"),
        ).cache()

        n_scored   = scored_df.count()
        n_churn    = scored_df.filter(F.col("prediction") == 1).count()
        n_retained = n_scored - n_churn

        # ── Sauvegarde CSV ────────────────────────────────────────────────────
        out.mkdir(parents=True, exist_ok=True)
        (
            scored_df
            .orderBy(F.col("churn_proba").desc())
            .toPandas()
            .to_csv(str(pred_csv), index=False)
        )

        # ── Top-N à risque ────────────────────────────────────────────────────
        (
            top_n_at_risk(scored_df, top_n)
            .toPandas()
            .to_csv(str(top_csv), index=False)
        )

        return StepResult(
            name="Scoring",
            success=True,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            output_path=str(pred_csv),
            metrics={
                "threshold":   threshold,
                "n_scored":    n_scored,
                "n_churn":     n_churn,
                "n_retained":  n_retained,
                "churn_rate":  round(n_churn / n_scored, 4) if n_scored > 0 else 0.0,
                "top_n_path":  str(top_csv),
            },
        ), scored_df

    except Exception as exc:
        return StepResult(
            name="Scoring",
            success=False,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            error=str(exc),
        ), None


# ──────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — Monitoring (drift PSI + rapport)
# ──────────────────────────────────────────────────────────────────────────────

def run_monitoring_step(
    spark: SparkSession,
    cfg: Config,
    ref_df: DataFrame,
    cur_df: DataFrame,
    scored_ref: Optional[DataFrame]  = None,
    scored_cur: Optional[DataFrame]  = None,
    output_dir: str                  = "output",
    model_metrics: Optional[Dict]    = None,
) -> StepResult:
    """
    Calcule le drift PSI entre les données de référence et courantes.

    Référence = train split  (données sur lesquelles le modèle a appris)
    Courant   = test split   (données "nouvelles" vues après entraînement)

    Étapes internes :
      1. PSI pour chaque feature (recency, frequency, monetary…)
      2. PSI sur la distribution des scores (churn_proba)
      3. Construction du rapport structuré
      4. Affichage console
      5. Sauvegarde JSON → output/monitoring/drift_report.json

    Returns
    -------
    StepResult (sans artefact DataFrame)
    """
    from src.monitoring.drift import (
        compute_feature_drift,
        compute_score_drift,
        drift_status,
    )
    from src.monitoring.report import build_report, print_report, save_report

    t0         = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        report_path = Path(output_dir) / "monitoring" / "drift_report.json"
        n_bins      = cfg.monitoring.n_bins
        warning     = cfg.monitoring.psi_threshold_warning
        critical    = cfg.monitoring.psi_threshold_critical

        # ── PSI features ──────────────────────────────────────────────────────
        feature_drift = compute_feature_drift(
            ref_df, cur_df, cfg.features.feature_cols, n_bins
        )

        # ── PSI scores ────────────────────────────────────────────────────────
        score_drift = 0.0
        if scored_ref is not None and scored_cur is not None:
            try:
                score_drift = compute_score_drift(
                    scored_ref, scored_cur, "churn_proba", n_bins
                )
            except Exception:
                import math
                score_drift = float("nan")

        # ── Rapport ───────────────────────────────────────────────────────────
        report = build_report(feature_drift, score_drift, model_metrics, cfg)
        print_report(report)
        save_report(report, str(report_path))

        # ── Métriques résumé ──────────────────────────────────────────────────
        n_warning  = sum(
            1 for p in feature_drift.values()
            if drift_status(p, warning, critical) == "WARNING"
        )
        n_critical = sum(
            1 for p in feature_drift.values()
            if drift_status(p, warning, critical) == "CRITICAL"
        )

        import math as _math
        score_psi_safe = (
            round(score_drift, 6) if not _math.isnan(score_drift) else None
        )
        return StepResult(
            name="Monitoring",
            success=True,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            output_path=str(report_path),
            metrics={
                "global_status":       report["global_status"],
                "n_features_stable":   report["summary"]["n_features_stable"],
                "n_features_warning":  n_warning,
                "n_features_critical": n_critical,
                "max_feature_psi":     report["summary"]["max_feature_psi"],
                "score_drift_psi":     score_psi_safe,
            },
        )

    except Exception as exc:
        return StepResult(
            name="Monitoring",
            success=False,
            elapsed_s=time.perf_counter() - t0,
            started_at=started_at,
            error=str(exc),
        )
