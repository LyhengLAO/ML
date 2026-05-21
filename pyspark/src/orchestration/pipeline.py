"""
Orchestrateur du pipeline ML complet.

run_full_pipeline() coordonne les 4 étapes dans l'ordre et retourne
un PipelineResult contenant les résultats de chaque étape.

Flux :
  DataPipeline  → Training  → Scoring  → Monitoring
       │               │           │           │
   features_df      model      scored_df    report JSON
   (Parquet)     (PipelineModel) (CSV)    (drift_report.json)

En cas d'échec d'une étape critique (DataPipeline ou Training),
le pipeline s'arrête et retourne immédiatement.
Les étapes Scoring et Monitoring sont non-bloquantes (leur échec
est enregistré mais le pipeline continue et retourne un résultat partiel).
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession

from src.config import Config
from src.orchestration.steps import (
    StepResult,
    run_data_step,
    run_monitoring_step,
    run_predict_step,
    run_train_step,
)


# ──────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Résultat global du pipeline.

    Attributes
    ----------
    steps          : liste des StepResult dans l'ordre d'exécution
    total_elapsed_s: durée totale en secondes
    success        : True si toutes les étapes critiques ont réussi
    started_at     : horodatage UTC de démarrage
    """
    steps:           List[StepResult]  = field(default_factory=list)
    total_elapsed_s: float             = 0.0
    success:         bool              = True
    started_at:      Optional[str]     = None

    def add(self, step: StepResult) -> None:
        self.steps.append(step)
        if not step.success:
            self.success = False

    def get(self, name: str) -> Optional[StepResult]:
        """Retourne le StepResult d'une étape par son nom."""
        for s in self.steps:
            if s.name == name:
                return s
        return None

    def print_summary(self) -> None:
        """Affiche un résumé tabulaire de toutes les étapes."""
        print()
        print("=" * 70)
        print(" RÉSUMÉ DU PIPELINE")
        print("=" * 70)
        for step in self.steps:
            print(f"  {step}")
        print(f"  {'─' * 60}")
        total_str  = f"{self.total_elapsed_s:.1f}s"
        status_str = "✅ SUCCÈS COMPLET" if self.success else "⚠️  PARTIEL / ÉCHEC"
        print(f"  Durée totale : {total_str:<8}  {status_str}")
        print()

    def as_dict(self) -> Dict[str, Any]:
        """Sérialise le résultat complet en dict JSON-compatible."""
        return {
            "started_at":      self.started_at,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "success":         self.success,
            "steps":           [s.as_dict() for s in self.steps],
        }

    def save(self, path: str) -> None:
        """Sauvegarde le résumé du pipeline en JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers d'affichage
# ──────────────────────────────────────────────────────────────────────────────

def _print_step_header(n: int, total: int, label: str) -> None:
    sep = "─" * max(0, 52 - len(label))
    print()
    print(f"── [{n}/{total}] {label} {sep}")


def _print_metrics(step: StepResult) -> None:
    if step.error:
        print(f"   ❌ ERREUR : {step.error}")
        return
    for key, val in step.metrics.items():
        if isinstance(val, float):
            print(f"   [info] {key} : {val:.4f}")
        elif val is not None:
            print(f"   [info] {key} : {val}")


# ──────────────────────────────────────────────────────────────────────────────
# run_full_pipeline — point d'entrée principal
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    spark: SparkSession,
    cfg: Config,
    csv_path: Optional[str]    = None,
    output_dir: str            = "output",
    skip_download: bool        = False,
    force_download: bool       = False,
    skip_train: bool           = False,
    threshold: float           = 0.5,
    top_n: int                 = 20,
    run_monitoring: bool       = True,
) -> PipelineResult:
    """
    Pipeline complet : DataPipeline → Training → Scoring → Monitoring.

    Parameters
    ----------
    spark          : SparkSession active
    cfg            : Config (depuis load_config)
    csv_path       : chemin CSV brut alternatif (override)
    output_dir     : dossier racine de sortie  (défaut: "output")
    skip_download  : ignorer le téléchargement si le CSV existe déjà
    force_download : re-télécharger même si le CSV est présent
    skip_train     : charger un modèle existant plutôt que re-entraîner
    threshold      : seuil de classification churn [0–1]  (défaut: 0.5)
    top_n          : nb de clients retenus dans top_at_risk.csv
    run_monitoring : activer l'étape de monitoring (défaut: True)

    Returns
    -------
    PipelineResult — résultat de toutes les étapes
    """
    total_steps = 4 if run_monitoring else 3
    t0_global   = time.perf_counter()
    started_at  = datetime.now(timezone.utc).isoformat()
    result      = PipelineResult(started_at=started_at)

    # Artefacts transmis entre étapes
    features_df: Optional[DataFrame]  = None
    model:       Optional[Any]        = None
    train_df:    Optional[DataFrame]  = None
    test_df:     Optional[DataFrame]  = None
    scored_df:   Optional[DataFrame]  = None

    # ── ÉTAPE 1 — DataPipeline ────────────────────────────────────────────────
    _print_step_header(1, total_steps, "DataPipeline")
    step, features_df = run_data_step(
        spark, cfg,
        csv_path=csv_path,
        skip_download=skip_download,
        force_download=force_download,
    )
    result.add(step)
    _print_metrics(step)

    if not step.success:
        result.total_elapsed_s = time.perf_counter() - t0_global
        result.print_summary()
        return result

    # ── ÉTAPE 2 — Training ────────────────────────────────────────────────────
    _print_step_header(2, total_steps, "Model Training")

    if skip_train:
        # Charger le modèle pré-entraîné
        from src.models.predict import load_model
        model_path = str(Path(output_dir) / "model")
        try:
            model = load_model(model_path)
            step  = StepResult(
                name="Training",
                success=True,
                elapsed_s=0.0,
                started_at=datetime.now(timezone.utc).isoformat(),
                output_path=model_path,
                metrics={"mode": "modèle existant chargé", "model_path": model_path},
            )
            print(f"   [info] Modèle existant chargé : {model_path}")
        except Exception as exc:
            step = StepResult(
                name="Training",
                success=False,
                elapsed_s=0.0,
                started_at=datetime.now(timezone.utc).isoformat(),
                error=f"Impossible de charger le modèle : {exc}",
            )
        result.add(step)
        if not step.success:
            result.total_elapsed_s = time.perf_counter() - t0_global
            result.print_summary()
            return result
    else:
        step, model, train_df, test_df = run_train_step(
            spark, cfg, features_df, output_dir=output_dir
        )
        result.add(step)
        _print_metrics(step)
        if not step.success:
            result.total_elapsed_s = time.perf_counter() - t0_global
            result.print_summary()
            return result

    # ── ÉTAPE 3 — Scoring ────────────────────────────────────────────────────
    _print_step_header(3, total_steps, "Batch Scoring")
    step, scored_df = run_predict_step(
        spark, cfg, model, features_df,
        output_dir=output_dir,
        threshold=threshold,
        top_n=top_n,
    )
    result.add(step)
    _print_metrics(step)
    # Scoring non-bloquant : on continue même en cas d'échec

    # ── ÉTAPE 4 — Monitoring ─────────────────────────────────────────────────
    if run_monitoring:
        _print_step_header(4, total_steps, "Monitoring & Drift PSI")

        if train_df is not None and test_df is not None:
            # Scores sur train et test pour comparer les distributions
            from src.models.predict import predict_batch
            try:
                scored_train = predict_batch(model, train_df)
                scored_test  = predict_batch(model, test_df)
            except Exception:
                scored_train = None
                scored_test  = None

            train_metrics = result.get("Training")
            model_metrics = train_metrics.metrics if train_metrics else {}

            step = run_monitoring_step(
                spark, cfg,
                ref_df=train_df,
                cur_df=test_df,
                scored_ref=scored_train,
                scored_cur=scored_test,
                output_dir=output_dir,
                model_metrics={
                    k: model_metrics.get(k)
                    for k in ("auc", "f1", "accuracy", "precision", "recall")
                    if k in model_metrics
                },
            )
        else:
            # Pas de split disponible (--skip-train) → monitoring sur features
            print("   [info] Pas de split train/test disponible.")
            print("   [info] Monitoring basé sur la moitié supérieure vs inférieure.")
            from pyspark.sql import functions as F
            n_half     = features_df.count() // 2
            ref_df_mon = features_df.orderBy(F.col("tenure_days").desc()).limit(n_half)
            cur_df_mon = features_df.orderBy(F.col("tenure_days").asc()).limit(n_half)
            step = run_monitoring_step(
                spark, cfg,
                ref_df=ref_df_mon,
                cur_df=cur_df_mon,
                output_dir=output_dir,
            )

        result.add(step)
        _print_metrics(step)

    # ── Sauvegarde du résumé pipeline ─────────────────────────────────────────
    result.total_elapsed_s = time.perf_counter() - t0_global
    pipeline_json = Path(output_dir) / "pipeline_run.json"
    result.save(str(pipeline_json))

    result.print_summary()
    return result
