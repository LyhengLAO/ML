"""
Détection de data drift via PSI (Population Stability Index).

Principe :
  PSI = Σ (P_current - P_ref) × ln(P_current / P_ref)

Interprétation standard :
  PSI < 0.10              → Distribution stable        (STABLE)
  0.10 ≤ PSI < 0.25       → Changement modéré          (WARNING)
  PSI ≥ 0.25              → Changement majeur           (CRITICAL)

Utilisation typique :
  - ref_df   = features calculées sur la période d'entraînement
  - cur_df   = features calculées sur une nouvelle période (production)
  Un PSI élevé indique que le modèle voit des données très différentes
  de celles sur lesquelles il a été entraîné → réévaluation nécessaire.

Toute la computation reste dans PySpark (Bucketizer + groupBy).
Seule la réduction finale (PSI scalaire) est faite en Python.
"""
import math
from typing import Dict, List

from pyspark.sql import DataFrame, functions as F


# ──────────────────────────────────────────────────────────────────────────────
# PSI pour une colonne numérique
# ──────────────────────────────────────────────────────────────────────────────

def compute_psi(
    ref_df: DataFrame,
    cur_df: DataFrame,
    col: str,
    n_bins: int = 10,
) -> float:
    """
    Calcule le PSI pour une colonne numérique en utilisant PySpark.

    Algorithme :
    1. Calcule les quantiles de la distribution de référence → bornes des bins
    2. Bucketize les deux DataFrames avec ces bornes (Bucketizer MLlib)
    3. Compte la fréquence par bin dans chaque dataset
    4. Applique la formule PSI en Python (données agrégées très petites)

    Parameters
    ----------
    ref_df  : DataFrame de référence (ex. données d'entraînement)
    cur_df  : DataFrame courant     (ex. nouvelles données en production)
    col     : colonne numérique à analyser
    n_bins  : nombre de bins (défaut 10)

    Returns
    -------
    float : PSI ≥ 0  (0.0 si calcul impossible)

    Raises
    ------
    Ne lève pas d'exception — retourne 0.0 en cas d'erreur.
    """
    from pyspark.ml.feature import Bucketizer

    EPS = 1e-10  # évite log(0)

    # Cast en double + supprimer nulls
    ref_clean = ref_df.select(F.col(col).cast("double").alias(col)).dropna()
    cur_clean = cur_df.select(F.col(col).cast("double").alias(col)).dropna()

    n_ref = ref_clean.count()
    n_cur = cur_clean.count()
    if n_ref == 0 or n_cur == 0:
        return 0.0

    # ── Calcul des bornes depuis la distribution de référence ─────────────────
    probs      = [round(i / n_bins, 10) for i in range(1, n_bins)]
    raw_splits = ref_clean.stat.approxQuantile(col, probs, 0.05)

    # Déduplication + sentinelles -inf / +inf
    splits = sorted(set(raw_splits))
    splits = [float("-inf")] + splits + [float("inf")]

    # Besoin d'au moins 2 buckets (3 splits) pour que PSI soit défini
    if len(splits) < 3:
        return 0.0

    n_buckets   = len(splits) - 1
    bucketizer  = Bucketizer(
        splits=splits,
        inputCol=col,
        outputCol="_bucket",
        handleInvalid="keep",   # lignes hors-range → bucket spécial NaN ignoré
    )

    # ── Comptage par bucket ───────────────────────────────────────────────────
    def _bucket_counts(df: DataFrame) -> Dict[int, int]:
        return {
            int(r["_bucket"]): r["count"]
            for r in (
                bucketizer.transform(df)
                .filter(F.col("_bucket").isNotNull())
                .groupBy("_bucket")
                .count()
                .collect()
            )
        }

    ref_counts = _bucket_counts(ref_clean)
    cur_counts = _bucket_counts(cur_clean)

    # ── Formule PSI ───────────────────────────────────────────────────────────
    psi = 0.0
    denom_ref = n_ref + EPS * n_buckets
    denom_cur = n_cur + EPS * n_buckets

    for i in range(n_buckets):
        ref_pct = (ref_counts.get(i, 0) + EPS) / denom_ref
        cur_pct = (cur_counts.get(i, 0) + EPS) / denom_cur
        psi    += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)

    return round(abs(psi), 6)


# ──────────────────────────────────────────────────────────────────────────────
# PSI pour toutes les colonnes de features
# ──────────────────────────────────────────────────────────────────────────────

def compute_feature_drift(
    ref_df: DataFrame,
    cur_df: DataFrame,
    feature_cols: List[str],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Calcule le PSI pour chaque colonne de feature.

    Parameters
    ----------
    ref_df       : DataFrame de référence
    cur_df       : DataFrame courant
    feature_cols : liste des colonnes à analyser
    n_bins       : nombre de bins PSI

    Returns
    -------
    Dict { nom_feature : psi_value }
    Les colonnes en erreur retournent float('nan').
    """
    results: Dict[str, float] = {}
    for col in feature_cols:
        try:
            results[col] = compute_psi(ref_df, cur_df, col, n_bins)
        except Exception:
            results[col] = float("nan")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# PSI sur la distribution des scores (churn_proba)
# ──────────────────────────────────────────────────────────────────────────────

def compute_score_drift(
    ref_scored: DataFrame,
    cur_scored: DataFrame,
    score_col: str = "churn_proba",
    n_bins: int = 10,
) -> float:
    """
    Calcule le PSI sur la distribution des probabilités de churn.

    Un score_drift élevé indique que le modèle produit des scores très
    différents sur les nouvelles données → signe de concept drift.

    Parameters
    ----------
    ref_scored : DataFrame scoré de référence (contient score_col)
    cur_scored : DataFrame scoré courant
    score_col  : colonne contenant la probabilité de churn
    n_bins     : nombre de bins PSI

    Returns
    -------
    float : PSI ≥ 0
    """
    return compute_psi(ref_scored, cur_scored, score_col, n_bins)


# ──────────────────────────────────────────────────────────────────────────────
# Interprétation du PSI
# ──────────────────────────────────────────────────────────────────────────────

def drift_status(
    psi: float,
    warning: float  = 0.10,
    critical: float = 0.25,
) -> str:
    """
    Convertit une valeur PSI en statut lisible.

    Returns
    -------
    'STABLE' | 'WARNING' | 'CRITICAL' | 'UNKNOWN'
    """
    if math.isnan(psi):
        return "UNKNOWN"
    if psi < warning:
        return "STABLE"
    if psi < critical:
        return "WARNING"
    return "CRITICAL"


def overall_drift_status(
    feature_drift: Dict[str, float],
    score_drift: float,
    warning: float  = 0.10,
    critical: float = 0.25,
) -> str:
    """
    Retourne le statut global = pire statut parmi features + score.
    """
    all_psi = list(feature_drift.values()) + [score_drift]
    valid   = [p for p in all_psi if not math.isnan(p)]
    if not valid:
        return "UNKNOWN"
    max_psi = max(valid)
    return drift_status(max_psi, warning, critical)
