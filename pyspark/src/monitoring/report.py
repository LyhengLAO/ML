"""
Génération du rapport de monitoring.

Fonctions :
  build_report(...)   → dict structuré (sérialisable JSON)
  print_report(...)   → affichage console formaté avec barres et icônes
  save_report(...)    → sauvegarde JSON sur disque
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.monitoring.drift import drift_status, overall_drift_status


# ──────────────────────────────────────────────────────────────────────────────
# Construction du rapport
# ──────────────────────────────────────────────────────────────────────────────

def build_report(
    feature_drift: Dict[str, float],
    score_drift: float,
    model_metrics: Optional[Dict[str, Any]] = None,
    cfg=None,
) -> Dict[str, Any]:
    """
    Construit un rapport de monitoring structuré et sérialisable en JSON.

    Parameters
    ----------
    feature_drift  : Dict {feature: psi}  depuis compute_feature_drift()
    score_drift    : PSI sur churn_proba  depuis compute_score_drift()
    model_metrics  : métriques du modèle  (auc, f1, accuracy…)
    cfg            : Config (pour lire les seuils PSI)

    Returns
    -------
    Dict prêt à être sérialisé en JSON.
    """
    warning  = cfg.monitoring.psi_threshold_warning  if cfg else 0.10
    critical = cfg.monitoring.psi_threshold_critical if cfg else 0.25

    # Statuts par feature
    feature_report: Dict[str, Any] = {}
    for col, psi in feature_drift.items():
        psi_val = round(psi, 6) if not math.isnan(psi) else None
        feature_report[col] = {
            "psi":    psi_val,
            "status": drift_status(psi, warning, critical),
        }

    # Score drift
    score_psi_val = round(score_drift, 6) if not math.isnan(score_drift) else None
    score_status  = drift_status(score_drift, warning, critical)

    # Statut global
    global_status = overall_drift_status(feature_drift, score_drift, warning, critical)

    # Résumé
    statuses = [v["status"] for v in feature_report.values()]
    summary = {
        "n_features_stable":   statuses.count("STABLE"),
        "n_features_warning":  statuses.count("WARNING"),
        "n_features_critical": statuses.count("CRITICAL"),
        "n_features_unknown":  statuses.count("UNKNOWN"),
        "max_feature_psi":     round(
            max((p for p in feature_drift.values() if not math.isnan(p)), default=0.0),
            6,
        ),
        "score_drift_psi": score_psi_val,
    }

    return {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "global_status": global_status,
        "thresholds": {
            "warning":  warning,
            "critical": critical,
        },
        "feature_drift": feature_report,
        "score_drift": {
            "psi":    score_psi_val,
            "status": score_status,
        },
        "model_metrics": model_metrics or {},
        "summary":       summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Affichage console
# ──────────────────────────────────────────────────────────────────────────────

_STATUS_ICON = {
    "STABLE":   "🟢",
    "WARNING":  "🟡",
    "CRITICAL": "🔴",
    "UNKNOWN":  "⚪",
}

_PSI_BAR_MAX = 0.50   # PSI au-delà duquel la barre est pleine


def _psi_bar(psi: Optional[float], width: int = 20) -> str:
    """Barre de progression ASCII proportionnelle au PSI."""
    if psi is None or math.isnan(psi):
        return "?" * width
    filled = int(min(psi, _PSI_BAR_MAX) / _PSI_BAR_MAX * width)
    return "█" * filled + "░" * (width - filled)


def print_report(report: Dict[str, Any]) -> None:
    """
    Affiche le rapport de monitoring dans la console avec icônes et barres.
    """
    gs   = report["global_status"]
    icon = _STATUS_ICON.get(gs, "?")

    print()
    print("=" * 70)
    print(" RAPPORT DE MONITORING — DATA DRIFT (PSI)")
    print("=" * 70)
    print(f"  Généré le     : {report['generated_at']}")
    print(f"  Statut global : {icon}  {gs}")
    thr = report["thresholds"]
    print(f"  Seuils        : WARNING ≥ {thr['warning']}   CRITICAL ≥ {thr['critical']}")

    # ── Feature drift ──────────────────────────────────────────────────────────
    print()
    print(f"  {'Feature':<16} {'PSI':>7}  {'Statut':<10}  {'Distribution drift'}")
    print(f"  {'─' * 16} {'─' * 7}  {'─' * 10}  {'─' * 20}")

    for col, info in report["feature_drift"].items():
        psi    = info["psi"]
        status = info["status"]
        si     = _STATUS_ICON.get(status, "?")
        bar    = _psi_bar(psi)
        psi_str = f"{psi:.4f}" if psi is not None else "  N/A "
        print(f"  {col:<16} {psi_str:>7}  {si} {status:<8}  {bar}")

    # ── Score drift ────────────────────────────────────────────────────────────
    print()
    sd     = report["score_drift"]
    si     = _STATUS_ICON.get(sd["status"], "?")
    psi_s  = f"{sd['psi']:.4f}" if sd["psi"] is not None else "  N/A "
    bar    = _psi_bar(sd["psi"])
    print(f"  {'churn_proba':<16} {psi_s:>7}  {si} {sd['status']:<8}  {bar}  ← score drift")

    # ── Résumé ─────────────────────────────────────────────────────────────────
    s = report["summary"]
    print()
    print("  Résumé :")
    print(f"    Features STABLE   : {s['n_features_stable']}")
    print(f"    Features WARNING  : {s['n_features_warning']}")
    print(f"    Features CRITICAL : {s['n_features_critical']}")
    print(f"    PSI max (features): {s['max_feature_psi']:.4f}")
    print(f"    Score drift PSI   : {s['score_drift_psi']}")

    # ── Métriques modèle ───────────────────────────────────────────────────────
    if report.get("model_metrics"):
        m = report["model_metrics"]
        print()
        print("  Métriques du modèle (jeu de test) :")
        for key in ("auc", "f1", "accuracy", "precision", "recall"):
            if key in m:
                print(f"    {key:<12} : {m[key]}")

    # ── Recommandation ─────────────────────────────────────────────────────────
    print()
    if gs == "CRITICAL":
        print("  ⚠️  RECOMMANDATION : drift critique détecté.")
        print("      → Réévaluer le modèle ou ré-entraîner sur des données récentes.")
        print("         python -m cli.pipeline --force-download")
    elif gs == "WARNING":
        print("  ⚠️  RECOMMANDATION : surveiller l'évolution du drift.")
        print("      → Vérifier les features WARNING et planifier un ré-entraînement.")
    else:
        print("  ✅ Distribution stable — aucune action requise.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde JSON
# ──────────────────────────────────────────────────────────────────────────────

def save_report(report: Dict[str, Any], path: str) -> None:
    """
    Sauvegarde le rapport de monitoring en JSON.

    Crée les dossiers parents si nécessaire.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
