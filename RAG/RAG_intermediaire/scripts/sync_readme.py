"""Régénère la table de métriques du README depuis results/metrics.json.

Source unique de vérité : la table du README provient TOUJOURS du fichier de
métriques produit par run_evaluation.py — jamais de valeurs saisies à la main.

    python scripts/sync_readme.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import comparison_markdown, update_readme_table  # noqa: E402


def main() -> int:
    metrics_path = ROOT / "results" / "metrics.json"
    if not metrics_path.exists():
        print("results/metrics.json introuvable — lancez d'abord run_evaluation.py")
        return 1

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    # On régénère la table à partir des agrégats (pas du markdown stocké),
    # pour garantir une cohérence stricte avec les chiffres.
    table = comparison_markdown(data["baseline"], data["optimized"])

    readme = str(ROOT / "README.md")
    if update_readme_table(readme, table):
        print("README.md synchronisé depuis results/metrics.json")
        mode = data.get("config", {}).get("mode", "?")
        print(f"  mode du run : {mode}")
        return 0
    print("Marqueurs <!--METRICS_START/END--> absents du README.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
