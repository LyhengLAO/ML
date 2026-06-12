"""Évaluation comparative complète : baseline vs optimisé.

Sauvegarde :
  - results/metrics.json        (agrégats des deux pipelines + delta + table MD)
  - results/metrics.csv         (une ligne par pipeline)
  - results/per_question_*.csv  (détail par question)
Et injecte la table comparative dans le README (entre les marqueurs METRICS).

Usage :
  python scripts/run_evaluation.py            # mode production (Ollama + MiniLM)
  python scripts/run_evaluation.py --offline  # mode offline (TF-IDF + extractif)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.evaluation import evaluate_pipeline, run_ragas  # noqa: E402
from src.factory import build_all  # noqa: E402
from src.llm import check_ollama  # noqa: E402
from src.utils import (  # noqa: E402
    comparison_markdown,
    save_json,
    save_metrics_csv,
    save_per_question_csv,
    update_readme_table,
)


def apply_offline(cfg):
    """Force les providers offline (aucun téléchargement, aucun Ollama)."""
    cfg.raw["embeddings"]["provider"] = "tfidf"
    cfg.raw["llm"]["provider"] = "extractive"
    cfg.raw["optimized"]["reranker"] = "tfidf"
    cfg.raw["evaluation"]["run_ragas"] = False
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                        help="Mode offline : TF-IDF + lecteur extractif (CI/démo).")
    args = parser.parse_args()

    cfg = load_config()
    if args.offline:
        cfg = apply_offline(cfg)
        print(">> Mode OFFLINE (TF-IDF + extractif)")

    ok, msg = check_ollama(cfg.llm)
    print(f">> {msg}")
    if not ok:
        print("   La génération échouera. Utilisez --offline ou démarrez Ollama.")
        return 1

    with open(cfg.eval_dataset, encoding="utf-8") as f:
        eval_items = json.load(f)["items"]
    print(f">> Jeu d'évaluation : {len(eval_items)} questions")

    print(">> Construction des pipelines...")
    built = build_all(cfg)
    embeddings = built.embeddings  # réutilise l'objet (fitté en mode TF-IDF)
    eval_cfg = cfg.evaluation
    k = eval_cfg["retrieval_k"]
    thr = eval_cfg.get("semantic_similarity_threshold", 0.6)

    print("\n>> Évaluation BASELINE")
    base_eval = evaluate_pipeline(built.baseline, eval_items, embeddings, k, thr)
    print("\n>> Évaluation OPTIMISÉ")
    opt_eval = evaluate_pipeline(built.optimized, eval_items, embeddings, k, thr)

    base_agg, opt_agg = base_eval["aggregate"], opt_eval["aggregate"]

    if eval_cfg.get("run_ragas", False):
        print("\n>> RAGAS (baseline)...")
        base_agg.update(run_ragas(base_eval["results"], eval_items, built.baseline.llm, embeddings))
        print(">> RAGAS (optimisé)...")
        opt_agg.update(run_ragas(opt_eval["results"], eval_items, built.optimized.llm, embeddings))

    results_dir = cfg.results_dir
    save_metrics_csv([base_agg, opt_agg], f"{results_dir}/metrics.csv")
    save_per_question_csv(base_eval["per_question"], f"{results_dir}/per_question_baseline.csv")
    save_per_question_csv(opt_eval["per_question"], f"{results_dir}/per_question_optimized.csv")

    table_md = comparison_markdown(base_agg, opt_agg)
    mode = "offline (TF-IDF + extractif)" if args.offline else f"{cfg.llm['model_name']} via Ollama"
    save_json(
        {
            "baseline": base_agg,
            "optimized": opt_agg,
            "config": {
                "mode": mode,
                "embeddings": cfg.embeddings.get("model_name") if not args.offline else "tfidf",
                "llm": cfg.llm.get("model_name") if not args.offline else "extractive",
                "retrieval_k": k,
                "n_docs": built.n_docs,
            },
            "comparison_markdown": table_md,
        },
        f"{results_dir}/metrics.json",
    )

    print("\n" + "=" * 70 + "\nRÉSULTATS COMPARATIFS\n" + "=" * 70)
    print(table_md)

    readme = str(Path(__file__).resolve().parents[1] / "README.md")
    if update_readme_table(readme, table_md):
        print("\n>> Table injectée dans README.md")
    print(f">> Métriques sauvegardées dans {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
