"""Sauvegarde des métriques (JSON + CSV) et génération de tableaux Markdown."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Any, path: str) ->None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_metric_csv(rows: list[dict], path: str) -> None:
    """Écrit une ligne par pipeline avec toutes les métriques en colonnes."""
    ensure_dir(str(Path(path).parent))
    if not rows:
        return
    # Union ordonnée des clés.
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_per_question_csv(records: list[dict], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
                   for k, v in r.items()}
            writer.writerow(row)


# Métriques pour lesquelles "plus c'est haut, mieux c'est".
HIGHER_IS_BETTER = {
    "hit_rate@k", "mrr@k", "precision@k", "context_recall",
    "answer_similarity", "answer_correctness", "rouge_l", "token_f1",
    "ragas_faithfulness", "ragas_answer_relevancy",
    "ragas_context_precision", "ragas_context_recall",
}


def comparison_markdown(baseline: dict, optimized: dict) -> str:
    """Tableau Markdown comparatif baseline vs optimisé, avec delta et %."""
    metrics = [k for k in baseline.keys()
               if k not in ("pipeline", "n_questions")
               and isinstance(baseline[k], (int, float))]

    lines = [
        "| Métrique | Baseline | Optimisé | Δ | Amélioration |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        b, o = baseline[m], optimized.get(m, 0)
        delta = o - b
        if m == "avg_latency_s":  # plus c'est bas, mieux c'est
            pct = (-delta / b * 100) if b else 0.0
            arrow = "🟢" if delta < 0 else ("🔴" if delta > 0 else "⚪")
        else:
            pct = (delta / b * 100) if b else (100.0 if o > 0 else 0.0)
            arrow = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
        lines.append(
            f"| `{m}` | {b:.3f} | {o:.3f} | {delta:+.3f} | {arrow} {pct:+.1f}% |"
        )
    return "\n".join(lines)


def update_readme_table(readme_path: str, table_md: str) -> bool:
    """Injecte le tableau entre les marqueurs <!--METRICS_START--> / END."""
    start, end = "<!--METRICS_START-->", "<!--METRICS_END-->"
    p = Path(readme_path)
    if not p.exists():
        return False
    content = p.read_text(encoding="utf-8")
    if start not in content or end not in content:
        return False
    before = content.split(start)[0]
    after = content.split(end)[1]
    new = f"{before}{start}\n\n{table_md}\n\n{end}{after}"
    p.write_text(new, encoding="utf-8")
    return True

