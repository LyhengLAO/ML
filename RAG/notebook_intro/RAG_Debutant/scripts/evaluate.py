"""
scripts/evaluate.py — Évaluation du pipeline RAG depuis le terminal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage :
    # Évaluation complète avec RAGAS (recommandé)
    python scripts/evaluate.py

    # Sans ground truth (Faithfulness + Answer Relevancy seulement)
    python scripts/evaluate.py --no-reference

    # Métriques simples sans RAGAS (100% local, pas d'API)
    python scripts/evaluate.py --simple

    # Sauvegarder le rapport JSON
    python scripts/evaluate.py --output reports/eval_v1.json

    # Utiliser vos propres questions depuis un fichier JSON
    python scripts/evaluate.py --questions my_questions.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator

logging.basicConfig(level=logging.WARNING)
console = Console()

# ── Dataset d'évaluation par défaut (NLP/RAG) ────────────────
DEFAULT_EVAL_SET = [
    {
        "question": "Qu'est-ce que le mécanisme d'attention dans les Transformers ?",
        "ground_truth": (
            "Le mécanisme d'attention permet au modèle de calculer un score de pertinence "
            "entre chaque paire de tokens (Query, Key, Value). Il produit une somme pondérée "
            "des vecteurs Value, où les poids sont déterminés par la similarité entre la Query "
            "et les Keys. Le mécanisme multi-têtes exécute plusieurs attentions en parallèle."
        ),
    },
    {
        "question": "Quelle est la différence entre BERT et GPT ?",
        "ground_truth": (
            "BERT est un encodeur bidirectionnel : il lit le texte dans les deux directions "
            "et est optimisé pour la compréhension. GPT est un décodeur auto-régressif : "
            "il prédit le token suivant et est optimisé pour la génération de texte. "
            "BERT utilise le masquage de tokens (MLM), GPT utilise la prédiction causale."
        ),
    },
    {
        "question": "Comment fonctionne la recherche par similarité cosinus ?",
        "ground_truth": (
            "La similarité cosinus mesure l'angle entre deux vecteurs. Elle est calculée "
            "comme le produit scalaire divisé par le produit des normes : cos(θ) = A·B / (||A||·||B||). "
            "Une valeur proche de 1 indique des vecteurs très similaires, proche de 0 indique "
            "des vecteurs orthogonaux. C'est la métrique standard pour la recherche sémantique."
        ),
    },
    {
        "question": "Qu'est-ce que le RAG et quels sont ses avantages ?",
        "ground_truth": (
            "Le RAG (Retrieval-Augmented Generation) combine la recherche d'information "
            "dans une base de connaissances externe avec la génération de texte par un LLM. "
            "Ses avantages sont : réduction des hallucinations, mise à jour facile des "
            "connaissances sans re-entraîner le modèle, traçabilité des sources, "
            "et personnalisation par domaine."
        ),
    },
    {
        "question": "À quoi sert ChromaDB dans un pipeline RAG ?",
        "ground_truth": (
            "ChromaDB est une base de données vectorielle open-source qui stocke les embeddings "
            "des chunks de documents. Elle permet la recherche sémantique par similarité cosinus "
            "très rapide sur des milliers de vecteurs. Elle stocke aussi les textes originaux "
            "et les métadonnées associées à chaque vecteur."
        ),
    },
]


def load_eval_set(filepath: str | None) -> tuple[list[str], list[str]]:
    """Charge un jeu d'évaluation depuis un fichier JSON ou utilise le défaut."""
    if filepath:
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        questions = [d["question"] for d in data]
        ground_truths = [d.get("ground_truth", "") for d in data]
        console.print(f"[dim]✓ {len(questions)} questions chargées depuis {filepath}[/dim]")
    else:
        questions = [d["question"] for d in DEFAULT_EVAL_SET]
        ground_truths = [d["ground_truth"] for d in DEFAULT_EVAL_SET]
        console.print(f"[dim]✓ {len(questions)} questions de démonstration utilisées[/dim]")

    return questions, ground_truths


def save_report(result, output_path: str):
    """Sauvegarde le rapport en JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_backend": config.LLM_BACKEND,
            "ollama_model": config.OLLAMA_MODEL,
            "search_type": config.SEARCH_TYPE,
            "top_k": config.TOP_K,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        "metrics": result if isinstance(result, dict) else result.to_dict(),
        "per_question": result.per_question if hasattr(result, "per_question") else result.get("per_question", []),
    }
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]✓ Rapport sauvegardé : {output_path}[/green]")


def display_simple_report(summary: dict):
    """Affiche un rapport pour les métriques simples."""
    table = Table(title="Métriques simples du pipeline", show_header=True, header_style="bold blue")
    table.add_column("Métrique", style="cyan")
    table.add_column("Valeur", justify="right", style="green")
    table.add_column("Interprétation", style="dim")

    bar = lambda v, mx=1.0: "█" * int((v/mx) * 10) + "░" * (10 - int((v/mx) * 10))

    rows = [
        ("Nb questions",         str(summary["n_questions"]),                      ""),
        ("Score retrieval moyen",f"{summary['avg_retrieval_score']:.3f}  {bar(summary['avg_retrieval_score'])}",  "pertinence chunks [0-1]"),
        ("Latence moyenne",      f"{summary['avg_latency_ms']:.0f} ms",             "temps de réponse"),
        ("Longueur réponse",     f"{summary['avg_answer_words']:.0f} mots",         "verbosité"),
        ("Diversité sources",    f"{summary['avg_source_diversity']:.1f} origines", "variété des sources"),
    ]
    if "avg_token_overlap" in summary:
        rows.append((
            "Token overlap F1",
            f"{summary['avg_token_overlap']:.3f}  {bar(summary['avg_token_overlap'])}",
            "chevauchement réponse/référence"
        ))

    for row in rows:
        table.add_row(*row)

    console.print(table)

    # Détail par question
    console.print("\n[bold]Détail par question :[/bold]")
    for i, r in enumerate(summary["per_question"], 1):
        score = r.get("avg_retrieval_score", 0)
        overlap = f"  overlap={r['token_overlap']:.3f}" if "token_overlap" in r else ""
        console.print(
            f"  [{i}] score={score:.3f}  latence={r['latency_ms']:.0f}ms"
            f"{overlap}  → {r['question'][:55]}..."
        )


def main():
    parser = argparse.ArgumentParser(description="Évaluation du pipeline RAG")
    parser.add_argument("--simple", action="store_true",
                        help="Métriques simples sans RAGAS (100%% local)")
    parser.add_argument("--no-reference", action="store_true",
                        help="Sans ground truth (Faithfulness + Answer Relevancy)")
    parser.add_argument("--questions", type=str, default=None,
                        help="Fichier JSON avec questions et ground_truths")
    parser.add_argument("--output", type=str, default=None,
                        help="Chemin du rapport JSON de sortie")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]📊 Évaluation du Pipeline RAG[/bold cyan]\n"
        f"Mode : [dim]{'simple (sans RAGAS)' if args.simple else 'no-reference' if args.no_reference else 'RAGAS complet'}[/dim]",
        border_style="cyan"
    ))

    # Charger le dataset
    questions, ground_truths = load_eval_set(args.questions)

    # Initialiser le pipeline
    console.print("\n[dim]Initialisation du pipeline RAG...[/dim]")
    try:
        rag = RAGPipeline()
    except Exception as e:
        console.print(f"[red] Erreur pipeline : {e}[/red]")
        console.print("[yellow]→ Avez-vous lancé : python scripts/ingest.py ?[/yellow]")
        sys.exit(1)

    n_docs = rag.vector_store._collection.count()
    if n_docs == 0:
        console.print("[red] Base vectorielle vide ! Lancez d'abord ingest.py[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Pipeline prêt — {n_docs} chunks indexés[/green]\n")
    evaluator = RAGEvaluator(rag)

    # Lancer l'évaluation
    if args.simple:
        result = evaluator.evaluate_simple(questions, ground_truths if not args.no_reference else None)
        display_simple_report(result)
        if args.output:
            save_report(result, args.output)

    elif args.no_reference:
        result = evaluator.evaluate_no_reference(questions)
        result.display()
        if args.output:
            save_report(result, args.output)

    else:
        result = evaluator.evaluate(questions, ground_truths)
        result.display()

        # Tableau détaillé par question
        if result.per_question:
            table = Table(title="Détail par question", show_header=True, header_style="bold blue")
            table.add_column("#", justify="right", style="dim")
            table.add_column("Question", max_width=38)
            table.add_column("Faith.", justify="right")
            table.add_column("Relev.", justify="right")
            table.add_column("Prec.", justify="right")
            table.add_column("Recall", justify="right")
            for i, row in enumerate(result.per_question, 1):
                table.add_row(
                    str(i),
                    row["question"][:38] + "...",
                    f"{row['faithfulness']:.2f}",
                    f"{row['answer_relevancy']:.2f}",
                    f"{row['context_precision']:.2f}",
                    f"{row['context_recall']:.2f}",
                )
            console.print(table)

        if args.output:
            save_report(result, args.output)


if __name__ == "__main__":
    main()