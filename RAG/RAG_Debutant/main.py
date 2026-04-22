"""
main.py — Point d'entrée RAG LangChain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage :
    python main.py                            # Mode interactif
    python main.py -q "Qu'est-ce que BERT ?" # Question directe
    python main.py --stream                   # Streaming token par token
    python main.py --demo                     # Questions de démonstration
    python main.py --stats                    # Stats du pipeline
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel

import config

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format="%(levelname)s — %(message)s")
for lib in ["chromadb", "sentence_transformers", "httpx", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

console = Console()

DEMO_QUESTIONS = [
    "Qu'est-ce que le mécanisme d'attention dans les Transformers ?",
    "Quelle est la différence entre BERT et GPT ?",
    "Comment fonctionne ChromaDB pour la recherche vectorielle ?",
    "Quels sont les avantages du RAG par rapport au fine-tuning ?",
    "Qu'est-ce que la similarité cosinus dans les embeddings ?",
]

BANNER = """
╔══════════════════════════════════════════════════════════╗
║            RAG PROJECT — LangChain + ChromaDB            ║
║          100% Open-Source  •  Local  •  Privé            ║
╚══════════════════════════════════════════════════════════╝
"""


def interactive_mode(rag, stream: bool = False):
    console.print(Panel.fit(
        "[bold cyan]💬 Mode Interactif[/bold cyan]\n"
        "Commandes : [bold]quit[/bold] | [bold]demo[/bold] | [bold]stats[/bold] | [bold]sources[/bold]",
        border_style="cyan"
    ))

    show_sources = False

    while True:
        try:
            question = console.input("\n[bold yellow]❓ Question :[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n👋 Au revoir !")
            break

        if not question:
            continue

        match question.lower():
            case "quit" | "exit" | "q":
                console.print("👋 Au revoir !")
                break
            case "demo":
                run_demo(rag)
                continue
            case "stats":
                print_stats(rag)
                continue
            case "sources":
                show_sources = not show_sources
                console.print(f"[dim]Affichage des sources : {'ON ✓' if show_sources else 'OFF'}[/dim]")
                continue

        try:
            if stream:
                console.print("\n[bold green]💬 Réponse :[/bold green]")
                for token in rag.stream(question):
                    print(token, end="", flush=True)
                print()
            elif show_sources:
                with console.status("Recherche + génération..."):
                    out = rag.query_with_sources(question)
                print(f"\n {out['answer']}")
                print(f"\n Sources :")
                for doc in out["source_documents"]:
                    title = doc.metadata.get("title", doc.metadata.get("filename", "?"))
                    print(f" • {str(title)[:70]}")
            else:
                with console.status("Recherche + génération..."):
                    result = rag.query(question)
                result.display()

        except Exception as e:
            console.print(f"[red] Erreur : {e}[/red]")


def run_demo(rag):
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"\n[dim]━━ Demo {i}/{len(DEMO_QUESTIONS)} ━━[/dim]")
        try:
            with console.status(f"{q[:50]}..."):
                result = rag.query(q)
            result.display()
        except Exception as e:
            console.print(f"[red]{e}[/red]")
        if i < len(DEMO_QUESTIONS):
            input("[dim]Entrée pour continuer...[/dim]")


def print_stats(rag):
    stats = rag.get_stats()
    console.print(Panel.fit(
        f"[bold] Pipeline Stats[/bold]\n\n"
        f"Documents indexés : [bold green]{stats['documents_indexed']}[/bold green]\n"
        f"Embedding         : {stats['embedding_model'].split('/')[-1]}\n"
        f"LLM backend       : {stats['llm_backend']}\n"
        f"Recherche         : {stats['search_type']}\n"
        f"Top-K             : {stats['top_k']}",
        border_style="blue"
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q")
    parser.add_argument("--stream", "-s", action="store_true")
    parser.add_argument("--demo", "-d", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--search", choices=["similarity", "mmr"], default=config.SEARCH_TYPE)
    args = parser.parse_args()

    console.print(BANNER, style="bold cyan")

    from src.rag_pipeline import RAGPipeline
    console.print("[dim]Initialisation du pipeline...[/dim]")

    try:
        rag = RAGPipeline(top_k=args.top_k, search_type=args.search)
    except Exception as e:
        console.print(f"[red] {e}[/red]")
        console.print("\n[yellow]Lancez d'abord :[/yellow]")
        console.print("  python scripts/ingest.py")
        sys.exit(1)

    n = rag.vector_store._collection.count()
    if n == 0:
        console.print("[yellow] Base vide ! Lancez : python scripts/ingest.py[/yellow]")
        sys.exit(1)

    console.print(f"[green] Prêt — {n} chunks indexés[/green]\n")

    if args.stats:
        print_stats(rag)
    elif args.demo:
        run_demo(rag)
    elif args.question:
        if args.stream:
            for token in rag.stream(args.question):
                print(token, end="", flush=True)
            print()
        else:
            result = rag.query(args.question)
            result.display()
    else:
        interactive_mode(rag, stream=args.stream)


if __name__ == "__main__":
    main()