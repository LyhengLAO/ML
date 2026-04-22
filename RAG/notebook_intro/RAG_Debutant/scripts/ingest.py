"""
scripts/ingest.py — Ingestion de documents (version LangChain)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage :
    python scripts/ingest.py                      # Tout ingérer depuis data/raw/
    python scripts/ingest.py --source wikipedia   # Wikipedia uniquement
    python scripts/ingest.py --source arxiv       # ArXiv uniquement
    python scripts/ingest.py --source hf          # HuggingFace Datasets
    python scripts/ingest.py --source local       # Fichiers locaux uniquement
    python scripts/ingest.py --reset              # Vider ChromaDB avant
    python scripts/ingest.py --stats              # Stats de la base existante
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from src.embeddings import get_embeddings
from src.vector_store import get_vector_store, add_documents
from src.chunker import split_documents
from src.document_loader import (
    load_directory,
    load_wikipedia,
    load_arxiv,
    load_huggingface_dataset,
)

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format="%(levelname)s — %(message)s")
for lib in ["chromadb", "sentence_transformers", "httpx"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

console = Console()


def ingest(source: str = "all", reset: bool = False):
    """Pipeline d'ingestion complet."""

    console.print(Panel.fit(
        "[bold cyan] Ingestion RAG — LangChain + ChromaDB[/bold cyan]\n"
        f"Source    : [dim]{source}[/dim]\n"
        f"Embedding : [dim]{config.EMBEDDING_MODEL.split('/')[-1]}[/dim]\n"
        f"ChromaDB  : [dim]{config.CHROMA_DIR}[/dim]",
        border_style="cyan"
    ))

    # ── Init ──────────────────────────────────────────────────
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    if reset:
        console.print("[yellow] Réinitialisation ChromaDB...[/yellow]")
        vector_store._client.delete_collection(config.COLLECTION_NAME)
        vector_store = get_vector_store(embeddings)

    all_docs = []

    # ── Sources ───────────────────────────────────────────────

    if source in ("local", "all"):
        console.print("\n[bold] Chargement fichiers locaux...[/bold]")
        if config.RAW_DIR.exists():
            docs = load_directory(config.RAW_DIR)
            all_docs.extend(docs)
            console.print(f"→ {len(docs)} documents")
        else:
            console.print(f"[dim]data/raw/ vide ou absent[/dim]")

    if source in ("wikipedia", "all"):
        console.print("\n[bold] Wikipedia...[/bold]")
        docs = load_wikipedia(config.WIKIPEDIA_TOPICS)
        all_docs.extend(docs)
        console.print(f" → {len(docs)} articles")

    if source in ("arxiv", "all"):
        console.print("\n[bold] ArXiv...[/bold]")
        docs = load_arxiv(config.ARXIV_QUERIES, config.ARXIV_MAX)
        all_docs.extend(docs)
        console.print(f"   → {len(docs)} papers")

    if source in ("hf", "all"):
        console.print("\n[bold] HuggingFace Datasets...[/bold]")
        docs = load_huggingface_dataset()
        all_docs.extend(docs)

    if not all_docs:
        console.print("[red] Aucun document chargé ![/red]")
        return

    console.print(f"\n[bold]Total chargé : {len(all_docs)} documents[/bold]")

    # ── Chunking ──────────────────────────────────────────────
    console.print("\n[bold] Découpage (RecursiveCharacterTextSplitter)...[/bold]")
    chunks = split_documents(all_docs, strategy="recursive")
    console.print(f" → {len(chunks)} chunks")

    # ── Indexation ────────────────────────────────────────────
    console.print("\n[bold] Création des embeddings + indexation ChromaDB...[/bold]")
    added = add_documents(vector_store, chunks)

    # ── Résumé ────────────────────────────────────────────────
    total = vector_store._collection.count()

    # Tableau des sources
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Métrique", style="cyan")
    table.add_column("Valeur", justify="right", style="green")
    table.add_row("Documents chargés", str(len(all_docs)))
    table.add_row("Chunks créés", str(len(chunks)))
    table.add_row("Chunks indexés", str(added))
    table.add_row("Total dans ChromaDB", str(total))
    table.add_row("Modèle embedding", config.EMBEDDING_MODEL.split("/")[-1])
    console.print(table)

    console.print(Panel.fit(
        "[bold green] Ingestion terminée ![/bold green]\n"
        "[dim]→ python main.py[/dim]",
        border_style="green"
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["all", "local", "wikipedia", "arxiv", "hf"],
        default="all"
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        vs = get_vector_store(get_embeddings())
        n = vs._collection.count()
        console.print(Panel.fit(
            f"[bold] ChromaDB Stats[/bold]\n\n"
            f"Collection : {config.COLLECTION_NAME}\n"
            f"Documents  : [bold green]{n}[/bold green]\n"
            f"Embedding  : {config.EMBEDDING_MODEL.split('/')[-1]}\n"
            f"Dossier    : {config.CHROMA_DIR}",
            border_style="blue"
        ))
        return

    ingest(source=args.source, reset=args.reset)


if __name__ == "__main__":
    main()