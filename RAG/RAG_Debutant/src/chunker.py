"""
src/chunker.py — Découpage de texte via LangChain Text Splitters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain propose plusieurs splitters selon le contexte :

  RecursiveCharacterTextSplitter  ← RECOMMANDÉ pour la majorité des cas
  CharacterTextSplitter           ← Simple, basé sur un séparateur unique
  TokenTextSplitter               ← Par nombre exact de tokens (tiktoken)
  MarkdownHeaderTextSplitter      ← Respecte la structure Markdown (h1,h2...)
  HTMLHeaderTextSplitter          ← Respecte la structure HTML
  SemanticChunker                 ← Découpage par sens (embedding-based)

Chaque splitter prend des Documents LangChain en entrée et retourne des Documents.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def get_splitter(strategy: str = "recursive", **kwargs):
    """
    Factory : retourne le bon splitter LangChain selon la stratégie.

    Args:
        strategy: "recursive" | "token" | "markdown" | "semantic"
        **kwargs: chunk_size, chunk_overlap, etc.

    Returns:
        Un TextSplitter LangChain
    """
    chunk_size    = kwargs.get("chunk_size", config.CHUNK_SIZE)
    chunk_overlap = kwargs.get("chunk_overlap", config.CHUNK_OVERLAP)

    if strategy == "recursive":
        # ── Meilleur choix par défaut ──
        # Essaie de couper sur \n\n, puis \n, puis ". ", puis " "
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            add_start_index=True,   # Ajoute "start_index" dans les métadonnées
        )

    elif strategy == "token":
        # ── Découpage par nombre de tokens ──
        # Utile pour respecter précisément les limites de contexte des LLMs
        from langchain.text_splitter import TokenTextSplitter
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name="cl100k_base",  # Tokenizer GPT-4
        )

    elif strategy == "markdown":
        # ── Respecte les headers Markdown ──
        # Idéal pour de la documentation technique
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        headers = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        return MarkdownHeaderTextSplitter(headers_to_split_on=headers)

    elif strategy == "semantic":
        # ── Découpage basé sur le sens ──
        # Regroupe les phrases par similarité sémantique
        # Nécessite sentence-transformers
        from langchain_experimental.text_splitter import SemanticChunker
        from src.embeddings import get_embeddings
        return SemanticChunker(
            embeddings=get_embeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}. Choix : recursive, token, markdown, semantic")


def split_documents(documents: list, strategy: str = "recursive", **kwargs) -> list:
    """
    Découpe une liste de Documents LangChain en chunks.

    Args:
        documents: list[Document] LangChain
        strategy: Stratégie de découpage
        **kwargs: chunk_size, chunk_overlap

    Returns:
        list[Document] — Chunks prêts à être indexés
    """
    splitter = get_splitter(strategy, **kwargs)
    chunks = splitter.split_documents(documents)

    # Filtrer les chunks trop courts
    min_size = kwargs.get("min_chunk_size", config.MIN_CHUNK_SIZE if hasattr(config, "MIN_CHUNK_SIZE") else 50)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= min_size]

    logger.info(
        f"{len(documents)} docs → {len(chunks)} chunks "
        f"[{strategy}, size={kwargs.get('chunk_size', config.CHUNK_SIZE)}]"
    )
    return chunks