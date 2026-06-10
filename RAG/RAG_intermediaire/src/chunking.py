"""Stratégies de découpage (chunking) des documents."""
"""ici comme on travaille sur une petit corpus donc la différence entre recursivechunking et sementicchunking ets negligeable"""
"""donc on utilise le plus simple"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Découpe récursif par caractères, en respectant au mieux la structure
    (paragraphes > phrases > mots). L'overlap préserve la continuité entre
    chunks voisins."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i
    return chunks
