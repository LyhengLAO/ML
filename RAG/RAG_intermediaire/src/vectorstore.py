"""Gestion du vector store ChromaDB (persistance locale, sans clé API)."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

def build_vectorstore(chunks: list[Document],
                      embeddings: Embeddings,
                      persist_dir: str,
                      collection_name: str) -> Chroma:
    """(Ré)indexe les chunks dans une collection Chroma persistée sur disque.

    La collection est recréée à zéro pour garantir un index propre et
    reproductible entre deux runs.
    """
    vs = Chroma(collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir
    )

    try:
        existing = vs.get()
        if existing and existing.get("ids"):
            vs.delete(ids=existing["ids"])
    except Exception:  # noqa: BLE001
        pass

    vs.add_documents(chunks)
    return vs

def load_vectorstore(
    embeddings: Embeddings,
    persist_dir: str,
    collection_name: str,
) -> Chroma:
    """Recharge une collection Chroma déjà persistée (sans réindexer)."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
