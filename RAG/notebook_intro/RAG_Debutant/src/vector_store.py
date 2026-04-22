"""
src/vector_store.py — Base vectorielle avec LangChain + ChromaDB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain fournit Chroma qui encapsule ChromaDB et expose :
  - add_documents()        ← Indexer des Documents
  - similarity_search()    ← Recherche par similarité cosinus
  - max_marginal_relevance_search() ← Recherche MMR (pertinence + diversité)
  - as_retriever()         ← Convertir en Retriever LangChain (pour les chains)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def get_vector_store(
    embeddings=None,
    collection_name: str = config.COLLECTION_NAME,
    persist_directory: str | Path = config.CHROMA_DIR,
):
    """
    Crée ou charge un VectorStore ChromaDB existant.

    Args:
        embeddings: Objet HuggingFaceEmbeddings (créé si None)
        collection_name: Nom de la collection ChromaDB
        persist_directory: Dossier de stockage local

    Returns:
        langchain_chroma.Chroma — VectorStore LangChain
    """
    from langchain_chroma import Chroma

    if embeddings is None:
        from src.embeddings import get_embeddings
        embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = vector_store._collection.count()
    logger.info(f"🗃️  ChromaDB '{collection_name}' — {count} documents")
    return vector_store


def add_documents(vector_store, documents: list, batch_size: int = 100) -> int:
    """
    Indexe des documents dans le VectorStore par batches.

    Args:
        vector_store: Chroma VectorStore
        documents: list[Document] LangChain (chunks)
        batch_size: Taille des batches

    Returns:
        Nombre de documents ajoutés
    """
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)
        total += len(batch)
        print(f"  Indexé : {total}/{len(documents)} chunks", end="\r")

    print(f"\n{total} chunks indexés dans ChromaDB")
    return total


def get_retriever(
    vector_store,
    search_type: str = config.SEARCH_TYPE,
    top_k: int = config.TOP_K,
):
    """
    Convertit le VectorStore en Retriever LangChain.

    Le Retriever est l'objet utilisé dans les chains LangChain (LCEL, etc.)
    Il expose une interface simple : retriever.invoke("question")

    Args:
        vector_store: Chroma VectorStore
        search_type: "similarity" | "mmr" | "similarity_score_threshold"
        top_k: Nombre de documents à récupérer

    Returns:
        VectorStoreRetriever — Retriever LangChain
    """
    search_kwargs = {"k": top_k}

    if search_type == "mmr":
        # MMR : Maximum Marginal Relevance
        # Équilibre pertinence ET diversité des résultats
        search_kwargs["fetch_k"] = top_k * 4    # Candidats avant sélection MMR
        search_kwargs["lambda_mult"] = 0.5       # 0=diversité pure, 1=pertinence pure

    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = config.SIMILARITY_THRESHOLD

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    logger.info(f"🔍 Retriever : type={search_type}, top_k={top_k}")
    return retriever