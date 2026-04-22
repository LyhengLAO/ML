"""
src/embeddings.py — Embeddings via LangChain + sentence-transformers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain fournit HuggingFaceEmbeddings qui encapsule sentence-transformers.
C'est l'objet utilisé directement par ChromaDB, les retrievers, etc.

Modèles recommandés (open-source, locaux) :
  all-MiniLM-L6-v2                       → rapide, 384 dims
  BAAI/bge-small-en-v1.5                 → meilleur MTEB, 384 dims
  paraphrase-multilingual-MiniLM-L12-v2  → multilingue (FR inclus)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def get_embeddings(
    model_name: str = config.EMBEDDING_MODEL,
    device: str = config.EMBEDDING_DEVICE,
):
    """
    Retourne un objet Embeddings LangChain (HuggingFaceEmbeddings).

    Cet objet est compatible avec tous les composants LangChain :
    Chroma, FAISS, retrievers, etc.

    Exemple :
        emb = get_embeddings()
        vectors = emb.embed_documents(["texte 1", "texte 2"])
        query_vec = emb.embed_query("ma question")
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    print(f"Chargement du modèle d'embedding : {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,   # Nécessaire pour la similarité cosinus
            "batch_size": 32,
        },
    )

    print(f"Embeddings prêts")
    return embeddings


# ── Catalogue des modèles disponibles ───────────────────────
MODELS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "best": "BAAI/bge-small-en-v1.5",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "large": "sentence-transformers/all-mpnet-base-v2",
}