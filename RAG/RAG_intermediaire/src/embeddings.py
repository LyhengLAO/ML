"""Fabrique d'embeddings : HuggingFace (production) ou TF-IDF (offline)."""
from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings


def build_embeddings(cfg: dict[str, Any]) -> Embeddings:
    """Construit le modèle d'embeddings selon cfg['provider'].

    - 'huggingface' (défaut) : sentence-transformers local (ex. all-MiniLM-L6-v2).
      Le même objet DOIT encoder corpus ET requêtes (géométrie partagée).
    - 'tfidf' : embeddings TF-IDF scikit-learn, 100 % offline (CI / démo).
      À fitter sur le corpus avant indexation (géré par factory.build_all).
    """
    provider = cfg.get("provider", "huggingface")

    if provider == "tfidf":
        from .offline import TfidfEmbeddings
        return TfidfEmbeddings()

    # Import paresseux : n'impose torch/sentence-transformers qu'en prod.
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=cfg["model_name"],
        model_kwargs={"device": cfg.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": cfg.get("normalize", True)},
    )
