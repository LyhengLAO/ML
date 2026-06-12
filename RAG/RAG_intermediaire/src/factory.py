"""Fabrique : assemble de bout en bout les pipelines baseline et optimisé.

Réutilisé par build_index.py, run_evaluation.py et l'app Streamlit pour éviter
toute duplication de logique d'indexation. Gère les providers prod (HF + Ollama)
et offline (TF-IDF + extractif).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .baseline_pipeline import BaselineRAGPipeline
from .chunking import split_documents
from .config import Config
from .data_loader import load_text_documents
from .embeddings import build_embeddings
from .llm import build_llm
from .optimized_pipeline import OptimizedRAGPipeline
from .vectorstore import build_vectorstore

@dataclass
class BuiltPipelines:
    baseline: BaselineRAGPipeline
    optimized: OptimizedRAGPipeline
    embeddings: Embeddings
    n_docs: int
    n_chunks_baseline: int
    n_chunks_optimized: int


def build_all(cfg: Config) -> BuiltPipelines:
    """Charge les données, indexe les deux collections et instancie les pipelines."""
    embeddings = build_embeddings(cfg.embeddings)
    llm = build_llm(cfg.llm)

    documents: list[Document] = load_text_documents(cfg.raw_data_dir)

    b_cfg, o_cfg = cfg.baseline, cfg.optimized
    baseline_chunks = split_documents(documents, b_cfg["chunk_size"], b_cfg["chunk_overlap"])
    optimized_chunks = split_documents(documents, o_cfg["chunk_size"], o_cfg["chunk_overlap"])

    # En mode TF-IDF, on fitte l'objet embeddings sur l'ensemble du corpus
    # pour un espace vectoriel cohérent entre les deux collections.
    if cfg.embeddings.get("provider", "huggingface") == "tfidf":
        all_texts = [c.page_content for c in baseline_chunks + optimized_chunks]
        embeddings.fit(all_texts)  # type: ignore[attr-defined]

    # En mode extractif, le multi-query (qui requiert un vrai LLM) est désactivé.
    if cfg.llm.get("provider", "ollama") == "extractive":
        o_cfg = {**o_cfg, "use_multi_query": False}

    # --- Baseline ---
    baseline_vs = build_vectorstore(
        baseline_chunks, embeddings, cfg.chroma_dir, b_cfg["collection_name"]
    )
    baseline = BaselineRAGPipeline(vectorstore=baseline_vs, llm=llm, k=b_cfg["retrieval_k"])

    # --- Optimisé ---
    optimized_vs = build_vectorstore(
        optimized_chunks, embeddings, cfg.chroma_dir, o_cfg["collection_name"]
    )
    optimized = OptimizedRAGPipeline(
        vectorstore=optimized_vs, chunks=optimized_chunks, llm=llm,
        cfg=o_cfg, embeddings=embeddings,
    )

    return BuiltPipelines(
        baseline=baseline,
        optimized=optimized,
        embeddings=embeddings,
        n_docs=len(documents),
        n_chunks_baseline=len(baseline_chunks),
        n_chunks_optimized=len(optimized_chunks),
    )
