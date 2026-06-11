"""Pipeline RAG OPTIMISÉ.

Améliorations vs baseline :
  1. Chunking plus fin avec overlap.
  2. Recherche HYBRIDE : dense (embeddings) + lexicale (BM25) via EnsembleRetriever.
  3. MULTI-QUERY : le LLM reformule la question (désactivé en mode extractif).
  4. RERANKING : cross-encoder (prod) ou similarité TF-IDF (offline), top_n.
  5. Prompt durci anti-hallucination.
"""
from __future__ import annotations

from typing import Any, Callable

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .base import BaseRAGPipeline
from .prompts import OPTIMIZED_PROMPT

class OptimizedRAGPipeline(BaseRAGPipeline):
    name = "optimized"

    def __init__(self, vectorstore, chunks, llm, cfg, embeddings=None):
        super().__init__(llm=llm, prompt=OPTIMIZED_PROMPT)

        # 1) Retriever dense (embeddings / Chroma)
        dense = vectorstore.as_retriever(search_kwargs={"k": cfg.get("retrieval_k_dense", 10)})

        # 2) Retriever lexical (BM25) sur les mêmes chunks
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = cfg.get("retrieval_k_bm25", 10) # == bm25 = BM25Retriever.from_documents(chunks, k=cfg.get("retrieval_k_bm25", 10))

        # 3) Fusion hybride dense + BM25
        ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=cfg.get("ensemble_weights", [0.5, 0.5]))

        # 4) Multi-query optionnel (reformulation par le LLM)
        self.base_retriever = ensemble
        if cfg.get("use_multi_query", True):
            from langchain.retrievers.multi_query import MultiQueryRetriever
            self.base_retriever = MultiQueryRetriever.from_llm(
                retriever=ensemble, llm=llm
            )

        # 5) Reranker : cross-encoder (prod) ou TF-IDF (offline)
        self.top_n = cfg.get("rerank_top_n", 4)
        self._reranker: Callable[[str, list], list] = self._build_reranker(cfg, chunks)

    def _build_reranker(self, cfg, chunks):
        reranker_type = cfg.get("reranker", "cross_encoder")

        if reranker_type == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from .offline import tfidf_rerank

            vec = TfidfVectorizer(stop_words="english")
            vec.fit([c.page_content for c in chunks])
            return lambda q, docs: tfidf_rerank(q, docs, self.top_n, vec)

        # Cross-encoder (production) — imports paresseux.
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        cross_encoder = HuggingFaceCrossEncoder(
            model_name=cfg.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=self.top_n)
        return lambda q, docs: list(compressor.compress_documents(docs, q))

    def retrieve(self, question: str) -> list[Document]:
        candidates = list(self.base_retriever.invoke(question))
        return self._reranker(question, candidates)
