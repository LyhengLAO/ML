"""
src/rag_pipeline.py — Pipeline RAG avec LangChain LCEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
LCEL = LangChain Expression Language
Syntaxe : composant1 | composant2 | composant3
 
Le pipeline RAG en LCEL :
 
  retriever | prompt | llm | parser
 
Deux variantes :
  1. RAGChain      — Réponse simple (string)
  2. RAGChainFull  — Réponse + sources (dict avec question/answer/sources)
"""
 
import logging
from dataclasses import dataclass, field
from typing import Iterator
import time
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
 
logger = logging.getLogger(__name__)
 
 
# ── Résultat d'une requête RAG ────────────────────────────────

@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_time: float = 0.0
    generation_time: float = 0.0
 
    @property
    def total_time(self):
        return round(self.retrieval_time + self.generation_time, 3)
 
    def display(self):
        print(f"\n{'═'*60}")
        print(f"{self.question}")
        print(f"{'─'*60}")
        print(f"{self.answer}")
        print(f"{'─'*60}")
        print(f"Sources ({len(self.sources)}) :")
        for i, src in enumerate(self.sources, 1):
            meta = src.get("metadata", {})
            title = meta.get("title", meta.get("filename", f"Source {i}"))
            origin = meta.get("origin", meta.get("source", "?"))
            print(f"  [{i}] {str(title)[:60]} — {origin}")
        print(f"{self.total_time}s (retrieval={self.retrieval_time}s | gen={self.generation_time}s)")
        print(f"{'═'*60}\n")

# ── Pipeline RAG principal ────────────────────────────────────
 
class RAGPipeline:
    """
    Pipeline RAG complet basé sur LangChain LCEL.
 
    Architecture :
                          ┌──────────────┐
      question  ──────────► retriever    │ ← ChromaDB (similarity / MMR)
                          │              │
                          ▼              │
                        context          │
                          │              │
                          ▼              │
                      ┌──────┐           │
      question ───────► prompt│          │
                      └──────┘           │
                          │              │
                          ▼              │
                       ┌─────┐           │
                       │ LLM │           │  (Ollama / HuggingFace)
                       └─────┘           │
                          │              │
                          ▼              │
                      ┌────────┐         │
                      │ Parser │         │  (StrOutputParser)
                      └────────┘         │
                          │              │
                          ▼              │
                        answer           │
                          └─────────────►┘
 
    Usage simple :
        rag = RAGPipeline()
        result = rag.query("Qu'est-ce que le Transformer ?")
        print(result.answer)
 
    Usage avancé :
        rag = RAGPipeline(search_type="mmr", top_k=3)
        for token in rag.stream("Ma question"):
            print(token, end="", flush=True)
    """
 
    def __init__(
        self,
        embeddings=None,
        vector_store=None,
        llm=None,
        search_type: str = config.SEARCH_TYPE,
        top_k: int = config.TOP_K,
    ):
        from src.embeddings import get_embeddings
        from src.vector_store import get_vector_store, get_retriever
        from src.llm import get_llm
 
        # Initialiser les composants
        self.embeddings = embeddings or get_embeddings()
        self.vector_store = vector_store or get_vector_store(self.embeddings)
        self.llm = llm or get_llm()
        self.retriever = get_retriever(self.vector_store, search_type, top_k)
        self.top_k = top_k
        self.search_type = search_type
 
        # Construire les chains LCEL
        self._chain_simple = self._build_simple_chain()
        self._chain_with_sources = self._build_chain_with_sources()
 
        logger.info(
            f"RAGPipeline prêt | "
            f"docs={self.vector_store._collection.count()} | "
            f"search={search_type} | top_k={top_k}"
        )

# ── Construction des chains LCEL ─────────────────────────
 
    def _build_simple_chain(self):
        """
        Chain LCEL simple : retourne uniquement la réponse texte.
 
        Syntaxe LCEL :
            {"context": retriever | format_docs, "question": passthrough}
            | prompt
            | llm
            | StrOutputParser()
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
 
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
 
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
 
    def _build_chain_with_sources(self):
        """
        Chain LCEL avancée : retourne réponse + documents sources.
 
        Utilise RunnableParallel pour exécuter retrieval et passthrough en parallèle,
        puis combine dans une étape finale.
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel
 
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
 
        # Récupérer contexte ET documents sources en parallèle
        retrieve_chain = RunnableParallel(
            context=self.retriever | self._format_docs,
            question=RunnablePassthrough(),
            source_documents=self.retriever,   # Documents bruts (avec métadonnées)
        )
 
        # Générer la réponse
        answer_chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )
 
        # Combiner tout
        def combine(inputs: dict) -> dict:
            return {
                "answer": answer_chain.invoke(
                    {"context": inputs["context"], "question": inputs["question"]}
                ),
                "source_documents": inputs["source_documents"],
            }
 
        return retrieve_chain | combine
 
    # ── Interface publique ────────────────────────────────────
 
    def query(self, question: str) -> RAGResult:
        """
        Exécute le pipeline RAG complet.
 
        Args:
            question: La question de l'utilisateur
 
        Returns:
            RAGResult avec réponse + sources + timings
        """
        if not question.strip():
            raise ValueError("La question ne peut pas être vide")
 
        t0 = time.time()
        docs_with_scores = self.vector_store.similarity_search_with_score(
            question, k=self.top_k
        )
        retrieval_time = round(time.time() - t0, 3)
 
        t1 = time.time()
        answer = self._chain_simple.invoke(question)
        generation_time = round(time.time() - t1, 3)
 
        sources = [
            {"content": d.page_content[:200], "metadata": d.metadata, "score": round(float(score), 4)}
            for d, score in docs_with_scores
        ]
 
        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
        )
 
    def query_with_sources(self, question: str) -> dict:
        """
        Variante retournant les documents sources complets.
 
        Returns:
            {"answer": str, "source_documents": list[Document]}
        """
        return self._chain_with_sources.invoke(question)
 
    def stream(self, question: str) -> Iterator[str]:
        """
        Génère la réponse token par token (streaming).
 
        Usage :
            for token in rag.stream("Ma question"):
                print(token, end="", flush=True)
        """
        yield from self._chain_simple.stream(question)
 
    # ── Ingestion ─────────────────────────────────────────────
 
    def ingest_documents(self, documents: list, strategy: str = "recursive") -> int:
        """
        Découpe et indexe des Documents LangChain.
 
        Args:
            documents: list[Document] LangChain
            strategy: Stratégie de chunking
 
        Returns:
            Nombre de chunks indexés
        """
        from src.chunker import split_documents
        from src.vector_store import add_documents
 
        chunks = split_documents(documents, strategy=strategy)
        return add_documents(self.vector_store, chunks)
 
    def ingest_directory(self, directory: str | Path, strategy: str = "recursive") -> int:
        """Charge et indexe tous les documents d'un répertoire."""
        from src.document_loader import load_directory
        docs = load_directory(directory)
        return self.ingest_documents(docs, strategy=strategy)
 
    # ── Utilitaires ───────────────────────────────────────────
 
    @staticmethod
    def _format_docs(docs: list) -> str:
        """Formate les documents récupérés en bloc de contexte."""
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            title = meta.get("title", meta.get("filename", f"Document {i}"))
            parts.append(f"[Source {i} — {title}]\n{doc.page_content}")
        return "\n\n".join(parts)
 
    def get_stats(self) -> dict:
        return {
            "documents_indexed": self.vector_store._collection.count(),
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_backend": config.LLM_BACKEND,
            "search_type": self.search_type,
            "top_k": self.top_k,
        }
