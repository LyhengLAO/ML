"""Classe de base commune aux pipelines RAG (logique de génération partagée)."""
from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class RAGResult:
    """Résultat d'une requête RAG."""

    question: str
    answer: str
    context: list[Document]
    latency_s: float

    @property
    def context_source(self) -> list[str]:
        """Liste ordonnée des fichiers sources des contextes récupérés."""
        return [c.metadata.get("source", "?") for c in self.contexts]
    
def format_context(docs: list[Document]) -> str:
    """Concatène les chunks récupérés en un bloc de contexte numéroté."""
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )

class BaseRAGPipeline:
    """Interface commune : les sous-classes implémentent retrieve()."""

    name: str = "base"

    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        self.llm = llm
        self.prompt = prompt

    def retrieve(self, question: str) -> list[Document]:  # noqa: D401
        raise NotImplementedError

    def generate(self, question: str, contexts: list[Document]) -> str:
        """Génère la réponse à partir des contextes récupérés."""
        messages = self.prompt.format_messages(
            context=format_context(contexts), question=question
        )
        resp = self.llm.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)

    def run(self, question: str) -> RAGResult:
        """Pipeline complet : retrieve -> generate, avec mesure de latence."""
        start = time.perf_counter()
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        latency = time.perf_counter() - start
        return RAGResult(
            question=question,
            answer=answer.strip(),
            contexts=contexts,
            latency_s=latency,
        )
