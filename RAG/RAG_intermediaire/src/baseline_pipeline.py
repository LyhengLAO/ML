"""Pipeline RAG BASELINE : chunking simple + retrieval dense top-k."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from .base import BaseRAGPipeline
from .prompts import BASELINE_PROMPT


class BaselineRAGPipeline(BaseRAGPipeline):
    """RAG naïf : similarité cosinus dense, top-k, prompt minimal."""

    name = "baseline"

    def __init__(self, vectorstore: Chroma, llm: BaseChatModel, k: int = 4):
        super().__init__(llm=llm, prompt=BASELINE_PROMPT)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)
