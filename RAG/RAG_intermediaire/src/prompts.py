"""Templates de prompts pour la génération RAG."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# Prompt baseline : minimal.
BASELINE_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using the context below.

Context:
{context}

Question: {question}

Answer:"""
)

# Prompt optimisé : consignes anti-hallucination + ancrage strict au contexte.
OPTIMIZED_PROMPT = ChatPromptTemplate.from_template(
    """You are a precise assistant. Answer the question using ONLY the context
provided below. If the context does not contain the answer, reply exactly:
"I cannot answer based on the provided context." Be concise and factual; do not
invent information.

Context:
{context}

Question: {question}

Answer:"""
)
