"""Providers OFFLINE (aucun téléchargement, aucune clé, aucun GPU).

Permettent d'exécuter tout le pipeline et l'évaluation sans Ollama ni
HuggingFace — idéal pour la CI, les tests, et la génération reproductible de la
table de métriques. En production on utilise MiniLM + Llama (cf. config.yaml).

  - TfidfEmbeddings : embeddings via TF-IDF scikit-learn (à fitter sur le corpus).
  - ExtractiveLLM   : "générateur" extractif déterministe (phrases du contexte).
  - tfidf_rerank    : reranking par similarité cosinus TF-IDF requête/document.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class TfidfEmbeddings(Embeddings):
    """Embeddings TF-IDF. Doit être fitté sur le corpus avant indexation."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbeddings":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def _vec(self, text: str) -> list[float]:
        m = self._vectorizer.transform([text])
        return m.toarray()[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            self.fit(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class ExtractiveLLM:
    """Lecteur extractif déterministe imitant l'interface chat (.invoke).

    Récupère la question et le contexte depuis le prompt formaté, puis renvoie
    les phrases du contexte les plus pertinentes (recouvrement lexical). C'est
    une vraie génération (déterministe), pas un appel LLM externe.
    """

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())

    def invoke(self, messages):
        content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        question = content.split("Question:")[-1].split("Answer:")[0].strip()
        context = ""
        if "Context:" in content:
            context = content.split("Context:")[1].split("Question:")[0]
        # Retire les marqueurs [n] (source: ...) du contexte.
        context = re.sub(r"\[\d+\]\s*\(source:[^)]*\)", " ", context)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.strip()) > 15]
        q_tokens = self._tokens(question)
        scored = []
        for s in sentences:
            overlap = len(q_tokens & self._tokens(s))
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = [s for sc, s in scored[: self.max_sentences] if sc > 0]
        answer = " ".join(best) if best else "I cannot answer based on the provided context."
        return SimpleNamespace(content=answer)


def tfidf_rerank(
    query: str, docs: list[Document], top_n: int, vectorizer
) -> list[Document]:
    """Reranking offline : score cosinus TF-IDF entre la requête et chaque doc."""
    if not docs:
        return docs
    from sklearn.metrics.pairwise import cosine_similarity

    q_vec = vectorizer.transform([query])
    d_vecs = vectorizer.transform([d.page_content for d in docs])
    scores = cosine_similarity(q_vec, d_vecs)[0]
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_n]]
