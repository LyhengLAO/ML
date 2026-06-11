"""Évaluation complète et comparative des pipelines RAG.

Trois familles de métriques :
  - RETRIEVAL (vérité terrain = source_docs) : Hit Rate@k, MRR@k, Precision@k, Recall.
  - GÉNÉRATION (vs réponse de référence) : similarité sémantique, ROUGE-L, token-F1.
  - PERFORMANCE : latence moyenne.
  - RAGAS (optionnel, LLM-judge) : faithfulness, answer_relevancy, context_precision/recall.

Aucune clé API requise : la similarité sémantique réutilise le modèle d'embeddings
local, et RAGAS (si activé) s'appuie sur le LLM Ollama local.
"""
from __future__ import annotations

import re
import statistics
from typing import Any

from langchain_core.embeddings import Embeddings

from .base import BaseRAGPipeline, RAGResult

# --------------------------------------------------------------------------- #
# Utilitaires texte                                                           #
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def token_f1(pred: str, ref: str) -> float:
    """F1 au niveau token entre prédiction et référence (style SQuAD)."""
    p_tokens, r_tokens = _normalize(pred), _normalize(ref)
    if not p_tokens or not r_tokens:
        return 0.0
    common: dict[str, int] = {}
    r_counts: dict[str, int] = {}
    for t in r_tokens:
        r_counts[t] = r_counts.get(t, 0) + 1
    overlap = 0
    p_counts: dict[str, int] = {}
    for t in p_tokens:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t, c in p_counts.items():
        overlap += min(c, r_counts.get(t, 0))
        common[t] = c
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L (F1) basé sur la plus longue sous-séquence commune (LCS)."""
    p, r = _normalize(pred), _normalize(ref)
    if not p or not r:
        return 0.0
    # LCS par programmation dynamique.
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            if p[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(p)][len(r)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(p)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Métriques de retrieval                                                       #
# --------------------------------------------------------------------------- #
def retrieval_metrics(retrieved_sources: list[str], gold: list[str], k: int) -> dict:
    top = retrieved_sources[:k]
    gold_set = set(gold)
    hit = 1.0 if any(s in gold_set for s in top) else 0.0

    mrr = 0.0
    for rank, s in enumerate(top, start=1):
        if s in gold_set:
            mrr = 1.0 / rank
            break

    relevant_in_top = sum(1 for s in top if s in gold_set)
    precision = relevant_in_top / k if k else 0.0
    found = len(gold_set & set(top))
    recall = found / len(gold_set) if gold_set else 0.0
    return {
        "hit_rate": hit,
        "mrr": mrr,
        "precision_at_k": precision,
        "context_recall": recall,
    }


# --------------------------------------------------------------------------- #
# Évaluation d'un pipeline                                                     #
# --------------------------------------------------------------------------- #
def evaluate_pipeline(
    pipeline: BaseRAGPipeline,
    eval_items: list[dict],
    embeddings: Embeddings,
    k: int,
    sim_threshold: float = 0.6,
    verbose: bool = True,
) -> dict[str, Any]:
    """Exécute le pipeline sur tout le jeu d'éval et agrège les métriques."""
    per_question: list[dict] = []
    results: list[RAGResult] = []

    for item in eval_items:
        res = pipeline.run(item["question"])
        results.append(res)

        rm = retrieval_metrics(res.context_sources, item["source_docs"], k)

        emb_pred = embeddings.embed_query(res.answer)
        emb_ref = embeddings.embed_query(item["ground_truth"])
        sem_sim = _cosine(emb_pred, emb_ref)

        record = {
            "id": item["id"],
            "question": item["question"],
            "answer": res.answer,
            "ground_truth": item["ground_truth"],
            "retrieved_sources": res.context_sources,
            "gold_sources": item["source_docs"],
            "latency_s": round(res.latency_s, 3),
            "semantic_similarity": round(sem_sim, 4),
            "answer_correct": float(sem_sim >= sim_threshold),
            "rouge_l": round(rouge_l(res.answer, item["ground_truth"]), 4),
            "token_f1": round(token_f1(res.answer, item["ground_truth"]), 4),
            **{kk: round(vv, 4) for kk, vv in rm.items()},
        }
        per_question.append(record)
        if verbose:
            print(f"  [{pipeline.name}] {item['id']} "
                  f"hit={rm['hit_rate']:.0f} sim={sem_sim:.2f} "
                  f"lat={res.latency_s:.2f}s")

    def avg(key: str) -> float:
        return round(statistics.mean(r[key] for r in per_question), 4)

    aggregate = {
        "pipeline": pipeline.name,
        "n_questions": len(per_question),
        "hit_rate@k": avg("hit_rate"),
        "mrr@k": avg("mrr"),
        "precision@k": avg("precision_at_k"),
        "context_recall": avg("context_recall"),
        "answer_similarity": avg("semantic_similarity"),
        "answer_correctness": avg("answer_correct"),
        "rouge_l": avg("rouge_l"),
        "token_f1": avg("token_f1"),
        "avg_latency_s": avg("latency_s"),
    }

    # RAGAS optionnel (LLM-judge) — n'interrompt jamais le run principal.
    return {"aggregate": aggregate, "per_question": per_question, "results": results}


# --------------------------------------------------------------------------- #
# RAGAS optionnel                                                              #
# --------------------------------------------------------------------------- #
def run_ragas(
    results: list[RAGResult],
    eval_items: list[dict],
    llm,
    embeddings: Embeddings,
) -> dict[str, float]:
    """Évalue avec RAGAS (faithfulness, answer_relevancy, context_precision/recall).

    Renvoie un dict vide si RAGAS n'est pas installé ou échoue.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except Exception as exc:  # noqa: BLE001
        print(f"[RAGAS] indisponible ({exc}) — métriques RAGAS ignorées.")
        return {}

    data = {
        "question": [it["question"] for it in eval_items],
        "answer": [r.answer for r in results],
        "contexts": [[c.page_content for c in r.contexts] for r in results],
        "ground_truth": [it["ground_truth"] for it in eval_items],
    }
    ds = Dataset.from_dict(data)
    try:
        scores = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
        )
        df = scores.to_pandas()
        return {
            "ragas_faithfulness": round(float(df["faithfulness"].mean()), 4),
            "ragas_answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
            "ragas_context_precision": round(float(df["context_precision"].mean()), 4),
            "ragas_context_recall": round(float(df["context_recall"].mean()), 4),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[RAGAS] échec de l'évaluation ({exc}).")
        return {}
