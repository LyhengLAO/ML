"""
src/evaluator.py — Évaluation du pipeline RAG avec RAGAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RAGAS (Retrieval Augmented Generation Assessment) est le framework
open-source standard pour évaluer un pipeline RAG.

4 métriques clés :
  ┌─────────────────────┬──────────────────────────────────────────────┐
  │ Faithfulness        │ La réponse est-elle fidèle au contexte ?     │
  │ Answer Relevancy    │ La réponse répond-elle bien à la question ?  │
  │ Context Precision   │ Les chunks récupérés sont-ils pertinents ?   │
  │ Context Recall      │ A-t-on récupéré tout le nécessaire ?         │
  └─────────────────────┴──────────────────────────────────────────────┘

Installation :
    pip install ragas datasets

Usage :
    from src.evaluator import RAGEvaluator
    evaluator = RAGEvaluator(rag_pipeline)
    report = evaluator.evaluate(questions, ground_truths)
    report.display()
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Résultat d'évaluation
# ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Résultat complet d'une évaluation RAGAS."""
    faithfulness: float        = 0.0   # [0,1] : réponse fondée sur le contexte ?
    answer_relevancy: float    = 0.0   # [0,1] : réponse pertinente pour la question ?
    context_precision: float   = 0.0   # [0,1] : chunks récupérés utiles ?
    context_recall: float      = 0.0   # [0,1] : tous les chunks nécessaires récupérés ?
    n_questions: int           = 0
    eval_time: float           = 0.0
    per_question: list[dict]   = field(default_factory=list)

    @property
    def ragas_score(self) -> float:
        """Score global RAGAS = moyenne harmonique des 4 métriques."""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        scores = [s for s in scores if s > 0]
        if not scores:
            return 0.0
        return round(len(scores) / sum(1/s for s in scores if s > 0), 4)

    def display(self):
        """Affiche un rapport d'évaluation formaté."""
        bar = lambda v: "█" * int(v * 20) + "░" * (20 - int(v * 20))

        print(f"\n{'═'*58}")
        print(f"  RAPPORT D'ÉVALUATION RAG — RAGAS")
        print(f"{'─'*58}")
        print(f"  Questions évaluées : {self.n_questions}")
        print(f"  Durée              : {self.eval_time:.1f}s")
        print(f"{'─'*58}")
        print(f"  SCORE RAGAS GLOBAL       : {self.ragas_score:.3f}  {bar(self.ragas_score)}")
        print(f"{'─'*58}")
        print(f"  Faithfulness             : {self.faithfulness:.3f}  {bar(self.faithfulness)}")
        print(f"  Answer Relevancy         : {self.answer_relevancy:.3f}  {bar(self.answer_relevancy)}")
        print(f"  Context Precision        : {self.context_precision:.3f}  {bar(self.context_precision)}")
        print(f"  Context Recall           : {self.context_recall:.3f}  {bar(self.context_recall)}")
        print(f"{'─'*58}")
        print(self._quality_label())
        print(f"{'═'*58}\n")

    def _quality_label(self) -> str:
        s = self.ragas_score
        if s >= 0.85:
            return "  Excellent  — pipeline production-ready"
        elif s >= 0.70:
            return "  Bon        — quelques ajustements recommandés"
        elif s >= 0.50:
            return "  Moyen      — chunking ou retrieval à revoir"
        else:
            return "  Faible     — pipeline à refactoriser"

    def to_dict(self) -> dict:
        return {
            "ragas_score": self.ragas_score,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "n_questions": self.n_questions,
            "eval_time": self.eval_time,
        }


# ─────────────────────────────────────────────────────────────
# Évaluateur principal
# ─────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Évalue un pipeline RAG avec le framework RAGAS.

    Exemple minimal :
        rag = RAGPipeline()
        evaluator = RAGEvaluator(rag)

        questions = [
            "Qu'est-ce que le mécanisme d'attention ?",
            "Quelle est la différence entre BERT et GPT ?",
        ]
        ground_truths = [
            "Le mécanisme d'attention calcule un score de pertinence entre chaque paire de tokens...",
            "BERT est un encodeur bidirectionnel, GPT est un décodeur auto-régressif...",
        ]

        report = evaluator.evaluate(questions, ground_truths)
        report.display()

    Exemple sans ground_truth (évaluation partielle) :
        report = evaluator.evaluate_no_reference(questions)
        # → Faithfulness + Answer Relevancy seulement
    """

    def __init__(self, rag_pipeline, llm_for_eval=None):
        """
        Args:
            rag_pipeline: Instance de RAGPipeline
            llm_for_eval: LLM utilisé pour scorer (None = même LLM que le pipeline)
        """
        self.pipeline = rag_pipeline
        self.llm_for_eval = llm_for_eval or rag_pipeline.llm

    # ── Évaluation complète (avec ground truth) ──────────────

    def evaluate(
        self,
        questions: list[str],
        ground_truths: list[str],
        verbose: bool = True,
    ) -> EvalResult:
        """
        Évaluation complète avec les 4 métriques RAGAS.

        Args:
            questions: Liste de questions
            ground_truths: Réponses de référence (une par question)
            verbose: Afficher la progression

        Returns:
            EvalResult avec toutes les métriques
        """
        assert len(questions) == len(ground_truths), \
            "questions et ground_truths doivent avoir la même longueur"

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Installer RAGAS : pip install ragas datasets\n"
                "Documentation : https://docs.ragas.io"
            )

        if verbose:
            print(f"\n Évaluation RAGAS — {len(questions)} question(s)...")

        t0 = time.time()

        # ── Étape 1 : Générer les réponses + récupérer les contextes ──
        eval_data = self._collect_pipeline_outputs(questions, ground_truths, verbose)

        # ── Étape 2 : Construire le Dataset RAGAS ──
        dataset = Dataset.from_dict({
            "question":         eval_data["questions"],
            "answer":           eval_data["answers"],
            "contexts":         eval_data["contexts"],      # list[list[str]]
            "ground_truth":     eval_data["ground_truths"],
        })

        # ── Étape 3 : Scorer avec RAGAS ──
        if verbose:
            print("  Calcul des métriques RAGAS...")

        ragas_result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self._get_ragas_llm(),
            embeddings=self._get_ragas_embeddings(),
            raise_exceptions=False,
        )

        eval_time = round(time.time() - t0, 2)

        # ── Étape 4 : Construire le rapport ──
        df = ragas_result.to_pandas()
        result = EvalResult(
            faithfulness=round(float(df["faithfulness"].mean()), 4),
            answer_relevancy=round(float(df["answer_relevancy"].mean()), 4),
            context_precision=round(float(df["context_precision"].mean()), 4),
            context_recall=round(float(df["context_recall"].mean()), 4),
            n_questions=len(questions),
            eval_time=eval_time,
            per_question=[
                {
                    "question": row["question"],
                    "faithfulness": round(row.get("faithfulness", 0), 4),
                    "answer_relevancy": round(row.get("answer_relevancy", 0), 4),
                    "context_precision": round(row.get("context_precision", 0), 4),
                    "context_recall": round(row.get("context_recall", 0), 4),
                }
                for _, row in df.iterrows()
            ]
        )

        return result

    # ── Évaluation sans ground truth (partielle) ─────────────

    def evaluate_no_reference(
        self,
        questions: list[str],
        verbose: bool = True,
    ) -> EvalResult:
        """
        Évaluation sans réponses de référence.
        Calcule uniquement : Faithfulness + Answer Relevancy.

        Utile quand vous n'avez pas de dataset annoté.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy
            from datasets import Dataset
        except ImportError:
            raise ImportError("pip install ragas datasets")

        if verbose:
            print(f"\n🔍 Évaluation sans référence — {len(questions)} question(s)...")

        t0 = time.time()
        eval_data = self._collect_pipeline_outputs(questions, verbose=verbose)

        dataset = Dataset.from_dict({
            "question": eval_data["questions"],
            "answer":   eval_data["answers"],
            "contexts": eval_data["contexts"],
        })

        ragas_result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self._get_ragas_llm(),
            embeddings=self._get_ragas_embeddings(),
            raise_exceptions=False,
        )

        df = ragas_result.to_pandas()
        return EvalResult(
            faithfulness=round(float(df["faithfulness"].mean()), 4),
            answer_relevancy=round(float(df["answer_relevancy"].mean()), 4),
            context_precision=0.0,  # Non calculé
            context_recall=0.0,     # Non calculé
            n_questions=len(questions),
            eval_time=round(time.time() - t0, 2),
        )

    # ── Métriques maison (sans RAGAS, 100% local) ────────────

    def evaluate_simple(
        self,
        questions: list[str],
        ground_truths: list[str] | None = None,
    ) -> dict:
        """
        Métriques simples sans RAGAS (utile hors connexion).

        Calcule :
          - retrieval_score   : pertinence moyenne des chunks (similarité cosinus)
          - answer_length     : longueur moyenne des réponses
          - source_diversity  : diversité des sources utilisées
          - latency_ms        : temps de réponse moyen en ms
          - token_overlap     : chevauchement de tokens réponse/référence (si ground_truth fourni)
        """
        results = []

        for i, question in enumerate(questions):
            t0 = time.time()
            rag_result = self.pipeline.query(question)
            latency = (time.time() - t0) * 1000

            sources = {
                r["metadata"].get("origin", r["metadata"].get("source", "?"))
                for r in rag_result.sources
            }

            entry = {
                "question": question,
                "answer": rag_result.answer,
                "n_sources": len(rag_result.sources),
                "source_diversity": len(sources),
                "avg_retrieval_score": round(
                    sum(r["score"] for r in rag_result.sources) / max(len(rag_result.sources), 1),
                    4,
                ),
                "answer_length": len(rag_result.answer.split()),
                "latency_ms": round(latency, 1),
            }

            # Chevauchement de tokens (token overlap) si ground truth fourni
            if ground_truths:
                entry["token_overlap"] = self._token_overlap(
                    rag_result.answer, ground_truths[i]
                )

            results.append(entry)
            logger.info(f"  [{i+1}/{len(questions)}] {question[:50]}...")

        # Agréger
        summary = {
            "n_questions":        len(questions),
            "avg_retrieval_score": round(sum(r["avg_retrieval_score"] for r in results) / len(results), 4),
            "avg_latency_ms":      round(sum(r["latency_ms"] for r in results) / len(results), 1),
            "avg_answer_words":    round(sum(r["answer_length"] for r in results) / len(results), 1),
            "avg_source_diversity":round(sum(r["source_diversity"] for r in results) / len(results), 2),
            "per_question":        results,
        }

        if ground_truths:
            summary["avg_token_overlap"] = round(
                sum(r.get("token_overlap", 0) for r in results) / len(results), 4
            )

        return summary

    # ── Helpers internes ─────────────────────────────────────

    def _collect_pipeline_outputs(
        self,
        questions: list[str],
        ground_truths: list[str] | None = None,
        verbose: bool = True,
    ) -> dict:
        """Exécute le pipeline sur chaque question et collecte les sorties."""
        answers, contexts = [], []

        for i, question in enumerate(questions):
            if verbose:
                print(f"  [{i+1}/{len(questions)}] {question[:60]}...")

            result = self.pipeline.query(question)
            answers.append(result.answer)

            # RAGAS attend une liste de strings (un string par chunk)
            contexts.append([src["content"] for src in result.sources])

        return {
            "questions":     questions,
            "answers":       answers,
            "contexts":      contexts,
            "ground_truths": ground_truths or [""] * len(questions),
        }

    def _get_ragas_llm(self):
        """Adapte le LLM LangChain pour RAGAS."""
        try:
            from ragas.llms import LangchainLLMWrapper
            return LangchainLLMWrapper(self.llm_for_eval)
        except Exception:
            return None  # RAGAS utilisera son LLM par défaut

    def _get_ragas_embeddings(self):
        """Adapte les embeddings LangChain pour RAGAS."""
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            return LangchainEmbeddingsWrapper(self.pipeline.embeddings)
        except Exception:
            return None

    @staticmethod
    def _token_overlap(answer: str, reference: str) -> float:
        """
        Chevauchement de tokens entre réponse et référence.
        Métrique simple similaire au F1-score token-level de SQuAD.
        """
        ans_tokens = set(answer.lower().split())
        ref_tokens = set(reference.lower().split())

        if not ans_tokens or not ref_tokens:
            return 0.0

        intersection = ans_tokens & ref_tokens
        precision = len(intersection) / len(ans_tokens)
        recall = len(intersection) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 4)