"""Construit (ou reconstruit) les index ChromaDB pour les deux pipelines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.factory import build_all  # noqa: E402
from src.llm import check_ollama  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if args.offline:
        cfg.raw["embeddings"]["provider"] = "tfidf"
        cfg.raw["llm"]["provider"] = "extractive"
        cfg.raw["optimized"]["reranker"] = "tfidf"
        print(">> Mode OFFLINE")

    ok, msg = check_ollama(cfg.llm)
    print(f">> {msg}")

    print(">> Indexation des collections ChromaDB...")
    built = build_all(cfg)
    print(f"   Documents       : {built.n_docs}")
    print(f"   Chunks baseline : {built.n_chunks_baseline}")
    print(f"   Chunks optimisé : {built.n_chunks_optimized}")
    print(f"   Persistance     : {cfg.chroma_dir}")
    print(">> Index prêt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
