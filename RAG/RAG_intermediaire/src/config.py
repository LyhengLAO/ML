"""Chargement et accès typé à la configuration YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _resolve(path: str) -> str:
    """Transforme un chemin relatif (depuis la config) en chemin absolu."""
    p = Path(path)
    return str(p if p.is_absolute() else PROJECT_ROOT / p)

@dataclass
class Config:
    """Wrapper léger autour du dict YAML avec accès par attributs."""

    raw: dict[str, Any] = field(default_factory=dict)

    # --- chemins ---
    @property
    def raw_data_dir(self) -> str:
        return _resolve(self.raw["paths"]["raw_data_dir"])

    @property
    def eval_dataset(self) -> str:
        return _resolve(self.raw["paths"]["eval_dataset"])

    @property
    def chroma_dir(self) -> str:
        return _resolve(self.raw["paths"]["chroma_dir"])

    @property
    def results_dir(self) -> str:
        return _resolve(self.raw["paths"]["results_dir"])

    # --- sous-sections ---
    @property
    def embeddings(self) -> dict[str, Any]:
        return self.raw["embeddings"]

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw["llm"]

    @property
    def baseline(self) -> dict[str, Any]:
        return self.raw["baseline"]

    @property
    def optimized(self) -> dict[str, Any]:
        return self.raw["optimized"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]


def load_config(path: str | None = None) -> Config:
    """Charge config/config.yaml (ou un chemin fourni)."""
    if path is None:
        path = os.environ.get("RAG_CONFIG", str(PROJECT_ROOT / "config" / "config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)