"""Fabrique du LLM : Ollama (production) ou lecteur extractif (offline)."""
from __future__ import annotations

from typing import Any

def build_llm(cfg: dict[str, Any]):
    """Construit le générateur selon cfg['provider'].

    - 'ollama' (défaut) : LLM local Llama 3.1 via Ollama (gratuit, sans clé).
    - 'extractive' : lecteur extractif déterministe, 100 % offline (CI / démo).
    """
    provider = cfg.get("provider", "ollama")

    if provider == "extractive":
        from .offline import ExtractiveLLM
        return ExtractiveLLM(max_sentences=cfg.get("max_sentences", 2))

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=cfg["model_name"],
        temperature=cfg.get("temperature", 0.0),
        base_url=cfg.get("base_url", "http://localhost:11434"),
        num_ctx=cfg.get("num_ctx", 4096),
    )


def check_ollama(cfg: dict[str, Any]) -> tuple[bool, str]:
    """Vérifie le serveur Ollama (ignoré si provider == 'extractive')."""
    if cfg.get("provider", "ollama") == "extractive":
        return True, "Mode offline (lecteur extractif) — Ollama non requis."

    import requests

    base = cfg.get("base_url", "http://localhost:11434").rstrip("/")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        wanted = cfg["model_name"]
        ok = any(m == wanted or m.startswith(wanted + ":") for m in models)
        if ok:
            return True, f"Ollama OK, modèle '{wanted}' disponible."
        return False, (
            f"Ollama répond mais le modèle '{wanted}' est absent. "
            f"Lancez : ollama pull {wanted}. Modèles présents : {models}"
        )
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"Impossible de joindre Ollama sur {base} ({exc}). "
            "Démarrez `ollama serve` ou passez en mode offline (--offline)."
        )