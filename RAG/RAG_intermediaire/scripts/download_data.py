"""(Optionnel) Télécharge des documents texte externes depuis Wikipedia.

Le projet fonctionne déjà avec le corpus fourni dans data/raw/. Ce script
permet d'enrichir le corpus avec de vraies pages Wikipédia (nécessite un accès
réseau, aucune clé API). Usage :

    python scripts/download_data.py
    python scripts/download_data.py --titles "Machine learning" "Transformer (deep learning architecture)"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TITLES = [
    "Machine learning",
    "Deep learning",
    "Transformer (deep learning architecture)",
    "Word embedding",
    "Vector database",
    "Retrieval-augmented generation",
    "Large language model",
]

API = "https://en.wikipedia.org/w/api.php"


def slugify(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return s or "doc"


def fetch_extract(title: str) -> str | None:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    try:
        r = requests.get(API, params=params, timeout=20,
                         headers={"User-Agent": "rag-portfolio/1.0"})
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Échec pour '{title}': {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", nargs="*", default=DEFAULT_TITLES)
    parser.add_argument("--max-chars", type=int, default=8000,
                        help="Tronque chaque article (0 = pas de troncature).")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for title in args.titles:
        print(f"- {title}")
        text = fetch_extract(title)
        if not text:
            continue
        if args.max_chars and len(text) > args.max_chars:
            text = text[: args.max_chars]
        path = out_dir / f"wiki_{slugify(title)}.txt"
        path.write_text(f"# {title}\n\n{text}", encoding="utf-8")
        print(f"  -> {path.relative_to(ROOT)} ({len(text)} caractères)")
        ok += 1

    print(f"\n{ok}/{len(args.titles)} documents téléchargés dans {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
