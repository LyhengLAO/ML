"""Chargement des documents texte externes depuis data/raw."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

# ici on utilise Document de langchain car avec ça le source de notre doc envoie juste le nom de fichier .txt ex : machine_learning.txt
# mais avec text_loader ça fait pareil mais la sourec envoie une chemin path de fichier ex: data/raw/machine_learning.txt
def load_text_documents(raw_dir: str) -> list[Document]:
    """Charge tous les fichiers .txt / .md d'un dossier en objets Document.

    Le nom de fichier est conservé dans metadata['source'] : il sert de
    'document id' pour les métriques de retrieval (hit_rate, MRR...).
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {raw_dir}")

    docs: list[Document] = []
    files = sorted([*raw_path.glob("*.txt"), *raw_path.glob("*.md")])
    if not files:
        raise FileNotFoundError(
            f"Aucun .txt/.md dans {raw_dir}. "
            "Ajoutez des documents ou lancez scripts/download_data.py."
        )

    for fp in files:
        text = fp.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": fp.name, "path": str(fp)},
            )
        )
    return docs

# Aternative avce text_loader de langchain
# from langchain_core.documents import Document
# from langchain_core.document_loaders import TextLoader, DirectoryLoader


# def load_text_documents(raw_dir: str) -> list[Document]:
#     raw_path = Path(raw_dir)
#     if not raw_path.exists():
#         raise FileNotFoundError(f"Dossier introuvable : {raw_dir}")

#     docs: list[Document] = []
#     for pattern in ("**/*.txt", "**/*.md"):
#         loader = DirectoryLoader(
#             raw_dir,
#             glob=pattern,
#             loader_cls=TextLoader,
#             loader_kwargs={"encoding": "utf-8"},
#             show_progress=False,
#         )
#         docs.extend(loader.load())

#     if not docs:
#         raise FileNotFoundError(
#             f"Aucun .txt/.md dans {raw_dir}. "
#             "Ajoutez des documents ou lancez scripts/download_data.py."
#         )

#     # On normalise 'source' en nom de fichier (id stable pour les métriques),
#     # et on garde le chemin complet à part.
#     for d in docs:
#         full_path = d.metadata.get("source", "")
#         d.metadata["path"] = full_path
#         d.metadata["source"] = Path(full_path).name

#     return docs