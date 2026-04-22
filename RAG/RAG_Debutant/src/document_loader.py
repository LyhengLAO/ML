"""
src/document_loader.py — Chargement de documents via LangChain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LangChain fournit des loaders pour tous les formats courants.
Chaque loader retourne une liste de langchain_core.documents.Document.

Formats supportés :
  .txt / .md   → TextLoader
  .pdf         → PyPDFLoader
  .docx        → Docx2txtLoader
  Dossier      → DirectoryLoader (détection automatique)
  Wikipedia    → WikipediaLoader
  ArXiv        → ArxivLoader
"""

import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ── Chargement depuis le disque ──────────────────────────────

def load_directory(directory: str | Path) -> list:
    """
    Charge tous les documents d'un répertoire avec détection automatique du format.

    Utilise DirectoryLoader de LangChain qui orchestre les bons loaders
    selon l'extension de chaque fichier.

    Returns:
        list[Document] — Documents LangChain avec page_content + metadata
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Répertoire introuvable : {directory}")

    all_docs = []

    # TXT & MD
    for glob in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(directory),
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            show_progress=True,
            silent_errors=True,
        )
        docs = loader.load()
        all_docs.extend(docs)
        logger.info(f"  {glob} : {len(docs)} documents")

    # PDF
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        pdf_loader = PyPDFDirectoryLoader(str(directory))
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        if pdf_docs:
            logger.info(f"  *.pdf : {len(pdf_docs)} pages")
    except Exception as e:
        logger.debug(f"PDF loader skipped : {e}")

    # Enrichir les métadonnées
    for doc in all_docs:
        _enrich_metadata(doc)

    logger.info(f"📂 Total : {len(all_docs)} documents chargés depuis {directory}")
    return all_docs


def load_file(filepath: str | Path) -> list:
    """Charge un fichier unique."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(filepath), encoding="utf-8", autodetect_encoding=True)

    elif ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(filepath))

    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(filepath))

    else:
        raise ValueError(f"Format non supporté : {ext}")

    docs = loader.load()
    for doc in docs:
        _enrich_metadata(doc)
    return docs


# ── Chargement depuis des sources web open-source ────────────

def load_wikipedia(topics: list[str], lang: str = "en") -> list:
    """
    Charge des articles Wikipedia via LangChain WikipediaLoader.

    Args:
        topics: Liste de titres d'articles Wikipedia
        lang: Langue ("en", "fr", etc.)

    Returns:
        list[Document]
    """
    from langchain_community.document_loaders import WikipediaLoader

    docs = []
    for topic in topics:
        try:
            loader = WikipediaLoader(
                query=topic,
                lang=lang,
                load_max_docs=1,
                doc_content_chars_max=50_000,
            )
            loaded = loader.load()
            docs.extend(loaded)
            print(f"Wikipedia : {topic} ({len(loaded)} page(s))")
        except Exception as e:
            print(f"Wikipedia '{topic}' : {e}")

    return docs


def load_arxiv(queries: list[str], max_docs: int = 3) -> list:
    """
    Charge des résumés ArXiv via LangChain ArxivLoader.

    Args:
        queries: Requêtes de recherche
        max_docs: Nombre max de papers par requête

    Returns:
        list[Document]
    """
    from langchain_community.document_loaders import ArxivLoader

    docs = []
    for query in queries:
        try:
            loader = ArxivLoader(
                query=query,
                load_max_docs=max_docs,
                load_all_available_meta=True,
            )
            loaded = loader.load()
            docs.extend(loaded)
            print(f"ArXiv : '{query}' → {len(loaded)} papers")
        except Exception as e:
            print(f"ArXiv '{query}' : {e}")

    return docs


def load_huggingface_dataset(dataset_name: str = "rajpurkar/squad", n: int = 300) -> list:
    """
    Charge un dataset HuggingFace et le convertit en Documents LangChain.

    Utilise SQuAD (CC BY-SA 4.0) par défaut : passages Wikipedia annotés.
    """
    from datasets import load_dataset
    from langchain_core.documents import Document

    print(f"Chargement dataset HuggingFace : {dataset_name}...")
    dataset = load_dataset(dataset_name, split=f"train[:{n}]")

    # Dédupliquer les contextes
    seen = set()
    docs = []
    for item in dataset:
        ctx = item["context"]
        if ctx not in seen:
            seen.add(ctx)
            docs.append(Document(
                page_content=ctx,
                metadata={
                    "source": f"huggingface/{dataset_name}",
                    "title": item["title"],
                    "dataset": dataset_name,
                }
            ))

    print(f"HuggingFace : {len(docs)} passages uniques")
    return docs


# ── Helpers ──────────────────────────────────────────────────

def _enrich_metadata(doc) -> None:
    """Ajoute des métadonnées utiles à un Document LangChain."""
    source = doc.metadata.get("source", "")
    if source:
        path = Path(source)
        doc.metadata.setdefault("filename", path.name)
        doc.metadata.setdefault("file_type", path.suffix.lstrip("."))
        # Inférer la source (wikipedia, arxiv, huggingface...)
        for keyword in ("wikipedia", "arxiv", "huggingface", "static"):
            if keyword in str(source).lower():
                doc.metadata.setdefault("origin", keyword)
                break
        else:
            doc.metadata.setdefault("origin", "local")

