"""
config.py — Configuration centralisée (version LangChain)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
CHROMA_DIR    = DATA_DIR / "chroma_db"

for d in [RAW_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Embeddings ──────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
    # "BAAI/bge-small-en-v1.5"           ← meilleur benchmark MTEB
    # "paraphrase-multilingual-MiniLM-L12-v2" ← multilingue (français ✓)
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── ChromaDB ────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_langchain")

# ── Chunking ────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# ── Retrieval ───────────────────────────────
TOP_K                = int(os.getenv("TOP_K", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))
SEARCH_TYPE          = os.getenv("SEARCH_TYPE", "mmr")   # "similarity" | "mmr"

# ── LLM ─────────────────────────────────────
LLM_BACKEND     = os.getenv("LLM_BACKEND", "ollama")     # "ollama" | "huggingface"
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
HF_MODEL_ID     = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_API_TOKEN    = os.getenv("HF_API_TOKEN", "")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TEMPERATURE    = float(os.getenv("TEMPERATURE", 0.1))

# ── Prompt ──────────────────────────────────
RAG_SYSTEM_PROMPT = """Tu es un assistant expert qui répond uniquement en te basant
sur les documents fournis en contexte. Si la réponse n'est pas dans le contexte,
dis clairement que tu ne sais pas. Sois précis et cite les informations des sources.

Contexte :
{context}"""

# ── Sources Wikipedia ────────────────────────
WIKIPEDIA_TOPICS = [
    "Transformer (machine learning model)",
    "Large language model",
    "Retrieval-augmented generation",
    "Word embedding",
    "Attention mechanism",
    "BERT (language model)",
    "GPT (language model)",
    "Vector database",
    "Natural language processing",
    "Semantic search",
]

ARXIV_QUERIES   = [
    "attention is all you need transformer",
    "retrieval augmented generation",
    "sentence bert embeddings",
]
ARXIV_MAX       = 5
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")