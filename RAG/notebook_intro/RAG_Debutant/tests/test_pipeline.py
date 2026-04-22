"""
tests/test_pipeline.py — Tests du pipeline RAG LangChain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Utilise des mocks LangChain pour ne pas nécessiter Ollama.

Lancer : pytest tests/ -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

from langchain_core.documents import Document


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def embeddings():
    """Embeddings réels (all-MiniLM-L6-v2)."""
    from src.embeddings import get_embeddings
    return get_embeddings()


@pytest.fixture
def tmp_vector_store(embeddings, tmp_path):
    """VectorStore ChromaDB temporaire."""
    from src.vector_store import get_vector_store
    return get_vector_store(
        embeddings=embeddings,
        collection_name="test_col",
        persist_directory=str(tmp_path / "chroma"),
    )


@pytest.fixture
def sample_docs():
    """Documents LangChain de test."""
    return [
        Document(
            page_content="Le Transformer est une architecture basée sur l'attention. Vaswani et al. 2017.",
            metadata={"source": "wikipedia", "title": "Transformer", "origin": "wikipedia"}
        ),
        Document(
            page_content="BERT est un modèle bidirectionnel pré-entraîné de Google.",
            metadata={"source": "wikipedia", "title": "BERT", "origin": "wikipedia"}
        ),
        Document(
            page_content="Le RAG combine la recherche sémantique et la génération LLM.",
            metadata={"source": "static", "title": "RAG Guide", "origin": "static"}
        ),
        Document(
            page_content="ChromaDB est une base vectorielle open-source qui stocke des embeddings.",
            metadata={"source": "static", "title": "ChromaDB", "origin": "static"}
        ),
        Document(
            page_content="GPT génère du texte de façon auto-régressive en prédisant le token suivant.",
            metadata={"source": "wikipedia", "title": "GPT", "origin": "wikipedia"}
        ),
    ]


@pytest.fixture
def mock_llm():
    """LLM mocké pour les tests (pas besoin d'Ollama)."""
    from langchain_core.language_models import BaseLLM
    from langchain_core.outputs import LLMResult, Generation

    class FakeLLM(BaseLLM):
        @property
        def _llm_type(self):
            return "fake"

        def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
            return LLMResult(
                generations=[[Generation(text="Réponse de test du LLM mocké.")]]
                * len(prompts)
            )

        def _stream(self, prompt, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import GenerationChunk
            for word in ["Réponse ", "de ", "test ", "streamée."]:
                yield GenerationChunk(text=word)

    return FakeLLM()


@pytest.fixture
def rag_pipeline(tmp_vector_store, embeddings, mock_llm, sample_docs):
    """Pipeline RAG complet pour les tests."""
    from src.vector_store import get_retriever, add_documents
    from src.chunker import split_documents
    from src.rag_pipeline import RAGPipeline, RAGResult

    # Indexer les documents de test
    chunks = split_documents(sample_docs, strategy="recursive",
                             chunk_size=200, chunk_overlap=20)
    add_documents(tmp_vector_store, chunks)

    retriever = get_retriever(tmp_vector_store, search_type="similarity", top_k=3)

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embeddings = embeddings
    pipeline.vector_store = tmp_vector_store
    pipeline.llm = mock_llm
    pipeline.retriever = retriever
    pipeline.top_k = 3
    pipeline.search_type = "similarity"
    pipeline._chain_simple = pipeline._build_simple_chain()
    pipeline._chain_with_sources = pipeline._build_chain_with_sources()

    return pipeline


# ── Tests Document Loader ─────────────────────────────────────

class TestDocumentLoader:

    def test_load_txt_file(self, tmp_path):
        """Charge un fichier TXT simple."""
        from src.document_loader import load_file
        f = tmp_path / "test.txt"
        f.write_text("Contenu de test pour le document loader.", encoding="utf-8")
        docs = load_file(f)
        assert len(docs) == 1
        assert "Contenu de test" in docs[0].page_content

    def test_load_directory(self, tmp_path):
        """Charge plusieurs fichiers d'un répertoire."""
        from src.document_loader import load_directory
        for i in range(3):
            (tmp_path / f"doc_{i}.txt").write_text(f"Document {i} avec du contenu.", encoding="utf-8")
        docs = load_directory(tmp_path)
        assert len(docs) >= 3

    def test_metadata_enriched(self, tmp_path):
        """Les métadonnées sont enrichies après chargement."""
        from src.document_loader import load_file
        f = tmp_path / "test_meta.txt"
        f.write_text("Contenu.", encoding="utf-8")
        docs = load_file(f)
        assert "filename" in docs[0].metadata


# ── Tests Chunker ────────────────────────────────────────────

class TestChunker:

    def test_recursive_split(self, sample_docs):
        """Le splitter récursif découpe correctement."""
        from src.chunker import split_documents
        chunks = split_documents(sample_docs[:1], strategy="recursive",
                                 chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c.page_content.strip()) > 0

    def test_metadata_preserved(self, sample_docs):
        """Les métadonnées du document source sont dans les chunks."""
        from src.chunker import split_documents
        chunks = split_documents(sample_docs[:1], strategy="recursive",
                                 chunk_size=200, chunk_overlap=20)
        for c in chunks:
            assert c.metadata.get("title") == "Transformer"

    def test_token_split(self, sample_docs):
        """Le splitter par tokens fonctionne."""
        from src.chunker import split_documents
        chunks = split_documents(sample_docs, strategy="token",
                                 chunk_size=50, chunk_overlap=5)
        assert len(chunks) >= 1

    def test_get_splitter_unknown_raises(self):
        """Une stratégie inconnue lève une erreur."""
        from src.chunker import get_splitter
        with pytest.raises(ValueError):
            get_splitter("unknown_strategy")


# ── Tests VectorStore ────────────────────────────────────────

class TestVectorStore:

    def test_add_and_count(self, tmp_vector_store, sample_docs):
        """L'ajout de documents augmente le compteur."""
        from src.vector_store import add_documents
        from src.chunker import split_documents
        initial = tmp_vector_store._collection.count()
        chunks = split_documents(sample_docs[:2], chunk_size=200, chunk_overlap=20)
        add_documents(tmp_vector_store, chunks)
        assert tmp_vector_store._collection.count() > initial

    def test_similarity_search(self, tmp_vector_store, sample_docs):
        """La recherche retourne des résultats pertinents."""
        from src.vector_store import add_documents
        from src.chunker import split_documents
        chunks = split_documents(sample_docs, chunk_size=300, chunk_overlap=20)
        add_documents(tmp_vector_store, chunks)

        results = tmp_vector_store.similarity_search("architecture Transformer attention", k=3)
        assert len(results) > 0
        assert any("Transformer" in r.page_content or "attention" in r.page_content
                   for r in results)

    def test_retriever_invoke(self, tmp_vector_store, sample_docs):
        """Le retriever retourne des Documents."""
        from src.vector_store import get_retriever, add_documents
        from src.chunker import split_documents
        chunks = split_documents(sample_docs, chunk_size=300, chunk_overlap=20)
        add_documents(tmp_vector_store, chunks)

        retriever = get_retriever(tmp_vector_store, search_type="similarity", top_k=3)
        docs = retriever.invoke("RAG retrieval generation")
        assert len(docs) > 0
        assert all(hasattr(d, "page_content") for d in docs)


# ── Tests RAGPipeline ────────────────────────────────────────

class TestRAGPipeline:

    def test_query_returns_result(self, rag_pipeline):
        """query() retourne un RAGResult valide."""
        from src.rag_pipeline import RAGResult
        result = rag_pipeline.query("Qu'est-ce que le Transformer ?")
        assert isinstance(result, RAGResult)
        assert len(result.answer) > 0
        assert len(result.sources) > 0

    def test_sources_have_metadata(self, rag_pipeline):
        """Les sources contiennent les métadonnées."""
        result = rag_pipeline.query("BERT modèle Google")
        for src in result.sources:
            assert "metadata" in src
            assert "content" in src

    def test_timing_recorded(self, rag_pipeline):
        """Les temps retrieval/génération sont enregistrés."""
        result = rag_pipeline.query("ChromaDB base vectorielle")
        assert result.retrieval_time >= 0
        assert result.generation_time >= 0
        assert result.total_time >= 0

    def test_stream_returns_tokens(self, rag_pipeline):
        """stream() retourne des tokens."""
        tokens = list(rag_pipeline.stream("Qu'est-ce que GPT ?"))
        assert len(tokens) > 0
        full = "".join(tokens)
        assert len(full) > 0

    def test_empty_question_raises(self, rag_pipeline):
        """Une question vide lève ValueError."""
        with pytest.raises(ValueError):
            rag_pipeline.query("")

    def test_get_stats(self, rag_pipeline):
        """get_stats() retourne un dict avec les clés attendues."""
        stats = rag_pipeline.get_stats()
        assert "documents_indexed" in stats
        assert "embedding_model" in stats
        assert "search_type" in stats
        assert stats["documents_indexed"] > 0

    def test_query_with_sources(self, rag_pipeline):
        """query_with_sources() retourne answer + source_documents."""
        out = rag_pipeline.query_with_sources("RAG retrieval")
        assert "answer" in out
        assert "source_documents" in out
        assert len(out["source_documents"]) > 0

    def test_display_no_error(self, rag_pipeline, capsys):
        """display() ne lève pas d'exception."""
        result = rag_pipeline.query("test question")
        result.display()
        captured = capsys.readouterr()
        assert "test question" in captured.out