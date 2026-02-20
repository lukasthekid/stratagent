"""Tests for RAG retriever and reranking."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from retrieval import retriever, retrieve_with_rerank, rerank, reset_retriever_cache


@pytest.fixture(autouse=True)
def _reset_retriever_cache():
    """Clear cached reranker so each test gets fresh mocks."""
    reset_retriever_cache()


@pytest.fixture
def sample_retrieved_docs() -> list[Document]:
    """Sample documents as returned by vector search."""
    return [
        Document(page_content="Dogs are loyal companions.", metadata={"source": "pets"}),
        Document(page_content="Cats enjoy independence.", metadata={"source": "pets"}),
        Document(page_content="Birds can sing melodies.", metadata={"source": "pets"}),
    ]


class TestRetriever:
    """Vector search retriever tests."""

    @patch("retrieval.retriever.get_vector_store")
    def test_retriever_returns_documents(
        self, mock_get_store: MagicMock, sample_retrieved_docs: list[Document]
    ) -> None:
        """Retriever returns documents from vector store."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = sample_retrieved_docs
        mock_get_store.return_value = mock_store

        result = retriever.invoke({"query": "pet animals", "k": 10})

        mock_store.similarity_search.assert_called_once_with("pet animals", k=10)
        assert result == sample_retrieved_docs
        assert len(result) == 3

    @patch("retrieval.retriever.get_vector_store")
    def test_retriever_uses_settings_default_k(self, mock_get_store: MagicMock) -> None:
        """Retriever uses settings.retrieval_top_k when k not provided."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_get_store.return_value = mock_store

        with patch("retrieval.retriever.settings") as mock_settings:
            mock_settings.retrieval_top_k = 50
            retriever.invoke({"query": "test"})

        mock_store.similarity_search.assert_called_once_with("test", k=50)

    @patch("retrieval.retriever.get_vector_store")
    def test_retriever_with_namespace(self, mock_get_store: MagicMock) -> None:
        """Retriever passes namespace to get_vector_store."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_get_store.return_value = mock_store

        retriever.invoke({"query": "test", "namespace": "my-ns"})

        mock_get_store.assert_called_once_with(namespace="my-ns")

    def test_retriever_empty_query_raises(self) -> None:
        """Empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.invoke({"query": ""})

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.invoke({"query": "   "})

    def test_retriever_invalid_k_raises(self) -> None:
        """Invalid k raises ValueError."""
        with pytest.raises(ValueError, match="k must be between 1 and 1000"):
            retriever.invoke({"query": "test", "k": 0})

        with pytest.raises(ValueError, match="k must be between 1 and 1000"):
            retriever.invoke({"query": "test", "k": 1001})


class TestRerank:
    """Reranking tests."""

    @patch("retrieval.retriever._get_reranker")
    def test_rerank_returns_top_k(
        self, mock_get_reranker: MagicMock, sample_retrieved_docs: list[Document]
    ) -> None:
        """Rerank returns top k documents by score."""
        mock_reranker = MagicMock()
        # Simulate scores: doc2 highest, doc0 mid, doc1 lowest
        mock_reranker.compute_score.return_value = [0.3, 0.9, 0.5]
        mock_get_reranker.return_value = mock_reranker

        result = rerank("pet animals", sample_retrieved_docs, k=2)

        assert len(result) == 2
        # Order by score desc: 0.9, 0.5, 0.3 -> doc1, doc2, doc0
        assert result[0].page_content == "Cats enjoy independence."
        assert result[1].page_content == "Birds can sing melodies."

    @patch("retrieval.retriever._get_reranker")
    def test_rerank_single_document_handles_float(
        self, mock_get_reranker: MagicMock
    ) -> None:
        """Rerank handles single pair returning float (not list)."""
        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = 0.85  # float for single pair
        mock_get_reranker.return_value = mock_reranker

        docs = [Document(page_content="Single doc.", metadata={})]
        result = rerank("query", docs, k=1)

        assert len(result) == 1
        assert result[0].page_content == "Single doc."

    def test_rerank_empty_documents_returns_empty(self) -> None:
        """Rerank with empty list returns empty list."""
        result = rerank("query", [], k=5)
        assert result == []

    def test_rerank_empty_query_raises(self) -> None:
        """Empty query raises ValueError."""
        docs = [Document(page_content="x", metadata={})]
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rerank("", docs)

    @patch("retrieval.retriever._get_reranker")
    def test_rerank_k_exceeds_docs_returns_all(
        self, mock_get_reranker: MagicMock, sample_retrieved_docs: list[Document]
    ) -> None:
        """Rerank with k > len(docs) returns all docs."""
        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = [0.5, 0.3, 0.7]
        mock_get_reranker.return_value = mock_reranker

        result = rerank("query", sample_retrieved_docs, k=10)

        assert len(result) == 3


class TestRetrieveWithRerank:
    """Full RAG pipeline tests."""

    @patch("retrieval.retriever.rerank")
    @patch("retrieval.retriever.get_vector_store")
    def test_retrieve_with_rerank_returns_reranked(
        self,
        mock_get_store: MagicMock,
        mock_rerank: MagicMock,
        sample_retrieved_docs: list[Document],
    ) -> None:
        """Full pipeline: vector search then rerank."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = sample_retrieved_docs
        mock_get_store.return_value = mock_store

        top_two = sample_retrieved_docs[:2]
        mock_rerank.return_value = top_two

        result = retrieve_with_rerank.invoke(
            {"query": "pet animals", "retrieval_k": 50, "rerank_k": 2}
        )

        mock_store.similarity_search.assert_called_once_with("pet animals", k=50)
        mock_rerank.assert_called_once_with(
            "pet animals", sample_retrieved_docs, k=2
        )
        assert result == top_two

    @patch("retrieval.retriever.get_vector_store")
    def test_retrieve_with_rerank_empty_candidates(
        self, mock_get_store: MagicMock
    ) -> None:
        """When no candidates found, returns empty list."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_get_store.return_value = mock_store

        result = retrieve_with_rerank.invoke({"query": "obscure query"})

        assert result == []

    def test_retrieve_with_rerank_empty_query_raises(self) -> None:
        """Empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieve_with_rerank.invoke({"query": ""})


class TestResetRetrieverCache:
    """Cache reset tests."""

    def test_reset_retriever_cache_clears_reranker(self) -> None:
        """reset_retriever_cache clears cached reranker."""
        # Trigger load
        with patch("retrieval.retriever.FlagReranker") as mock_cls:
            mock_reranker = MagicMock()
            mock_reranker.compute_score.return_value = [0.5]
            mock_cls.return_value = mock_reranker

            rerank("q", [Document(page_content="x", metadata={})])

        reset_retriever_cache()

        # Next rerank should create new instance
        with patch("retrieval.retriever.FlagReranker") as mock_cls2:
            mock_reranker2 = MagicMock()
            mock_reranker2.compute_score.return_value = [0.8]
            mock_cls2.return_value = mock_reranker2

            result = rerank("q", [Document(page_content="y", metadata={})])

        mock_cls2.assert_called_once()
        assert result[0].page_content == "y"
