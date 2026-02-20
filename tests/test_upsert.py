"""Tests for document upsert to Pinecone."""

from unittest.mock import MagicMock, patch

import pytest

from ingestion import upsert_documents
from ingestion.upsert import reset_upsert_cache
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def _reset_upsert_cache():
    """Clear cached model/index so each test gets fresh mocks."""
    reset_upsert_cache()


class TestUpsertDocuments:
    """Upsert logic tests (mocked embeddings and Pinecone)."""

    @patch("ingestion.upsert.PineconeVectorStore")
    @patch("ingestion.upsert.Pinecone")
    @patch("ingestion.upsert.HuggingFaceEmbeddings")
    def test_upsert_returns_ids(self, mock_embeddings_cls, mock_pinecone_cls, mock_store_cls, sample_documents):
        mock_store = MagicMock()
        mock_store.add_documents.return_value = ["id-1", "id-2", "id-3"]
        mock_store_cls.return_value = mock_store

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        ids = upsert_documents(sample_documents)

        mock_embeddings_cls.assert_called_once()
        mock_store.add_documents.assert_called_once()
        docs_passed = mock_store.add_documents.call_args.kwargs["documents"]
        assert len(docs_passed) >= 2  # Chunked from 2 docs
        assert ids == ["id-1", "id-2", "id-3"]

    @patch("ingestion.upsert.PineconeVectorStore")
    @patch("ingestion.upsert.Pinecone")
    @patch("ingestion.upsert.HuggingFaceEmbeddings")
    def test_upsert_chunks_documents(self, mock_embeddings_cls, mock_pinecone_cls, mock_store_cls, sample_documents):
        mock_store = MagicMock()
        mock_store.add_documents.return_value = []
        mock_store_cls.return_value = mock_store

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        upsert_documents(sample_documents, chunk_size=20, chunk_overlap=5)

        docs_passed = mock_store.add_documents.call_args.kwargs["documents"]
        # With chunk_size=20, 2 short docs should produce multiple chunks
        assert len(docs_passed) > 2
        for doc in docs_passed:
            assert isinstance(doc, Document)
            assert doc.page_content
            assert "source" in doc.metadata

    @patch("ingestion.upsert.PineconeVectorStore")
    @patch("ingestion.upsert.Pinecone")
    @patch("ingestion.upsert.HuggingFaceEmbeddings")
    def test_upsert_uses_settings(self, mock_embeddings_cls, mock_pinecone_cls, mock_store_cls, sample_documents):
        mock_store = MagicMock()
        mock_store.add_documents.return_value = ["id-1"]
        mock_store_cls.return_value = mock_store

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        upsert_documents(sample_documents)

        mock_embeddings_cls.assert_called_once()
        call_kwargs = mock_embeddings_cls.call_args.kwargs
        assert "model_name" in call_kwargs

        mock_pinecone_cls.assert_called_once()
        assert "api_key" in mock_pinecone_cls.call_args.kwargs

    @patch("ingestion.upsert.PineconeVectorStore")
    @patch("ingestion.upsert.Pinecone")
    @patch("ingestion.upsert.HuggingFaceEmbeddings")
    def test_upsert_with_namespace(self, mock_embeddings_cls, mock_pinecone_cls, mock_store_cls, sample_documents):
        mock_store = MagicMock()
        mock_store.add_documents.return_value = ["id-1"]
        mock_store_cls.return_value = mock_store

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        upsert_documents(sample_documents, namespace="test-ns")

        mock_store_cls.assert_called_once()
        call_kwargs = mock_store_cls.call_args.kwargs
        assert call_kwargs["namespace"] == "test-ns"

    @patch("ingestion.upsert.PineconeVectorStore")
    @patch("ingestion.upsert.Pinecone")
    @patch("ingestion.upsert.HuggingFaceEmbeddings")
    def test_upsert_single_document(self, mock_embeddings_cls, mock_pinecone_cls, mock_store_cls):
        docs = [Document(page_content="Single doc.", metadata={"source": "single"})]
        mock_store = MagicMock()
        mock_store.add_documents.return_value = ["id-1"]
        mock_store_cls.return_value = mock_store

        mock_index = MagicMock()
        mock_pinecone_cls.return_value.Index.return_value = mock_index

        ids = upsert_documents(docs)
        assert ids == ["id-1"]


class TestUpsertErrors:
    """Error handling tests."""

    def test_empty_documents_raises(self):
        with pytest.raises(ValueError, match="Documents list cannot be empty"):
            upsert_documents([])
