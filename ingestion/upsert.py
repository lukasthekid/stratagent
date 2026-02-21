"""Document chunking and vector store upsert."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import torch

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Cached instances (reused across calls to avoid expensive model load and connection setup)
_embedding_model: PineconeEmbeddings | None = None
_pinecone_index: Any = None
_vector_store: PineconeVectorStore | None = None

def _get_embedding_model() -> PineconeEmbeddings:
    """Lazy-load and cache the embedding model (avoids ~10–30s load per call)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = PineconeEmbeddings(
            model=settings.embedding_model,
            pinecone_api_key=settings.pinecone_api_key,
        )
    return _embedding_model


def _get_pinecone_index():
    """Lazy-load and cache the Pinecone index (reuses connection, pool_threads for parallel upserts)."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(
            api_key=settings.pinecone_api_key,
            pool_threads=settings.pinecone_pool_threads,
        )

        index_name = settings.pinecone_index_name
        #create the Index
        if index_name not in pc.list_indexes().names():
            pc.create_index(name=index_name,
                            dimension=settings.embedding_dimensions,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                            )
        _pinecone_index = pc.Index(name=index_name)
    return _pinecone_index

def get_vector_store(namespace:str = None) -> PineconeVectorStore:
    global _vector_store
    if _vector_store is None:
        model = _get_embedding_model()
        index = _get_pinecone_index()

        store_kwargs: dict[str, Any] = {"embedding": model, "index": index}
        if namespace is not None:
            store_kwargs["namespace"] = namespace
        _vector_store = PineconeVectorStore(**store_kwargs)

    return _vector_store



def reset_upsert_cache() -> None:
    """Clear cached model and index. Use in tests or when settings change."""
    global _embedding_model, _pinecone_index
    _embedding_model = None
    _pinecone_index = None


def upsert_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    namespace: str | None = None,
    batch_size: int | None = None,
    embedding_chunk_size: int | None = None,
) -> list[str]:
    """Chunk documents and upsert them into Pinecone.

    Documents are split with RecursiveCharacterTextSplitter, embedded via
    HuggingFace (batched), and stored in the configured Pinecone index.
    Embedding model and Pinecone client are cached for reuse across calls.

    Args:
        documents: LangChain Documents to chunk and upsert.
        chunk_size: Max characters per chunk. Default 512.
        chunk_overlap: Overlap between chunks. Default 50.
        namespace: Optional Pinecone namespace. Defaults to index default.
        batch_size: Pinecone upsert batch size. Default from settings (64).
        embedding_chunk_size: Docs per embedding batch. Default 500.

    Returns:
        List of Pinecone document IDs for the upserted chunks.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty.")

    batch_size = batch_size or settings.upsert_batch_size
    embedding_chunk_size = embedding_chunk_size or 500

    logger.info(
        "Upserting %d document(s) with chunk_size=%d, overlap=%d, batch=%d",
        len(documents), chunk_size, chunk_overlap, batch_size,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Split into %d chunk(s)", len(chunks))

    storage = get_vector_store()

    ids = storage.add_documents(
        documents=chunks,
        batch_size=batch_size,
        embedding_chunk_size=embedding_chunk_size,
    )

    logger.info("Upserted %d chunk(s) to Pinecone", len(ids))
    return ids



