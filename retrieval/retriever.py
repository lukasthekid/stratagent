"""RAG retriever with vector search and optional reranking."""

import logging
from typing import Any

from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from langchain_core.runnables import chain

from config import settings
from ingestion import get_vector_store

logger = logging.getLogger(__name__)

# Lazy-loaded reranker (avoids expensive model load at import time)
_reranker: FlagReranker | None = None


import threading
_reranker_lock = threading.Lock()

def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:  # double-checked locking
                _reranker = FlagReranker(settings.reranker_model, use_fp16=True)
    return _reranker


def reset_retriever_cache() -> None:
    global _reranker
    with _reranker_lock:
        _reranker = None


def _vector_search(
    query: str,
    k: int,
    namespace: str | None = None,
) -> list[Document]:
    """Internal: perform vector similarity search."""
    store = get_vector_store(namespace=namespace)
    results = store.similarity_search_with_score(query, k=k)
    candidates = [doc for doc, score in results if score >= settings.retriever_threshold]  # cosine threshold
    return candidates


def _unpack_retriever_input(input: str | dict) -> tuple[str, int | None, str | None]:
    """Unpack chain input: dict from invoke or string for direct call."""
    if isinstance(input, dict):
        return (
            input.get("query", ""),
            input.get("k"),
            input.get("namespace"),
        )
    return (str(input), None, None)


@chain
def retriever(
    query: str,
    k: int | None = None,
    namespace: str | None = None,
) -> list[Document]:
    """Retrieve documents via vector similarity search (no reranking).

    Args:
        query: Search query string (or dict with query, k, namespace when invoked).
        k: Number of documents to return. Defaults to settings.retrieval_top_k.
        namespace: Optional Pinecone namespace. Must match ingestion namespace.

    Returns:
        List of Documents ordered by similarity.

    Raises:
        ValueError: If query is empty or k is invalid.
    """
    query, k, namespace = _unpack_retriever_input(query)
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    k_val = k if k is not None else settings.retrieval_top_k
    if k_val < 1 or k_val > 1000:
        raise ValueError("k must be between 1 and 1000")

    docs = _vector_search(query, k_val, namespace)
    logger.debug("Retrieved %d documents for query (k=%d)", len(docs), k_val)
    return docs


def rerank(
    query: str,
    documents: list[Document],
    k: int | None = None,
) -> list[Document]:
    """Rerank documents by relevance to the query using a cross-encoder.

    Args:
        query: Search query string.
        documents: Documents to rerank (from vector search or elsewhere).
        k: Number of top documents to return. Defaults to settings.rerank_top_k.

    Returns:
        Top-k documents ordered by reranker score (descending).
    """
    if not documents:
        return []
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    k = k if k is not None else settings.rerank_top_k
    k = min(k, len(documents))

    pairs = [(query, d.page_content) for d in documents]
    reranker_model = _get_reranker()
    scores = reranker_model.compute_score(pairs, normalize=True)

    # FlagReranker returns float for single pair, list for multiple
    if isinstance(scores, (int, float)):
        scores = [scores]

    docs_with_scores = list(zip(documents, scores))
    reranked = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    top_docs = []
    for doc, score in reranked[:k]:
        #relevance threshold
        if score > settings.reranker_threshold:
            doc.metadata["rerank_score"] = round(score, 4)
            top_docs.append(doc)

    logger.debug("Reranked %d documents to top %d", len(documents), k)
    return top_docs


def _unpack_rerank_input(
    input: str | dict,
) -> tuple[str, int | None, int | None, str | None]:
    """Unpack chain input for retrieve_with_rerank."""
    if isinstance(input, dict):
        return (
            input.get("query", ""),
            input.get("retrieval_k"),
            input.get("rerank_k"),
            input.get("namespace"),
        )
    return (str(input), None, None, None)


@chain
def retrieve_with_rerank(
    query: str,
    retrieval_k: int | None = None,
    rerank_k: int | None = None,
    namespace: str | None = None,
) -> list[Document]:
    """Retrieve documents via vector search, then rerank for relevance.

    Two-stage RAG pipeline: (1) vector similarity search for candidates,
    (2) cross-encoder reranking for final ordering.

    Args:
        query: Search query string (or dict when invoked).
        retrieval_k: Number of candidates from vector search. Defaults to
            settings.retrieval_top_k.
        rerank_k: Number of final documents after reranking. Defaults to
            settings.rerank_top_k.
        namespace: Optional Pinecone namespace.

    Returns:
        Top rerank_k documents ordered by reranker score.
    """
    query, retrieval_k, rerank_k, namespace = _unpack_rerank_input(query)
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    k_retrieval = retrieval_k if retrieval_k is not None else settings.retrieval_top_k
    candidates = _vector_search(query, k_retrieval, namespace)
    if not candidates:
        logger.debug("No candidates retrieved for query")
        return []

    return rerank(query, candidates, k=rerank_k)
