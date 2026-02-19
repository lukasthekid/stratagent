"""Base loader interface."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from ingestion.models import Document


@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol for document loaders."""

    def load(self) -> list[Document]:
        """Load and return standardized documents."""
        ...
