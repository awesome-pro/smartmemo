"""Embedding providers and vector indexes."""

from smartmemo.embedding.service import (
    EmbeddingService,
    FaissVectorIndex,
    HashEmbeddingProvider,
    InMemoryVectorIndex,
    SearchCandidate,
    SentenceTransformerEmbeddingProvider,
)

__all__ = [
    "EmbeddingService",
    "FaissVectorIndex",
    "HashEmbeddingProvider",
    "InMemoryVectorIndex",
    "SearchCandidate",
    "SentenceTransformerEmbeddingProvider",
]
