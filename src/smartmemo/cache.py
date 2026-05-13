"""Public cache facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from smartmemo.classifier import ClassifierService
from smartmemo.embedding import (
    EmbeddingService,
    FaissVectorIndex,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from smartmemo.exceptions import MissingDependencyError
from smartmemo.models import CacheConfig, CacheResult, CacheStats, ClassifierConfig
from smartmemo.orchestrator import CacheOrchestrator
from smartmemo.store import SQLiteCacheStore
from smartmemo.types import EmbeddingProvider, LLMFunction


class SmartMemo:
    """Async-first semantic cache facade.

    Without a classifier checkpoint, SmartMemo uses cosine similarity as a
    measured baseline. With a classifier checkpoint, cosine search selects
    candidates and the learned classifier decides cache hits.
    """

    def __init__(
        self,
        *,
        domain: str,
        config: CacheConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        classifier: ClassifierConfig | None = None,
        store: SQLiteCacheStore | None = None,
        use_faiss: bool = True,
    ) -> None:
        self.domain = domain
        self.config = config or CacheConfig()
        self.store = store or SQLiteCacheStore(self.config.db_path)
        provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            self.config.embedding_model,
            dim=self.config.embedding_dim,
        )
        if use_faiss:
            try:
                index = FaissVectorIndex(provider.dim)
            except MissingDependencyError:
                index = InMemoryVectorIndex(provider.dim)
        else:
            index = InMemoryVectorIndex(provider.dim)
        embedding_service = EmbeddingService(provider, index)
        classifier_service = self._build_classifier_service(classifier)
        self._orchestrator = CacheOrchestrator(
            domain=domain,
            config=self.config,
            store=self.store,
            embedding_service=embedding_service,
            classifier_service=classifier_service,
        )

    async def get_or_call(
        self,
        *,
        prompt: str,
        llm_function: LLMFunction,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        return await self._orchestrator.get_or_call(
            prompt=prompt,
            llm_function=llm_function,
            model=model,
            metadata=metadata,
        )

    async def report_bad_hit(self, query_id: UUID, reason: str | None = None) -> bool:
        return self._orchestrator.report_bad_hit(query_id, reason=reason)

    async def report_good_hit(self, query_id: UUID) -> bool:
        return self._orchestrator.report_good_hit(query_id)

    def stats(self) -> CacheStats:
        return self._orchestrator.stats()

    def export_feedback_pairs(self, path: Path | str, *, split: str = "train") -> int:
        return self._orchestrator.export_feedback_pairs(str(path), split=split)

    def close(self) -> None:
        self.store.close()

    def _build_classifier_service(
        self,
        classifier: ClassifierConfig | None,
    ) -> ClassifierService | None:
        if classifier is None or classifier.model_path is None:
            return None
        threshold = (
            classifier.threshold
            if classifier.threshold is not None
            else self.config.classifier_threshold
        )
        return ClassifierService(
            classifier.model_path,
            device=classifier.device,
            threshold=threshold,
        )
