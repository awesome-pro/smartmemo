from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from smartmemo import CacheConfig, SmartMemo
from smartmemo.embedding import EmbeddingService, InMemoryVectorIndex
from smartmemo.orchestrator import CacheOrchestrator
from smartmemo.store import SQLiteCacheStore
from smartmemo.types import FloatVector


class ToyEmbeddingProvider:
    dim = 4

    def embed(self, text: str) -> FloatVector:
        match text:
            case "alpha":
                return np.array([1, 0, 0, 0], dtype=np.float32)
            case "alpha duplicate":
                return np.array([1, 0, 0, 0], dtype=np.float32)
            case "near alpha":
                return np.array([0.8, 0.2, 0, 0], dtype=np.float32)
            case "beta":
                return np.array([0, 1, 0, 0], dtype=np.float32)
            case _:
                return np.array([0, 0, 1, 0], dtype=np.float32)


class RejectingClassifier:
    threshold = 0.8

    def predict_batch(self, pairs: Sequence[tuple[FloatVector, FloatVector]]) -> list[float]:
        return [0.2 for _ in pairs]


class NearAlphaClassifier:
    threshold = 0.8

    def predict_batch(self, pairs: Sequence[tuple[FloatVector, FloatVector]]) -> list[float]:
        near_alpha = np.array([0.8, 0.2, 0, 0], dtype=np.float32)
        near_alpha = near_alpha / np.linalg.norm(near_alpha)
        return [0.95 if np.allclose(pair[1], near_alpha) else 0.2 for pair in pairs]


def build_classifier_orchestrator(
    *,
    tmp_path: Path,
    cache_config: CacheConfig,
    classifier_service: RejectingClassifier | NearAlphaClassifier,
) -> tuple[CacheOrchestrator, SQLiteCacheStore, EmbeddingService]:
    provider = ToyEmbeddingProvider()
    store = SQLiteCacheStore(tmp_path / "classifier-cache.db")
    embedding_service = EmbeddingService(provider, InMemoryVectorIndex(provider.dim))
    orchestrator = CacheOrchestrator(
        domain="test",
        config=cache_config,
        store=store,
        embedding_service=embedding_service,
        classifier_service=classifier_service,
    )
    return orchestrator, store, embedding_service


def add_seed_entry(
    *,
    store: SQLiteCacheStore,
    embedding_service: EmbeddingService,
    prompt: str,
    response: str,
) -> None:
    embedding = embedding_service.embed(prompt)
    entry_id = store.add(
        prompt=prompt,
        embedding=embedding,
        response=response,
        model="unit-test",
    )
    embedding_service.add(entry_id, embedding)


async def test_get_or_call_misses_then_hits(cache: SmartMemo) -> None:
    calls = 0

    async def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return f"fresh:{prompt}"

    first = await cache.get_or_call(prompt="alpha", llm_function=llm, model="unit-test")
    second = await cache.get_or_call(prompt="alpha duplicate", llm_function=llm)

    assert first.was_cache_hit is False
    assert second.was_cache_hit is True
    assert second.response == "fresh:alpha"
    assert second.cache_entry_id == first.cache_entry_id
    assert second.classifier_score is None
    assert second.cost_saved_usd > 0
    assert calls == 1


async def test_threshold_rejects_low_similarity(cache: SmartMemo) -> None:
    calls = 0

    def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return f"fresh:{prompt}"

    first = await cache.get_or_call(prompt="alpha", llm_function=llm)
    second = await cache.get_or_call(prompt="beta", llm_function=llm)

    assert first.was_cache_hit is False
    assert second.was_cache_hit is False
    assert second.similarity_score == 0.0
    assert calls == 2


async def test_classifier_rejects_high_cosine_candidate(
    cache_config: CacheConfig,
    tmp_path: Path,
) -> None:
    orchestrator, store, embedding_service = build_classifier_orchestrator(
        tmp_path=tmp_path,
        cache_config=cache_config,
        classifier_service=RejectingClassifier(),
    )
    add_seed_entry(
        store=store,
        embedding_service=embedding_service,
        prompt="alpha",
        response="cached:alpha",
    )
    calls = 0

    async def llm(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return f"fresh:{prompt}"

    result = await orchestrator.get_or_call(prompt="alpha duplicate", llm_function=llm)

    assert result.was_cache_hit is False
    assert result.response == "fresh:alpha duplicate"
    assert result.similarity_score == 1.0
    assert result.classifier_score == 0.2
    assert calls == 1
    store.close()


async def test_classifier_selects_best_equivalent_candidate(
    cache_config: CacheConfig,
    tmp_path: Path,
) -> None:
    orchestrator, store, embedding_service = build_classifier_orchestrator(
        tmp_path=tmp_path,
        cache_config=cache_config,
        classifier_service=NearAlphaClassifier(),
    )
    add_seed_entry(
        store=store,
        embedding_service=embedding_service,
        prompt="alpha",
        response="cached:alpha",
    )
    add_seed_entry(
        store=store,
        embedding_service=embedding_service,
        prompt="near alpha",
        response="cached:near-alpha",
    )

    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    result = await orchestrator.get_or_call(prompt="alpha duplicate", llm_function=llm)

    assert result.was_cache_hit is True
    assert result.response == "cached:near-alpha"
    assert result.similarity_score is not None
    assert result.similarity_score < 1.0
    assert result.classifier_score == 0.95
    store.close()


async def test_feedback_updates_hit_entry(cache: SmartMemo) -> None:
    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    await cache.get_or_call(prompt="alpha", llm_function=llm)
    hit = await cache.get_or_call(prompt="alpha duplicate", llm_function=llm)

    assert await cache.report_bad_hit(hit.query_id, reason="wrong answer") is True
    assert hit.cache_entry_id is not None
    entry = cache.store.get(hit.cache_entry_id)
    assert entry is not None
    assert entry.feedback_negative_count == 1


async def test_stats_track_runtime_counts(cache: SmartMemo) -> None:
    async def llm(prompt: str) -> str:
        return f"fresh:{prompt}"

    await cache.get_or_call(prompt="alpha", llm_function=llm)
    await cache.get_or_call(prompt="alpha duplicate", llm_function=llm)
    stats = cache.stats()

    assert stats.total_entries == 1
    assert stats.total_lookups == 2
    assert stats.cache_hits == 1
    assert stats.cache_misses == 1
    assert stats.hit_rate == 0.5
