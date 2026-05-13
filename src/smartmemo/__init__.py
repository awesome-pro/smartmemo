"""SmartMemo public API."""

from smartmemo.cache import SmartMemo
from smartmemo.models import (
    CacheConfig,
    CacheEntry,
    CacheResult,
    CacheStats,
    ClassifierConfig,
    EvictionPolicy,
)

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "CacheResult",
    "CacheStats",
    "ClassifierConfig",
    "SmartMemo",
    "EvictionPolicy",
]
