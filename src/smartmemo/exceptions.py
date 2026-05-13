"""Project-specific exceptions."""


class SmartMemoError(Exception):
    """Base exception for SmartMemo errors."""


class MissingDependencyError(SmartMemoError):
    """Raised when an optional dependency is required but not installed."""


class CacheStoreError(SmartMemoError):
    """Raised for persistence-layer failures."""
