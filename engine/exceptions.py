from __future__ import annotations


class EngineError(Exception):
    """Base exception for the independent engine."""


class ValidationError(EngineError):
    """Raised when an input or contract is invalid."""


class ArtifactError(EngineError):
    """Raised when an artifact cannot be loaded or persisted."""
