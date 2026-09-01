from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import Any, Callable, Iterator


ProgressCallback = Callable[[dict[str, Any]], None]

_progress_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "materials_agent_progress_callback",
    default=None,
)
_progress_started_at: ContextVar[float | None] = ContextVar(
    "materials_agent_progress_started_at",
    default=None,
)


@contextmanager
def progress_context(callback: ProgressCallback) -> Iterator[None]:
    """Attach a request-local progress sink without changing every API signature."""
    callback_token = _progress_callback.set(callback)
    started_token = _progress_started_at.set(monotonic())
    try:
        yield
    finally:
        _progress_started_at.reset(started_token)
        _progress_callback.reset(callback_token)


def emit_progress(
    stage: str,
    status: str,
    title: str,
    message: str,
    **details: Any,
) -> None:
    """Emit a structured, user-safe execution update when streaming is active.

    This deliberately carries auditable work stages, not hidden model reasoning.
    Calls are no-ops for the existing synchronous endpoint.
    """
    callback = _progress_callback.get()
    if callback is None:
        return
    started_at = _progress_started_at.get()
    elapsed_ms = (
        max(0, round((monotonic() - started_at) * 1000))
        if started_at is not None
        else 0
    )
    event: dict[str, Any] = {
        "schema_version": "1.1",
        "source": "backend",
        "stage": str(stage),
        "status": str(status),
        "title": str(title),
        "message": str(message),
        "elapsed_ms": elapsed_ms,
    }
    event.update({key: value for key, value in details.items() if value is not None})
    try:
        callback(event)
    except Exception:
        # UI progress must never break the actual analysis request.
        return
