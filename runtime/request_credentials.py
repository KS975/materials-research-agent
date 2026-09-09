from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_request_authorization: ContextVar[str | None] = ContextVar(
    "request_authorization", default=None
)
_request_platform_company_id: ContextVar[str | None] = ContextVar(
    "request_platform_company_id", default=None
)


def extract_bearer_token(authorization: str | None) -> str | None:
    value = str(authorization or "").strip()
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.casefold() == "bearer" and token.strip():
        return token.strip()
    return value


@contextmanager
def request_authorization_scope(
    authorization: str | None,
    platform_company_id: str | None = None,
) -> Iterator[None]:
    """Keep raw request identity values alive only for this HTTP call."""

    token = extract_bearer_token(authorization)
    reset = _request_authorization.set(token)
    company_reset = _request_platform_company_id.set(
        str(platform_company_id or "").strip() or None
    )
    try:
        yield
    finally:
        _request_platform_company_id.reset(company_reset)
        _request_authorization.reset(reset)


def get_request_authorization() -> str | None:
    return _request_authorization.get()


def get_request_platform_company_id() -> str | None:
    return _request_platform_company_id.get()
