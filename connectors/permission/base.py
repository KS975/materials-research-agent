from __future__ import annotations

from typing import Protocol

from fastapi import Request

from schemas.user_context import UserContext


class PermissionAdapter(Protocol):
    def resolve(self, request: Request) -> UserContext:
        ...
