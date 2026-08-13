from __future__ import annotations

from fastapi import HTTPException, Request, status

from schemas.user_context import UserContext


class PlatformPermissionAdapter:
    """Production adapter placeholder with fail-closed behavior.

    The actual MatCloud/材数智元 token/user-context contract has not been supplied yet.
    Do not infer it. Until implemented, this adapter always rejects access.
    """

    def resolve(self, request: Request) -> UserContext:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PlatformPermissionAdapter 尚未接入真实材数智元权限接口",
        )
