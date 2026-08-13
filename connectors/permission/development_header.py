from __future__ import annotations

from fastapi import HTTPException, Request, status

from schemas.user_context import UserContext


class DevelopmentHeaderPermissionAdapter:
    """Temporary local integration adapter.

    This is NOT the production MatCloud permission implementation.
    It requires explicit company and project scope on every request.
    """

    def resolve(self, request: Request) -> UserContext:
        user_id = (request.headers.get("X-User-Id") or "").strip()
        company_id = (request.headers.get("X-Company-Id") or "").strip()
        raw_projects = (request.headers.get("X-Project-Ids") or "").strip()

        if not user_id or not company_id or not raw_projects:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "development_header 模式必须提供 "
                    "X-User-Id、X-Company-Id、X-Project-Ids"
                ),
            )

        try:
            project_ids = tuple(
                sorted({int(item.strip()) for item in raw_projects.split(",") if item.strip()})
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Project-Ids 必须是逗号分隔的整数",
            ) from exc

        if not project_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="项目权限范围不能为空",
            )

        return UserContext(
            user_id=user_id,
            company_id=company_id,
            project_ids=project_ids,
            permission_source="development_header",
        )
