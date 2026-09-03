from __future__ import annotations

import base64
import json
from collections.abc import Mapping

from fastapi import HTTPException, Request, status

from app.config import Settings
from schemas.user_context import UserContext


class PlatformPermissionAdapter:
    """Resolve MatCloud's trusted forwarded request context.

    The adapter never persists or returns the Bearer token. Signature validation
    belongs to the unit's authentication gateway; direct trust is disabled by
    default and must be explicitly enabled after that gateway is in place.
    """

    _MAX_HEADER_LENGTH = 8192

    def __init__(self, settings: Settings):
        self.settings = settings

    def resolve(self, request: Request) -> UserContext:
        if not self.settings.platform_trust_forwarded_headers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "platform 权限模式尚未信任上游身份网关；请确认网关已验证 "
                    "Bearer Token 后设置 PLATFORM_TRUST_FORWARDED_HEADERS=true"
                ),
            )

        # The deployed MatCloud UI consistently sends authorization and
        # company-id. Organization headers are page/context dependent, so they
        # must narrow an existing company scope when present instead of being
        # prerequisites for establishing the user's identity.
        authorization = self._required_header(request, "authorization")
        company_id = self._required_header(request, "company-id")
        organization_id = self._optional_header(request, "organization-id")
        organization_level = self._optional_header(request, "organization-level")

        scheme, separator, token = authorization.partition(" ")
        if not separator or scheme.casefold() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authorization 必须使用 Bearer Token",
            )

        payload = self._decode_jwt_payload(token.strip())
        user_id = self._find_user_id(payload)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "已收到平台 Token，但其中没有可用的稳定用户标识；"
                    "需要平台 Token 提供 userId/user_id/sub/id claim"
                ),
            )

        return UserContext(
            user_id=user_id,
            company_id=company_id,
            project_ids=(),
            permission_source="platform_forwarded_headers",
            all_projects=True,
            organization_id=organization_id,
            organization_level=organization_level,
        )

    @classmethod
    def _required_header(cls, request: Request, name: str) -> str:
        value = (request.headers.get(name) or "").strip()
        if not value:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "平台请求缺少必需 Header：authorization、company-id"
                ),
            )
        if len(value) > cls._MAX_HEADER_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"平台 Header {name} 长度异常",
            )
        return value

    @classmethod
    def _optional_header(cls, request: Request, name: str) -> str | None:
        value = (request.headers.get(name) or "").strip()
        if not value:
            return None
        if len(value) > cls._MAX_HEADER_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"平台 Header {name} 长度异常",
            )
        return value

    @staticmethod
    def _decode_jwt_payload(token: str) -> Mapping[str, object]:
        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="平台 Authorization 不是有效的 JWT Bearer Token",
            )
        try:
            encoded = parts[1] + "=" * (-len(parts[1]) % 4)
            decoded = base64.urlsafe_b64decode(encoded.encode("ascii"))
            payload = json.loads(decoded.decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="平台 JWT Payload 无法解析",
            ) from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="平台 JWT Payload 格式无效",
            )
        return payload

    def _find_user_id(self, payload: Mapping[str, object]) -> str | None:
        claims = [
            item.strip()
            for item in self.settings.platform_jwt_user_claims.split(",")
            if item.strip()
        ]
        for claim in claims:
            value = payload.get(claim)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()

        # Some gateways put identity claims below ``user`` or ``data.user``.
        nested_candidates = [payload.get("user")]
        data = payload.get("data")
        if isinstance(data, Mapping):
            nested_candidates.append(data.get("user"))
        for candidate in nested_candidates:
            if not isinstance(candidate, Mapping):
                continue
            for claim in claims:
                value = candidate.get(claim)
                if isinstance(value, (str, int)) and str(value).strip():
                    return str(value).strip()
        return None
