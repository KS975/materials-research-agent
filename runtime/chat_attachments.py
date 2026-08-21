from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from file_processing.models import ParsedDocument
from schemas.user_context import UserContext


@dataclass(frozen=True, slots=True)
class ChatAttachment:
    attachment_id: str
    filename: str
    media_type: str
    parser: str
    page_count: int | None
    char_count: int
    chunk_count: int
    created_at: str
    expires_at: str
    chunks: tuple[dict, ...]


class ChatAttachmentStore:
    """Temporary current-chat attachment store.

    This is intentionally NOT the V0.1.2 long-term Knowledge Index. Files are
    kept in a local runtime directory with TTL and are always ownership scoped.
    """

    def __init__(self, root_dir: str, ttl_minutes: int = 180):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(minutes=max(5, ttl_minutes))

    def save(
        self,
        *,
        ctx: UserContext,
        filename: str,
        media_type: str,
        original_bytes: bytes,
        parsed: ParsedDocument,
    ) -> ChatAttachment:
        self.cleanup_expired()
        attachment_id = uuid4().hex
        created = datetime.now(UTC)
        expires = created + self.ttl
        folder = self.root / attachment_id
        folder.mkdir(parents=True, exist_ok=False)

        suffix = Path(filename).suffix.lower()
        source_path = folder / f"source{suffix}"
        source_path.write_bytes(original_bytes)

        payload = {
            "attachment_id": attachment_id,
            "filename": filename,
            "media_type": media_type,
            "parser": parsed.parser,
            "page_count": parsed.page_count,
            "char_count": len(parsed.text),
            "chunk_count": len(parsed.chunks),
            "created_at": created.isoformat(),
            "expires_at": expires.isoformat(),
            "owner": {
                "user_id": ctx.user_id,
                "company_id": ctx.company_id,
                "project_ids": list(ctx.project_ids),
                "all_projects": bool(ctx.all_projects),
            },
            "chunks": [chunk.to_dict() for chunk in parsed.chunks],
        }
        (folder / "parsed.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self._to_attachment(payload)

    def get(self, attachment_id: str, ctx: UserContext) -> ChatAttachment:
        self.cleanup_expired()
        folder = self.root / attachment_id
        meta_path = folder / "parsed.json"
        if not meta_path.exists():
            raise FileNotFoundError("当前 Chat 附件不存在或已过期")

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        owner = payload.get("owner") or {}
        if owner.get("user_id") != ctx.user_id or owner.get("company_id") != ctx.company_id:
            raise PermissionError("无权访问该 Chat 附件")

        original_projects = {int(x) for x in owner.get("project_ids") or []}
        original_all_projects = bool(owner.get("all_projects", False))
        current_projects = set(ctx.project_ids)

        # A current all-projects grant can access any attachment owned by the
        # same user/company. Narrower current scope may not open an attachment
        # that was created under a broader all-projects scope.
        if original_all_projects and not ctx.all_projects:
            raise PermissionError("当前项目权限范围不能访问该 Chat 附件")
        if (
            not ctx.all_projects
            and original_projects
            and not original_projects.issubset(current_projects)
        ):
            raise PermissionError("当前项目权限范围不能访问该 Chat 附件")

        expires_at = datetime.fromisoformat(payload["expires_at"])
        if expires_at <= datetime.now(UTC):
            shutil.rmtree(folder, ignore_errors=True)
            raise FileNotFoundError("当前 Chat 附件已过期")

        return self._to_attachment(payload)

    def delete(self, attachment_id: str, ctx: UserContext) -> None:
        self.get(attachment_id, ctx)
        shutil.rmtree(self.root / attachment_id, ignore_errors=True)

    def cleanup_expired(self) -> None:
        now = time.time()
        for folder in self.root.iterdir():
            if not folder.is_dir():
                continue
            meta_path = folder / "parsed.json"
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                expires = datetime.fromisoformat(payload["expires_at"]).timestamp()
            except Exception:
                # Broken temporary records older than the TTL are safe to remove.
                if now - folder.stat().st_mtime > self.ttl.total_seconds():
                    shutil.rmtree(folder, ignore_errors=True)
                continue
            if expires <= now:
                shutil.rmtree(folder, ignore_errors=True)

    @staticmethod
    def _to_attachment(payload: dict) -> ChatAttachment:
        return ChatAttachment(
            attachment_id=str(payload["attachment_id"]),
            filename=str(payload["filename"]),
            media_type=str(payload.get("media_type") or ""),
            parser=str(payload.get("parser") or ""),
            page_count=payload.get("page_count"),
            char_count=int(payload.get("char_count") or 0),
            chunk_count=int(payload.get("chunk_count") or 0),
            created_at=str(payload["created_at"]),
            expires_at=str(payload["expires_at"]),
            chunks=tuple(payload.get("chunks") or []),
        )
