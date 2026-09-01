from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
from typing import Any
import uuid

from schemas.user_context import UserContext


CONVERSATION_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
CLIENT_MESSAGE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,95}")


class ChatHistoryError(RuntimeError):
    pass


class ChatHistoryNotFoundError(ChatHistoryError):
    pass


class ChatHistoryPermissionError(ChatHistoryError):
    pass


class ChatHistoryValidationError(ChatHistoryError):
    pass


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


class ChatHistoryStore:
    """Atomic JSON conversation history, isolated by user and company."""

    schema_version = 1

    def __init__(self, root: str | Path, *, max_messages: int = 400):
        self.root = Path(root)
        self.max_messages = max(20, min(int(max_messages), 2000))
        self._lock = threading.RLock()

    @staticmethod
    def new_conversation_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _conversation_id(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if not CONVERSATION_ID_PATTERN.fullmatch(normalized):
            raise ChatHistoryValidationError("conversation_id 格式无效")
        return normalized

    @staticmethod
    def _client_message_id(value: str) -> str:
        normalized = str(value or "").strip()
        if not CLIENT_MESSAGE_ID_PATTERN.fullmatch(normalized):
            raise ChatHistoryValidationError("client_message_id 格式无效")
        return normalized

    @staticmethod
    def _scope_dir(ctx: UserContext) -> str:
        raw = f"{ctx.company_id}\0{ctx.user_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _path(self, ctx: UserContext, conversation_id: str) -> Path:
        safe_id = self._conversation_id(conversation_id)
        return self.root / self._scope_dir(ctx) / f"{safe_id}.json"

    def append_exchange(
        self,
        *,
        ctx: UserContext,
        conversation_id: str,
        client_message_id: str,
        user_content: str,
        assistant_content: str,
        assistant_meta: dict[str, Any] | None = None,
        evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        conversation_id = self._conversation_id(conversation_id)
        client_message_id = self._client_message_id(client_message_id)
        user_content = str(user_content or "").strip()
        assistant_content = str(assistant_content or "").strip()
        if not user_content or not assistant_content:
            raise ChatHistoryValidationError("不能保存空白聊天消息")

        path = self._path(ctx, conversation_id)
        with self._lock:
            if path.exists():
                data = self._read_owned(path, ctx)
            else:
                created_at = _now()
                data = {
                    "schema_version": self.schema_version,
                    "conversation_id": conversation_id,
                    "owner": {
                        "user_id": ctx.user_id,
                        "company_id": ctx.company_id,
                        "organization_id": ctx.organization_id,
                        "organization_level": ctx.organization_level,
                    },
                    "title": self._default_title(user_content),
                    "created_at": created_at,
                    "updated_at": created_at,
                    "messages": [],
                }

            messages = data.setdefault("messages", [])
            if any(item.get("client_message_id") == client_message_id for item in messages):
                return self._summary(data)

            timestamp = _now()
            messages.extend([
                {
                    "message_id": f"{client_message_id}.user",
                    "client_message_id": client_message_id,
                    "role": "user",
                    "content": user_content,
                    "created_at": timestamp,
                },
                {
                    "message_id": f"{client_message_id}.assistant",
                    "client_message_id": client_message_id,
                    "role": "assistant",
                    "content": assistant_content,
                    "meta": _jsonable(assistant_meta or {}),
                    "evidence": _jsonable((evidence or [])[:50]),
                    "created_at": timestamp,
                },
            ])
            if len(messages) > self.max_messages:
                data["messages"] = messages[-self.max_messages :]
                data["messages_trimmed"] = True
            data["updated_at"] = timestamp
            self._write(path, data)
            return self._summary(data)

    def list_conversations(
        self,
        ctx: UserContext,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        folder = self.root / self._scope_dir(ctx)
        if not folder.exists():
            return {"total": 0, "conversations": []}
        items = []
        with self._lock:
            for path in folder.glob("*.json"):
                if not CONVERSATION_ID_PATTERN.fullmatch(path.stem):
                    continue
                try:
                    data = self._read_owned(path, ctx)
                except ChatHistoryError:
                    continue
                items.append(self._summary(data))
        items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return {
            "total": len(items),
            "conversations": items[offset : offset + limit],
        }

    def get_conversation(self, ctx: UserContext, conversation_id: str) -> dict[str, Any]:
        path = self._path(ctx, conversation_id)
        with self._lock:
            return deepcopy(self._read_owned(path, ctx))

    def rename(self, ctx: UserContext, conversation_id: str, title: str) -> dict[str, Any]:
        title = " ".join(str(title or "").split()).strip()
        if not title or len(title) > 120:
            raise ChatHistoryValidationError("会话标题必须为1到120个字符")
        path = self._path(ctx, conversation_id)
        with self._lock:
            data = self._read_owned(path, ctx)
            data["title"] = title
            data["updated_at"] = _now()
            self._write(path, data)
            return self._summary(data)

    def delete(self, ctx: UserContext, conversation_id: str) -> None:
        path = self._path(ctx, conversation_id)
        with self._lock:
            self._read_owned(path, ctx)
            try:
                path.unlink()
            except OSError as exc:
                raise ChatHistoryError(f"聊天记录删除失败：{type(exc).__name__}") from exc

    def _read_owned(self, path: Path, ctx: UserContext) -> dict[str, Any]:
        if not path.exists():
            raise ChatHistoryNotFoundError("聊天会话不存在")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChatHistoryError(f"聊天记录读取失败：{type(exc).__name__}") from exc
        owner = data.get("owner") or {}
        if owner.get("user_id") != ctx.user_id or owner.get("company_id") != ctx.company_id:
            raise ChatHistoryPermissionError("无权访问该聊天会话")
        return data

    @staticmethod
    def _default_title(message: str) -> str:
        compact = " ".join(message.split())
        return compact[:40] + ("…" if len(compact) > 40 else "")

    @staticmethod
    def _summary(data: dict[str, Any]) -> dict[str, Any]:
        messages = data.get("messages") or []
        return {
            "conversation_id": data.get("conversation_id"),
            "title": data.get("title") or "未命名会话",
            "message_count": len(messages),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "last_message_preview": str(messages[-1].get("content") or "")[:120]
            if messages else "",
        }

    @staticmethod
    def _write(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(
            f".json.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(_jsonable(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise ChatHistoryError(f"聊天记录写入失败：{type(exc).__name__}") from exc
