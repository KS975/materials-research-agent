from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from schemas.chat_ui import ChatUIRequest, ChatUIResponse
from schemas.user_context import UserContext


WORKFLOW_SCHEMA_VERSION = 3
WORKFLOW_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,63}")


class ChatUIWorkflowError(RuntimeError):
    pass


class ChatUIWorkflowNotFoundError(ChatUIWorkflowError):
    pass


class ChatUIWorkflowPermissionError(ChatUIWorkflowError):
    pass


class ChatUIWorkflowConflictError(ChatUIWorkflowError):
    pass


class ChatUIWorkflowCheckpointError(ChatUIWorkflowError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _fingerprint(body: ChatUIRequest, ctx: UserContext) -> str:
    payload = {
        "message": body.message,
        "history": [item.model_dump(mode="json") for item in body.history],
        "attachment_ids": list(body.attachment_ids),
        "attachment_reference_mode": body.attachment_reference_mode,
        "scope": {
            "user_id": ctx.user_id,
            "company_id": ctx.company_id,
            "project_ids": list(ctx.project_ids),
            "all_projects": ctx.all_projects,
        },
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ChatUIWorkflowStore:
    """Company/user-scoped atomic checkpoints for Chat UI workflows.

    Only Agent runtime files are written. Business MySQL remains read-only.
    The complete response is checkpointed after dispatch so an explicit resume
    can replay it without repeating an LLM or database call.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_response_chars: int = 2_000_000,
        checkpoint_retries: int = 3,
        lease_seconds: int = 120,
        ttl_hours: int = 72,
        cleanup_interval_seconds: int = 900,
    ) -> None:
        self.root = Path(root)
        self.max_response_chars = max(100_000, min(int(max_response_chars), 10_000_000))
        self.checkpoint_retries = max(1, min(int(checkpoint_retries), 5))
        self.lease_seconds = max(10, min(int(lease_seconds), 3600))
        self.ttl_hours = max(1, min(int(ttl_hours), 720))
        self.cleanup_interval_seconds = max(60, int(cleanup_interval_seconds))
        self._last_cleanup_monotonic = 0.0
        self._lock = threading.RLock()

    @staticmethod
    def _safe_workflow_id(value: str | None) -> str:
        workflow_id = str(value or "").strip()
        if not WORKFLOW_ID_PATTERN.fullmatch(workflow_id):
            raise ChatUIWorkflowConflictError(
                "workflow_id 必须为8到64位字母、数字、点、下划线或短横线"
            )
        return workflow_id

    @staticmethod
    def _scope_dir(ctx: UserContext) -> str:
        raw = f"{ctx.company_id}\0{ctx.user_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _path(self, ctx: UserContext, workflow_id: str) -> Path:
        return self.root / self._scope_dir(ctx) / f"{workflow_id}.json"

    def begin(
        self,
        *,
        body: ChatUIRequest,
        ctx: UserContext,
    ) -> dict[str, Any]:
        self._maybe_cleanup()
        requested = str(body.workflow_id or "").strip()
        resume = bool(body.resume_workflow)
        if resume and not requested:
            raise ChatUIWorkflowConflictError(
                "resume_workflow=true 时必须提供 workflow_id"
            )
        workflow_id = (
            self._safe_workflow_id(requested)
            if requested
            else str(uuid.uuid4())
        )
        path = self._path(ctx, workflow_id)
        fingerprint = _fingerprint(body, ctx)
        now = _utc_now()
        with self._lock:
            if path.exists():
                data = self._read(path)
                self._verify(data, ctx, fingerprint)
                if not resume:
                    raise ChatUIWorkflowConflictError(
                        "workflow_id 已存在；如需恢复请设置 resume_workflow=true"
                    )
                if data.get("status") == "RUNNING" and self._lease_active(data):
                    raise ChatUIWorkflowConflictError(
                        "工作流仍在运行或租约尚未过期，拒绝并发恢复"
                    )
                data["resume_count"] = int(data.get("resume_count", 0)) + 1
                data["status"] = "RUNNING"
                data["last_resumed_at"] = now
                self._event(data, "WORKFLOW_RESUMED", {})
                self._write(path, data)
                return {
                    "workflow_id": workflow_id,
                    "resuming": True,
                    "cached_response": deepcopy(data.get("execution_response")),
                    "resume_count": data["resume_count"],
                }
            if resume:
                raise ChatUIWorkflowNotFoundError(
                    f"工作流检查点不存在：{workflow_id}"
                )
            data = {
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "workflow_id": workflow_id,
                "user_id": ctx.user_id,
                "company_id": ctx.company_id,
                "request_fingerprint": fingerprint,
                "status": "RUNNING",
                "stage": "created",
                "primary_family": "",
                "deterministic_kind": "",
                "semantic_family": "",
                "resume_count": 0,
                "created_at": now,
                "updated_at": now,
                "last_resumed_at": None,
                "execution_response": None,
                "last_error": None,
                "events": [],
            }
            self._event(data, "WORKFLOW_CREATED", {})
            self._write(path, data)
            return {
                "workflow_id": workflow_id,
                "resuming": False,
                "cached_response": None,
                "resume_count": 0,
            }

    def record_stage(
        self,
        *,
        workflow_id: str,
        ctx: UserContext,
        stage: str,
        state: dict[str, Any],
        response: ChatUIResponse | None = None,
    ) -> None:
        path = self._path(ctx, self._safe_workflow_id(workflow_id))
        with self._lock:
            data = self._read_owned(path, ctx)
            data["stage"] = str(stage)
            data["primary_family"] = str(state.get("primary_family") or "")
            data["deterministic_kind"] = str(state.get("deterministic_kind") or "")
            data["semantic_family"] = str(state.get("semantic_family") or "")
            data["last_error"] = None
            if response is not None:
                try:
                    payload = response.model_dump(mode="json")
                except Exception:
                    payload = response.model_dump(mode="python")
                payload = _jsonable(payload)
                encoded = json.dumps(payload, ensure_ascii=False, default=str)
                if len(encoded) <= self.max_response_chars:
                    data["execution_response"] = _jsonable(payload)
                    data["response_checkpointed"] = True
                else:
                    data["response_checkpointed"] = False
                    data["response_checkpoint_warning"] = (
                        "响应超过检查点上限，恢复时不能直接复用完整响应"
                    )
            self._event(data, "STAGE_COMPLETED", {"stage": str(stage)})
            self._write(path, data)

    def pause(
        self,
        *,
        workflow_id: str,
        ctx: UserContext,
        state: dict[str, Any],
    ) -> None:
        path = self._path(ctx, self._safe_workflow_id(workflow_id))
        with self._lock:
            data = self._read_owned(path, ctx)
            data["status"] = "PAUSED"
            data["stage"] = "primary_classified"
            data["primary_family"] = str(state.get("primary_family") or "")
            data["deterministic_kind"] = str(state.get("deterministic_kind") or "")
            self._event(data, "WORKFLOW_PAUSED", {"after": "classify_primary"})
            self._write(path, data)

    def finish(
        self,
        *,
        workflow_id: str,
        ctx: UserContext,
        state: dict[str, Any],
        response: ChatUIResponse,
    ) -> None:
        self.record_stage(
            workflow_id=workflow_id,
            ctx=ctx,
            stage="completed",
            state=state,
            response=response,
        )
        path = self._path(ctx, self._safe_workflow_id(workflow_id))
        with self._lock:
            data = self._read_owned(path, ctx)
            data["status"] = "SUCCEEDED"
            self._event(data, "WORKFLOW_COMPLETED", {})
            self._write(path, data)

    def fail(
        self,
        *,
        workflow_id: str,
        ctx: UserContext,
        error: Exception,
    ) -> None:
        path = self._path(ctx, self._safe_workflow_id(workflow_id))
        with self._lock:
            if not path.exists():
                return
            data = self._read_owned(path, ctx)
            data["status"] = "FAILED"
            data["last_error"] = {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "timestamp": _utc_now(),
            }
            self._event(data, "WORKFLOW_FAILED", data["last_error"])
            self._write(path, data)

    def status(self, workflow_id: str, ctx: UserContext) -> dict[str, Any]:
        path = self._path(ctx, self._safe_workflow_id(workflow_id))
        with self._lock:
            data = self._read_owned(path, ctx)
        return {
            "schema_version": data.get("schema_version"),
            "workflow_id": data.get("workflow_id"),
            "status": data.get("status"),
            "stage": data.get("stage"),
            "primary_family": data.get("primary_family"),
            "deterministic_kind": data.get("deterministic_kind"),
            "semantic_family": data.get("semantic_family"),
            "resume_count": data.get("resume_count", 0),
            "response_checkpointed": bool(data.get("execution_response")),
            "last_error": deepcopy(data.get("last_error")),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        }

    def _read_owned(self, path: Path, ctx: UserContext) -> dict[str, Any]:
        if not path.exists():
            raise ChatUIWorkflowNotFoundError("工作流检查点不存在")
        data = self._read(path)
        if data.get("company_id") != ctx.company_id or data.get("user_id") != ctx.user_id:
            raise ChatUIWorkflowPermissionError("无权访问该工作流检查点")
        return data

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ChatUIWorkflowNotFoundError("工作流检查点不存在") from exc
        except (OSError, ValueError) as exc:
            raise ChatUIWorkflowCheckpointError(
                f"工作流检查点读取失败：{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _verify(data: dict[str, Any], ctx: UserContext, fingerprint: str) -> None:
        if data.get("company_id") != ctx.company_id or data.get("user_id") != ctx.user_id:
            raise ChatUIWorkflowPermissionError("无权恢复该工作流")
        if data.get("request_fingerprint") != fingerprint:
            raise ChatUIWorkflowConflictError(
                "恢复请求与原始问题、附件或权限范围不一致"
            )

    def _lease_active(self, data: dict[str, Any]) -> bool:
        try:
            updated = datetime.fromisoformat(str(data.get("updated_at") or ""))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            return age < self.lease_seconds
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _event(data: dict[str, Any], event_type: str, payload: dict[str, Any]) -> None:
        events = data.setdefault("events", [])
        events.append({
            "event_id": len(events) + 1,
            "event_type": event_type,
            "timestamp": _utc_now(),
            "payload": _jsonable(payload),
        })
        if len(events) > 100:
            data["events"] = events[-100:]

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        data["updated_at"] = _utc_now()
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error: OSError | None = None
        safe_data = _jsonable(data)
        for attempt in range(1, self.checkpoint_retries + 1):
            tmp = path.with_suffix(f".json.tmp.{os.getpid()}.{threading.get_ident()}")
            try:
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(safe_data, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
                return
            except OSError as exc:
                last_error = exc
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt < self.checkpoint_retries:
                    time.sleep(0.05 * attempt)
        raise ChatUIWorkflowCheckpointError(
            "工作流检查点写入失败："
            f"{type(last_error).__name__}: {last_error}"
        )

    def _maybe_cleanup(self) -> None:
        now_mono = time.monotonic()
        if now_mono - self._last_cleanup_monotonic < self.cleanup_interval_seconds:
            return
        with self._lock:
            if now_mono - self._last_cleanup_monotonic < self.cleanup_interval_seconds:
                return
            self._last_cleanup_monotonic = now_mono
            if not self.root.exists():
                return
            cutoff = time.time() - self.ttl_hours * 3600
            for path in self.root.glob("*/*.json"):
                if not re.fullmatch(r"[0-9a-f]{24}", path.parent.name):
                    continue
                if not WORKFLOW_ID_PATTERN.fullmatch(path.stem):
                    continue
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    continue
