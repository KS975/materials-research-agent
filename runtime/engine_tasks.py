from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterator
import uuid

from agent.core import AgentCore
from agent.scenario_composer import ScenarioPlan, SkillPlanStep
from runtime.progress import progress_context
from schemas.user_context import UserContext


TASK_SCHEMA_VERSION = 1
TASK_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,63}")


class EngineTaskError(RuntimeError):
    pass


class EngineTaskNotFoundError(EngineTaskError):
    pass


class EngineTaskPermissionError(EngineTaskError):
    pass


class EngineTaskConflictError(EngineTaskError):
    pass


class EngineTaskCancelled(RuntimeError):
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


@dataclass(frozen=True, slots=True)
class EngineTaskRequest:
    intent: str
    tool_name: str
    tool_args: dict[str, Any]
    conversation_id: str = ""


class EngineTaskStore:
    """Company/user-scoped, atomic checkpoints for asynchronous engine tasks."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_result_chars: int = 2_000_000,
        checkpoint_retries: int = 3,
        lease_seconds: int = 300,
        max_events: int = 200,
    ) -> None:
        self.root = Path(root)
        self.max_result_chars = max(100_000, min(int(max_result_chars), 10_000_000))
        self.checkpoint_retries = max(1, min(int(checkpoint_retries), 5))
        self.lease_seconds = max(10, min(int(lease_seconds), 3600))
        self.max_events = max(20, min(int(max_events), 1000))
        self._lock = threading.RLock()

    @staticmethod
    def _safe_task_id(value: str) -> str:
        task_id = str(value or "").strip()
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise EngineTaskConflictError(
                "task_id 必须为8到64位字母、数字、点、下划线或短横线"
            )
        return task_id

    @staticmethod
    def _scope_dir(ctx: UserContext) -> str:
        raw = f"{ctx.company_id}\0{ctx.user_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _path(self, ctx: UserContext, task_id: str) -> Path:
        return self.root / self._scope_dir(ctx) / f"{task_id}.json"

    def create(
        self,
        *,
        ctx: UserContext,
        request: EngineTaskRequest,
        plan: ScenarioPlan,
    ) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = _utc_now()
        data = {
            "schema_version": TASK_SCHEMA_VERSION,
            "task_id": task_id,
            "user_id": ctx.user_id,
            "company_id": ctx.company_id,
            "project_ids": list(ctx.project_ids),
            "all_projects": bool(ctx.all_projects),
            "permission_source": ctx.permission_source,
            "organization_id": ctx.organization_id,
            "organization_level": ctx.organization_level,
            "intent": request.intent,
            "tool_name": request.tool_name,
            "tool_args": _jsonable(dict(request.tool_args)),
            "conversation_id": str(request.conversation_id or ""),
            "status": "AWAITING_APPROVAL" if plan.requires_approval else "QUEUED",
            "scenario_plan": plan.to_dict(),
            "approval": None,
            "step_checkpoints": {},
            "progress_events": [],
            "current_stage": "task_created",
            "result": None,
            "answer": None,
            "error": None,
            "resume_count": 0,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
        }
        self._write(self._path(ctx, task_id), data)
        return self.public_status(task_id, ctx)

    def public_status(self, task_id: str, ctx: UserContext) -> dict[str, Any]:
        with self._lock:
            data = self._read_owned(
                self._path(ctx, self._safe_task_id(task_id)), ctx
            )
        terminal = data.get("status") in {"SUCCEEDED", "FAILED", "CANCELLED"}
        return {
            "schema_version": data.get("schema_version"),
            "task_id": data.get("task_id"),
            "status": data.get("status"),
            "intent": data.get("intent"),
            "tool_name": data.get("tool_name"),
            "scenario_plan": deepcopy(data.get("scenario_plan") or {}),
            "current_stage": data.get("current_stage"),
            "progress_events": deepcopy(data.get("progress_events") or []),
            "completed_steps": sorted((data.get("step_checkpoints") or {}).keys()),
            "resume_count": int(data.get("resume_count", 0) or 0),
            "approval": deepcopy(data.get("approval")),
            "answer": data.get("answer") if terminal else None,
            "result": data.get("result") if terminal else None,
            "error": deepcopy(data.get("error")) if terminal else None,
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "started_at": data.get("started_at"),
            "finished_at": data.get("finished_at"),
        }

    def internal_task(self, task_id: str, ctx: UserContext) -> dict[str, Any]:
        with self._lock:
            return deepcopy(
                self._read_owned(self._path(ctx, self._safe_task_id(task_id)), ctx)
            )

    def mark_running(self, task_id: str, ctx: UserContext) -> None:
        def mutate(data: dict[str, Any]) -> None:
            if data.get("status") not in {"QUEUED", "CANCEL_REQUESTED"}:
                return
            if data.get("status") == "CANCEL_REQUESTED":
                data["status"] = "CANCELLED"
                data["finished_at"] = _utc_now()
                self._event(data, "TASK_CANCELLED", {"before": "worker_start"})
                return
            data["status"] = "RUNNING"
            data["started_at"] = data.get("started_at") or _utc_now()
            self._event(data, "TASK_STARTED", {})

        self._mutate(task_id, ctx, mutate)

    def append_event(self, task_id: str, ctx: UserContext, event: dict[str, Any]) -> None:
        def mutate(data: dict[str, Any]) -> None:
            self._event(data, "PROGRESS", dict(event))
            data["current_stage"] = str(
                event.get("stage") or data.get("current_stage") or "running"
            )

        self._mutate(task_id, ctx, mutate)

    def record_step(
        self,
        task_id: str,
        ctx: UserContext,
        step: SkillPlanStep,
        result: dict[str, Any],
    ) -> None:
        def mutate(data: dict[str, Any]) -> None:
            if data.get("status") != "RUNNING":
                return
            payload = _jsonable(result)
            encoded = json.dumps(payload, ensure_ascii=False, default=str)
            if len(encoded) > self.max_result_chars:
                raise EngineTaskConflictError(
                    "Skill 结果超过任务检查点上限，拒绝写入不完整检查点"
                )
            checkpoints = data.setdefault("step_checkpoints", {})
            checkpoints[step.step_id] = {
                "step_id": step.step_id,
                "skill_name": step.skill_name,
                "operation": step.operation,
                "status": str(result.get("status") or "OK"),
                "result": payload,
            }
            self._event(
                data,
                "SKILL_CHECKPOINTED",
                {"step_id": step.step_id, "operation": step.operation},
            )

        self._mutate(task_id, ctx, mutate)

    def step_result(
        self,
        task_id: str,
        ctx: UserContext,
        step_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            data = self._read_owned(
                self._path(ctx, self._safe_task_id(task_id)), ctx
            )
        checkpoint = dict(data.get("step_checkpoints") or {}).get(step_id)
        if not isinstance(checkpoint, dict):
            return None
        result = checkpoint.get("result")
        return deepcopy(result) if isinstance(result, dict) else None

    def finish(self, task_id: str, ctx: UserContext, result: dict[str, Any]) -> None:
        def mutate(data: dict[str, Any]) -> None:
            if data.get("status") == "CANCEL_REQUESTED":
                data["status"] = "CANCELLED"
                data["finished_at"] = _utc_now()
                self._event(data, "TASK_CANCELLED", {"before": "finish"})
                return
            if data.get("status") != "RUNNING":
                return
            payload = _jsonable(result)
            encoded = json.dumps(payload, ensure_ascii=False, default=str)
            if len(encoded) > self.max_result_chars:
                raise EngineTaskConflictError("任务最终结果超过检查点上限")
            data["status"] = "SUCCEEDED"
            data["result"] = payload
            data["answer"] = str(payload.get("answer") or "")
            data["finished_at"] = _utc_now()
            self._event(data, "TASK_COMPLETED", {})

        self._mutate(task_id, ctx, mutate)

    def fail(self, task_id: str, ctx: UserContext, error: BaseException) -> None:
        def mutate(data: dict[str, Any]) -> None:
            if data.get("status") == "CANCEL_REQUESTED":
                data["status"] = "CANCELLED"
            elif data.get("status") != "RUNNING":
                return
            else:
                data["status"] = "FAILED"
            data["error"] = {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "timestamp": _utc_now(),
            }
            data["finished_at"] = _utc_now()
            self._event(data, "TASK_FAILED", deepcopy(data["error"]))

        self._mutate(task_id, ctx, mutate)

    def request_cancel(self, task_id: str, ctx: UserContext) -> dict[str, Any]:
        def mutate(data: dict[str, Any]) -> None:
            status = str(data.get("status"))
            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                raise EngineTaskConflictError("任务已结束，不能取消")
            if status == "AWAITING_APPROVAL":
                data["status"] = "CANCELLED"
                data["finished_at"] = _utc_now()
                self._event(data, "TASK_CANCELLED", {"from": "AWAITING_APPROVAL"})
            elif status in {"QUEUED", "RUNNING"}:
                data["status"] = "CANCEL_REQUESTED"
                self._event(data, "TASK_CANCEL_REQUESTED", {})
            elif status != "CANCEL_REQUESTED":
                raise EngineTaskConflictError(f"当前状态不支持取消：{status}")

        self._mutate(task_id, ctx, mutate)
        return self.public_status(task_id, ctx)

    def approve(self, task_id: str, ctx: UserContext, reason: str = "") -> None:
        def mutate(data: dict[str, Any]) -> None:
            if data.get("status") != "AWAITING_APPROVAL":
                raise EngineTaskConflictError("任务不处于待审批状态")
            data["status"] = "QUEUED"
            data["approval"] = {
                "approved_by": ctx.user_id,
                "reason": str(reason or "")[:500],
                "approved_at": _utc_now(),
            }
            self._event(data, "TASK_APPROVED", deepcopy(data["approval"]))

        self._mutate(task_id, ctx, mutate)

    def resume(self, task_id: str, ctx: UserContext) -> None:
        def mutate(data: dict[str, Any]) -> None:
            status = str(data.get("status"))
            if status not in {"FAILED", "CANCELLED", "INTERRUPTED"}:
                raise EngineTaskConflictError(
                    f"当前状态不支持恢复：{status}"
                )
            data["status"] = "QUEUED"
            data["resume_count"] = int(data.get("resume_count", 0) or 0) + 1
            data["error"] = None
            data["result"] = None
            data["answer"] = None
            data["finished_at"] = None
            self._event(
                data,
                "TASK_RESUMED",
                {"resume_count": data["resume_count"]},
            )

        self._mutate(task_id, ctx, mutate)

    def recover_interrupted(self) -> int:
        if not self.root.exists():
            return 0
        recovered = 0
        for path in self.root.glob("*/*.json"):
            if not re.fullmatch(r"[0-9a-f]{24}", path.parent.name):
                continue
            if not TASK_ID_PATTERN.fullmatch(path.stem):
                continue
            try:
                with self._lock:
                    data = self._read(path)
                    if data.get("status") not in {"QUEUED", "RUNNING"}:
                        continue
                    if self._lease_active(data):
                        continue
                    data["status"] = "INTERRUPTED"
                    data["error"] = {
                        "type": "TaskInterrupted",
                        "message": "Worker 进程中断，任务已转入可恢复状态。",
                        "timestamp": _utc_now(),
                    }
                    self._event(data, "TASK_INTERRUPTED", {})
                    self._write(path, data)
                    recovered += 1
            except (OSError, ValueError):
                continue
        return recovered

    def is_cancel_requested(self, task_id: str, ctx: UserContext) -> bool:
        with self._lock:
            data = self._read_owned(
                self._path(ctx, self._safe_task_id(task_id)), ctx
            )
        return data.get("status") == "CANCEL_REQUESTED"

    def _mutate(
        self,
        task_id: str,
        ctx: UserContext,
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        path = self._path(ctx, self._safe_task_id(task_id))
        with self._lock:
            data = self._read_owned(path, ctx)
            mutate(data)
            self._write(path, data)

    def _read_owned(self, path: Path, ctx: UserContext) -> dict[str, Any]:
        if not path.exists():
            raise EngineTaskNotFoundError("任务不存在")
        data = self._read(path)
        if data.get("company_id") != ctx.company_id or data.get("user_id") != ctx.user_id:
            raise EngineTaskPermissionError("无权访问该任务")
        return data

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EngineTaskNotFoundError("任务不存在") from exc
        except (OSError, ValueError) as exc:
            raise EngineTaskConflictError(
                f"任务检查点读取失败：{type(exc).__name__}: {exc}"
            ) from exc

    def _lease_active(self, data: dict[str, Any]) -> bool:
        try:
            updated = datetime.fromisoformat(str(data.get("updated_at") or ""))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            return (
                datetime.now(timezone.utc) - updated
            ).total_seconds() < self.lease_seconds
        except (TypeError, ValueError):
            return True

    def _event(self, data: dict[str, Any], event_type: str, payload: dict[str, Any]) -> None:
        events = data.setdefault("progress_events", [])
        events.append({
            "event_id": len(events) + 1,
            "event_type": event_type,
            "timestamp": _utc_now(),
            "payload": _jsonable(payload),
        })
        if len(events) > self.max_events:
            data["progress_events"] = events[-self.max_events:]

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        data["updated_at"] = _utc_now()
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_data = _jsonable(data)
        last_error: OSError | None = None
        for attempt in range(1, self.checkpoint_retries + 1):
            tmp = path.with_suffix(f".json.tmp.{os.getpid()}.{threading.get_ident()}")
            try:
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(
                        safe_data,
                        handle,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
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
        raise EngineTaskConflictError(
            f"任务检查点写入失败：{type(last_error).__name__}: {last_error}"
        )


class EngineTaskManager:
    """Run scenario workflows in a bounded worker pool with Skill checkpoints."""

    def __init__(
        self,
        *,
        core: AgentCore,
        store: EngineTaskStore,
        worker_count: int = 2,
    ) -> None:
        if not 1 <= int(worker_count) <= 16:
            raise ValueError("worker_count 必须在1到16之间")
        self.core = core
        self.store = store
        self._executor = ThreadPoolExecutor(
            max_workers=int(worker_count),
            thread_name_prefix="engine-task-worker",
        )
        self._futures: dict[str, Future[None]] = {}
        self._futures_lock = threading.Lock()

    def submit(
        self,
        *,
        ctx: UserContext,
        request: EngineTaskRequest,
    ) -> dict[str, Any]:
        plan = self.core.scenario_composer.compose(
            intent=request.intent,
            tool_name=request.tool_name,
            tool_args=request.tool_args,
        )
        status = self.store.create(ctx=ctx, request=request, plan=plan)
        if status["status"] != "AWAITING_APPROVAL":
            self._schedule(str(status["task_id"]), ctx)
        return status

    def status(self, task_id: str, ctx: UserContext) -> dict[str, Any]:
        return self.store.public_status(task_id, ctx)

    def cancel(self, task_id: str, ctx: UserContext) -> dict[str, Any]:
        return self.store.request_cancel(task_id, ctx)

    def approve(self, task_id: str, ctx: UserContext, reason: str = "") -> dict[str, Any]:
        self.store.approve(task_id, ctx, reason=reason)
        self._schedule(task_id, ctx)
        return self.status(task_id, ctx)

    def resume(self, task_id: str, ctx: UserContext) -> dict[str, Any]:
        self.store.resume(task_id, ctx)
        self._schedule(task_id, ctx)
        return self.status(task_id, ctx)

    def recover_interrupted(self) -> int:
        return self.store.recover_interrupted()

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def _schedule(self, task_id: str, ctx: UserContext) -> None:
        with self._futures_lock:
            existing = self._futures.get(task_id)
            if existing is not None and not existing.done():
                return
            future = self._executor.submit(self._run, task_id, ctx)
            self._futures[task_id] = future

    def _run(self, task_id: str, ctx: UserContext) -> None:
        try:
            self.store.mark_running(task_id, ctx)
            task = self.store.internal_task(task_id, ctx)
            if task.get("status") == "CANCELLED":
                return

            request = EngineTaskRequest(
                intent=str(task.get("intent") or ""),
                tool_name=str(task.get("tool_name") or ""),
                tool_args=dict(task.get("tool_args") or {}),
                conversation_id=str(task.get("conversation_id") or ""),
            )
            plan = self.core.scenario_composer.compose(
                intent=request.intent,
                tool_name=request.tool_name,
                tool_args=request.tool_args,
            )
            execution_args = {
                **request.tool_args,
                "_workflow_id": task_id,
                "_conversation_id": request.conversation_id or task_id,
            }

            def progress(event: dict[str, Any]) -> None:
                self.store.append_event(task_id, ctx, event)

            def execute_skill(
                step: SkillPlanStep,
                args: dict[str, Any],
                user_ctx: UserContext,
            ) -> dict[str, Any]:
                if self.store.is_cancel_requested(task_id, ctx):
                    raise EngineTaskCancelled("用户已请求取消任务")
                checkpoint = self.store.step_result(task_id, ctx, step.step_id)
                if checkpoint is not None:
                    return checkpoint
                result = self.core.execute_skill_step(step, dict(args), user_ctx)
                self.store.record_step(task_id, ctx, step, result)
                return result

            with progress_context(progress):
                result = self.core.workflow_orchestrator.execute(
                    plan=plan,
                    tool_args=execution_args,
                    ctx=ctx,
                    execute_skill=execute_skill,
                )
            self.store.finish(task_id, ctx, result)
        except EngineTaskCancelled as exc:
            self.store.fail(task_id, ctx, exc)
        except Exception as exc:
            self.store.fail(task_id, ctx, exc)


@contextmanager
def managed_engine_task_manager(
    *,
    core: AgentCore,
    store: EngineTaskStore,
    worker_count: int = 2,
) -> Iterator[EngineTaskManager]:
    manager = EngineTaskManager(core=core, store=store, worker_count=worker_count)
    try:
        yield manager
    finally:
        manager.shutdown(wait=True)
