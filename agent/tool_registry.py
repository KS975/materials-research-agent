from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Callable
import uuid

from jsonschema import Draft202012Validator
from schemas.user_context import UserContext


ToolAuditSink = Callable[[dict[str, Any]], None]
ToolPermissionChecker = Callable[
    ["ToolSpec", dict[str, Any], UserContext | None],
    bool,
]


class ToolValidationError(ValueError):
    """Structured input-contract violation at the Tool boundary."""

    def __init__(self, message: str, *, errors: list[dict[str, Any]]):
        super().__init__(message)
        self.errors = errors


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    input_schema: dict[str, Any] | None = None
    access_scope: str = "legacy"
    handler_context: str = "auto"


class ToolRegistry:
    """Unified Tool boundary for schema, permission, and audit concerns."""

    def __init__(
        self,
        *,
        audit_sink: ToolAuditSink | None = None,
        permission_checker: ToolPermissionChecker | None = None,
    ):
        self._tools: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self._audit_sink = audit_sink
        self._permission_checker = permission_checker

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        *,
        input_schema: dict[str, Any] | None = None,
        access_scope: str = "legacy",
        handler_context: str | None = None,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"工具已注册：{name}")
        if access_scope not in {"legacy", "user_context", "host_scoped"}:
            raise ValueError(f"未知 Tool 权限范围：{access_scope}")
        if handler_context is None:
            handler_context = (
                "required"
                if access_scope == "user_context"
                else "omitted"
                if access_scope == "host_scoped"
                else "auto"
            )
        if handler_context not in {"auto", "required", "omitted"}:
            raise ValueError(f"未知 Tool 上下文模式：{handler_context}")
        if input_schema is not None:
            try:
                Draft202012Validator.check_schema(input_schema)
            except Exception as exc:
                raise ValueError(
                    f"Tool {name} 的 input_schema 不是合法 JSON Schema：{exc}"
                ) from exc

        spec = ToolSpec(
            name=name,
            description=description,
            handler=handler,
            input_schema=input_schema,
            access_scope=access_scope,
            handler_context=handler_context,
        )
        self._tools[name] = spec
        if input_schema is not None:
            self._validators[name] = Draft202012Validator(input_schema)

    def execute(self, name: str, **kwargs: Any) -> Any:
        call_id = str(uuid.uuid4())
        started_at = _utc_now()
        started_mono = monotonic()
        ctx = kwargs.pop("ctx", None)
        invocation_args = dict(kwargs)
        spec = self._tools.get(name)

        try:
            if spec is None:
                raise KeyError(f"未知工具：{name}")
            self._authorize(spec, invocation_args, ctx)
            self._validate(name, spec, invocation_args)
            self._audit(
                "TOOL_CALL_STARTED",
                call_id,
                spec,
                ctx,
                invocation_args,
                started_at,
                started_mono,
            )

            handler_kwargs = dict(invocation_args)
            if spec.handler_context == "required":
                handler_kwargs["ctx"] = ctx
            elif spec.handler_context == "auto" and ctx is not None:
                handler_kwargs["ctx"] = ctx

            result = spec.handler(**handler_kwargs)
            result_status = (
                str(result.get("status")) if isinstance(result, dict) else None
            )
            self._audit(
                "TOOL_CALL_COMPLETED",
                call_id,
                spec,
                ctx,
                invocation_args,
                started_at,
                started_mono,
                result_status=result_status,
            )
            return result
        except Exception as exc:
            self._audit(
                "TOOL_CALL_FAILED",
                call_id,
                spec,
                ctx,
                invocation_args,
                started_at,
                started_mono,
                error=exc,
                requested_tool_name=name,
            )
            raise

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "access_scope": spec.access_scope,
                "has_input_schema": spec.input_schema is not None,
            }
            for spec in self._tools.values()
        ]

    def _authorize(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        ctx: UserContext | None,
    ) -> None:
        if spec.access_scope == "user_context":
            if not isinstance(ctx, UserContext):
                raise PermissionError(f"Tool {spec.name} 必须携带 UserContext。")
        elif spec.access_scope == "host_scoped":
            if not isinstance(ctx, UserContext):
                raise PermissionError(f"Tool {spec.name} 必须由宿主携带 UserContext。")
            if ctx.all_projects or len(ctx.project_ids) != 1:
                raise PermissionError(
                    f"Tool {spec.name} 只能在已收敛为单一 Project 的宿主工作流中调用。"
                )
        if (
            self._permission_checker is not None
            and not self._permission_checker(spec, args, ctx)
        ):
            raise PermissionError(f"Tool {spec.name} 未通过权限策略。")

    def _validate(
        self,
        name: str,
        spec: ToolSpec,
        args: dict[str, Any],
    ) -> None:
        validator = self._validators.get(name)
        if validator is None:
            return
        errors = [
            {
                "path": "$." + ".".join(str(item) for item in error.absolute_path),
                "message": error.message,
            }
            for error in validator.iter_errors(args)
        ]
        if errors:
            raise ToolValidationError(
                f"Tool {name} 输入 Schema 校验失败。",
                errors=errors,
            )

    def _audit(
        self,
        event_type: str,
        call_id: str,
        spec: ToolSpec | None,
        ctx: UserContext | None,
        args: dict[str, Any],
        started_at: str,
        started_mono: float,
        *,
        result_status: str | None = None,
        error: BaseException | None = None,
        requested_tool_name: str = "",
    ) -> None:
        if self._audit_sink is None:
            return
        event: dict[str, Any] = {
            "schema_version": 1,
            "event_type": event_type,
            "call_id": call_id,
            "tool_name": spec.name if spec is not None else requested_tool_name,
            "tool_access_scope": spec.access_scope if spec is not None else "UNKNOWN",
            "schema_validation": (
                "VALID"
                if spec is not None and spec.input_schema is not None
                else "MISSING"
            ),
            "user_id": ctx.user_id if isinstance(ctx, UserContext) else None,
            "company_id": ctx.company_id if isinstance(ctx, UserContext) else None,
            "project_ids": list(ctx.project_ids) if isinstance(ctx, UserContext) else [],
            "all_projects": ctx.all_projects if isinstance(ctx, UserContext) else None,
            "argument_names": sorted(args.keys()),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_ms": max(0, round((monotonic() - started_mono) * 1000)),
            "result_status": result_status,
        }
        if error is not None:
            event["error"] = {
                "type": type(error).__name__,
                "message": str(error)[:2000],
            }
        self._audit_sink(event)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
