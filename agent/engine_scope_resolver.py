from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from dataclasses import replace

from agent.engine_workflow_types import ArtifactScope
from schemas.user_context import UserContext


class EngineScopeResolver:
    """Resolve one authorized Company/Project/session execution scope."""

    def __init__(self, artifact_root: str | Path):
        self.artifact_root = Path(artifact_root).resolve()

    def resolve(
        self,
        ctx: UserContext,
        args: dict[str, Any],
        workflow_id: str,
        conversation_id: str,
    ) -> ArtifactScope:
        raw_project_id = args.get("project_id")
        if raw_project_id is None:
            if len(ctx.project_ids) == 1:
                raw_project_id = ctx.project_ids[0]
            else:
                raise ValueError(
                    "无法唯一确定 Project；请明确 project_id 后再执行引擎工作流。"
                )
        try:
            project_id = int(raw_project_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("project_id 必须是整数。") from exc
        if not ctx.can_access_project(project_id):
            raise PermissionError(
                f"当前用户无权访问 Company={ctx.company_id} Project={project_id}。"
            )

        project_ctx = replace(
            ctx,
            project_ids=(project_id,),
            all_projects=False,
        )
        project_root = (
            self.artifact_root
            / "companies"
            / self._safe_token(ctx.company_id, "company")
            / "projects"
            / self._safe_token(f"project_{project_id}", "project")
        )
        conversation_token = conversation_id or workflow_id or "adhoc"
        session_root = project_root / "sessions" / self._session_token(
            conversation_token
        )
        return ArtifactScope(
            ctx=project_ctx,
            company_id=ctx.company_id,
            project_id=project_id,
            project_root=project_root,
            model_registry_path=project_root / "models" / "model-registry.json",
            session_root=session_root,
            conversation_id=conversation_token,
        )

    @staticmethod
    def _safe_token(value: Any, kind: str) -> str:
        text = str(value or "").strip()
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")[:64]
        if not token:
            token = kind
        if token != text:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
            return f"{token[:52]}_{digest}"
        return token

    @staticmethod
    def _session_token(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
