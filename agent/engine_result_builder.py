from __future__ import annotations

from typing import Any

from agent.engine_workflow_types import ArtifactScope


class EngineResultBuilder:
    """Build stable public workflow envelopes for engine executions."""

    def success(
        self,
        scope: ArtifactScope,
        intent: str,
        result: dict[str, Any],
        steps: list[str],
        answer: str,
        warnings: list[Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": intent,
            "status": "OK",
            "scope": self.public_scope(scope),
            "steps": [{"name": name, "status": "COMPLETED"} for name in steps],
            "result": result,
            "answer": answer,
            "evidence": [{
                "source": "engine_workflow",
                "company_id": scope.company_id,
                "project_id": scope.project_id,
                "artifact_root": str(scope.project_root),
            }],
            "warnings": list(warnings or []),
        }

    def blocked(
        self,
        scope: ArtifactScope,
        intent: str,
        result: dict[str, Any],
        answer: str,
        warnings: list[Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": intent,
            "status": "BLOCKED",
            "scope": self.public_scope(scope),
            "steps": [{"name": "modeling_gate", "status": "BLOCKED"}],
            "result": result,
            "answer": answer,
            "evidence": [{
                "source": "engine_modeling_gate",
                "company_id": scope.company_id,
                "project_id": scope.project_id,
            }],
            "warnings": list(warnings or []),
        }

    def model_required(
        self,
        scope: ArtifactScope,
        target_metric: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": "ensure_model",
            "status": "MODEL_REQUIRED",
            "scope": self.public_scope(scope),
            "steps": [{"name": "query_model_registry", "status": "COMPLETED"}],
            "result": {
                "target_metric": target_metric,
                "available_model_count": len(records),
            },
            "answer": (
                f"Project {scope.project_id} 当前没有可用于“{target_metric}”的已注册模型。"
                "请先明确发起建模；本次预测或优化不会自动训练。"
            ),
            "evidence": [{
                "source": "model_registry",
                "company_id": scope.company_id,
                "project_id": scope.project_id,
            }],
            "warnings": [],
        }

    def failure(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "workflow": "engine_workflow",
            "status": "ERROR",
            "error": {"code": str(code), "message": message},
            "steps": [],
            "result": {},
            "answer": message,
            "evidence": [],
            "warnings": [],
        }
        if details is not None:
            payload["details"] = details
        return payload

    @staticmethod
    def public_scope(scope: ArtifactScope) -> dict[str, Any]:
        return {
            "company_id": scope.company_id,
            "project_id": scope.project_id,
            "conversation_id": scope.conversation_id,
        }
