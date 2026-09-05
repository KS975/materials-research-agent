from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from agent.engine_model_selector import EngineModelSelector
from agent.engine_result_builder import EngineResultBuilder
from agent.engine_scope_resolver import EngineScopeResolver
from agent.engine_snapshot_service import EngineSnapshotService
from agent.engine_workflow_types import (
    ArtifactScope as _ArtifactScope,
    EngineWorkflowToolError,
    SourceSnapshot as _SourceSnapshot,
)
from schemas.user_context import UserContext


_PUBLIC_TOOL_BY_INTENT = {
    "ensure_model": "list_artifacts",
    "engine_prepare_dataset": "preprocess_dataset",
    "automl_training": "train_model",
    "predict_performance": "predict_model",
    "optimize_formula": "optimize_formula",
    "recommend_next_experiments": "recommend_next_experiments",
}

_PUBLIC_ARG_KEYS = {
    "project_id",
    "target_metric",
    "target_section",
    "target_unit",
    "preprocessing_config",
    "algorithms",
    "training_config",
    "model_id",
    "model_version",
    "inputs",
    "sample_identifier",
    "objectives",
    "variables",
    "hard_constraints",
    "soft_constraints",
    "top_n",
    "random_seed",
    "max_evaluations",
    "time_limit",
    "preference",
    "model_quality_gate",
    "algorithm_override",
    "acquisition",
}


class EngineWorkflowAdapter:
    """Deterministic host adapter for the framework-neutral engine tools.

    The intent model may describe business targets and constraints, but never
    file locations. This adapter narrows UserContext to one project, owns all
    artifact paths, and invokes the public tools in a fixed order.
    """

    def __init__(
        self,
        registry: Any,
        *,
        artifact_root: str | Path,
        enabled: bool = True,
        max_source_rows: int = 5000,
        default_algorithms: Iterable[str] = (),
        allowed_model_statuses: Iterable[str] = ("CANDIDATE",),
    ) -> None:
        self.registry = registry
        self.enabled = bool(enabled)
        self.max_source_rows = int(max_source_rows)
        self.default_algorithms = tuple(
            str(item).strip() for item in default_algorithms if str(item).strip()
        )
        self.allowed_model_statuses = {
            str(item).strip().upper()
            for item in allowed_model_statuses
            if str(item).strip()
        }
        if not self.allowed_model_statuses:
            raise ValueError("allowed_model_statuses must not be empty")
        self.model_selector = EngineModelSelector(self.allowed_model_statuses)
        self.snapshot_service = EngineSnapshotService(
            registry,
            max_source_rows=self.max_source_rows,
        )
        self.scope_resolver = EngineScopeResolver(artifact_root)
        self.result_builder = EngineResultBuilder()

    def execute(
        self,
        intent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: UserContext,
    ) -> dict[str, Any]:
        if intent not in _PUBLIC_TOOL_BY_INTENT:
            return self._failure("UNSUPPORTED_INTENT", f"未知引擎工作流意图：{intent}")
        if not self.enabled:
            return self._failure(
                "ENGINE_WORKFLOW_DISABLED", "当前部署未启用独立引擎工作流。"
            )
        if tool_name != _PUBLIC_TOOL_BY_INTENT[intent]:
            return self._failure(
                "INVALID_TOOL_BINDING",
                f"意图 {intent} 只允许调用 {_PUBLIC_TOOL_BY_INTENT[intent]}。",
            )

        raw_args = dict(tool_args or {})
        workflow_id = str(raw_args.pop("_workflow_id", "") or "")
        conversation_id = str(raw_args.pop("_conversation_id", "") or "")
        args = {
            str(key): value
            for key, value in raw_args.items()
            if key in _PUBLIC_ARG_KEYS
        }
        try:
            scope = self._resolve_scope(ctx, args, workflow_id, conversation_id)
            if intent == "ensure_model":
                return self._execute_ensure_model(scope, args)
            if intent == "engine_prepare_dataset":
                return self._execute_prepare(scope, args)
            if intent == "automl_training":
                return self._execute_training(scope, args)
            if intent == "predict_performance":
                return self._execute_prediction(scope, args)
            return self._execute_optimization(scope, args, intent)
        except PermissionError:
            raise
        except EngineWorkflowToolError as exc:
            error = dict(exc.result.get("error") or {})
            return self._failure(
                str(error.get("code") or "ENGINE_TOOL_ERROR"),
                str(error.get("message") or "引擎工具执行失败。"),
                details={"tool_result": exc.result},
            )
        except Exception as exc:
            return self._failure(
                "ENGINE_WORKFLOW_ERROR",
                f"引擎工作流执行失败：{type(exc).__name__}: {exc}",
            )

    def _resolve_scope(
        self,
        ctx: UserContext,
        args: dict[str, Any],
        workflow_id: str,
        conversation_id: str,
    ) -> _ArtifactScope:
        return self.scope_resolver.resolve(ctx, args, workflow_id, conversation_id)

    def _execute_prepare(
        self,
        scope: _ArtifactScope,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot, target_column, metadata = self._build_dataset_inputs(scope, args)
        user_config = self._object_arg(args.get("preprocessing_config"))
        self._apply_dataset_field_config(
            user_config, snapshot, target_column
        )
        result = self._run_tool(
            "preprocess_dataset",
            {
                "input_uri": str(snapshot.path),
                "config": user_config,
                "metadata": metadata,
                "source_hash": snapshot.source_hash,
                "output_dir": str(scope.session_root / "datasets"),
                "result_mode": "summary",
            },
            scope,
        )
        gate = dict(result.get("final_gate") or {})
        artifact = dict(result.get("dataset_artifact") or {})
        answer = (
            f"已完成 Project {scope.project_id} 的建模数据准备。"
            f"数据门禁 {gate.get('decision', 'UNKNOWN')}，"
            + (
                f"数据集 {artifact.get('dataset_id')} / {artifact.get('version')} 已保存。"
                if artifact
                else "未生成正式 Dataset Artifact。"
            )
        )
        return self._success(
            scope,
            "engine_prepare_dataset",
            result,
            ["list_authorized_samples", "preprocess_dataset"],
            answer,
            snapshot.warnings,
        )

    def _execute_ensure_model(
        self,
        scope: _ArtifactScope,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        targets: list[str] = []
        if args.get("target_metric") is not None:
            targets.append(self._required_string(args.get("target_metric")))
        else:
            objectives = args.get("objectives")
            if not isinstance(objectives, list) or not objectives:
                return self._failure(
                    "INVALID_INPUT",
                    "确认模型需要 target_metric 或 objectives。",
                )
            for objective in objectives:
                if not isinstance(objective, dict):
                    return self._failure(
                        "INVALID_INPUT",
                        "objectives 必须是 JSON 对象数组。",
                    )
                targets.append(
                    self._required_string(
                        objective.get("target_name") or objective.get("target_metric")
                    )
                )

        records = self._model_records(scope)
        selected_models: list[dict[str, Any]] = []
        for target in targets:
            selected = self._select_model(
                records,
                target,
                model_id=args.get("model_id"),
                version=args.get("model_version"),
            )
            if selected is None:
                return self._model_required(scope, target, records)
            selected_models.append({
                "requested_target": target,
                "model_id": str(selected.get("model_id") or ""),
                "version": str(selected.get("version") or ""),
                "target_name": str(selected.get("target_name") or ""),
                "dataset_artifact_id": selected.get("dataset_artifact_id"),
                "status": str(selected.get("status") or "CANDIDATE"),
            })

        return self._success(
            scope,
            "ensure_model",
            {
                "selected_model_count": len(selected_models),
                "selected_models": selected_models,
            },
            ["list_artifacts", "select_model"],
            f"Project {scope.project_id} 已确认 {len(selected_models)} 个可用模型。",
        )

    def _execute_training(
        self,
        scope: _ArtifactScope,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot, target_column, metadata = self._build_dataset_inputs(scope, args)
        user_config = self._object_arg(args.get("preprocessing_config"))
        self._apply_dataset_field_config(
            user_config, snapshot, target_column
        )
        preprocess = self._run_tool(
            "preprocess_dataset",
            {
                "input_uri": str(snapshot.path),
                "config": user_config,
                "metadata": metadata,
                "source_hash": snapshot.source_hash,
                "output_dir": str(scope.session_root / "datasets"),
                "result_mode": "summary",
            },
            scope,
        )
        gate = dict(preprocess.get("final_gate") or {})
        artifact = dict(preprocess.get("dataset_artifact") or {})
        if gate.get("decision") == "FAIL" or not artifact:
            return self._blocked(
                scope,
                "automl_training",
                preprocess,
                "建模数据门禁未通过，已按失败关闭策略阻止训练。",
                snapshot.warnings,
            )

        training_config = self._object_arg(args.get("training_config"))
        training_config["target_names"] = [target_column]
        algorithms = self._string_list(args.get("algorithms"))
        if algorithms:
            training_config["algorithms"] = algorithms
        elif self.default_algorithms:
            training_config["algorithms"] = list(self.default_algorithms)

        result = self._run_tool(
            "train_model",
            {
                "dataset_artifact_uri": str(artifact.get("artifact_dir")),
                "config": training_config,
                "output_dir": str(scope.project_root / "models"),
                "model_registry_path": str(scope.model_registry_path),
                "result_mode": "summary",
            },
            scope,
        )
        training_run = dict(result.get("training_run") or {})
        models = list(training_run.get("model_artifacts") or [])
        model_ids = [str(item.get("model_id") or "") for item in models]
        answer = (
            f"Project {scope.project_id} 模型训练完成，"
            f"已注册 {len(models)} 个候选模型"
            + (f"（{', '.join(filter(None, model_ids))}）。" if model_ids else "。")
        )
        merged_result = {
            "preprocessing": preprocess,
            "training": result,
        }
        return self._success(
            scope,
            "automl_training",
            merged_result,
            ["list_authorized_samples", "preprocess_dataset", "train_model"],
            answer,
            snapshot.warnings,
        )

    def _execute_prediction(
        self,
        scope: _ArtifactScope,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        target_metric = self._required_string(args.get("target_metric"))
        records = self._model_records(scope)
        selected = self._select_model(
            records,
            target_metric,
            model_id=args.get("model_id"),
            version=args.get("model_version"),
        )
        if selected is None:
            return self._model_required(scope, target_metric, records)

        feature_names = [
            str(item) for item in selected.get("feature_names") or []
        ]
        if args.get("sample_identifier") is not None:
            sample = self._sample_context(scope, args.get("sample_identifier"))
            inputs = [self._record_for_model(sample, feature_names)]
        else:
            raw_inputs = args.get("inputs")
            if not isinstance(raw_inputs, list) or not raw_inputs:
                return self._failure(
                    "INVALID_INPUT", "预测需要 inputs 或 sample_identifier。"
                )
            inputs = [
                self._align_record(dict(item), feature_names)
                if isinstance(item, dict)
                else item
                for item in raw_inputs
            ]

        result = self._run_tool(
            "predict_model",
            {
                "model_registry_path": str(scope.model_registry_path),
                "model_selector": {
                    "model_id": str(selected.get("model_id") or ""),
                    "target_name": str(selected.get("target_name") or ""),
                    "version": str(selected.get("version") or ""),
                },
                "inputs": inputs,
                "result_mode": "summary",
            },
            scope,
        )
        model = dict(result.get("model") or {})
        preview = list(result.get("prediction_preview") or [])
        first_prediction = dict(preview[0]) if preview else {}
        answer = (
            f"已使用模型 {model.get('model_id')} / {model.get('version')} "
            f"完成 {result.get('prediction_count', 0)} 条预测。"
        )
        if first_prediction:
            answer += (
                f"首条 {model.get('target_name')} 预测值为 "
                f"{first_prediction.get('predicted_value')}，适用域 "
                f"{first_prediction.get('applicability_domain')}。"
            )
        return self._success(
            scope,
            "predict_performance",
            result,
            ["list_artifacts", "select_model", "validate_input", "predict_model"],
            answer,
        )

    def _execute_optimization(
        self,
        scope: _ArtifactScope,
        args: dict[str, Any],
        intent: str,
    ) -> dict[str, Any]:
        objectives = args.get("objectives")
        if not isinstance(objectives, list) or not objectives:
            return self._failure("INVALID_INPUT", "优化请求必须包含 objectives。")
        objectives = [dict(item) for item in objectives if isinstance(item, dict)]
        if not objectives:
            return self._failure("INVALID_INPUT", "objectives 必须是 JSON 对象数组。")

        records = self._model_records(scope)
        selected_models: dict[str, dict[str, Any]] = {}
        missing_targets: list[str] = []
        rewritten_objectives: list[dict[str, Any]] = []
        for objective in objectives:
            requested_target = self._required_string(
                objective.get("target_name") or objective.get("target_metric")
            )
            selected = self._select_model(
                records,
                requested_target,
                model_id=args.get("model_id"),
                version=args.get("model_version"),
            )
            if selected is None:
                missing_targets.append(requested_target)
                continue
            selected_models[requested_target] = selected
            rewritten = dict(objective)
            rewritten["target_name"] = str(selected.get("target_name") or "")
            rewritten_objectives.append(rewritten)
        if missing_targets:
            return self._model_required(scope, missing_targets[0], records)

        dataset_ids = {
            str(item.get("dataset_artifact_id"))
            for item in selected_models.values()
            if item.get("dataset_artifact_id") is not None
        }
        if len(dataset_ids) > 1:
            return self._failure(
                "MODEL_DATASET_MISMATCH", "多目标优化选择的模型必须来自同一 Dataset。"
            )

        feature_names = sorted({
            str(item)
            for model in selected_models.values()
            for item in model.get("feature_names") or []
        })
        snapshot = self._authorized_snapshot(scope)
        history_rows = []
        for record in snapshot.records:
            try:
                history_rows.append(self._align_record(record, feature_names))
            except ValueError:
                continue

        target_names = [str(item.get("target_name")) for item in rewritten_objectives]
        historical_experiments = []
        for row in history_rows:
            observed = self._observed_targets(row, target_names)
            if observed is None:
                continue
            historical_experiments.append({
                "experiment_id": f"eln_sample_{row.get('sample_id', len(historical_experiments) + 1)}",
                "values": row,
                "observed_values": observed,
            })
        if intent == "recommend_next_experiments" and not historical_experiments:
            return self._failure(
                "INSUFFICIENT_OBSERVED_HISTORY",
                "授权项目中没有覆盖全部优化目标的完整实测历史，无法执行下一批实验推荐。",
            )

        target_mappings: dict[str, dict[str, str]] = {}
        for objective, rewritten in zip(objectives, rewritten_objectives):
            original = self._required_string(
                objective.get("target_name") or objective.get("target_metric")
            )
            selected = selected_models[original]
            selected_target = str(rewritten.get("target_name") or "")
            target_mappings[selected_target] = {
                "model_id": str(selected.get("model_id") or ""),
                "version": str(selected.get("version") or ""),
            }
        request: dict[str, Any] = {
            "mode": (
                "recommend_next_experiments"
                if intent == "recommend_next_experiments"
                else "recommend_recipe"
            ),
            "objectives": rewritten_objectives,
            "model_registry_path": str(scope.model_registry_path),
            "model_selection": {
                "strategy": (
                    "explicit_model_id" if args.get("model_id") else "latest_valid"
                ),
                "target_mappings": target_mappings,
            },
            "historical_candidates": history_rows,
            "historical_experiments": historical_experiments,
        }
        for key in (
            "variables", "hard_constraints", "soft_constraints", "top_n",
            "random_seed", "max_evaluations", "time_limit", "preference",
            "model_quality_gate", "algorithm_override", "acquisition",
        ):
            if args.get(key) is not None:
                request[key] = args.get(key)
        request["variables"] = self._map_variables(
            request.get("variables"), feature_names
        )
        request["hard_constraints"] = self._map_constraints(
            request.get("hard_constraints"), feature_names
        )
        request["soft_constraints"] = self._map_constraints(
            request.get("soft_constraints"), feature_names
        )

        result = self._run_tool(
            _PUBLIC_TOOL_BY_INTENT[intent],
            {
                "request": request,
                "output_dir": str(scope.session_root / "optimizations"),
                "result_mode": "summary",
            },
            scope,
        )
        visualization = self._visualization(scope, result)
        if visualization is not None:
            result = {**result, "visualization_datasets": visualization}
        selected_count = len(result.get("selected_candidates") or [])
        answer = (
            f"Project {scope.project_id} 优化完成，返回 {selected_count} 个候选方案；"
            "完整约束、适用域和排序依据见结构化结果。"
        )
        warnings = list(result.get("warnings") or []) + list(snapshot.warnings)
        return self._success(
            scope,
            intent,
            result,
            ["list_artifacts", "ensure_model", "load_history", _PUBLIC_TOOL_BY_INTENT[intent]],
            answer,
            warnings,
        )

    def _build_dataset_inputs(
        self,
        scope: _ArtifactScope,
        args: dict[str, Any],
    ) -> tuple[_SourceSnapshot, str, dict[str, Any]]:
        return self.snapshot_service.build_dataset_inputs(scope, args)

    def _authorized_snapshot(self, scope: _ArtifactScope) -> _SourceSnapshot:
        return self.snapshot_service.snapshot(scope)

    def _apply_dataset_field_config(
        self,
        user_config: dict[str, Any],
        snapshot: _SourceSnapshot,
        target_column: str,
    ) -> None:
        self.snapshot_service.apply_dataset_field_config(
            user_config, snapshot, target_column
        )

    def _flatten_samples(
        self,
        samples: Iterable[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        return self.snapshot_service.flatten_samples(samples)

    def _model_records(self, scope: _ArtifactScope) -> list[dict[str, Any]]:
        result = self._run_tool(
            "list_artifacts",
            {
                "dataset_roots": [],
                "model_registry_paths": [str(scope.model_registry_path)],
            },
            scope,
        )
        return [dict(item) for item in result.get("models") or []]

    def _sample_context(
        self,
        scope: _ArtifactScope,
        identifier: Any,
    ) -> dict[str, Any]:
        return self.snapshot_service.sample_context(scope, identifier)

    def _visualization(
        self,
        scope: _ArtifactScope,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        artifacts = dict(result.get("artifact_ids") or {})
        source = str(artifacts.get("optimization_result") or "")
        if not source:
            return None
        chart = self._run_tool(
            "get_chart_data",
            {"input_uri": source, "source_kind": "optimization"},
            scope,
        )
        return dict(chart.get("result") or chart)

    def _run_tool(
        self,
        name: str,
        payload: dict[str, Any],
        scope: _ArtifactScope | None = None,
    ) -> dict[str, Any]:
        result = self.registry.execute(
            name,
            payload=payload,
            ctx=scope.ctx if scope is not None else None,
        )
        if not isinstance(result, dict) or result.get("status") != "OK":
            raise EngineWorkflowToolError(
                result if isinstance(result, dict) else {"error": {
                    "code": "TOOL_EXECUTION_ERROR",
                    "message": "engine tool returned a non-object response",
                }}
            )
        return dict(result.get("result") or {})

    def _select_model(
        self,
        records: list[dict[str, Any]],
        target_metric: str,
        *,
        model_id: Any = None,
        version: Any = None,
    ) -> dict[str, Any] | None:
        return self.model_selector.select(
            records,
            target_metric,
            model_id=model_id,
            version=version,
        )

    def _record_for_model(
        self,
        sample: dict[str, Any],
        feature_names: list[str],
    ) -> dict[str, Any]:
        sample_row, _ = self._flatten_samples([sample])
        if not sample_row:
            raise ValueError("样品数据无法转换为模型输入。")
        return self._align_record(sample_row[0], feature_names)

    def _align_record(
        self,
        record: dict[str, Any],
        feature_names: list[str],
    ) -> dict[str, Any]:
        return self.model_selector.align_record(record, feature_names)

    def _observed_targets(
        self,
        row: dict[str, Any],
        target_names: list[str],
    ) -> dict[str, float] | None:
        return self.model_selector.observed_targets(row, target_names)

    def _map_variables(
        self,
        variables: Any,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        return self.model_selector.map_variables(variables, feature_names)

    def _map_constraints(
        self,
        constraints: Any,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        return self.model_selector.map_constraints(constraints, feature_names)

    @staticmethod
    def _resolve_feature_name(name: Any, feature_names: list[str]) -> str:
        return EngineModelSelector.resolve_feature_name(name, feature_names)

    def _success(
        self,
        scope: _ArtifactScope,
        intent: str,
        result: dict[str, Any],
        steps: list[str],
        answer: str,
        warnings: list[Any] | None = None,
    ) -> dict[str, Any]:
        return self.result_builder.success(
            scope, intent, result, steps, answer, warnings
        )

    def _blocked(
        self,
        scope: _ArtifactScope,
        intent: str,
        result: dict[str, Any],
        answer: str,
        warnings: list[Any] | None = None,
    ) -> dict[str, Any]:
        return self.result_builder.blocked(
            scope, intent, result, answer, warnings
        )

    def _model_required(
        self,
        scope: _ArtifactScope,
        target_metric: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.result_builder.model_required(scope, target_metric, records)

    def _failure(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.result_builder.failure(
            code, message, details=details
        )

    @staticmethod
    def _public_scope(scope: _ArtifactScope) -> dict[str, Any]:
        return EngineResultBuilder.public_scope(scope)

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return value

    @staticmethod
    def _object_arg(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("配置参数必须是 JSON 对象。")
        return dict(value)

    @staticmethod
    def _required_string(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("缺少必填业务字段。")
        return text

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if not isinstance(value, list):
            raise ValueError("算法列表必须是字符串数组。")
        return [str(item).strip() for item in value if str(item).strip()]
