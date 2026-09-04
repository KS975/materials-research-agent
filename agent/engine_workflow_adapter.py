from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from agent.field_catalog import (
    bind_metric_to_catalog,
    build_material_field_catalog,
    normalize_field_name,
)
from schemas.user_context import UserContext


_PUBLIC_TOOL_BY_INTENT = {
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


class EngineWorkflowToolError(RuntimeError):
    """Carry a structured engine-tool failure without leaking an exception."""

    def __init__(self, result: dict[str, Any]):
        error = dict(result.get("error") or {})
        super().__init__(str(error.get("message") or "engine tool failed"))
        self.result = result


@dataclass(frozen=True, slots=True)
class _ArtifactScope:
    ctx: UserContext
    company_id: str
    project_id: int
    project_root: Path
    model_registry_path: Path
    session_root: Path
    conversation_id: str


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: Path
    source_hash: str
    records: list[dict[str, Any]]
    numeric_feature_fields: list[str]
    sample_count: int
    warnings: list[str]


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
    ) -> None:
        self.registry = registry
        self.enabled = bool(enabled)
        self.max_source_rows = int(max_source_rows)
        self.default_algorithms = tuple(
            str(item).strip() for item in default_algorithms if str(item).strip()
        )
        self.artifact_root = Path(artifact_root).resolve()

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
        return _ArtifactScope(
            ctx=project_ctx,
            company_id=ctx.company_id,
            project_id=project_id,
            project_root=project_root,
            model_registry_path=project_root / "models" / "model-registry.json",
            session_root=session_root,
            conversation_id=conversation_token,
        )

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
        target_metric = self._required_string(args.get("target_metric"))
        snapshot = self._authorized_snapshot(scope)
        catalog = build_material_field_catalog({
            "samples": self._catalog_source(snapshot.records),
            "total_matches": snapshot.sample_count,
            "scan_complete": True,
            "scan_truncated": False,
            "warnings": [],
        })
        binding = bind_metric_to_catalog(
            target_metric,
            catalog,
            section=str(args.get("target_section") or "auto"),
        )
        if binding.get("status") != "ok":
            candidates = ", ".join(
                str(item) for item in binding.get("candidates") or []
            )
            suffix = f"候选字段：{candidates}。" if candidates else ""
            raise ValueError(
                f"目标字段“{target_metric}”未唯一绑定到授权数据字段。{suffix}"
            )
        section = str(binding.get("section"))
        normalized = normalize_field_name(binding.get("canonical"))
        entry = self._catalog_entry(catalog, section, normalized)
        if int(entry.get("ambiguous_sample_count") or 0) > 0:
            raise ValueError(f"目标字段 {target_metric} 在授权数据中存在同名多条记录。")
        units = [str(item) for item in entry.get("units") or []]
        requested_unit = str(args.get("target_unit") or "").strip()
        if len(units) > 1:
            raise ValueError(
                f"目标字段 {target_metric} 存在多种单位：{', '.join(units)}。"
            )
        if requested_unit and units and requested_unit not in units:
            raise ValueError(
                f"目标单位 {requested_unit} 与数据记录单位 {units[0]} 不一致。"
            )

        column = self._dynamic_column(section, normalized, catalog)
        metadata = {
            "target_fields": [column],
            "identifier_fields": ["sample_id"],
            "units": {column: requested_unit or (units[0] if units else "")},
        }
        return snapshot, column, metadata

    def _authorized_snapshot(self, scope: _ArtifactScope) -> _SourceSnapshot:
        source = self.registry.execute(
            "list_samples_for_analysis",
            keyword="",
            ctx=scope.ctx,
            limit=500,
        )
        if not isinstance(source, dict) or source.get("status") != "ok":
            raise ValueError("授权项目数据快照读取失败。")
        if not source.get("scan_complete", True):
            raise ValueError("授权项目数据扫描未完整结束，已阻止建模或优化。")
        sample_count = int(source.get("count") or 0)
        if sample_count <= 0:
            raise ValueError("当前授权项目没有可用于建模或优化的样品数据。")
        if sample_count > self.max_source_rows:
            raise ValueError(
                f"授权样品数 {sample_count} 超过配置上限 {self.max_source_rows}。"
            )
        records, warnings = self._flatten_samples(source.get("samples") or [])
        if not records:
            raise ValueError("授权样品字段解析后没有可用数据行。")
        numeric_feature_fields = self._coerce_numeric_columns(records)
        serialized = json.dumps(
            records, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        data_hash = hashlib.sha256(serialized).hexdigest()
        source_dir = scope.session_root / "source_snapshots"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / f"authorized_project_{data_hash[:20]}.csv"
        if not source_path.exists():
            pd.DataFrame.from_records(records).to_csv(
                source_path, index=False, encoding="utf-8"
            )
        file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        return _SourceSnapshot(
            path=source_path,
            source_hash=file_hash,
            records=records,
            numeric_feature_fields=numeric_feature_fields,
            sample_count=sample_count,
            warnings=warnings,
        )

    @staticmethod
    def _apply_dataset_field_config(
        user_config: dict[str, Any],
        snapshot: _SourceSnapshot,
        target_column: str,
    ) -> None:
        user_config["target_fields"] = [target_column]
        user_config["identifier_fields"] = ["sample_id"]
        if user_config.get("feature_fields") is None:
            user_config["feature_fields"] = [
                item for item in snapshot.numeric_feature_fields
                if item != target_column and not item.startswith("performance.")
            ]
            return
        feature_fields = [
            str(item).strip() for item in user_config["feature_fields"]
            if str(item).strip()
        ]
        if not feature_fields:
            raise ValueError("feature_fields 不能为空。")
        if target_column in feature_fields:
            raise ValueError("目标字段不能同时作为特征字段。")
        user_config["feature_fields"] = feature_fields

    @staticmethod
    def _coerce_numeric_columns(
        records: list[dict[str, Any]],
    ) -> list[str]:
        columns = sorted({
            key for record in records for key in record
        })
        blocked = {
            "sample_id", "sample_name", "project_id", "sample_type", "create_time"
        }
        numeric_columns: list[str] = []
        for column in columns:
            if column in blocked:
                continue
            values = [record.get(column) for record in records]
            non_missing = [value for value in values if value is not None]
            if not non_missing:
                continue
            parsed: list[Decimal | None] = []
            for value in values:
                if value is None:
                    parsed.append(None)
                    continue
                try:
                    candidate = Decimal(str(value).strip())
                    parsed.append(candidate if candidate.is_finite() else None)
                except (InvalidOperation, ValueError):
                    parsed.append(None)
            parsed_non_missing = [item for item in parsed if item is not None]
            if len(parsed_non_missing) != len(non_missing):
                continue
            numeric_columns.append(column)
            for record, value in zip(records, parsed):
                if value is None:
                    record[column] = None
                elif value == value.to_integral_value():
                    record[column] = int(value)
                else:
                    record[column] = float(value)
        return numeric_columns

    def _flatten_samples(
        self,
        samples: Iterable[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        records: list[dict[str, Any]] = []
        ambiguous: set[str] = set()
        for sample_item in samples:
            sample = dict(sample_item.get("sample") or {})
            row: dict[str, Any] = {
                "sample_id": sample.get("id"),
                "sample_name": sample.get("name"),
                "project_id": sample.get("project_id"),
                "sample_type": sample.get("sample_type"),
                "create_time": sample.get("create_time"),
            }
            for section in ("formula", "process", "performance"):
                grouped: dict[str, list[dict[str, Any]]] = {}
                for field in sample_item.get(section) or []:
                    name = str(field.get("name") or field.get("raw_key") or "").strip()
                    if name:
                        grouped.setdefault(normalize_field_name(name), []).append(field)
                for normalized, fields in grouped.items():
                    display_name = str(
                        fields[0].get("name") or fields[0].get("raw_key") or normalized
                    )
                    column = f"{section}.{display_name}"
                    if len(fields) == 1:
                        row[column] = self._scalar(fields[0].get("value"))
                    else:
                        ambiguous.add(f"{section}.{display_name}")
            for name, value in dict(sample_item.get("conditions") or {}).items():
                row[f"condition.{name}"] = self._scalar(value)
            records.append(row)
        warnings = (
            [f"存在同名多条动态字段，已置为缺失：{', '.join(sorted(ambiguous))}"]
            if ambiguous
            else []
        )
        return records, warnings

    def _catalog_source(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Rebuild the resolver-shaped payload only for catalog construction.
        return [
            {
                "sample": {
                    "id": record.get("sample_id"),
                    "name": record.get("sample_name"),
                    "project_id": record.get("project_id"),
                    "sample_type": record.get("sample_type"),
                    "create_time": record.get("create_time"),
                },
                "formula": [{"name": key.split(".", 1)[1], "value": value}
                            for key, value in record.items()
                            if key.startswith("formula.")],
                "process": [{"name": key.split(".", 1)[1], "value": value}
                            for key, value in record.items()
                            if key.startswith("process.")],
                "performance": [{"name": key.split(".", 1)[1], "value": value}
                                for key, value in record.items()
                                if key.startswith("performance.")],
                "conditions": {
                    key.split(".", 1)[1]: value for key, value in record.items()
                    if key.startswith("condition.")
                },
            }
            for record in records
        ]

    def _model_records(self, scope: _ArtifactScope) -> list[dict[str, Any]]:
        result = self._run_tool(
            "list_artifacts",
            {
                "dataset_roots": [],
                "model_registry_paths": [str(scope.model_registry_path)],
            },
        )
        return [dict(item) for item in result.get("models") or []]

    def _sample_context(
        self,
        scope: _ArtifactScope,
        identifier: Any,
    ) -> dict[str, Any]:
        result = self.registry.execute(
            "get_sample_context",
            identifier=identifier,
            ctx=scope.ctx,
        )
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise ValueError("未在当前 Company + Project 权限范围内找到待预测样品。")
        return result

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
        )
        return dict(chart.get("result") or chart)

    def _run_tool(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.registry.execute(name, payload=payload)
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
        candidates = [
            item for item in records
            if str(item.get("status") or "CANDIDATE").upper() != "DEPRECATED"
        ]
        if model_id is not None:
            candidates = [
                item for item in candidates
                if str(item.get("model_id") or "") == str(model_id)
            ]
        if version is not None:
            candidates = [
                item for item in candidates
                if str(item.get("version") or "") == str(version)
            ]
        requested = normalize_field_name(target_metric)
        requested_suffix = normalize_field_name(target_metric.rsplit(".", 1)[-1])
        matched = [
            item for item in candidates
            if normalize_field_name(item.get("target_name")) == requested
            or normalize_field_name(
                str(item.get("target_name") or "").rsplit(".", 1)[-1]
            ) == requested_suffix
        ]
        if len(matched) > 1:
            return None
        candidates = matched
        if not candidates:
            return None
        return max(
            enumerate(candidates),
            key=lambda pair: (
                str(pair[1].get("created_at") or ""),
                self._version_number(pair[1].get("version")),
                pair[0],
            ),
        )[1]

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
        aligned: dict[str, Any] = {}
        missing: list[str] = []
        for feature_name in feature_names:
            if feature_name in record:
                aligned[feature_name] = record[feature_name]
                continue
            requested = normalize_field_name(feature_name)
            requested_suffix = normalize_field_name(
                feature_name.rsplit(".", 1)[-1]
            )
            matches = [
                key for key, value in record.items()
                if value is not None
                and (
                    normalize_field_name(key) == requested
                    or normalize_field_name(key.rsplit(".", 1)[-1])
                    == requested_suffix
                )
            ]
            if len(matches) == 1:
                aligned[feature_name] = record[matches[0]]
            else:
                missing.append(feature_name)
        if missing:
            raise ValueError(f"模型输入缺少字段：{', '.join(missing)}")
        return aligned

    def _observed_targets(
        self,
        row: dict[str, Any],
        target_names: list[str],
    ) -> dict[str, float] | None:
        observed: dict[str, float] = {}
        for target_name in target_names:
            value = row.get(target_name)
            if value is None:
                return None
            try:
                parsed = Decimal(str(value).strip())
            except (InvalidOperation, ValueError):
                return None
            if not parsed.is_finite():
                return None
            observed[target_name] = float(parsed)
        return observed

    def _map_variables(
        self,
        variables: Any,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        if variables is None:
            return []
        if not isinstance(variables, list):
            raise ValueError("variables 必须是 JSON 对象数组。")
        mapped = []
        for item in variables:
            if not isinstance(item, dict):
                raise ValueError("每个变量必须是 JSON 对象。")
            copied = dict(item)
            copied["name"] = self._resolve_feature_name(
                copied.get("name"), feature_names
            )
            mapped.append(copied)
        return mapped

    def _map_constraints(
        self,
        constraints: Any,
        feature_names: list[str],
    ) -> list[dict[str, Any]]:
        if constraints is None:
            return []
        if not isinstance(constraints, list):
            raise ValueError("约束必须是 JSON 对象数组。")
        mapped = []
        for item in constraints:
            if not isinstance(item, dict):
                raise ValueError("每条约束必须是 JSON 对象。")
            copied = dict(item)
            variables = [
                self._resolve_feature_name(name, feature_names)
                for name in copied.get("variables") or []
            ]
            if variables:
                copied["variables"] = variables
            mapped.append(copied)
        return mapped

    @staticmethod
    def _resolve_feature_name(name: Any, feature_names: list[str]) -> str:
        requested = str(name or "").strip()
        if not requested:
            raise ValueError("变量名不能为空。")
        if requested in feature_names:
            return requested
        requested_normalized = normalize_field_name(requested)
        requested_suffix = normalize_field_name(requested.rsplit(".", 1)[-1])
        matches = [
            feature_name for feature_name in feature_names
            if normalize_field_name(feature_name) == requested_normalized
            or normalize_field_name(feature_name.rsplit(".", 1)[-1])
            == requested_suffix
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"变量 {requested} 未用于当前模型。")
        raise ValueError(f"变量 {requested} 匹配到多个模型字段，请明确字段区段。")

    def _success(
        self,
        scope: _ArtifactScope,
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
            "scope": self._public_scope(scope),
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

    def _blocked(
        self,
        scope: _ArtifactScope,
        intent: str,
        result: dict[str, Any],
        answer: str,
        warnings: list[Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": intent,
            "status": "BLOCKED",
            "scope": self._public_scope(scope),
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

    def _model_required(
        self,
        scope: _ArtifactScope,
        target_metric: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow": "ensure_model",
            "status": "MODEL_REQUIRED",
            "scope": self._public_scope(scope),
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

    def _failure(
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
    def _public_scope(scope: _ArtifactScope) -> dict[str, Any]:
        return {
            "company_id": scope.company_id,
            "project_id": scope.project_id,
            "conversation_id": scope.conversation_id,
        }

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

    @staticmethod
    def _version_number(value: Any) -> int:
        try:
            return int(str(value).lstrip("vV"))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _catalog_entry(
        catalog: dict[str, Any],
        section: str,
        normalized: str,
    ) -> dict[str, Any]:
        for item in (catalog.get("sections") or {}).get(section) or []:
            if normalize_field_name(item.get("name")) == normalized:
                return dict(item)
        raise ValueError("目标字段目录绑定失败。")

    @staticmethod
    def _dynamic_column(
        section: str,
        normalized: str,
        catalog: dict[str, Any],
    ) -> str:
        sections = catalog.get("sections") or {}
        entry = next((
            dict(item) for item in sections.get(section) or []
            if normalize_field_name(item.get("name")) == normalized
        ), None)
        if entry is None:
            raise ValueError("目标字段目录绑定失败。")
        name = str(entry.get("name") or normalized)
        return f"{section}.{name}"
