from __future__ import annotations

import re
from typing import Any

from data.dynamic_fields import DynamicFieldResolver
from data.json_utils import decode_json_mapping, decode_json_value
from data.mysql.repositories import (
    ArchiveRepository,
    ExperimentRepository,
    ProjectRepository,
    SampleRepository,
)
from schemas.user_context import UserContext


class MaterialsTools:

    @staticmethod
    def _compare_test_conditions(
            left: dict,
            right: dict,
    ) -> dict:
        left = left or {}
        right = right or {}

        if not left and not right:
            return {
                "left": {},
                "right": {},
                "status": "missing_both",
                "same": None,
                "comparable": False,
            }

        if not left:
            return {
                "left": {},
                "right": right,
                "status": "missing_left",
                "same": None,
                "comparable": False,
            }

        if not right:
            return {
                "left": left,
                "right": {},
                "status": "missing_right",
                "same": None,
                "comparable": False,
            }

        same = left == right

        return {
            "left": left,
            "right": right,
            "status": "same" if same else "different",
            "same": same,
            "comparable": same,
        }
    def __init__(
        self,
        samples: SampleRepository,
        projects: ProjectRepository,
        archives: ArchiveRepository,
        experiments: ExperimentRepository,
        resolver: DynamicFieldResolver,
    ):
        self.samples = samples
        self.projects = projects
        self.archives = archives
        self.experiments = experiments
        self.resolver = resolver

    @staticmethod
    def _normalize_sample_identifier(identifier: str | int) -> str | int:
        """Defensive normalization at the DB-tool boundary.

        The router should already extract entities, but a fallback route or LLM
        must never be able to turn “查看样品3811具体信息” into an exact-name DB
        lookup for that whole phrase. Only unwrap a single standalone numeric ID;
        names such as ABS-051 / B251218-6 are intentionally left untouched.
        """
        if isinstance(identifier, int):
            return identifier
        value = str(identifier or "").strip()
        if value.isdigit():
            return int(value)
        numeric_ids = re.findall(
            r"(?<![A-Za-z0-9_.-])(\d{3,})(?![A-Za-z0-9_.-])",
            value,
        )
        if len(numeric_ids) == 1 and any(
            marker in value
            for marker in ("样品", "样本", "编号", "查看", "查询", "查", "具体信息", "详细信息")
        ):
            return int(numeric_ids[0])
        return value

    def _locate_sample(self, identifier: str | int, ctx: UserContext) -> dict[str, Any]:
        requested_identifier = identifier
        identifier = self._normalize_sample_identifier(identifier)
        if isinstance(identifier, int) or str(identifier).strip().isdigit():
            row = self.samples.get_by_id(int(identifier), ctx)
            if not row:
                result = {"status": "not_found", "identifier": str(identifier)}
                if str(requested_identifier).strip() != str(identifier):
                    result["requested_identifier"] = str(requested_identifier)
                return result
            return {"status": "ok", "sample": row}

        matches = self.samples.find_exact_name(str(identifier).strip(), ctx)
        if not matches:
            return {"status": "not_found", "identifier": str(identifier)}
        if len(matches) > 1:
            return {
                "status": "ambiguous",
                "identifier": str(identifier),
                "candidates": [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "project_id": row.get("project_id"),
                        "create_time": row.get("create_time"),
                    }
                    for row in matches
                ],
            }
        return {"status": "ok", "sample": matches[0]}

    def get_sample_context(self, identifier: str | int, ctx: UserContext) -> dict[str, Any]:
        located = self._locate_sample(identifier, ctx)
        if located["status"] != "ok":
            return located

        sample = located["sample"]
        sample_id = int(sample["id"])
        project = self.projects.get_by_id(sample.get("project_id"), ctx)
        archive = self.archives.get_sample_archive(sample_id, ctx.company_id)

        formula = self.resolver.resolve_formula(sample.get("recipes"), ctx.company_id)
        process = self.resolver.resolve_dynamic(
            sample.get("craft_param"), ctx.company_id, "process"
        )
        performance = self.resolver.resolve_dynamic(
            sample.get("performances"), ctx.company_id, "performance"
        )
        service_performance = self.resolver.resolve_dynamic(
            sample.get("service_performances"),
            ctx.company_id,
            "service_performance",
        )

        synthesis = self.experiments.list_synthesis(sample_id)
        verify_items = self.experiments.list_verify_items(sample_id)

        evidence = [
            {"source": "eln_sample", "record_id": sample_id},
        ]
        if project:
            evidence.append({"source": "mat_project", "record_id": project["id"]})
        if archive:
            evidence.append({"source": "archive_data", "record_id": archive["id"]})
        evidence.extend(
            {"source": "eln_synthesis_exp", "record_id": row["id"]} for row in synthesis
        )
        evidence.extend(
            {
                "source": "eln_verify_item",
                "record_id": row.get("verify_item_id"),
            }
            for row in verify_items
            if row.get("verify_item_id") is not None
        )

        unresolved = [
            item["raw_key"]
            for group in (formula, process, performance, service_performance)
            for item in group
            if not item["resolved"]
        ]

        return {
            "status": "ok",
            "sample": {
                "id": sample_id,
                "name": sample.get("name"),
                "project_id": sample.get("project_id"),
                "company": sample.get("company"),
                "sample_type": sample.get("sample_type"),
                "describe": sample.get("describe"),
                "create_time": sample.get("create_time"),
                "update_time": sample.get("update_time"),
            },
            "project": project,
            "formula": formula,
            "process": process,
            "performance": performance,
            "service_performance": service_performance,
            "conditions": decode_json_mapping(sample.get("conditions")),
            "recipe_batches": decode_json_mapping(sample.get("recipe_batches")),
            "craft_detail": decode_json_value(sample.get("craft_detail")),
            "archive": archive,
            "synthesis_records": synthesis,
            "verify_items": verify_items,
            "evidence": evidence,
            "warnings": (
                [f"存在未解析动态字段：{', '.join(unresolved)}"] if unresolved else []
            ),
        }

    def get_formula(self, identifier: str | int, ctx: UserContext) -> dict[str, Any]:
        result = self.get_sample_context(identifier, ctx)
        if result.get("status") != "ok":
            return result
        return {
            "status": "ok",
            "sample": result["sample"],
            "formula": result["formula"],
            "recipe_batches": result["recipe_batches"],
            "evidence": result["evidence"],
            "warnings": result["warnings"],
        }

    def get_process(self, identifier: str | int, ctx: UserContext) -> dict[str, Any]:
        result = self.get_sample_context(identifier, ctx)
        if result.get("status") != "ok":
            return result
        return {
            "status": "ok",
            "sample": result["sample"],
            "process": result["process"],
            "craft_detail": result["craft_detail"],
            "synthesis_records": result["synthesis_records"],
            "evidence": result["evidence"],
            "warnings": result["warnings"],
        }

    def get_performance(self, identifier: str | int, ctx: UserContext) -> dict[str, Any]:
        result = self.get_sample_context(identifier, ctx)
        if result.get("status") != "ok":
            return result
        return {
            "status": "ok",
            "sample": result["sample"],
            "performance": result["performance"],
            "service_performance": result["service_performance"],
            "conditions": result["conditions"],
            "verify_items": result["verify_items"],
            "evidence": result["evidence"],
            "warnings": result["warnings"],
        }

    def find_samples(self, keyword: str, ctx: UserContext, limit: int = 20) -> dict[str, Any]:
        rows = self.samples.find(keyword, ctx, limit=limit)
        return {
            "status": "ok",
            "keyword": keyword,
            "count": len(rows),
            "samples": rows,
            "evidence": [{"source": "eln_sample", "record_id": row["id"]} for row in rows],
            "warnings": [],
        }

    def list_samples_for_analysis(
        self,
        keyword: str,
        ctx: UserContext,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Read the complete authorized set in bounded keyset pages."""
        try:
            page_size = max(1, min(int(limit), 500))
        except (TypeError, ValueError):
            page_size = 500
        total_matches = self.samples.count_for_analysis(keyword, ctx)
        rows = []
        seen_ids: set[int] = set()
        before_id: int | None = None
        page_count = 0
        pagination_aborted = False
        while True:
            page = self.samples.list_for_analysis(
                keyword,
                ctx,
                limit=page_size,
                before_id=before_id,
            )
            if not page:
                break
            page_count += 1
            page_ids = [int(row["id"]) for row in page]
            fresh_rows = [row for row in page if int(row["id"]) not in seen_ids]
            rows.extend(fresh_rows)
            seen_ids.update(int(row["id"]) for row in fresh_rows)

            next_before_id = min(page_ids)
            if before_id is not None and next_before_id >= before_id:
                pagination_aborted = True
                break
            before_id = next_before_id
            if len(page) < page_size:
                break
        raw_mappings = []
        formula_ids: set[int] = set()
        dynamic_ids: set[int] = set()
        for row in rows:
            formula_raw = decode_json_mapping(row.get("recipes"))
            process_raw = decode_json_mapping(row.get("craft_param"))
            performance_raw = decode_json_mapping(row.get("performances"))
            raw_mappings.append((formula_raw, process_raw, performance_raw))
            for key in formula_raw:
                match = re.fullmatch(r"R3-(\d+)", str(key))
                if match:
                    formula_ids.add(int(match.group(1)))
            for mapping, pattern in (
                (process_raw, r"S(\d+)"),
                (performance_raw, r"P(\d+)"),
            ):
                for key in mapping:
                    match = re.fullmatch(pattern, str(key))
                    if match:
                        dynamic_ids.add(int(match.group(1)))

        formula_definitions = self.resolver.materials.get_sample_materials(
            formula_ids, ctx.company_id
        )
        dynamic_definitions = self.resolver.columns.get_by_ids(
            dynamic_ids, ctx.company_id
        )
        samples = []
        unresolved = []
        for row, mappings in zip(rows, raw_mappings):
            sample_id = int(row["id"])
            formula_raw, process_raw, performance_raw = mappings
            formula = self._resolve_prefetched_fields(
                formula_raw,
                pattern=r"R3-(\d+)",
                definitions=formula_definitions,
                source="sample_materials",
            )
            process = self._resolve_prefetched_fields(
                process_raw,
                pattern=r"S(\d+)",
                definitions=dynamic_definitions,
                source="data_column",
            )
            performance = self._resolve_prefetched_fields(
                performance_raw,
                pattern=r"P(\d+)",
                definitions=dynamic_definitions,
                source="data_column",
            )
            unresolved_fields = [
                item.get("raw_key")
                for group in (formula, process, performance)
                for item in group
                if not item.get("resolved")
            ]
            if unresolved_fields:
                unresolved.append({"sample_id": sample_id, "fields": unresolved_fields})
            samples.append({
                "sample": {
                    "id": sample_id,
                    "name": row.get("name"),
                    "project_id": row.get("project_id"),
                    "sample_type": row.get("sample_type"),
                    "create_time": row.get("create_time"),
                },
                "formula": formula,
                "process": process,
                "performance": performance,
                "conditions": decode_json_mapping(row.get("conditions")),
            })
        warnings = []
        if unresolved:
            warnings.append("部分样品存在未解析动态字段。")
        if pagination_aborted:
            warnings.append("样品分页游标未继续前进，已停止读取以避免重复扫描。")
        elif total_matches != len(samples):
            warnings.append(
                f"分页开始时匹配 {total_matches} 条，实际读取 {len(samples)} 条；"
                "扫描期间数据库记录可能发生变化。"
            )
        return {
            "status": "ok",
            "keyword": str(keyword or "").strip(),
            "count": len(samples),
            "total_matches": total_matches,
            "scan_limit": None,
            "scan_page_size": page_size,
            "scan_page_count": page_count,
            "scan_complete": not pagination_aborted,
            "scan_truncated": pagination_aborted,
            "samples": samples,
            "similar_names": (
                self.samples.suggest_similar_names(keyword, ctx)
                if keyword and not samples
                else []
            ),
            "unresolved_dynamic_fields": unresolved,
            "evidence": [
                {"source": "eln_sample", "record_id": item["sample"]["id"]}
                for item in samples
            ],
            "warnings": warnings,
        }

    @staticmethod
    def _resolve_prefetched_fields(
        mapping: dict[str, Any],
        *,
        pattern: str,
        definitions: dict[int, dict[str, Any]],
        source: str,
    ) -> list[dict[str, Any]]:
        resolved = []
        for raw_key, value in mapping.items():
            match = re.fullmatch(pattern, str(raw_key))
            field_id = int(match.group(1)) if match else None
            definition = definitions.get(field_id) if field_id is not None else None
            resolved.append({
                "raw_key": str(raw_key),
                "field_id": field_id,
                "name": definition.get("name") if definition else None,
                "unit": definition.get("unit") if definition else None,
                "value": value,
                "resolved": bool(definition),
                "source": source if definition else None,
            })
        return resolved

    @staticmethod
    def _keyed(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(item.get("name") or item.get("raw_key")): item for item in items}

    @classmethod
    def _diff_fields(
        cls,
        left: list[dict[str, Any]],
        right: list[dict[str, Any]],
    ) -> dict[str, Any]:
        a = cls._keyed(left)
        b = cls._keyed(right)
        keys = sorted(set(a) | set(b))
        changed, same = [], []
        for key in keys:
            av = a.get(key)
            bv = b.get(key)
            left_unit = av.get("unit") if av else None
            right_unit = bv.get("unit") if bv else None
            unit_match = None
            if av is not None and bv is not None:
                if left_unit and right_unit:
                    unit_match = str(left_unit) == str(right_unit)
                elif left_unit is None and right_unit is None:
                    unit_match = True
            payload = {
                "field": key,
                "left": av.get("value") if av else None,
                "right": bv.get("value") if bv else None,
                "unit": left_unit or right_unit,
                "left_unit": left_unit,
                "right_unit": right_unit,
                "unit_match": unit_match,
                "left_present": av is not None,
                "right_present": bv is not None,
            }
            (same if av and bv and av.get("value") == bv.get("value") else changed).append(payload)
        return {"changed": changed, "same": same}

    def compare_samples(
        self,
        left_identifier: str | int,
        right_identifier: str | int,
        ctx: UserContext,
    ) -> dict[str, Any]:
        left = self.get_sample_context(left_identifier, ctx)
        right = self.get_sample_context(right_identifier, ctx)
        if left.get("status") != "ok":
            return {"status": "left_error", "left": left}
        if right.get("status") != "ok":
            return {"status": "right_error", "right": right}

        return {
            "status": "ok",
            "left_sample": left["sample"],
            "right_sample": right["sample"],
            "formula_diff": self._diff_fields(left["formula"], right["formula"]),
            "process_diff": self._diff_fields(left["process"], right["process"]),
            "performance_diff": self._diff_fields(left["performance"], right["performance"]),
            "service_performance_diff": self._diff_fields(
                left["service_performance"], right["service_performance"]
            ),
            "test_conditions": self._compare_test_conditions(
                left["conditions"],
                right["conditions"],
            ),
            "evidence": [*left["evidence"], *right["evidence"]],
            "warnings": [*left["warnings"], *right["warnings"]],
        }
