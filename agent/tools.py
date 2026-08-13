from __future__ import annotations

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

    def _locate_sample(self, identifier: str | int, ctx: UserContext) -> dict[str, Any]:
        if isinstance(identifier, int) or str(identifier).strip().isdigit():
            row = self.samples.get_by_id(int(identifier), ctx)
            if not row:
                return {"status": "not_found", "identifier": str(identifier)}
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
            payload = {
                "field": key,
                "left": av.get("value") if av else None,
                "right": bv.get("value") if bv else None,
                "unit": (av or bv or {}).get("unit"),
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
