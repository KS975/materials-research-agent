from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

from agent.engine_tool_registration import register_engine_tools
from agent.material_tool_registration import (
    MATERIAL_TOOL_SPECS,
    register_material_tools,
)
from agent.tool_registry import ToolRegistry, ToolValidationError
from runtime.tool_audit import JsonlToolAuditStore
from schemas.user_context import UserContext


class FakeMaterialsTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def compare_samples(
        self,
        left_identifier: str,
        right_identifier: str,
        ctx: UserContext,
    ) -> dict[str, Any]:
        self.calls.append(
            ("compare_samples", {"left": left_identifier, "right": right_identifier})
        )
        return {"status": "ok"}

    def __getattr__(self, name: str):
        def handler(**kwargs: Any) -> dict[str, Any]:
            self.calls.append((name, dict(kwargs)))
            return {"status": "ok"}

        return handler


class ToolRegistryGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = UserContext(
            user_id="user-1",
            company_id="company_a",
            project_ids=(1,),
            permission_source="test",
        )

    def test_material_and_engine_input_schemas_are_valid(self) -> None:
        from jsonschema import Draft202012Validator

        for name, spec in MATERIAL_TOOL_SPECS.items():
            with self.subTest(tool=name):
                Draft202012Validator.check_schema(spec["input_schema"])

    def test_material_tool_rejects_unexpected_argument_before_handler(self) -> None:
        tools = FakeMaterialsTools()
        registry = ToolRegistry()
        register_material_tools(registry, tools)

        with self.assertRaises(ToolValidationError):
            registry.execute(
                "compare_samples",
                ctx=self.ctx,
                left_identifier="A",
                right_identifier="B",
                target_metric="impact",
            )
        self.assertEqual(tools.calls, [])

    def test_host_scoped_engine_tool_requires_narrowed_project_context(self) -> None:
        registry = ToolRegistry()
        register_engine_tools(registry)
        broad_ctx = UserContext(
            user_id="user-1",
            company_id="company_a",
            project_ids=(1, 2),
            permission_source="test",
        )

        with self.assertRaises(PermissionError):
            registry.execute(
                "list_artifacts",
                ctx=broad_ctx,
                payload={"dataset_roots": []},
            )

    def test_schema_permission_and_audit_are_visible_in_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = JsonlToolAuditStore(root / "audit")
            registry = ToolRegistry(audit_sink=audit.record)
            tools = FakeMaterialsTools()
            register_material_tools(registry, tools)

            result = registry.execute(
                "compare_samples",
                ctx=self.ctx,
                left_identifier="A",
                right_identifier="B",
            )
            self.assertEqual(result["status"], "ok")

            files = list((root / "audit").glob("*/*.jsonl"))
            self.assertEqual(len(files), 1)
            events = [
                json.loads(line)
                for line in files[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [item["event_type"] for item in events],
                ["TOOL_CALL_STARTED", "TOOL_CALL_COMPLETED"],
            )
            self.assertEqual(events[0]["tool_name"], "compare_samples")
            self.assertEqual(events[0]["schema_validation"], "VALID")
            self.assertEqual(events[0]["company_id"], "company_a")
            self.assertEqual(events[0]["argument_names"], [
                "left_identifier",
                "right_identifier",
            ])
            self.assertNotIn("arguments", events[0])


if __name__ == "__main__":
    unittest.main()
