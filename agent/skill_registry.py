from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class SkillRegistryError(ValueError):
    """Raised when a Skill contract or dispatch violates the registry."""


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """Declarative contract for one atomic business Skill.

    A Skill owns business operations (the legacy ``intent`` values), not raw
    natural-language routing.  The contract is deliberately JSON-safe so it can
    be checkpointed, audited and shown in the SSE execution trace.
    """

    name: str
    display_name: str
    description: str
    intents: frozenset[str]
    tool_allowlist: frozenset[str] = frozenset()
    allows_no_tool: bool = False
    workflow: tuple[str, ...] = ()
    input_required_by_intent: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    output_required: tuple[str, ...] = ("status",)
    evidence_rules: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()
    approval_points: tuple[str, ...] = ()
    error_strategy: str = "fail_closed"
    default_executor_family: str = "material_tool"
    executor_family_by_intent: Mapping[str, str] = field(default_factory=dict)

    def executor_family_for(self, intent: str) -> str:
        return str(
            self.executor_family_by_intent.get(
                intent,
                self.default_executor_family,
            )
        )

    def validate_dispatch(
        self,
        *,
        intent: str,
        tool_name: str | None,
        tool_args: Mapping[str, Any] | None,
    ) -> None:
        if intent not in self.intents:
            raise SkillRegistryError(
                f"Skill {self.name} 不支持 operation={intent}"
            )
        if tool_name is None:
            if not self.allows_no_tool:
                raise SkillRegistryError(
                    f"Skill {self.name} 的 operation={intent} 缺少受控 Tool"
                )
        elif tool_name not in self.tool_allowlist:
            raise SkillRegistryError(
                f"Skill {self.name} 不允许调用 Tool={tool_name}"
            )

        args = dict(tool_args or {})
        missing = [
            key
            for key in self.input_required_by_intent.get(intent, ())
            if args.get(key) is None or str(args.get(key)).strip() == ""
        ]
        if missing:
            raise SkillRegistryError(
                f"Skill {self.name}/{intent} 缺少输入字段：{', '.join(missing)}"
            )

    def validate_output(self, result: Any) -> None:
        if not isinstance(result, Mapping):
            raise SkillRegistryError(
                f"Skill {self.name} 输出必须是结构化对象"
            )
        missing = [key for key in self.output_required if key not in result]
        if missing:
            raise SkillRegistryError(
                f"Skill {self.name} 输出缺少字段：{', '.join(missing)}"
            )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "operations": sorted(self.intents),
            "tool_allowlist": sorted(self.tool_allowlist),
            "allows_no_tool": self.allows_no_tool,
            "workflow": list(self.workflow),
            "input_schema": {
                key: {"required": list(value)}
                for key, value in self.input_required_by_intent.items()
            },
            "output_schema": {"required": list(self.output_required)},
            "evidence_rules": list(self.evidence_rules),
            "guardrails": list(self.guardrails),
            "approval_points": list(self.approval_points),
            "error_strategy": self.error_strategy,
        }


class SkillRegistry:
    """Register Skill contracts and own the intent-to-Skill binding."""

    def __init__(self, specs: Iterable[SkillSpec] | None = None) -> None:
        self._skills: dict[str, SkillSpec] = {}
        self._intent_owner: dict[str, str] = {}
        for spec in specs or ():
            self.register(spec)

    def register(self, spec: SkillSpec) -> None:
        name = str(spec.name or "").strip()
        if not name:
            raise SkillRegistryError("Skill name 不能为空")
        if name in self._skills:
            raise SkillRegistryError(f"Skill 已注册：{name}")
        duplicates = sorted(
            intent for intent in spec.intents if intent in self._intent_owner
        )
        if duplicates:
            owners = ", ".join(
                f"{intent}->{self._intent_owner[intent]}" for intent in duplicates
            )
            raise SkillRegistryError(f"operation 只能归属一个 Skill：{owners}")
        self._skills[name] = spec
        for intent in spec.intents:
            self._intent_owner[intent] = name

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[str(name)]
        except KeyError as exc:
            raise SkillRegistryError(f"未知 Skill：{name}") from exc

    def resolve(self, intent: str) -> SkillSpec:
        owner = self._intent_owner.get(str(intent))
        if owner is None:
            raise SkillRegistryError(f"没有 Skill 注册 operation={intent}")
        return self._skills[owner]

    def validate_dispatch(
        self,
        *,
        intent: str,
        tool_name: str | None,
        tool_args: Mapping[str, Any] | None,
        expected_skill: str | None = None,
    ) -> SkillSpec:
        spec = self.resolve(intent)
        if expected_skill is not None and spec.name != expected_skill:
            raise SkillRegistryError(
                "Skill 计划与注册表不一致："
                f"planned={expected_skill}, registered={spec.name}"
            )
        spec.validate_dispatch(
            intent=intent,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        return spec

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            self._skills[name].to_public_dict()
            for name in sorted(self._skills)
        ]
