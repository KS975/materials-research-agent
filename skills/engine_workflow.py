from __future__ import annotations

from typing import Any

from runtime.progress import emit_progress
from schemas.user_context import UserContext


class EngineWorkflowSkill:
    """Dispatch the five public engine intents to the deterministic adapter."""

    intents = {
        "engine_prepare_dataset",
        "automl_training",
        "predict_performance",
        "optimize_formula",
        "recommend_next_experiments",
    }

    _labels = {
        "engine_prepare_dataset": "准备建模数据集",
        "automl_training": "自动机器学习建模",
        "predict_performance": "模型性能预测",
        "optimize_formula": "配方或工艺优化",
        "recommend_next_experiments": "下一批实验推荐",
    }

    def __init__(self, adapter: Any):
        self.adapter = adapter

    def can_handle(self, intent: str) -> bool:
        return intent in self.intents

    def execute_intent(
        self,
        intent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: UserContext,
    ) -> dict[str, Any]:
        if not self.can_handle(intent):
            raise ValueError(f"EngineWorkflowSkill 不支持 operation={intent}")
        label = self._labels[intent]
        emit_progress(
            "engine_workflow",
            "running",
            label,
            "正在执行权限收敛、受控 Tool 编排和结果登记。",
            detail_items=[
                {"label": "执行器", "value": "EngineWorkflowAdapter"},
                {"label": "入口 Tool", "value": tool_name},
            ],
        )
        try:
            result = self.adapter.execute(intent, tool_name, dict(tool_args), ctx)
        except Exception as exc:
            emit_progress(
                "engine_workflow",
                "failed",
                label,
                f"{label}执行失败：{exc}",
            )
            raise
        status = str(result.get("status") or "OK")
        emit_progress(
            "engine_workflow",
            "completed" if status in {"OK", "MODEL_REQUIRED", "BLOCKED"} else "failed",
            label,
            f"{label}已结束，状态 {status}。",
            detail_items=[{"label": "状态", "value": status}],
        )
        return result
