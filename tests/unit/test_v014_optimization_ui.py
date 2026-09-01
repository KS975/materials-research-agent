import json
from pathlib import Path

import pytest
import runtime.v014_ui as v014_ui

from runtime.v014_ui import (
    V014UIError,
    infer_batch_size,
    infer_bo_target_metric,
    looks_like_inverse_design,
    looks_like_next_experiments,
    run_inverse_design_for_ui,
)
from runtime.progress import progress_context


def _write_gate(root: Path, project_id: int, metric: str):
    path = root / "v013" / "gates" / f"project_{project_id}_{metric}_modeling_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "stage": "V0.1.3-B_modeling_gate",
                "project_id": project_id,
                "target_metric": metric,
                "decision": "PASS",
                "training_allowed": True,
                "official_model_allowed": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_detects_inverse_design_without_confusing_normal_comparison():
    assert looks_like_inverse_design("冲击强度 >= 43、MFR >= 8.5，推荐5组方案")
    assert not looks_like_inverse_design("3811 的冲击强度比 3809 低很多")


def test_detects_next_experiment_request():
    assert looks_like_next_experiments("以冲击强度为目标，下一轮推荐5组实验")
    assert not looks_like_next_experiments("帮我推荐5组配方")


def test_infers_metric_from_gate_and_message(tmp_path):
    _write_gate(tmp_path, 9018, "冲击强度")
    _write_gate(tmp_path, 9018, "MFR")
    assert infer_bo_target_metric(
        "以冲击强度为目标，下一轮推荐5组实验",
        tmp_path,
        9018,
    ) == "冲击强度"


def test_single_gate_metric_can_be_defaulted(tmp_path):
    _write_gate(tmp_path, 9018, "冲击强度")
    assert infer_bo_target_metric("下一轮推荐5组实验", tmp_path, 9018) == "冲击强度"


def test_ambiguous_multi_metric_request_requires_explicit_target(tmp_path):
    _write_gate(tmp_path, 9018, "冲击强度")
    _write_gate(tmp_path, 9018, "MFR")
    with pytest.raises(V014UIError):
        infer_bo_target_metric("下一轮推荐5组实验", tmp_path, 9018)


def test_batch_size_parser():
    assert infer_batch_size("下一轮推荐5组实验") == 5
    assert infer_batch_size("下一轮做3个测试") == 3


@pytest.mark.parametrize(
    "message",
    [
        "Project 9016：冲击强度 >= 43、MFR >= 8.5，推荐5组方案",
        "给我推荐五组冲击强度>=43，MFR>=8.5的方案",
        "给我推荐五组project 9016里冲击强度>=43，MFR>=8.5的方案",
    ],
)
def test_t17_ui_binds_screenshot_phrasings_to_real_gate_metrics(
    tmp_path,
    monkeypatch,
    message,
):
    _write_gate(tmp_path, 9016, "冲击强度")
    _write_gate(tmp_path, 9016, "MFR")

    captured = {}
    monkeypatch.setattr(
        v014_ui,
        "_resolve_search_space",
        lambda *_: tmp_path / "search_space.json",
    )
    monkeypatch.setattr(
        v014_ui,
        "_resolve_inverse_dataset",
        lambda *_: tmp_path / "dataset.csv",
    )
    monkeypatch.setattr(v014_ui, "_read_json", lambda *_: {})
    monkeypatch.setattr(v014_ui, "load_search_space", lambda *_: object())

    def fake_run_inverse_design(**kwargs):
        captured["request"] = kwargs["request"]
        return {"answer": "ok"}

    monkeypatch.setattr(
        v014_ui,
        "run_inverse_design",
        fake_run_inverse_design,
    )

    progress_events = []
    with progress_context(progress_events.append):
        report = run_inverse_design_for_ui(
            runtime_root=tmp_path,
            project_id=9016,
            message=message,
        )

    assert report["answer"] == "ok"
    assert [
        item.metric for item in captured["request"].objectives
    ] == ["冲击强度", "MFR"]
    assert captured["request"].recommendation_count == 5
    assert [event["stage"] for event in progress_events] == [
        "optimization_inputs",
        "optimization_inputs",
        "objective_binding",
        "inverse_design",
    ]
    assert all(event["source"] == "backend" for event in progress_events)
