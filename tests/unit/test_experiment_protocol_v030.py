from copy import deepcopy
import json

import pytest

from experiments.protocol import (
    ExperimentProtocolBuilder,
    ExperimentProtocolConflictError,
    ExperimentProtocolStore,
    ExperimentProtocolValidationError,
    convert_unit,
    validate_protocol_document,
)
from scripts.build_v030_t27_fixture import candidates, fixture_template


def test_valid_candidate_builds_ready_protocol():
    doc = ExperimentProtocolBuilder(fixture_template()).build(candidates()["valid"])
    assert doc["status"] == "READY"
    assert doc["validation"]["issues"] == []
    assert len(doc["material_recipe"]) == 3
    assert len(doc["process_steps"]) == 2
    assert len(doc["measurement_steps"]) == 1
    validate_protocol_document(doc)


def test_unit_normalization_is_explicit_and_correct():
    builder = ExperimentProtocolBuilder(fixture_template())
    doc = builder.build(candidates()["valid"])
    params = {x["name"]: x for x in doc["process_parameters"]}
    assert params["加工温度"]["value"] == pytest.approx(230.0)
    assert params["混炼时间"]["value"] == pytest.approx(120.0)
    assert convert_unit(850, "kPa", "MPa") == pytest.approx(0.85)


def test_same_input_is_deterministic():
    builder = ExperimentProtocolBuilder(fixture_template())
    one = builder.build(candidates()["valid"])
    two = builder.build(candidates()["valid"])
    assert one == two
    assert one["protocol_id"] == two["protocol_id"]
    assert one["content_sha256"] == two["content_sha256"]


def test_missing_required_parameter_is_blocked():
    doc = ExperimentProtocolBuilder(fixture_template()).build(candidates()["missing"])
    assert doc["status"] == "BLOCKED"
    assert "MISSING_REQUIRED_PARAMETER" in {x["code"] for x in doc["validation"]["issues"]}


def test_unsupported_candidate_unit_is_blocked():
    doc = ExperimentProtocolBuilder(fixture_template()).build(candidates()["bad_unit"])
    assert doc["status"] == "BLOCKED"
    assert "UNSUPPORTED_UNIT" in {x["code"] for x in doc["validation"]["issues"]}


def test_safety_limit_and_categorical_choice_are_blocked():
    builder = ExperimentProtocolBuilder(fixture_template())
    unsafe = builder.build(candidates()["unsafe"])
    category = builder.build(candidates()["bad_category"])
    assert unsafe["status"] == "BLOCKED"
    assert "SAFETY_LIMIT" in {x["code"] for x in unsafe["validation"]["issues"]}
    assert category["status"] == "BLOCKED"
    assert "INVALID_CHOICE" in {x["code"] for x in category["validation"]["issues"]}


def test_optional_missing_parameter_is_allowed():
    doc = ExperimentProtocolBuilder(fixture_template()).build(candidates()["optional_missing"])
    assert doc["status"] == "READY"
    assert "增韧剂" not in {x["name"] for x in doc["material_recipe"]}


def test_protocol_store_is_immutable_and_idempotent(tmp_path):
    builder = ExperimentProtocolBuilder(fixture_template())
    doc = builder.build(candidates()["valid"])
    store = ExperimentProtocolStore(tmp_path)
    first = store.save(doc)
    second = store.save(doc)
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert store.load(doc["protocol_id"]) == doc

    # Same ID with a tampered document is rejected before overwrite.
    tampered = deepcopy(doc)
    tampered["material_recipe"][0]["value"] = 99
    with pytest.raises(ExperimentProtocolValidationError):
        store.save(tampered)


def test_document_tampering_is_detected():
    doc = ExperimentProtocolBuilder(fixture_template()).build(candidates()["valid"])
    bad = deepcopy(doc)
    bad["process_parameters"][0]["value"] += 1
    with pytest.raises(ExperimentProtocolValidationError):
        validate_protocol_document(bad)


def test_invalid_template_unit_or_step_reference_is_rejected():
    template = fixture_template()
    template["parameters"][0]["canonical_unit"] = "mystery_unit"
    with pytest.raises(ExperimentProtocolValidationError):
        ExperimentProtocolBuilder(template)

    template = fixture_template()
    template["process_steps"][0]["parameters"].append("不存在参数")
    with pytest.raises(ExperimentProtocolValidationError):
        ExperimentProtocolBuilder(template)
