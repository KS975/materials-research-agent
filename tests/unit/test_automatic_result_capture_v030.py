from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import (
    AutomaticResultCaptureService,
    CampaignStore,
    ExperimentalResultService,
    ResultCaptureConflictError,
    ResultCaptureIntegrityError,
    ResultCaptureValidationError,
    ResultNotReadyError,
    SafetyInterlock,
    SafetyStopActiveError,
    SimulatorDeviceAdapter,
    normalize_device_result_for_t20,
    verify_device_result_integrity,
)
from experiments.protocol import sha256_json
from scripts.build_v030_t31_fixture import overlimit_protocol, safety_policy
from scripts.build_v030_t32_fixture import (
    CAMPAIGN_ID,
    TARGET,
    campaign_create,
    planned_experiments,
    protocols,
    round_plan,
    simulator_profile,
)


def _setup_campaign(tmp_path):
    store = CampaignStore(tmp_path)
    service = ExperimentalResultService(str(tmp_path))
    c = campaign_create()
    store.create(
        campaign_id=c["campaign_id"],
        project_id=c["project_id"],
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    r = store.add_round(c["campaign_id"], plan=round_plan())
    service.register_planned_experiments(
        c["campaign_id"],
        round_id=r["round_id"],
        experiments=planned_experiments(),
    )
    store.transition_round(
        c["campaign_id"],
        round_id=r["round_id"],
        new_status="RUNNING",
        reason="test",
    )
    return store, service, r["round_id"]


def _complete(protocol):
    adapter = SimulatorDeviceAdapter(simulator_profile())
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    adapter.run_to_completion()
    return adapter


def _reseal(result):
    result = deepcopy(result)
    result.pop("result_id", None)
    result.pop("content_sha256", None)
    digest = sha256_json(result)
    result["result_id"] = "simres_" + digest[:20]
    result["content_sha256"] = digest
    return result


def _experiment(store, round_id, index=0):
    return store.load(CAMPAIGN_ID)["rounds"][0]["experiments"][index]


def test_device_result_integrity_passes():
    adapter = _complete(protocols()[0])
    assert verify_device_result_integrity(adapter.read_result()) is True


def test_successful_capture_flows_into_t20(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    capture = AutomaticResultCaptureService(tmp_path)
    out = capture.capture(CAMPAIGN_ID, round_id=rid, adapter=adapter, protocol=protocol)
    result = out["t20"]["experiment"]["result"]
    assert result["training_eligible"] is True
    assert result["measurements"][TARGET] == adapter.read_result()["outputs"][0]["value"]
    assert "SIMULATOR_FIXTURE" in result["notes"]


def test_first_capture_transitions_round_to_partially_completed(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    AutomaticResultCaptureService(tmp_path).capture(
        CAMPAIGN_ID, round_id=rid, adapter=_complete(protocol), protocol=protocol
    )
    assert store.load(CAMPAIGN_ID)["rounds"][0]["status"] == "PARTIALLY_COMPLETED"


def test_capture_replay_is_idempotent(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    capture = AutomaticResultCaptureService(tmp_path)
    first = capture.capture(CAMPAIGN_ID, round_id=rid, adapter=adapter, protocol=protocol)
    second = capture.capture(CAMPAIGN_ID, round_id=rid, adapter=adapter, protocol=protocol)
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    receipt_dir = tmp_path/"v030"/"result_capture"/CAMPAIGN_ID/rid
    assert len(list(receipt_dir.glob("*.json"))) == 1


def test_all_results_make_round_closable_but_t32_does_not_close(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    capture = AutomaticResultCaptureService(tmp_path)
    for protocol in protocols():
        capture.capture(CAMPAIGN_ID, round_id=rid, adapter=_complete(protocol), protocol=protocol)
    summary = service.summary(CAMPAIGN_ID, round_id=rid)
    assert summary["progress"]["completed"] == 4
    assert summary["progress"]["pending"] == 0
    assert summary["can_close_round"] is True
    assert store.load(CAMPAIGN_ID)["rounds"][0]["status"] == "PARTIALLY_COMPLETED"


def test_t21_report_refreshes_after_capture(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    capture = AutomaticResultCaptureService(tmp_path)
    for protocol in protocols()[:2]:
        out = capture.capture(CAMPAIGN_ID, round_id=rid, adapter=_complete(protocol), protocol=protocol)
    report = capture.evaluations.evaluate(CAMPAIGN_ID, round_id=rid, metric=TARGET, persist=True)
    assert report["counts"]["evaluated"] == 2
    assert report["aggregate"]["mae"] >= 0
    assert Path(report["report_json"]).exists()


def test_result_not_ready_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = SimulatorDeviceAdapter(simulator_profile())
    adapter.connect(); adapter.prepare(protocol); adapter.submit_protocol(protocol); adapter.start()
    with pytest.raises(ResultNotReadyError):
        AutomaticResultCaptureService(tmp_path).capture(
            CAMPAIGN_ID, round_id=rid, adapter=adapter, protocol=protocol
        )


def test_tampered_result_hash_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["outputs"][0]["value"] = 999.0
    with pytest.raises(ResultCaptureIntegrityError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_wrong_candidate_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["candidate_id"] = "WRONG"
    result = _reseal(result)
    with pytest.raises(ResultCaptureConflictError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_wrong_protocol_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["protocol_id"] = "proto_wrong"
    result = _reseal(result)
    with pytest.raises(ResultCaptureConflictError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_wrong_unit_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["outputs"][0]["unit"] = "J/m"
    result = _reseal(result)
    with pytest.raises(ResultCaptureConflictError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_wrong_condition_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["outputs"][0]["condition_signature"] = "OTHER_25C"
    result = _reseal(result)
    with pytest.raises(ResultCaptureConflictError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_missing_metric_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["outputs"] = []
    result = _reseal(result)
    with pytest.raises(ResultCaptureValidationError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_duplicate_metric_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["outputs"].append(deepcopy(result["outputs"][0]))
    result = _reseal(result)
    with pytest.raises(ResultCaptureValidationError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_nonfinite_measurement_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["outputs"][0]["value"] = float("inf")
    # Skip reseal because sha256_json itself correctly refuses non-finite JSON;
    # exercise semantic normalization through a finite-hashable string instead.
    result["outputs"][0]["value"] = "not-a-number"
    result = _reseal(result)
    with pytest.raises(ResultCaptureValidationError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_unverified_real_measurement_source_is_blocked(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    result = adapter.read_result()
    result["measurement_origin"] = "REAL_DEVICE"
    result["synthetic"] = False
    result["is_real_measurement"] = True
    result = _reseal(result)
    with pytest.raises(ResultCaptureValidationError):
        normalize_device_result_for_t20(
            result,
            experiment=_experiment(store, rid),
            protocol_id=protocol["protocol_id"],
            device_id=adapter.device_id,
        )


def test_safety_stop_blocks_capture(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    adapter = _complete(protocol)
    safety = SafetyInterlock(
        interlock_id="T32_SAFETY",
        policy=safety_policy(adapter.device_id),
        runtime_root=tmp_path,
    )
    safety.check_protocol(overlimit_protocol())
    with pytest.raises(SafetyStopActiveError):
        AutomaticResultCaptureService(tmp_path).capture(
            CAMPAIGN_ID,
            round_id=rid,
            adapter=adapter,
            protocol=protocol,
            safety_interlock=safety,
        )


def test_receipt_contains_explicit_fixture_provenance(tmp_path):
    store, service, rid = _setup_campaign(tmp_path)
    protocol = protocols()[0]
    out = AutomaticResultCaptureService(tmp_path).capture(
        CAMPAIGN_ID, round_id=rid, adapter=_complete(protocol), protocol=protocol
    )
    receipt = json.loads(Path(out["receipt_json"]).read_text(encoding="utf-8"))
    assert receipt["measurement_origin"] == "SIMULATOR_FIXTURE"
    assert receipt["synthetic"] is True
    assert receipt["is_real_measurement"] is False
    assert "不是真实材料实验结果" in receipt["fixture_warning"]
