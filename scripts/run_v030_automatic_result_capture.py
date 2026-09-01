from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

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
)
from experiments.protocol import sha256_json
from scripts.build_v030_t31_fixture import overlimit_protocol, safety_policy
from scripts.build_v030_t32_fixture import (
    CAMPAIGN_ID,
    ROUND_CONDITION,
    TARGET,
    UNIT,
    campaign_create,
    planned_experiments,
    protocols,
    round_plan,
    simulator_profile,
)


def _complete(protocol):
    adapter = SimulatorDeviceAdapter(simulator_profile())
    adapter.connect()
    adapter.prepare(protocol)
    adapter.submit_protocol(protocol)
    adapter.start()
    adapter.run_to_completion()
    return adapter


def _reseal(result: dict) -> dict:
    result = deepcopy(result)
    result.pop("result_id", None)
    result.pop("content_sha256", None)
    digest = sha256_json(result)
    result["result_id"] = "simres_" + digest[:20]
    result["content_sha256"] = digest
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    root = Path(args.runtime_root)

    store = CampaignStore(root)
    result_service = ExperimentalResultService(str(root))
    capture = AutomaticResultCaptureService(root)
    cid = CAMPAIGN_ID

    if args.reset:
        for target in (
            store.campaign_dir(cid),
            root / "v030" / "result_capture" / cid,
            root / "v020" / "evaluations" / cid,
            root / "v030" / "safety" / "V030_T32_LATCHED",
        ):
            if target.exists():
                shutil.rmtree(target)

    c = campaign_create()
    store.create(
        campaign_id=cid,
        project_id=c["project_id"],
        name=c["name"],
        target_metrics=c["target_metrics"],
        metadata=c["metadata"],
    )
    round_record = store.add_round(cid, plan=round_plan())
    rid = round_record["round_id"]
    result_service.register_planned_experiments(
        cid,
        round_id=rid,
        experiments=planned_experiments(),
    )
    store.transition_round(
        cid,
        round_id=rid,
        new_status="RUNNING",
        reason="T32 automatic result capture start",
    )

    print("V0.3-T32 AUTOMATIC RESULT CAPTURE")
    print(f"campaign_id: {cid}")
    print(f"round_id: {rid}")
    print()
    print("BOUNDARY")
    print("device_adapter: SimulatorDeviceAdapter only")
    print("result_origin: SIMULATOR_FIXTURE")
    print("synthetic: true")
    print("is_real_measurement: false")
    print("manual_result_submission_required: false")
    print()

    docs = protocols()
    actual_values = []
    first_adapter = None

    for index, protocol in enumerate(docs, start=1):
        adapter = _complete(protocol)
        if first_adapter is None:
            first_adapter = adapter
        result = capture.capture(
            cid,
            round_id=rid,
            adapter=adapter,
            protocol=protocol,
        )
        exp = result["t20"]["experiment"]
        value = exp["result"]["measurements"][TARGET]
        actual_values.append(value)
        print(f"CAPTURE {index}")
        print(f"candidate_id: {protocol['candidate_id']}")
        print(f"measurement_{TARGET}: {value:.6f} {UNIT}")
        print(f"t20_training_eligible: {str(exp['result']['training_eligible']).lower()}")
        print("measurement_origin: SIMULATOR_FIXTURE")
        print("is_real_measurement: false")
        if index == 1:
            current = store.load(cid)
            print(f"round_status_after_first_capture: {current['rounds'][0]['status']}")
            replay = capture.capture(
                cid,
                round_id=rid,
                adapter=adapter,
                protocol=protocol,
            )
            print(f"capture_replay_idempotent: {str(replay['idempotent_replay']).lower()}")
        print()

    summary = result_service.summary(cid, round_id=rid)
    latest_eval = capture.evaluations.evaluate(
        cid,
        round_id=rid,
        metric=TARGET,
        persist=True,
    )
    campaign = store.load(cid)
    round_status = campaign["rounds"][0]["status"]

    print("T20 ROUND RESULT STATE")
    print(f"completed: {summary['progress']['completed']}")
    print(f"pending: {summary['progress']['pending']}")
    print(f"training_eligible: {summary['progress']['training_eligible']}")
    print(f"can_close_round: {str(summary['can_close_round']).lower()}")
    print(f"round_status: {round_status}")
    print("t32_auto_closes_round: false")
    print()

    print("T21 PREDICTION VS MEASUREMENT")
    print(f"evaluated: {latest_eval['counts']['evaluated']}")
    print(f"MAE: {latest_eval['aggregate']['mae']:.6f}")
    print(f"RMSE: {latest_eval['aggregate']['rmse']:.6f}")
    r2 = latest_eval['aggregate']['r2']
    print(f"R2: {r2:.6f}" if r2 is not None else "R2: null")
    print(f"report_json: {latest_eval['report_json']}")
    print()

    # Boundary: result not ready.
    pending_adapter = SimulatorDeviceAdapter(simulator_profile())
    pending_adapter.connect()
    pending_adapter.prepare(docs[0])
    pending_adapter.submit_protocol(docs[0])
    pending_adapter.start()
    not_ready_code = None
    try:
        capture.capture(
            cid,
            round_id=rid,
            adapter=pending_adapter,
            protocol=docs[0],
        )
    except ResultNotReadyError as exc:
        not_ready_code = exc.code

    # Semantic validation cases use a correctly resealed simulator result so
    # they test the intended boundary rather than only hash integrity.
    source_result = first_adapter.read_result()
    experiment = store.load(cid)["rounds"][0]["experiments"][0]

    wrong_unit = deepcopy(source_result)
    wrong_unit["outputs"][0]["unit"] = "J/m"
    wrong_unit = _reseal(wrong_unit)
    unit_code = None
    try:
        normalize_device_result_for_t20(
            wrong_unit,
            experiment=experiment,
            protocol_id=docs[0]["protocol_id"],
            device_id=first_adapter.device_id,
        )
    except ResultCaptureConflictError as exc:
        unit_code = exc.code

    wrong_condition = deepcopy(source_result)
    wrong_condition["outputs"][0]["condition_signature"] = "DIFFERENT_25C"
    wrong_condition = _reseal(wrong_condition)
    condition_code = None
    try:
        normalize_device_result_for_t20(
            wrong_condition,
            experiment=experiment,
            protocol_id=docs[0]["protocol_id"],
            device_id=first_adapter.device_id,
        )
    except ResultCaptureConflictError as exc:
        condition_code = exc.code

    tampered = deepcopy(source_result)
    tampered["outputs"][0]["value"] = 999.0
    integrity_code = None
    try:
        normalize_device_result_for_t20(
            tampered,
            experiment=experiment,
            protocol_id=docs[0]["protocol_id"],
            device_id=first_adapter.device_id,
        )
    except ResultCaptureIntegrityError as exc:
        integrity_code = exc.code

    fake_real = deepcopy(source_result)
    fake_real["measurement_origin"] = "REAL_DEVICE"
    fake_real["synthetic"] = False
    fake_real["is_real_measurement"] = True
    fake_real = _reseal(fake_real)
    real_code = None
    try:
        normalize_device_result_for_t20(
            fake_real,
            experiment=experiment,
            protocol_id=docs[0]["protocol_id"],
            device_id=first_adapter.device_id,
        )
    except ResultCaptureValidationError as exc:
        real_code = exc.code

    # T31 latch blocks capture even when a device has a completed result.
    safety = SafetyInterlock(
        interlock_id="V030_T32_LATCHED",
        policy=safety_policy(first_adapter.device_id),
        runtime_root=root,
    )
    safety.check_protocol(overlimit_protocol())
    safety_code = None
    try:
        capture.capture(
            cid,
            round_id=rid,
            adapter=first_adapter,
            protocol=docs[0],
            safety_interlock=safety,
        )
    except SafetyStopActiveError as exc:
        safety_code = exc.code

    print("BOUNDARY REJECTIONS")
    print(f"result_not_ready: {not_ready_code}")
    print(f"unit_mismatch: {unit_code}")
    print(f"condition_mismatch: {condition_code}")
    print(f"tampered_result: {integrity_code}")
    print(f"unverified_real_source: {real_code}")
    print(f"latched_safety_stop: {safety_code}")
    print()

    receipt_dir = root / "v030" / "result_capture" / cid / rid
    receipts = sorted(receipt_dir.glob("*.json"))
    print("AUDIT")
    print(f"capture_receipts: {len(receipts)}")
    print(f"receipt_dir: {receipt_dir}")
    print("atomic_receipt_write: true")
    print()
    print("EXECUTION BOUNDARY")
    print("T32 automatically maps validated simulator COMPLETED results into V0.2 T20.")
    print("T21 is refreshed after capture; T32 does not close the Round or create a new Round.")
    print("Simulator fixture results are training-eligible inside this engineering fixture only; they are not real material measurements.")
    print("T33 will own autonomous round completion and transition orchestration.")

    if summary["progress"]["completed"] != 4:
        raise SystemExit("ERROR: expected 4 completed T20 results")
    if summary["progress"]["pending"] != 0:
        raise SystemExit("ERROR: pending should be 0")
    if not summary["can_close_round"]:
        raise SystemExit("ERROR: round should be closable")
    if round_status != "PARTIALLY_COMPLETED":
        raise SystemExit("ERROR: T32 must not auto-close the round")
    if latest_eval["counts"]["evaluated"] != 4:
        raise SystemExit("ERROR: T21 expected 4 evaluated samples")
    if len(receipts) != 4:
        raise SystemExit("ERROR: expected 4 capture receipts")
    expected_codes = {
        "result_not_ready": "RESULT_NOT_READY",
        "unit_mismatch": "RESULT_CAPTURE_CONFLICT",
        "condition_mismatch": "RESULT_CAPTURE_CONFLICT",
        "tampered_result": "RESULT_CAPTURE_INTEGRITY_ERROR",
        "unverified_real_source": "RESULT_CAPTURE_VALIDATION_ERROR",
        "latched_safety_stop": "SAFETY_STOP_ACTIVE",
    }
    actual_codes = {
        "result_not_ready": not_ready_code,
        "unit_mismatch": unit_code,
        "condition_mismatch": condition_code,
        "tampered_result": integrity_code,
        "unverified_real_source": real_code,
        "latched_safety_stop": safety_code,
    }
    if actual_codes != expected_codes:
        raise SystemExit(f"ERROR: boundary codes mismatch: {actual_codes}")

    print()
    print("V0.3-T32 AUTOMATIC RESULT CAPTURE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
