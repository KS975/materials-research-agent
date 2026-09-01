from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from experiments.protocol import (
    ExperimentProtocolBuilder,
    ExperimentProtocolStore,
    validate_protocol_document,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    runtime = Path(args.runtime_root)
    fixture = runtime / "v030" / "fixtures" / "t27"
    template_path = fixture / "protocol_template.json"
    candidates_path = fixture / "candidates.json"
    if not template_path.exists() or not candidates_path.exists():
        raise SystemExit("ERROR: 请先运行 python -m scripts.build_v030_t27_fixture --reset")

    if args.reset:
        protocols = runtime / "v030" / "protocols"
        if protocols.exists():
            shutil.rmtree(protocols)

    template = json.loads(template_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    builder = ExperimentProtocolBuilder(template)
    store = ExperimentProtocolStore(runtime)

    valid = builder.build(candidates["valid"])
    valid_again = builder.build(candidates["valid"])
    validate_protocol_document(valid)
    first_save = store.save(valid)
    replay_save = store.save(valid_again)

    values = {x["name"]: x for x in valid["process_parameters"]}
    missing = builder.build(candidates["missing"])
    bad_unit = builder.build(candidates["bad_unit"])
    unsafe = builder.build(candidates["unsafe"])
    bad_category = builder.build(candidates["bad_category"])
    optional_missing = builder.build(candidates["optional_missing"])

    print("V0.3-T27 EXPERIMENT PROTOCOL")
    print()
    print("VALID CANDIDATE")
    print(f"status: {valid['status']}")
    print(f"protocol_id: {valid['protocol_id']}")
    print(f"material_recipe_items: {len(valid['material_recipe'])}")
    print(f"process_parameters: {len(valid['process_parameters'])}")
    print(f"process_steps: {len(valid['process_steps'])}")
    print(f"measurement_steps: {len(valid['measurement_steps'])}")
    print()
    print("UNIT NORMALIZATION")
    print(f"加工温度: {values['加工温度']['value']:.6f} {values['加工温度']['unit']}")
    print(f"混炼时间: {values['混炼时间']['value']:.6f} {values['混炼时间']['unit']}")
    print(f"螺杆转速: {values['螺杆转速']['value']} {values['螺杆转速']['unit']}")
    print()
    print("DETERMINISM + PERSISTENCE")
    print(f"same_protocol_id: {str(valid['protocol_id'] == valid_again['protocol_id']).lower()}")
    print(f"same_document: {str(valid == valid_again).lower()}")
    print(f"persisted_replay_idempotent: {str(replay_save['idempotent_replay']).lower()}")
    print(f"protocol_json: {first_save['protocol_path']}")
    print()
    print("BLOCKING CASES")
    for label, doc in (
        ("missing_required", missing),
        ("unsupported_unit", bad_unit),
        ("safety_limit", unsafe),
        ("invalid_choice", bad_category),
    ):
        codes = [x["code"] for x in doc["validation"]["issues"]]
        print(f"{label}: {doc['status']} codes={json.dumps(codes, ensure_ascii=False)}")
    print()
    print("OPTIONAL PARAMETER")
    print(f"optional_missing_status: {optional_missing['status']}")
    print()
    print("EXECUTION BOUNDARY")
    print("READY protocol may proceed to T28 Device Adapter.")
    print("BLOCKED protocol must never be submitted to a device.")
    print("T27 does not connect to or control any real device.")

    assert valid["status"] == "READY"
    assert abs(values["加工温度"]["value"] - 230.0) < 1e-9
    assert abs(values["混炼时间"]["value"] - 120.0) < 1e-9
    assert valid == valid_again
    assert replay_save["idempotent_replay"] is True
    assert missing["status"] == "BLOCKED"
    assert bad_unit["status"] == "BLOCKED"
    assert unsafe["status"] == "BLOCKED"
    assert bad_category["status"] == "BLOCKED"
    assert optional_missing["status"] == "READY"

    print()
    print("V0.3-T27 EXPERIMENT PROTOCOL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
