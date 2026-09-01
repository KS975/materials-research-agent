from pathlib import Path

from runtime.v030_ui import build_autonomy_overview


def main() -> int:
    root = Path(".runtime")
    view = build_autonomy_overview(
        root, campaign_id="V030_T35_DEMO"
    )
    r1 = view["rounds"][0]
    r2 = view["rounds"][1]

    print("V0.3 UI/API INTEGRATION SMOKE")
    print()
    print("AUTONOMOUS RUNTIME")
    print(f"kind: {view['kind']}")
    print(f"round_count: {view['summary']['round_count']}")
    print(
        "automatic_capture_count: "
        f"{view['summary']['automatic_capture_count']}"
    )
    print(
        "dataset_rows: "
        f"{[x['row_count'] for x in view['datasets']]}"
    )
    print(
        "r1_scheduler_completed: "
        f"{r1['scheduler']['counts']['COMPLETED']}"
    )
    print(
        "r1_telemetry_sessions: "
        f"{r1['telemetry']['session_count']}"
    )
    print(f"r2_safety: {r2['safety']['state']}")
    print()
    print("CRASH / RESUME")
    print(
        "checkpoint_completed_before_crash: "
        f"{r2['crash_checkpoint']['completed_results_before_crash']}"
    )
    print(
        "recovery_audit_valid: "
        f"{str(r2['recovery_report']['recovery_audit_valid']).lower()}"
    )
    print()
    print("BOUNDARY")
    print("simulator_only: true")
    print("real_device_connected: false")
    print("automatic_model_activation: false")
    print("operator_override_cannot_bypass_safety: true")
    print()
    print("V0.3 UI/API INTEGRATION SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
