from __future__ import annotations

import argparse
import json
import sys

from demo import MondayDemoService, MondayDemoError


def _configure_utf8_stdio() -> None:
    """Protect the top-level runner from Windows cp1252 stdout/stderr."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main() -> int:
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the stable Monday V0.1.1 -> V0.3 demo runtime."
        )
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset deterministic V0.2/V0.3 demo campaigns before rerun.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Do not execute fixture pipelines; only inspect current readiness.",
    )
    parser.add_argument(
        "--encoding-check",
        action="store_true",
        help="Only verify the UTF-8 child Python used by Monday Demo.",
    )
    args = parser.parse_args()

    service = MondayDemoService()

    if args.encoding_check:
        result = service.encoding_preflight()
        print("MONDAY DEMO UTF-8 PREFLIGHT")
        print(result["stdout"].rstrip())
        print("command:", " ".join(str(x) for x in (result.get("child_command") or [])))
        print("status:", "PASS" if result["pass"] else "FAIL")
        return 0 if result["pass"] else 1

    try:
        report = (
            service.status()
            if args.status_only
            else service.prepare(reset=args.reset)
        )
    except MondayDemoError as exc:
        print("MONDAY DEMO PREPARE FAILED")
        print(str(exc))
        return 1

    print("MONDAY V0.1.1 -> V0.3 DEMO")
    print()
    print(f"status: {report['status']}")
    print(
        "prepared_internal_versions: "
        f"{report['prepared_internal_versions']}"
    )
    print()

    print("VERSION MATRIX")
    for item in report["versions"]:
        print(
            f"{item['version']}: "
            f"{item['status']} · {item['title']}"
        )
        if item.get("project_id"):
            print(f"  project_id: {item['project_id']}")
        print(
            "  capabilities: "
            + " / ".join(item["capabilities"])
        )
    print()

    print("MAIN CLOSED-LOOP STORY")
    print(
        "Data -> Reality Check -> Modeling Gate -> Train/CV/AD -> "
        "BO -> Campaign -> Experiment -> Result Capture -> "
        "Dataset Version -> Retrain -> Next BO -> V0.3 Autonomous Run"
    )
    print()

    v020 = next(
        x for x in report["versions"]
        if x["version"] == "V0.2"
    )
    v030 = next(
        x for x in report["versions"]
        if x["version"] == "V0.3"
    )
    print("AUTO-LEARNING DEMO")
    print(
        "V0.2 dataset rows: "
        f"{v020.get('summary', {}).get('dataset_row_counts')}"
    )
    print(
        "V0.2 model auto activation: false "
        "(retrain/compare automatic, promotion approval human)"
    )
    print(
        "V0.3 autonomous dataset rows: "
        f"{v030.get('summary', {}).get('dataset_row_counts')}"
    )
    print(
        "V0.3 automatic captures: "
        f"{v030.get('summary', {}).get('automatic_captures')}"
    )
    print()

    print("DEMO PROJECT IDS")
    print(
        ",".join(
            str(x)
            for x in report["demo_scope_project_ids"]
        )
    )
    print()

    print("BOUNDARY")
    print("V0.1.1 business MySQL: READ ONLY")
    print("V0.1.2 RAG: live external capability, not fabricated")
    print("V0.1.3/V0.1.4 fixtures: engineering demo only")
    print("V0.2/V0.3 measurements: deterministic synthetic fixture")
    print("V0.3: Simulator only, real_device_connected=false")
    print("Model promotion: no automatic approval")
    print("Safety: operator cannot bypass")
    print()

    print(f"report_path: {report.get('report_path') or '.runtime/demo/monday_v030/monday_demo_report.json'}")
    print(f"report_sha256: {report['report_sha256']}")
    print()
    if report["status"] == "READY":
        print("MONDAY DEMO PREPARE PASS")
        return 0

    print(
        "MONDAY DEMO INTERNAL RUNTIME NOT FULLY PREPARED. "
        "Run without --status-only."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
