from __future__ import annotations

from pathlib import Path

from runtime.v014_ui import (
    run_inverse_design_for_ui,
    run_next_experiments_for_ui,
)


def main() -> None:
    root = Path(".runtime")

    inverse = run_inverse_design_for_ui(
        runtime_root=root,
        project_id=9016,
        message="冲击强度 >= 43、MFR >= 8.5，推荐5组方案",
        candidate_count=600,
        random_state=42,
    )
    assert inverse["status"] == "SUCCESS"
    assert inverse["counts"]["recommended"] == 5
    assert all(
        card["applicability_domain"]["status"] == "IN_DOMAIN"
        for card in inverse["design_cards"]
    )

    bo = run_next_experiments_for_ui(
        runtime_root=root,
        project_id=9018,
        target_metric="冲击强度",
        batch_size=5,
        candidate_count=900,
        random_state=42,
    )
    assert bo["status"] == "SUCCESS"
    assert len(bo["next_experiments"]) == 5
    assert all(
        row["applicability_domain"]["status"] != "OUT_OF_DOMAIN"
        for row in bo["next_experiments"]
    )

    print("V0.1.4 UI/API SMOKE")
    print(
        "T17:",
        inverse["status"],
        "recommended=",
        inverse["counts"]["recommended"],
        "pareto=",
        inverse["counts"]["pareto_front"],
    )
    print(
        "T18:",
        bo["status"],
        "next_experiments=",
        len(bo["next_experiments"]),
        "eligible=",
        bo["candidate_filtering"]["eligible_for_bo"],
    )
    print("V0.1.4 UI/API INTEGRATION PASS")


if __name__ == "__main__":
    main()
