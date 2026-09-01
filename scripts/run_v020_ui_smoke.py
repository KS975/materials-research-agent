from __future__ import annotations

import argparse
from runtime.v020_ui import build_campaign_overview


def main():
    parser = argparse.ArgumentParser(description="V0.2 UI/API smoke")
    parser.add_argument("--runtime-root", default=".runtime")
    args = parser.parse_args()

    final = build_campaign_overview(args.runtime_root, campaign_id="V020_T26_DEMO")
    feedback = build_campaign_overview(args.runtime_root, campaign_id="V020_T24_DEMO")

    print("V0.2 UI/API SMOKE")
    print()
    print("T26 CLOSED LOOP VIEW")
    print("campaign_status:", final["campaign"]["status"])
    print("round_count:", final["campaign"]["round_count"])
    print("dataset_rows:", [x["row_count"] for x in final["datasets"]])
    print("evaluation_MAE:", round(final["evaluation"]["aggregate"]["mae"], 6))
    print("model_decision:", final["model_promotion"]["decision"])
    print("checkpoint_status:", final["checkpoint"]["status"])
    print("end_to_end:", final["end_to_end"]["decision"])
    print()
    print("T24 FEEDBACK VIEW")
    print("round_status:", feedback["latest_round"]["status"])
    print("pending_experiments:", len(feedback["latest_round"]["pending_experiments"]))
    print("can_start:", str(feedback["latest_round"]["can_start"]).lower())
    print()
    assert final["end_to_end"]["decision"] == "PASS"
    assert final["campaign"]["round_count"] == 3
    assert feedback["latest_round"]["status"] == "PLANNED"
    assert len(feedback["latest_round"]["pending_experiments"]) == 5
    print("V0.2 UI/API INTEGRATION PASS")


if __name__ == "__main__":
    main()
