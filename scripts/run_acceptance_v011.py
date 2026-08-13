from __future__ import annotations

import argparse
import os

from app.container import ApplicationContainer
from app.config import get_settings
from schemas.user_context import UserContext


def main():
    parser = argparse.ArgumentParser(
        description="V0.1.1 local acceptance runner (real MySQL)"
    )
    parser.add_argument("--sample-a", required=True)
    parser.add_argument("--sample-b", required=True)
    parser.add_argument(
        "--why",
        default=None,
        help="Optional performance question, e.g. '为什么 样品A 密度差下降？'",
    )
    args = parser.parse_args()

    user_id = os.getenv("DEV_USER_ID", "").strip()
    company_id = os.getenv("DEV_COMPANY_ID", "").strip()
    project_ids = tuple(
        int(x.strip())
        for x in os.getenv("DEV_PROJECT_IDS", "").split(",")
        if x.strip()
    )
    if not user_id or not company_id or not project_ids:
        raise SystemExit(
            "请设置 DEV_USER_ID / DEV_COMPANY_ID / DEV_PROJECT_IDS"
        )

    ctx = UserContext(
        user_id=user_id,
        company_id=company_id,
        project_ids=project_ids,
        permission_source="acceptance_script",
    )
    app = ApplicationContainer(get_settings())

    cases = [
        ("T01", f"查 {args.sample_a} 的完整研发上下文"),
        ("T02", f"比较 {args.sample_a} 和 {args.sample_b}"),
    ]
    if args.why:
        cases.append(("T03", args.why))

    for case_id, message in cases:
        print("=" * 80)
        print(case_id, message)
        state = app.agent.chat(message, ctx)
        print("intent:", state.get("intent"))
        print("tool:", state.get("tool_name"))
        print("answer:")
        print(state.get("answer"))
        print("warnings:", state.get("warnings"))


if __name__ == "__main__":
    main()
