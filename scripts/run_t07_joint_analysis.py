from __future__ import annotations

import argparse
import os

import httpx
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="T07 acceptance: MySQL facts + historical Qdrant RAG"
    )
    parser.add_argument(
        "--message",
        default=(
            "3811 的冲击强度比 3809 低很多，历史上有没有类似问题？"
            "结合数据库数据和历史报告分析一下。"
        ),
    )
    parser.add_argument("--project-id", type=int, default=115)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    user_id = os.getenv("DEV_USER_ID", "").strip()
    company_id = os.getenv("DEV_COMPANY_ID", "").strip()
    projects = os.getenv("DEV_PROJECT_IDS", "").strip()

    if not user_id or not company_id or not projects:
        print("ERROR: configure DEV_USER_ID, DEV_COMPANY_ID, DEV_PROJECT_IDS in .env")
        return 2

    message = args.message
    if args.project_id is not None and str(args.project_id) not in message:
        message = f"项目{args.project_id}：{message}"

    response = httpx.post(
        args.base_url.rstrip("/") + "/api/v1/chat-ui",
        headers={
            "X-User-Id": user_id,
            "X-Company-Id": company_id,
            "X-Project-Ids": projects,
        },
        json={
            "message": message,
            "history": [],
            "attachment_ids": [],
        },
        timeout=180.0,
    )

    print("http_status:", response.status_code)
    if response.status_code != 200:
        print(response.text)
        return 1

    data = response.json()
    print("intent:", data.get("intent"))
    print("router:", data.get("router"))
    print("tool:", data.get("tool_name"))
    print("answer:")
    print(data.get("answer", ""))

    evidence = data.get("evidence") or []
    mysql = [
        item for item in evidence
        if item.get("evidence_type") == "mysql"
    ]
    history = [
        item for item in evidence
        if item.get("evidence_type") == "knowledge_index"
        or item.get("source") == "knowledge_index"
    ]

    print("evidence_count:", len(evidence))
    print("mysql_evidence_count:", len(mysql))
    print("history_evidence_count:", len(history))

    for index, item in enumerate(evidence, start=1):
        if item.get("evidence_type") == "knowledge_index" or item.get("source") == "knowledge_index":
            print(
                f"{index}. HISTORY "
                f"score={item.get('score')} "
                f"project={item.get('project_id')} "
                f"file={item.get('filename')} "
                f"page={item.get('page')} "
                f"paragraph={item.get('paragraph_start')}-{item.get('paragraph_end')} "
                f"chunk={item.get('chunk_index')}"
            )
        else:
            print(
                f"{index}. MYSQL "
                f"source={item.get('source')} "
                f"record_id={item.get('record_id')}"
            )

    if data.get("intent") != "joint_mysql_knowledge_analysis":
        print("FAIL: intent is not joint_mysql_knowledge_analysis")
        return 1
    if not mysql:
        print("FAIL: no MySQL evidence")
        return 1
    if not history:
        print("FAIL: no historical Knowledge Index evidence")
        return 1
    if any(
        item.get("project_id") not in (None, args.project_id)
        for item in history
    ):
        print("FAIL: historical evidence leaked another project")
        return 1

    print("T07 MYSQL + HISTORICAL RAG PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
