from __future__ import annotations

import argparse
import os

import httpx
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="T06 acceptance: Chat -> historical Qdrant RAG -> Evidence"
    )
    parser.add_argument(
        "--message",
        default="历史有没有类似问题？",
    )
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    args = parser.parse_args()

    user_id = os.getenv("DEV_USER_ID", "").strip()
    company_id = os.getenv("DEV_COMPANY_ID", "").strip()
    projects = os.getenv("DEV_PROJECT_IDS", "").strip()

    if not user_id or not company_id or not projects:
        print("ERROR: configure DEV_USER_ID, DEV_COMPANY_ID, DEV_PROJECT_IDS in .env")
        return 2

    message = args.message
    if args.project_id is not None and f"{args.project_id}" not in message:
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
        timeout=120.0,
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
    print("evidence_count:", len(data.get("evidence") or []))

    for index, item in enumerate(data.get("evidence") or [], start=1):
        print(
            f"{index}. source={item.get('source')} "
            f"score={item.get('score')} "
            f"project={item.get('project_id')} "
            f"file={item.get('filename')} "
            f"page={item.get('page')} "
            f"paragraph={item.get('paragraph_start')}-{item.get('paragraph_end')} "
            f"chunk={item.get('chunk_index')}"
        )

    if data.get("intent") != "search_historical_knowledge":
        print("FAIL: intent is not search_historical_knowledge")
        return 1

    evidence = data.get("evidence") or []
    if evidence and any(item.get("source") != "knowledge_index" for item in evidence):
        print("FAIL: evidence contains a non-knowledge source")
        return 1

    print("T06 CHAT HISTORICAL RAG PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
