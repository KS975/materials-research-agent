from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx


_RECORD_ID = re.compile(r"\b(?:mysql|vector|upload|dialog)-[0-9a-f]{20}\b")


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    scenario_id: int
    message: str
    required_args: tuple[str, ...] = ()


CASES: tuple[AcceptanceCase, ...] = (
    AcceptanceCase("S1-01", 1, "查找与 EXP-128 相似的配方，并结合历史案例。", ("identifier",)),
    AcceptanceCase("S1-02", 1, "找与 EXP-128 配方相近的历史样品，并结合历史资料。", ("identifier",)),
    AcceptanceCase("S1-03", 1, "查组分和 EXP-128 最接近的配方，并结合历史资料。", ("identifier",)),

    AcceptanceCase("S2-01", 2, "找与 EXP-128 工艺最像的样品，并结合历史资料。", ("identifier",)),
    AcceptanceCase("S2-02", 2, "查与 EXP-128 综合条件相近的实验，并结合历史案例。", ("identifier",)),
    AcceptanceCase("S2-03", 2, "查找和 EXP-128 配方性能都接近的历史样品，并结合资料。", ("identifier",)),

    AcceptanceCase(
        "S3-01",
        3,
        "查找密度差大于 100 且持液量小于 0.1 的历史样品，并结合历史资料。",
        ("filters",),
    ),
    AcceptanceCase(
        "S3-02",
        3,
        "查找密度差大于 150 的样品，并结合历史案例。",
        ("filters",),
    ),
    AcceptanceCase(
        "S3-03",
        3,
        "查找持液量在 0.05 到 0.1 之间的样品，并结合历史资料。",
        ("filters",),
    ),

    AcceptanceCase("S4-01", 4, "查 EXP-128 的完整研发上下文，并结合历史资料综合判断。", ("identifier",)),
    AcceptanceCase("S4-02", 4, "查看 EXP-127 的配方、工艺和性能，并结合历史资料。", ("identifier",)),
    AcceptanceCase("S4-03", 4, "查 EXP-129 对应样品和测试结果，并结合历史案例。", ("identifier",)),

    AcceptanceCase("S5-01", 5, "查水用在哪些样品、用量多少以及性能如何，并结合历史案例。", ("material_name",)),
    AcceptanceCase("S5-02", 5, "查询 P507+煤油 的原料使用效果，并结合历史资料。", ("material_name",)),
    AcceptanceCase("S5-03", 5, "使用水的样品性能怎么样？结合历史资料判断。", ("material_name",)),

    AcceptanceCase(
        "S6-01",
        6,
        "查找 P507+煤油 替代 水的历史记录，并结合历史资料。",
        ("original_material", "replacement_material"),
    ),
    AcceptanceCase(
        "S6-02",
        6,
        "有没有用水替换 P507+煤油 的配方记录？结合历史案例说明。",
        ("original_material", "replacement_material"),
    ),
    AcceptanceCase(
        "S6-03",
        6,
        "查这两种原料的替代后性能变化：P507+煤油 和 水，并结合资料。",
        ("original_material", "replacement_material"),
    ),

    AcceptanceCase("S7-01", 7, "查找失败配方和失败实验，并结合历史资料。"),
    AcceptanceCase("S7-02", 7, "以前类似萃取配方路线为什么失败？结合历史案例。"),
    AcceptanceCase("S7-03", 7, "查失败后调整过的实验记录，并结合内部历史资料。"),

    AcceptanceCase("S12-01", 12, "查找开裂异常案例，并结合历史资料。", ("phenomenon",)),
    AcceptanceCase("S12-02", 12, "查析出或变色的历史失效案例，并结合资料。", ("phenomenon",)),
    AcceptanceCase("S12-03", 12, "查粘接失效类似案例，并结合内部资料。", ("phenomenon",)),

    AcceptanceCase("S13-01", 13, "与竞品 ABS 的性能差距是多少？结合历史数据判断。", ("competitor_name",)),
    AcceptanceCase("S13-02", 13, "历史上哪些路线最接近竞品 PC/ABS？结合资料。", ("competitor_name",)),
    AcceptanceCase("S13-03", 13, "查竞品对标的内部样品和资料。"),

    AcceptanceCase(
        "S14-01",
        14,
        "新项目要求密度差大于 100 且持液量小于 0.1，请结合历史和失败记录给首轮方案。",
        ("filters",),
    ),
    AcceptanceCase(
        "S14-02",
        14,
        "新项目冷启动：目标密度差大于 150，请结合历史资料给第一轮方案。",
        ("filters",),
    ),
    AcceptanceCase(
        "S14-03",
        14,
        "新项目从零开始，请结合历史项目、失败记录和资料给首轮实验建议。",
    ),

    AcceptanceCase("S17-01", 17, "去年做过哪些方案？结合数据库和历史资料。"),
    AcceptanceCase("S17-02", 17, "这个项目里水为什么后来停用？结合历史资料。"),
    AcceptanceCase("S17-03", 17, "项目知识问答：当前授权项目的结论和风险有哪些？结合资料。"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage-3 read-only research scenarios against a live backend."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--platform-company-id", required=True)
    parser.add_argument("--business-company-id", required=True)
    parser.add_argument("--project-ids", default="*")
    parser.add_argument("--user-id", default="stage3-acceptance-user")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Only run the given case ID; may be repeated.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run_case(
    client: httpx.Client,
    *,
    case: AcceptanceCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    token = os.environ.get("STAGE3_PLATFORM_TOKEN", "").strip()
    if not token:
        raise RuntimeError("STAGE3_PLATFORM_TOKEN is required")

    headers = {
        "Authorization": f"Bearer {token}",
        "Company-Id": args.platform_company_id,
        "X-User-Id": args.user_id,
        "X-Company-Id": args.business_company_id,
        "X-Project-Ids": args.project_ids,
    }
    payload = {
        "message": case.message,
        "conversation_id": None,
        "client_message_id": f"stage3-{case.case_id.lower()}-{int(time.time())}",
    }
    started = time.monotonic()
    response = client.post(
        f"{args.base_url}/api/v1/chat-ui",
        headers=headers,
        json=payload,
        timeout=args.timeout,
    )
    elapsed = round(time.monotonic() - started, 3)
    if response.status_code != 200:
        return {
            "case_id": case.case_id,
            "expected_scenario_id": case.scenario_id,
            "message": case.message,
            "http_status": response.status_code,
            "elapsed_seconds": elapsed,
            "route_pass": False,
            "slot_pass": False,
            "evidence_pass": False,
            "report_pass": False,
            "error": response.text[:1000],
        }

    body = response.json()
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    workflow = data.get("research_workflow") or {}
    mysql = data.get("mysql_result") or {}
    vector = data.get("vector_result") or {}
    frame = data.get("evidence_frame") or {}
    synthesis = data.get("synthesis") or {}
    tool_args = body.get("tool_args") or {}
    source_summary = frame.get("source_summary") or {}
    scores = [
        float(hit.get("score"))
        for hit in vector.get("hits") or []
        if isinstance(hit, dict)
        and isinstance(hit.get("score"), (int, float))
    ]
    citations = sorted(set(_RECORD_ID.findall(str(body.get("answer") or ""))))
    scenario_id = workflow.get("scenario_id")
    route_pass = (
        body.get("intent") == "hybrid_research_qa"
        and scenario_id == case.scenario_id
        and workflow.get("execution_status") == "SUPPORTED"
    )
    slot_pass = all(
        _non_empty(tool_args.get(key)) for key in case.required_args
    )
    mysql_evidence_available = (
        str(mysql.get("status") or "") == "ok"
        or (
            int(source_summary.get("mysql", 0) or 0) > 0
            and len(mysql.get("warnings") or []) > 0
        )
    )
    evidence_pass = (
        mysql_evidence_available
        and vector.get("status") == "ok"
        and int(source_summary.get("mysql", 0) or 0) > 0
        and (
            int(source_summary.get("vector_api", 0) or 0) > 0
            or len(vector.get("warnings") or []) > 0
        )
    )
    report_pass = (
        synthesis.get("status") == "ok"
        and bool(citations)
        and bool(body.get("answer"))
    )
    return {
        "case_id": case.case_id,
        "expected_scenario_id": case.scenario_id,
        "message": case.message,
        "http_status": response.status_code,
        "elapsed_seconds": elapsed,
        "intent": body.get("intent"),
        "tool_args": tool_args,
        "workflow": {
            "workflow_id": workflow.get("workflow_id"),
            "scenario_id": scenario_id,
            "scenario_name": workflow.get("scenario_name"),
            "execution_status": workflow.get("execution_status"),
        },
        "structured_strategy": data.get("structured_strategy"),
        "mysql": {
            "status": mysql.get("status"),
            "analysis_type": mysql.get("analysis_type"),
            "count": mysql.get("count", mysql.get("total_matching_sample_count")),
            "warning_count": len(mysql.get("warnings") or []),
        },
        "vector": {
            "status": vector.get("status"),
            "hit_count": vector.get("hit_count"),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "warning_count": len(vector.get("warnings") or []),
        },
        "evidence_frame": {
            "source_summary": source_summary,
            "conflict_count": len(frame.get("conflicts") or []),
            "warning_count": len(frame.get("warnings") or []),
        },
        "source_relation": data.get("source_relation"),
        "synthesis": {
            "status": synthesis.get("status"),
            "mode": synthesis.get("mode"),
            "attempts": synthesis.get("attempts"),
        },
        "answer_chars": len(str(body.get("answer") or "")),
        "citation_count": len(citations),
        "warnings": body.get("warnings") or [],
        "route_pass": route_pass,
        "slot_pass": slot_pass,
        "evidence_pass": evidence_pass,
        "report_pass": report_pass,
        "pass": route_pass and slot_pass and evidence_pass and report_pass,
    }


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return bool(str(value).strip())


def main() -> None:
    args = parse_args()
    cases = CASES[: args.limit] if args.limit else CASES
    if args.only:
        selected = set(args.only)
        cases = [case for case in cases if case.case_id in selected]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {
                pool.submit(
                    run_case,
                    client,
                    case=case,
                    args=args,
                ): case
                for case in cases
            }
            for future in as_completed(futures):
                case = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "case_id": case.case_id,
                        "expected_scenario_id": case.scenario_id,
                        "message": case.message,
                        "route_pass": False,
                        "slot_pass": False,
                        "evidence_pass": False,
                        "report_pass": False,
                        "pass": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                print(
                    f"{row['case_id']} {'PASS' if row.get('pass') else 'FAIL'} "
                    f"scenario={((row.get('workflow') or {}).get('scenario_id'))}",
                    flush=True,
                )

    rows.sort(key=lambda item: item["case_id"])
    summary = {
        "schema_version": 1,
        "total": len(rows),
        "passed": sum(1 for row in rows if row.get("pass")),
        "failed": sum(1 for row in rows if not row.get("pass")),
        "by_scenario": {
            str(scenario_id): {
                "total": sum(
                    1 for row in rows if row.get("expected_scenario_id") == scenario_id
                ),
                "passed": sum(
                    1
                    for row in rows
                    if row.get("expected_scenario_id") == scenario_id and row.get("pass")
                ),
            }
            for scenario_id in sorted({row["expected_scenario_id"] for row in rows})
        },
        "rows": rows,
    }
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"summary: {summary['passed']}/{summary['total']} passed; "
        f"output={output}"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
