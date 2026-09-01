from __future__ import annotations

from demo import MondayDemoService


def build_monday_demo_overview() -> dict:
    report = MondayDemoService().status()
    external = [
        item
        for item in report["versions"]
        if item["version"] in {"V0.1.1", "V0.1.2"}
    ]
    internal = [
        item
        for item in report["versions"]
        if item["version"] not in {"V0.1.1", "V0.1.2"}
    ]

    if report["status"] == "READY":
        answer = (
            "周一 Demo Runtime 已准备完成："
            "V0.1.3 → V0.3 的建模、BO、反馈学习和自主实验闭环均有"
            "确定性验收结果；V0.1.1/V0.1.2 保留真实 MySQL / RAG 的 live 演示边界。"
        )
    else:
        answer = (
            "周一 Demo Runtime 尚未完整准备。"
            "请先运行 python -m scripts.prepare_monday_demo --reset。"
        )

    return {
        **report,
        "answer": answer,
        "external_versions": external,
        "internal_versions": internal,
    }
