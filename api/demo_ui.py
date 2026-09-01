from __future__ import annotations

from fastapi import APIRouter

from runtime.monday_demo_ui import build_monday_demo_overview


router = APIRouter(
    prefix="/api/v1/demo-ui",
    tags=["demo-ui"],
)


@router.get("/status")
def demo_status():
    data = build_monday_demo_overview()
    return {
        "answer": data["answer"],
        "data": data,
    }
