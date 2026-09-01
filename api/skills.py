from __future__ import annotations

from fastapi import APIRouter, Depends

from api.chat import resolve_user_context
from app.container import ApplicationContainer, get_container
from schemas.user_context import UserContext


router = APIRouter(prefix="/api/v1", tags=["skills"])


@router.get("/skills")
def list_skills(
    _: UserContext = Depends(resolve_user_context),
    container: ApplicationContainer = Depends(get_container),
):
    """Return the public, credential-free Skill Registry contract."""
    return {
        "version": "skill-registry-v1",
        "architecture": "intent -> scenario_composer -> atomic_skill -> tool_registry",
        "skills": container.skill_registry.list_skills(),
    }

