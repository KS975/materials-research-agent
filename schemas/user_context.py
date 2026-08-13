from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserContext:
    user_id: str
    company_id: str
    project_ids: tuple[int, ...]
    permission_source: str

    def can_access_project(self, project_id: int | None) -> bool:
        if project_id is None:
            return False
        return project_id in self.project_ids
