from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UserContext:
    user_id: str
    company_id: str
    project_ids: tuple[int, ...]
    permission_source: str
    # development_header may explicitly grant all projects inside the current
    # company. Company scope remains mandatory and is still enforced by every
    # business-MySQL repository.
    all_projects: bool = False

    def can_access_project(self, project_id: int | None) -> bool:
        if project_id is None:
            return False
        return self.all_projects or project_id in self.project_ids
