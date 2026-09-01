from __future__ import annotations

import os
from pathlib import Path


def resolve_company_data_runtime_root(
    runtime_root: str | Path | None = None,
) -> Path:
    """Resolve the same company-data runtime root everywhere.

    Priority:
    1. explicit function/CLI value
    2. COMPANY_DATA_RUNTIME_ROOT environment variable
    3. <project-root>/.runtime

    This intentionally matches the API/UI company-data runtime behavior.
    """
    if runtime_root is not None:
        text = str(runtime_root).strip()
        if text:
            return Path(text).expanduser().resolve()

    override = os.getenv("COMPANY_DATA_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    return Path(__file__).resolve().parents[1] / ".runtime"
