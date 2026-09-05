from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any


class ToolAuditWriteError(RuntimeError):
    pass


class JsonlToolAuditStore:
    """Append-only, company/user-scoped Tool call audit logs.

    The store is deliberately host-neutral: it records argument names and
    outcomes, never business values or tool payloads.
    """

    def __init__(self, root: str | Path, *, retries: int = 3) -> None:
        self.root = Path(root)
        self.retries = max(1, min(int(retries), 5))
        self._lock = threading.RLock()

    def record(self, event: dict[str, Any]) -> None:
        user_id = str(event.get("user_id") or "unknown")
        company_id = str(event.get("company_id") or "unknown")
        scope_raw = f"{company_id}\0{user_id}".encode("utf-8")
        scope = hashlib.sha256(scope_raw).hexdigest()[:24]
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.root / scope / f"tool-audit-{day}.jsonl"
        line = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )

        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            last_error: OSError | None = None
            for attempt in range(1, self.retries + 1):
                try:
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    return
                except OSError as exc:
                    last_error = exc
            raise ToolAuditWriteError(
                f"Tool 审计写入失败：{type(last_error).__name__}: {last_error}"
            ) from last_error
