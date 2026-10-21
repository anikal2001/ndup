"""Volume lease so two pods cannot start Postgres on one data directory.

Phase 2 adds the heartbeat supervisor. This is the rule the entrypoint uses.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from ndup.errors import NdUpError


class LeaseError(NdUpError):
    """Another live pod holds the volume lease."""


def acquire_lease(
    path: Path,
    pod_id: str,
    now: datetime,
    *,
    stale_after: timedelta = timedelta(minutes=2),
) -> None:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        holder = str(raw.get("pod_id") or "")
        heartbeat_raw = raw.get("heartbeat")
        heartbeat = datetime.fromisoformat(str(heartbeat_raw)) if heartbeat_raw else None
        if holder and holder != pod_id and heartbeat is not None and now - heartbeat < stale_after:
            raise LeaseError(f"Lease held by pod {holder}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pod_id": pod_id, "heartbeat": now.isoformat()}),
        encoding="utf-8",
    )


def lease_held_by(path: Path, pod_id: str, now: datetime, stale_after: timedelta) -> bool:
    if not path.is_file():
        return False
    raw = json.loads(path.read_text(encoding="utf-8"))
    if str(raw.get("pod_id") or "") != pod_id:
        return False
    heartbeat = datetime.fromisoformat(str(raw["heartbeat"]))
    return now - heartbeat < stale_after
