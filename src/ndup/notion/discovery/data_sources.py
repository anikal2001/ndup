"""Data-source enumeration and the adaptive poll interval."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID


def data_sources_of(database: dict[str, Any]) -> list[tuple[UUID, str | None]]:
    raw = database.get("data_sources")
    found: list[tuple[UUID, str | None]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                source_id = UUID(str(item.get("id")))
            except ValueError:
                continue
            name = item.get("name")
            found.append((source_id, str(name) if isinstance(name, str) else None))
    return found


def next_poll_interval(
    last_change: datetime | None,
    now: datetime,
    *,
    hot_window: timedelta = timedelta(days=7),
    cold: timedelta = timedelta(days=1),
) -> timedelta:
    """Hourly while the source changed inside the hot window, otherwise daily.

    Bounded to [1 hour, 7 days].
    """
    if last_change is not None and now - last_change <= hot_window:
        interval = timedelta(hours=1)
    else:
        interval = cold
    return min(max(interval, timedelta(hours=1)), timedelta(days=7))
