"""Search is a hint. Stop at the watermark minus the lag buffer."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def edited_at(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("last_edited_time")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def past_watermark(
    edited: datetime | None,
    watermark: datetime | None,
    lag: timedelta,
) -> bool:
    """True when a descending search cursor has reached the hint boundary."""
    if watermark is None or edited is None:
        return False
    return edited < watermark - lag
