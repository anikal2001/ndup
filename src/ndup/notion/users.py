"""Notion user records. Unknown ids display as ``Unknown user (a1b2…)``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ndup.model import UserRow


def parse_user(payload: dict[str, Any], now: datetime | None = None) -> UserRow | None:
    try:
        user_id = UUID(str(payload.get("id")))
    except ValueError:
        return None
    person_raw = payload.get("person")
    person = person_raw if isinstance(person_raw, dict) else {}
    email_raw = person.get("email")
    email = email_raw if isinstance(email_raw, str) else None
    name = payload.get("name") if isinstance(payload.get("name"), str) else None
    return UserRow(
        id=user_id,
        type=str(payload.get("type")) if payload.get("type") else None,
        name=name,
        email=email,
        avatar_url=payload.get("avatar_url") if isinstance(payload.get("avatar_url"), str) else None,
        updated_at=now or datetime.now(UTC),
    )


def display_user(user: UserRow | None, user_id: UUID | None) -> str:
    if user is not None and user.name:
        return user.name
    if user_id is None:
        return "Unknown user"
    return f"Unknown user ({str(user_id)[:4]}…)"
