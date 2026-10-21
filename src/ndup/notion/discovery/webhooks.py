"""Webhook signature check and event-to-task mapping.

The HTTP receiver is Phase 2. These functions are the part that can be tested
without a public URL.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class WebhookAction:
    entity_id: UUID | None
    kind: str
    priority: int
    delete: bool


def verify_signature(body: bytes, header: str, secret: str) -> bool:
    """Notion signs the raw body with HMAC-SHA256, hex-encoded."""
    if not header or not secret:
        return False
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    provided = header.strip()
    if provided.startswith("sha256="):
        provided = provided.split("=", 1)[1]
    return hmac.compare_digest(digest, provided)


def action_for(event_type: str, entity_id: UUID | None) -> WebhookAction | None:
    deleted = event_type in {"page.deleted", "database.deleted"}
    kind = "page"
    if event_type.startswith("data_source"):
        kind = "data_source"
    elif event_type.startswith("database"):
        kind = "database"
    if event_type in {
        "page.content_updated",
        "page.created",
        "page.deleted",
        "page.undeleted",
        "page.moved",
        "page.properties_updated",
        "data_source.content_updated",
        "data_source.schema_updated",
        "database.deleted",
    }:
        return WebhookAction(entity_id=entity_id, kind=kind, priority=0, delete=deleted)
    return None
