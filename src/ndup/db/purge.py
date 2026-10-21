"""Document purge. One transaction, no leftover text or vectors.

``ndup purge --doc`` calls :meth:`purge`. The memory store and Postgres store
both implement it. Title text is not written to the audit row.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID


class PurgeStore(Protocol):
    def purge(self, doc_id: UUID, now: datetime) -> None: ...


def purge_document(store: PurgeStore, doc_id: UUID, now: datetime) -> None:
    store.purge(doc_id, now)


def embedding_rows(store: Any, model_id: int) -> int:
    cache = store.embedding_cache(model_id)
    return len(cache)
