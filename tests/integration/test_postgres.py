"""WP0.3 schema, roles, HNSW, and purge. Skips without Postgres."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
import pytest

from ndup.db.postgres import PostgresStore
from ndup.model import DocumentDraft

pytestmark = pytest.mark.integration

URL = os.environ.get("NDUP_TEST_DATABASE_URL")


@pytest.fixture
def store() -> PostgresStore:
    if not URL:
        pytest.skip("NDUP_TEST_DATABASE_URL is not set")
    opened = PostgresStore.open(URL)
    opened.migrate()
    opened.conn.execute(
        """
        TRUNCATE document, data_source, root, embedding, doc_embedding, embedding_model,
          doc_pair, cluster, cluster_member, cluster_review, sync_task, audit_log
        RESTART IDENTITY CASCADE
        """
    )
    opened.conn.commit()
    yield opened
    opened.close()


def test_migrate_is_idempotent(store: PostgresStore) -> None:
    assert store.pgvector_version() is not None
    again = store.migrate()
    assert again == []


def test_web_role_cannot_delete(store: PostgresStore) -> None:
    import psycopg

    store.conn.execute(
        "INSERT INTO document (id, kind, title) VALUES (%s, 'page', 'secret')",
        (UUID(int=1),),
    )
    store.conn.commit()
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with store.conn.transaction():
            store.conn.execute("SET LOCAL ROLE ndup_web")
            store.conn.execute("DELETE FROM document")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with store.conn.transaction():
            store.conn.execute("SET LOCAL ROLE ndup_web")
            store.conn.execute("UPDATE document SET title = 'changed'")


def test_hnsw_plan_and_purge(store: PostgresStore) -> None:
    model_id = store.register_model("fake-bow-v1", 8, "onnx-int8", {"tau_chunk": 0.7})
    vector = np.ones(8, dtype=np.float32)
    doc_id = UUID(int=2)
    store.commit_document(
        DocumentDraft(
            id=doc_id,
            kind="page",
            title="Page",
            markdown="body text",
            status="active",
            content_hash=b"hash-hash-hash-hash-hash-hash",
        ),
        datetime.now(UTC),
    )
    store.put_embeddings(model_id, {b"embed-hash-value!!": vector})
    store.remember_source_hash(model_id, doc_id, b"source", vector)
    plan = store.explain_doc_knn(model_id, 8, vector)
    assert "hnsw" in plan.lower() or "index" in plan.lower()
    store.purge(doc_id, datetime.now(UTC))
    assert store.embedding_cache(model_id) == {}
    stored = store.get_document(doc_id)
    assert stored is not None
    assert stored.markdown is None
    assert stored.title is None
