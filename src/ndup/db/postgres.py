"""Postgres store. The supported backend for solo and team installs.

The memory store mirrors this surface so unit tests do not need a server.
HNSW indexes are created when a model is registered, not in the migration.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
from psycopg.types.json import Jsonb

from ndup.analysis.stable_keys import review_after
from ndup.analysis.types import (
    AnalysisResult,
    ClusterRecord,
    MemberRecord,
    PairRecord,
    PassageMatch,
    PrevCluster,
)
from ndup.db.migrate import connect
from ndup.db.migrate import migrate as apply_migrations
from ndup.db.migrate import migration_status as migration_rows
from ndup.db.queue import complete_task, fail_task, lease_task
from ndup.embed.registry import hnsw_index_sql
from ndup.model import (
    BlockRecord,
    ChunkDraft,
    DataSourceRow,
    DocumentDraft,
    IgnoreRule,
    ModelRow,
    ReviewRow,
    RootRow,
    SyncTask,
    TokenSpan,
    UserRow,
)

_DOC_COLUMNS = (
    "id",
    "kind",
    "parent_type",
    "parent_id",
    "data_source_id",
    "root_id",
    "area",
    "breadcrumb",
    "title",
    "title_hash",
    "url",
    "created_time",
    "created_by",
    "last_edited_time",
    "last_edited_by",
    "last_content_change",
    "properties",
    "markdown",
    "content_hash",
    "token_count",
    "lang",
    "contributors",
    "is_stub",
    "status",
    "discovered_via",
    "first_seen_at",
    "last_synced_at",
)


class PostgresStore:
    def __init__(self, conn: Any, url: str) -> None:
        self.conn = conn
        self.url = url

    @classmethod
    def open(cls, url: str) -> PostgresStore:
        return cls(connect(url), url)

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> list[str]:
        return apply_migrations(self.conn, with_roles=True)

    def migration_status(self) -> list[tuple[str, bool]]:
        return migration_rows(self.conn)

    def pgvector_version(self) -> str | None:
        row = self.conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        self.conn.rollback()
        if row is None:
            return None
        return str(row["extversion"])

    def chunks_missing_hash(self) -> int:
        row = self.conn.execute(
            "SELECT count(*) AS n FROM chunk WHERE text_hash IS NULL OR embed_hash IS NULL"
        ).fetchone()
        self.conn.rollback()
        return int(row["n"]) if row else 0

    def get_state(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT value FROM sync_state WHERE key = %s", (key,)).fetchone()
        self.conn.rollback()
        if row is None or not isinstance(row["value"], dict):
            return None
        return dict(row["value"])

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO sync_state (key, value) VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (key, Jsonb(value)),
        )
        self.conn.commit()

    def upsert_root(self, root: RootRow) -> None:
        current = self.conn.execute("SELECT kind FROM root WHERE id = %s", (root.id,)).fetchone()
        if current is not None and current["kind"] == "configured":
            self.conn.rollback()
            return
        self.conn.execute(
            """
            INSERT INTO root (id, label, kind, excluded, created_at)
            VALUES (%s, %s, %s, %s, COALESCE(%s, now()))
            ON CONFLICT (id) DO UPDATE
              SET label = EXCLUDED.label, kind = EXCLUDED.kind, excluded = EXCLUDED.excluded
              WHERE root.kind <> 'configured'
            """,
            (root.id, root.label, root.kind, root.excluded, root.created_at),
        )
        self.conn.commit()

    def list_roots(self) -> list[RootRow]:
        rows = self.conn.execute("SELECT * FROM root").fetchall()
        self.conn.rollback()
        return [
            RootRow(
                id=row["id"],
                label=row["label"],
                kind=row["kind"],
                excluded=bool(row["excluded"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def upsert_data_source(self, source: DataSourceRow) -> None:
        self.conn.execute(
            """
            INSERT INTO data_source (
              id, database_id, title, row_count, template_title, poll_interval,
              next_poll_at, last_polled_at, last_change_seen, watermark
            ) VALUES (%s, %s, %s, %s, %s, make_interval(secs => %s), %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              database_id = EXCLUDED.database_id,
              title = EXCLUDED.title,
              row_count = EXCLUDED.row_count,
              template_title = EXCLUDED.template_title,
              poll_interval = EXCLUDED.poll_interval,
              next_poll_at = EXCLUDED.next_poll_at,
              last_polled_at = EXCLUDED.last_polled_at,
              last_change_seen = EXCLUDED.last_change_seen,
              watermark = EXCLUDED.watermark
            """,
            (
                source.id,
                source.database_id,
                source.title,
                source.row_count,
                source.template_title,
                source.poll_interval_seconds,
                source.next_poll_at,
                source.last_polled_at,
                source.last_change_seen,
                source.watermark,
            ),
        )
        self.conn.commit()

    def list_data_sources(self) -> list[DataSourceRow]:
        rows = self.conn.execute("SELECT * FROM data_source").fetchall()
        self.conn.rollback()
        found: list[DataSourceRow] = []
        for row in rows:
            interval = row["poll_interval"]
            seconds = int(interval.total_seconds()) if isinstance(interval, timedelta) else 3600
            found.append(
                DataSourceRow(
                    id=row["id"],
                    database_id=row["database_id"],
                    title=row["title"],
                    row_count=int(row["row_count"] or 0),
                    template_title=row["template_title"],
                    poll_interval_seconds=seconds,
                    next_poll_at=row["next_poll_at"],
                    last_polled_at=row["last_polled_at"],
                    last_change_seen=row["last_change_seen"],
                    watermark=row["watermark"],
                )
            )
        return found

    def upsert_task(self, task: SyncTask) -> None:
        with self.conn.transaction():
            current = self._task(task.doc_id)
            if current is None:
                self._insert_task(task)
                return
            if task.priority < current.priority:
                current.priority = task.priority
            if current.state == "done" and task.discovered_via == "webhook":
                current.state = "pending"
                current.priority = task.priority
            current.remote_last_edited = task.remote_last_edited or current.remote_last_edited
            current.discovered_via = task.discovered_via or current.discovered_via
            self._update_task(current)

    def list_tasks(self) -> list[SyncTask]:
        rows = self.conn.execute("SELECT * FROM sync_task").fetchall()
        self.conn.rollback()
        return [_task_from_row(row) for row in rows]

    def known_ids(self) -> set[UUID]:
        rows = self.conn.execute("SELECT doc_id AS id FROM sync_task UNION SELECT id FROM document").fetchall()
        self.conn.rollback()
        return {row["id"] for row in rows}

    def claim_task(self, now: datetime, lease_for: timedelta) -> SyncTask | None:
        with self.conn.transaction():
            row = self.conn.execute(
                """
                SELECT * FROM sync_task
                WHERE (
                  state = 'pending'
                  AND (next_attempt_at IS NULL OR next_attempt_at <= %s)
                ) OR (
                  state = 'in_progress'
                  AND leased_until IS NOT NULL
                  AND leased_until <= %s
                )
                ORDER BY priority ASC, remote_last_edited DESC NULLS LAST, doc_id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (now, now),
            ).fetchone()
            if row is None:
                return None
            task = _task_from_row(row)
            lease_task(task, now, lease_for)
            self._update_task(task)
            return task

    def complete_task(self, doc_id: UUID) -> None:
        with self.conn.transaction():
            task = self._task(doc_id)
            if task is None:
                return
            complete_task(task)
            self._update_task(task)

    def fail_task(self, doc_id: UUID, error: str, now: datetime) -> None:
        with self.conn.transaction():
            task = self._task(doc_id)
            if task is None:
                return
            fail_task(task, error, now)
            self._update_task(task)

    def exclude_document(self, doc_id: UUID, discovered_via: str, now: datetime) -> None:
        with self.conn.transaction():
            self.conn.execute("DELETE FROM chunk WHERE doc_id = %s", (doc_id,))
            self.conn.execute("DELETE FROM block WHERE doc_id = %s", (doc_id,))
            self.conn.execute("DELETE FROM doc_link WHERE src = %s", (doc_id,))
            self.conn.execute(
                """
                INSERT INTO document (
                  id, kind, title, markdown, properties, status, discovered_via,
                  first_seen_at, last_synced_at, is_stub
                ) VALUES (%s, 'page', NULL, NULL, '{}'::jsonb, 'excluded', %s, %s, %s, false)
                ON CONFLICT (id) DO UPDATE SET
                  title = NULL,
                  markdown = NULL,
                  properties = '{}'::jsonb,
                  status = 'excluded',
                  last_synced_at = EXCLUDED.last_synced_at
                """,
                (doc_id, discovered_via, now, now),
            )
            task = self._task(doc_id)
            if task is not None:
                complete_task(task)
                self._update_task(task)

    def commit_document(self, doc: DocumentDraft, now: datetime) -> None:
        with self.conn.transaction():
            previous = self.get_document(doc.id, _commit=False)
            if (
                previous is not None
                and previous.content_hash is not None
                and previous.content_hash == doc.content_hash
                and previous.status == "active"
            ):
                self.conn.execute(
                    """
                    UPDATE document SET
                      last_edited_time = %s,
                      last_edited_by = %s,
                      last_synced_at = %s,
                      area = COALESCE(%s, area)
                    WHERE id = %s
                    """,
                    (doc.last_edited_time, doc.last_edited_by, now, doc.area, doc.id),
                )
                return
            if previous is not None and previous.content_hash not in {None, doc.content_hash}:
                doc.last_content_change = now
            elif previous is not None:
                doc.last_content_change = previous.last_content_change
            doc.first_seen_at = previous.first_seen_at if previous is not None else now
            doc.last_synced_at = now
            self._write_document(doc)
            self.conn.execute("DELETE FROM chunk WHERE doc_id = %s", (doc.id,))
            self.conn.execute("DELETE FROM block WHERE doc_id = %s", (doc.id,))
            self.conn.execute("DELETE FROM doc_link WHERE src = %s", (doc.id,))
            for chunk in doc.chunks:
                chunk.chunk_id = self._insert_chunk(doc.id, chunk)
            for block in doc.blocks:
                self._insert_block(doc.id, block)
            for dst in doc.links:
                self.conn.execute(
                    "INSERT INTO doc_link (src, dst) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (doc.id, dst),
                )
            if doc.content_hash is not None:
                self.conn.execute(
                    """
                    INSERT INTO document_version (
                      doc_id, content_hash, last_edited_time, last_edited_by, synced_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (doc.id, doc.content_hash, doc.last_edited_time, doc.last_edited_by, now),
                )

    def get_document(self, doc_id: UUID, *, _commit: bool = True) -> DocumentDraft | None:
        row = self.conn.execute("SELECT * FROM document WHERE id = %s", (doc_id,)).fetchone()
        if row is None:
            if _commit:
                self.conn.rollback()
            return None
        chunks = self.conn.execute(
            "SELECT * FROM chunk WHERE doc_id = %s ORDER BY ord",
            (doc_id,),
        ).fetchall()
        blocks = self.conn.execute("SELECT * FROM block WHERE doc_id = %s", (doc_id,)).fetchall()
        links = self.conn.execute("SELECT dst FROM doc_link WHERE src = %s", (doc_id,)).fetchall()
        if _commit:
            self.conn.rollback()
        return _document_from_row(row, chunks, blocks, [item["dst"] for item in links])

    def list_documents(self) -> list[DocumentDraft]:
        rows = self.conn.execute("SELECT * FROM document").fetchall()
        chunk_rows = self.conn.execute("SELECT * FROM chunk ORDER BY doc_id, ord").fetchall()
        block_rows = self.conn.execute("SELECT * FROM block").fetchall()
        link_rows = self.conn.execute("SELECT src, dst FROM doc_link").fetchall()
        self.conn.rollback()
        chunks: dict[UUID, list[Any]] = {}
        for row in chunk_rows:
            chunks.setdefault(row["doc_id"], []).append(row)
        blocks: dict[UUID, list[Any]] = {}
        for row in block_rows:
            blocks.setdefault(row["doc_id"], []).append(row)
        links: dict[UUID, list[UUID]] = {}
        for row in link_rows:
            links.setdefault(row["src"], []).append(row["dst"])
        return [
            _document_from_row(
                row,
                chunks.get(row["id"], []),
                blocks.get(row["id"], []),
                links.get(row["id"], []),
            )
            for row in rows
        ]

    def upsert_user(self, user: UserRow) -> None:
        self.conn.execute(
            """
            INSERT INTO notion_user (id, type, name, email, avatar_url, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              type = EXCLUDED.type, name = EXCLUDED.name, email = EXCLUDED.email,
              avatar_url = EXCLUDED.avatar_url, updated_at = EXCLUDED.updated_at
            """,
            (user.id, user.type, user.name, user.email, user.avatar_url, user.updated_at),
        )
        self.conn.commit()

    def register_model(
        self,
        model: str,
        dims: int,
        runtime: str,
        thresholds: dict[str, Any],
    ) -> int:
        row = self.conn.execute(
            """
            INSERT INTO embedding_model (model, dims, runtime, storage, is_active, thresholds)
            VALUES (%s, %s, %s, 'halfvec', true, %s)
            ON CONFLICT (model, dims, runtime) DO UPDATE SET
              thresholds = EXCLUDED.thresholds,
              is_active = true
            RETURNING id
            """,
            (model, dims, runtime, Jsonb(thresholds)),
        ).fetchone()
        model_id = int(row["id"])
        self.conn.execute(
            "UPDATE embedding_model SET is_active = false WHERE id <> %s",
            (model_id,),
        )
        self.conn.commit()
        for table in ("embedding", "doc_embedding"):
            self.conn.execute(hnsw_index_sql(table=table, model_id=model_id, dims=dims))
        self.conn.commit()
        return model_id

    def get_model(self, model: str, runtime: str) -> ModelRow | None:
        row = self.conn.execute(
            """
            SELECT * FROM embedding_model
            WHERE model = %s AND runtime = %s
            ORDER BY id DESC LIMIT 1
            """,
            (model, runtime),
        ).fetchone()
        self.conn.rollback()
        return None if row is None else _model_from_row(row)

    def active_model(self) -> ModelRow | None:
        row = self.conn.execute("SELECT * FROM embedding_model WHERE is_active ORDER BY id DESC LIMIT 1").fetchone()
        self.conn.rollback()
        return None if row is None else _model_from_row(row)

    def list_models(self) -> list[ModelRow]:
        rows = self.conn.execute("SELECT * FROM embedding_model ORDER BY id").fetchall()
        self.conn.rollback()
        return [_model_from_row(row) for row in rows]

    def put_embeddings(self, model_id: int, vectors: dict[bytes, np.ndarray]) -> None:
        with self.conn.transaction():
            for digest, vector in vectors.items():
                self.conn.execute(
                    """
                    INSERT INTO embedding (model_id, embed_hash, vec)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (model_id, embed_hash) DO UPDATE SET vec = EXCLUDED.vec
                    """,
                    (model_id, digest, _half(vector)),
                )

    def embedding_cache(self, model_id: int) -> dict[bytes, np.ndarray]:
        rows = self.conn.execute(
            "SELECT embed_hash, vec FROM embedding WHERE model_id = %s",
            (model_id,),
        ).fetchall()
        self.conn.rollback()
        return {bytes(row["embed_hash"]): _vec(row["vec"]) for row in rows}

    def remember_source_hash(
        self,
        model_id: int,
        doc_id: UUID,
        digest: bytes,
        vector: np.ndarray | None = None,
    ) -> None:
        model = self.conn.execute(
            "SELECT dims FROM embedding_model WHERE id = %s",
            (model_id,),
        ).fetchone()
        dims = int(model["dims"]) if model else 768
        if vector is None:
            vector = np.zeros(dims, dtype=np.float32)
        self.conn.execute(
            """
            INSERT INTO doc_embedding (model_id, doc_id, vec, source_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_id, doc_id) DO UPDATE SET
              vec = EXCLUDED.vec, source_hash = EXCLUDED.source_hash
            """,
            (model_id, doc_id, _half(np.asarray(vector, dtype=np.float32)), digest),
        )
        self.conn.commit()

    def source_hash(self, model_id: int, doc_id: UUID) -> bytes | None:
        row = self.conn.execute(
            "SELECT source_hash FROM doc_embedding WHERE model_id = %s AND doc_id = %s",
            (model_id, doc_id),
        ).fetchone()
        self.conn.rollback()
        if row is None or row["source_hash"] is None:
            return None
        return bytes(row["source_hash"])

    def gc_embeddings(self, model_id: int) -> int:
        removed = self.conn.execute(
            """
            DELETE FROM embedding e
            WHERE e.model_id = %s
              AND NOT EXISTS (SELECT 1 FROM chunk c WHERE c.embed_hash = e.embed_hash)
            """,
            (model_id,),
        ).rowcount
        self.conn.execute(
            """
            DELETE FROM doc_embedding d
            WHERE d.model_id = %s
              AND NOT EXISTS (
                SELECT 1 FROM document doc WHERE doc.id = d.doc_id AND doc.status = 'active'
              )
            """,
            (model_id,),
        )
        self.conn.commit()
        return int(removed)

    def load_previous_clusters(self) -> list[PrevCluster]:
        clusters = self.list_clusters()
        reviews = {review.stable_key: review for review in self._reviews()}
        previous: list[PrevCluster] = []
        for cluster in clusters:
            review = reviews.get(cluster.stable_key)
            created = cluster.created_at or datetime.now().astimezone()
            previous.append(
                PrevCluster(
                    stable_key=cluster.stable_key,
                    members={member.doc_id for member in cluster.members if not member.is_satellite},
                    created_at=created,
                    note=None if review is None else review.note,
                )
            )
        self.conn.rollback()
        return previous

    def save_analysis(self, result: AnalysisResult, now: datetime) -> None:
        with self.conn.transaction():
            model = self.conn.execute(
                "SELECT id FROM embedding_model WHERE is_active ORDER BY id DESC LIMIT 1"
            ).fetchone()
            model_id = None if model is None else int(model["id"])
            run = self.conn.execute(
                """
                INSERT INTO analysis_run (model_id, mode, params, started_at, finished_at, stats)
                VALUES (%s, 'full', %s, %s, %s, %s)
                RETURNING id
                """,
                (model_id, Jsonb(result.thresholds), now, now, Jsonb(result.stats)),
            ).fetchone()
            run_id = int(run["id"])
            self.conn.execute("DELETE FROM passage_match")
            self.conn.execute("DELETE FROM doc_pair")
            self.conn.execute("DELETE FROM cluster_member")
            self.conn.execute("DELETE FROM cluster")
            for pair in result.pairs:
                self.conn.execute(
                    """
                    INSERT INTO doc_pair (
                      doc_a, doc_b, doc_sim, cov_ab, cov_ba, lex_jaccard, lex_cont_ab,
                      lex_cont_ba, score, relation, first_run, last_run
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        pair.doc_a,
                        pair.doc_b,
                        pair.doc_sim,
                        pair.cov_ab,
                        pair.cov_ba,
                        pair.lex_jaccard,
                        pair.lex_cont_ab,
                        pair.lex_cont_ba,
                        pair.score,
                        pair.relation,
                        run_id,
                        run_id,
                    ),
                )
                for passage in pair.passages:
                    self.conn.execute(
                        """
                        INSERT INTO passage_match (doc_a, doc_b, chunk_a, chunk_b, sim, lex)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            pair.doc_a,
                            pair.doc_b,
                            passage.chunk_a,
                            passage.chunk_b,
                            passage.sim,
                            passage.lex,
                        ),
                    )
            for cluster in result.clusters:
                self.conn.execute(
                    """
                    INSERT INTO cluster (
                      stable_key, label, size, member_hash, oldest_doc, newest_doc,
                      active_copies, areas, last_run, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        cluster.stable_key,
                        cluster.label,
                        cluster.size,
                        cluster.member_hash,
                        cluster.oldest_doc,
                        cluster.newest_doc,
                        cluster.active_copies,
                        cluster.areas,
                        run_id,
                        now,
                    ),
                )
                for member in cluster.members:
                    self.conn.execute(
                        """
                        INSERT INTO cluster_member (
                          stable_key, doc_id, relation_to_cluster, is_satellite
                        ) VALUES (%s, %s, %s, %s)
                        """,
                        (
                            cluster.stable_key,
                            member.doc_id,
                            member.relation_to_cluster,
                            member.is_satellite,
                        ),
                    )
            for event in result.events:
                self.conn.execute(
                    """
                    INSERT INTO cluster_event (stable_key, doc_id, event, run_id, at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (event.stable_key, event.doc_id, event.event, run_id, now),
                )
            reviews = {review.stable_key: review for review in self._reviews()}
            for key, note in result.review_notes.items():
                review = reviews.get(key)
                if review is None:
                    reviews[key] = ReviewRow(key, "open", None, note, None, None)
                else:
                    review.note = f"{review.note}\n{note}".strip() if review.note else note
            for cluster in result.clusters:
                review = reviews.get(cluster.stable_key)
                if review is None:
                    continue
                review.status = review_after(review.status, review.reviewed_member_hash, cluster.member_hash)
                self._write_review(review)

    def list_clusters(self) -> list[ClusterRecord]:
        rows = self.conn.execute("SELECT * FROM cluster").fetchall()
        members = self.conn.execute("SELECT * FROM cluster_member").fetchall()
        first_seen = self.conn.execute(
            "SELECT stable_key, min(at) AS at FROM cluster_event GROUP BY stable_key"
        ).fetchall()
        self.conn.rollback()
        grouped: dict[UUID, list[MemberRecord]] = {}
        for row in members:
            grouped.setdefault(row["stable_key"], []).append(
                MemberRecord(
                    doc_id=row["doc_id"],
                    relation_to_cluster=row["relation_to_cluster"] or "related",
                    is_satellite=bool(row["is_satellite"]),
                )
            )
        created = {row["stable_key"]: row["at"] for row in first_seen}
        return [
            ClusterRecord(
                stable_key=row["stable_key"],
                label=row["label"] or "",
                size=int(row["size"] or 0),
                member_hash=bytes(row["member_hash"] or b""),
                oldest_doc=row["oldest_doc"],
                newest_doc=row["newest_doc"],
                active_copies=int(row["active_copies"] or 0),
                areas=list(row["areas"] or []),
                members=grouped.get(row["stable_key"], []),
                created_at=created.get(row["stable_key"], row["updated_at"]),
            )
            for row in rows
        ]

    def get_cluster(self, key: UUID) -> ClusterRecord | None:
        for cluster in self.list_clusters():
            if cluster.stable_key == key:
                return cluster
        return None

    def list_pairs(self) -> list[PairRecord]:
        rows = self.conn.execute("SELECT * FROM doc_pair").fetchall()
        passages = self.conn.execute("SELECT * FROM passage_match").fetchall()
        self.conn.rollback()
        grouped: dict[tuple[UUID, UUID], list[PassageMatch]] = {}
        for row in passages:
            grouped.setdefault((row["doc_a"], row["doc_b"]), []).append(
                PassageMatch(
                    chunk_a=int(row["chunk_a"]),
                    chunk_b=int(row["chunk_b"]),
                    sim=float(row["sim"] or 0),
                    lex=float(row["lex"] or 0),
                )
            )
        return [
            PairRecord(
                doc_a=row["doc_a"],
                doc_b=row["doc_b"],
                doc_sim=float(row["doc_sim"] or 0),
                cov_ab=float(row["cov_ab"] or 0),
                cov_ba=float(row["cov_ba"] or 0),
                lex_jaccard=float(row["lex_jaccard"] or 0),
                lex_cont_ab=float(row["lex_cont_ab"] or 0),
                lex_cont_ba=float(row["lex_cont_ba"] or 0),
                score=float(row["score"] or 0),
                relation=row["relation"] or "related",
                median_match=0.0,
                passages=grouped.get((row["doc_a"], row["doc_b"]), []),
            )
            for row in rows
        ]

    def review(
        self,
        key: UUID,
        status: str,
        note: str | None,
        by: str | None,
        now: datetime,
    ) -> None:
        cluster = self.get_cluster(key)
        reviewed_hash = None if cluster is None else cluster.member_hash
        self._write_review(
            ReviewRow(
                stable_key=key,
                status=status,
                reviewed_member_hash=reviewed_hash,
                note=note,
                reviewed_by=by,
                reviewed_at=now,
            )
        )
        self.conn.commit()

    def ignore(self, kind: str, a: UUID, b: UUID | None, by: str | None, now: datetime) -> None:
        with self.conn.transaction():
            self.conn.execute(
                """
                INSERT INTO ignore_rule (kind, a, b, created_by, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (kind, a, b) DO NOTHING
                """,
                (kind, a, b, by, now),
            )
            if kind == "doc":
                self.conn.execute("UPDATE document SET status = 'ignored' WHERE id = %s", (a,))
            if kind == "root":
                self.conn.execute("UPDATE root SET excluded = true WHERE id = %s", (a,))

    def list_ignores(self) -> list[IgnoreRule]:
        rows = self.conn.execute("SELECT * FROM ignore_rule").fetchall()
        self.conn.rollback()
        return [
            IgnoreRule(
                kind=row["kind"],
                a=row["a"],
                b=row["b"],
                created_by=row["created_by"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_label(
        self,
        doc_a: UUID,
        doc_b: UUID,
        label: str,
        by: str | None,
        now: datetime,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO calibration_label (doc_a, doc_b, label, labeled_by, labeled_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_a, doc_b) DO UPDATE SET
              label = EXCLUDED.label, labeled_by = EXCLUDED.labeled_by, labeled_at = EXCLUDED.labeled_at
            """,
            (doc_a, doc_b, label, by, now),
        )
        self.conn.commit()

    def list_labels(self) -> list[tuple[UUID, UUID, str]]:
        rows = self.conn.execute("SELECT doc_a, doc_b, label FROM calibration_label").fetchall()
        self.conn.rollback()
        return [(row["doc_a"], row["doc_b"], row["label"]) for row in rows]

    def purge(self, doc_id: UUID, now: datetime) -> None:
        with self.conn.transaction():
            doc = self.conn.execute("SELECT id FROM document WHERE id = %s", (doc_id,)).fetchone()
            if doc is None:
                return
            touched = self.conn.execute(
                "SELECT stable_key FROM cluster_member WHERE doc_id = %s",
                (doc_id,),
            ).fetchall()
            self.conn.execute(
                """
                UPDATE document SET
                  markdown = NULL, properties = '{}'::jsonb, title = NULL,
                  breadcrumb = '{}', contributors = '{}', status = 'gone', last_synced_at = %s
                WHERE id = %s
                """,
                (now, doc_id),
            )
            self.conn.execute("DELETE FROM chunk WHERE doc_id = %s", (doc_id,))
            self.conn.execute("DELETE FROM block WHERE doc_id = %s", (doc_id,))
            self.conn.execute("DELETE FROM doc_link WHERE src = %s OR dst = %s", (doc_id, doc_id))
            self.conn.execute("DELETE FROM doc_pair WHERE doc_a = %s OR doc_b = %s", (doc_id, doc_id))
            self.conn.execute("DELETE FROM cluster_member WHERE doc_id = %s", (doc_id,))
            for row in touched:
                key = row["stable_key"]
                self.conn.execute(
                    """
                    INSERT INTO cluster_event (stable_key, doc_id, event, at)
                    VALUES (%s, %s, 'left', %s)
                    """,
                    (key, doc_id, now),
                )
                remaining = self.conn.execute(
                    "SELECT doc_id FROM cluster_member WHERE stable_key = %s",
                    (key,),
                ).fetchall()
                if not remaining:
                    self.conn.execute("DELETE FROM cluster WHERE stable_key = %s", (key,))
                    continue
                from ndup.analysis.stable_keys import member_hash

                digest = member_hash([item["doc_id"] for item in remaining])
                self.conn.execute(
                    "UPDATE cluster SET member_hash = %s, size = %s, updated_at = %s WHERE stable_key = %s",
                    (digest, len(remaining), now, key),
                )
                review = self.conn.execute(
                    "SELECT * FROM cluster_review WHERE stable_key = %s",
                    (key,),
                ).fetchone()
                if review is not None:
                    status = review_after(review["status"], _bytes(review["reviewed_member_hash"]), digest)
                    self.conn.execute(
                        "UPDATE cluster_review SET status = %s WHERE stable_key = %s",
                        (status, key),
                    )
            model = self.conn.execute(
                "SELECT id FROM embedding_model WHERE is_active ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if model is not None:
                self.conn.execute(
                    """
                    DELETE FROM embedding e
                    WHERE e.model_id = %s
                      AND NOT EXISTS (SELECT 1 FROM chunk c WHERE c.embed_hash = e.embed_hash)
                    """,
                    (int(model["id"]),),
                )
                self.conn.execute(
                    "DELETE FROM doc_embedding WHERE doc_id = %s",
                    (doc_id,),
                )
            self.conn.execute(
                """
                INSERT INTO audit_log (at, user_email, action, target_kind, target_id)
                VALUES (%s, NULL, 'purge', 'document', %s)
                """,
                (now, doc_id),
            )

    def add_audit(
        self,
        action: str,
        target_kind: str,
        target_id: UUID,
        user: str | None,
        now: datetime,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO audit_log (at, user_email, action, target_kind, target_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (now, user, action, target_kind, target_id),
        )
        self.conn.commit()

    def task_counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT state, count(*) AS n FROM sync_task GROUP BY state").fetchall()
        self.conn.rollback()
        return {str(row["state"]): int(row["n"]) for row in rows}

    def reindex(self) -> int:
        models = self.list_models()
        self.conn.execute("SET max_parallel_maintenance_workers = 4")
        for model in models:
            self.conn.execute(f"DROP INDEX IF EXISTS emb_hnsw_m{int(model.id)}")
            self.conn.execute(f"DROP INDEX IF EXISTS docemb_hnsw_m{int(model.id)}")
            self.conn.execute(hnsw_index_sql(table="embedding", model_id=model.id, dims=model.dims))
            self.conn.execute(hnsw_index_sql(table="doc_embedding", model_id=model.id, dims=model.dims))
        self.conn.commit()
        return len(models)

    def explain_doc_knn(self, model_id: int, dims: int, vector: np.ndarray) -> str:
        self.conn.execute("SET hnsw.ef_search = 100")
        self.conn.execute("SET hnsw.iterative_scan = relaxed_order")
        self.conn.execute("SET enable_seqscan = off")
        half = _half(vector)
        rows = self.conn.execute(
            f"""
            EXPLAIN SELECT doc_id
            FROM doc_embedding
            WHERE model_id = {int(model_id)}
            ORDER BY vec::halfvec({int(dims)}) <=> %s
            LIMIT 60
            """,
            (half,),
        ).fetchall()
        self.conn.rollback()
        return "\n".join(" ".join(str(value) for value in row.values()) for row in rows)

    def dump(self, *, state_only: bool) -> dict[str, Any]:
        payload = {
            "roots": [_jsonable(dict(row)) for row in self.conn.execute("SELECT * FROM root").fetchall()],
            "reviews": [_jsonable(dict(row)) for row in self.conn.execute("SELECT * FROM cluster_review").fetchall()],
            "ignores": [_jsonable(dict(row)) for row in self.conn.execute("SELECT * FROM ignore_rule").fetchall()],
            "labels": [_jsonable(dict(row)) for row in self.conn.execute("SELECT * FROM calibration_label").fetchall()],
            "audit": [_jsonable(dict(row)) for row in self.conn.execute("SELECT * FROM audit_log").fetchall()],
            "sync_state": [_jsonable(dict(row)) for row in self.conn.execute("SELECT * FROM sync_state").fetchall()],
        }
        if not state_only:
            payload["documents"] = self.conn.execute("SELECT count(*) AS n FROM document").fetchone()["n"]
            payload["pairs"] = self.conn.execute("SELECT count(*) AS n FROM doc_pair").fetchone()["n"]
            payload["clusters"] = self.conn.execute("SELECT count(*) AS n FROM cluster").fetchone()["n"]
            payload["embeddings"] = self.conn.execute("SELECT count(*) AS n FROM embedding").fetchone()["n"]
        self.conn.rollback()
        return payload

    def _task(self, doc_id: UUID) -> SyncTask | None:
        row = self.conn.execute("SELECT * FROM sync_task WHERE doc_id = %s", (doc_id,)).fetchone()
        return None if row is None else _task_from_row(row)

    def _insert_task(self, task: SyncTask) -> None:
        self.conn.execute(
            """
            INSERT INTO sync_task (
              doc_id, kind, priority, remote_last_edited, discovered_via, state,
              attempts, next_attempt_at, last_error, leased_until
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                task.doc_id,
                task.kind,
                task.priority,
                task.remote_last_edited,
                task.discovered_via,
                task.state,
                task.attempts,
                task.next_attempt_at,
                task.last_error,
                task.leased_until,
            ),
        )

    def _update_task(self, task: SyncTask) -> None:
        self.conn.execute(
            """
            UPDATE sync_task SET
              kind = %s, priority = %s, remote_last_edited = %s, discovered_via = %s,
              state = %s, attempts = %s, next_attempt_at = %s, last_error = %s, leased_until = %s
            WHERE doc_id = %s
            """,
            (
                task.kind,
                task.priority,
                task.remote_last_edited,
                task.discovered_via,
                task.state,
                task.attempts,
                task.next_attempt_at,
                task.last_error,
                task.leased_until,
                task.doc_id,
            ),
        )

    def _write_document(self, doc: DocumentDraft) -> None:
        values = (
            doc.id,
            doc.kind,
            doc.parent_type,
            doc.parent_id,
            doc.data_source_id,
            doc.root_id,
            doc.area,
            doc.breadcrumb,
            doc.title,
            doc.title_hash,
            doc.url,
            doc.created_time,
            doc.created_by,
            doc.last_edited_time,
            doc.last_edited_by,
            doc.last_content_change,
            Jsonb(doc.properties),
            doc.markdown,
            doc.content_hash,
            doc.token_count,
            doc.lang,
            doc.contributors,
            doc.is_stub,
            doc.status,
            doc.discovered_via,
            doc.first_seen_at,
            doc.last_synced_at,
        )
        assignments = ", ".join(f"{column} = EXCLUDED.{column}" for column in _DOC_COLUMNS if column != "id")
        placeholders = ", ".join(["%s"] * len(_DOC_COLUMNS))
        self.conn.execute(
            f"""
            INSERT INTO document ({", ".join(_DOC_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT (id) DO UPDATE SET {assignments}
            """,
            values,
        )

    def _insert_chunk(self, doc_id: UUID, chunk: ChunkDraft) -> int:
        row = self.conn.execute(
            """
            INSERT INTO chunk (
              doc_id, ord, heading_path, text, embed_text, start_char, end_char, block_ids,
              span_ids, span_tokens, token_count, text_hash, embed_hash, minhash, lang,
              weight, is_boilerplate, boilerplate_reason, is_synced, earliest_block_time
            ) VALUES (
              %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                doc_id,
                chunk.ord,
                chunk.heading_path,
                chunk.text,
                chunk.embed_text,
                chunk.start_char,
                chunk.end_char,
                chunk.block_ids,
                [span.span_id for span in chunk.spans],
                [span.tokens for span in chunk.spans],
                chunk.token_count,
                chunk.text_hash,
                chunk.embed_hash,
                chunk.minhash,
                chunk.lang,
                chunk.weight,
                chunk.is_boilerplate,
                chunk.boilerplate_reason,
                chunk.is_synced,
                chunk.earliest_block_time,
            ),
        ).fetchone()
        return int(row["id"])

    def _insert_block(self, doc_id: UUID, block: BlockRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO block (
              id, doc_id, type, start_char, end_char, created_time, created_by,
              last_edited_time, last_edited_by, is_synced
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              doc_id = EXCLUDED.doc_id, type = EXCLUDED.type,
              start_char = EXCLUDED.start_char, end_char = EXCLUDED.end_char
            """,
            (
                block.id,
                doc_id,
                block.type,
                block.start_char,
                block.end_char,
                block.created_time,
                block.created_by,
                block.last_edited_time,
                block.last_edited_by,
                block.is_synced,
            ),
        )

    def _reviews(self) -> list[ReviewRow]:
        rows = self.conn.execute("SELECT * FROM cluster_review").fetchall()
        return [
            ReviewRow(
                stable_key=row["stable_key"],
                status=row["status"] or "open",
                reviewed_member_hash=_bytes(row["reviewed_member_hash"]),
                note=row["note"],
                reviewed_by=row["reviewed_by"],
                reviewed_at=row["reviewed_at"],
            )
            for row in rows
        ]

    def _write_review(self, review: ReviewRow) -> None:
        self.conn.execute(
            """
            INSERT INTO cluster_review (
              stable_key, status, reviewed_member_hash, note, reviewed_by, reviewed_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stable_key) DO UPDATE SET
              status = EXCLUDED.status,
              reviewed_member_hash = EXCLUDED.reviewed_member_hash,
              note = EXCLUDED.note,
              reviewed_by = EXCLUDED.reviewed_by,
              reviewed_at = EXCLUDED.reviewed_at
            """,
            (
                review.stable_key,
                review.status,
                review.reviewed_member_hash,
                review.note,
                review.reviewed_by,
                review.reviewed_at,
            ),
        )


def _task_from_row(row: dict[str, Any]) -> SyncTask:
    return SyncTask(
        doc_id=row["doc_id"],
        kind=row["kind"],
        priority=int(row["priority"]),
        remote_last_edited=row["remote_last_edited"],
        discovered_via=row["discovered_via"],
        state=row["state"] or "pending",
        attempts=int(row["attempts"] or 0),
        next_attempt_at=row["next_attempt_at"],
        last_error=row["last_error"],
        leased_until=row["leased_until"],
    )


def _model_from_row(row: dict[str, Any]) -> ModelRow:
    thresholds = row["thresholds"] if isinstance(row["thresholds"], dict) else {}
    return ModelRow(
        id=int(row["id"]),
        model=row["model"],
        dims=int(row["dims"]),
        runtime=row["runtime"],
        thresholds=thresholds,
        is_active=bool(row["is_active"]),
    )


def _document_from_row(
    row: dict[str, Any],
    chunks: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    links: list[UUID],
) -> DocumentDraft:
    return DocumentDraft(
        id=row["id"],
        kind=row["kind"],
        parent_type=row["parent_type"],
        parent_id=row["parent_id"],
        data_source_id=row["data_source_id"],
        root_id=row["root_id"],
        area=row["area"],
        breadcrumb=list(row["breadcrumb"] or []),
        title=row["title"],
        title_hash=_bytes(row["title_hash"]),
        url=row["url"],
        created_time=row["created_time"],
        created_by=row["created_by"],
        last_edited_time=row["last_edited_time"],
        last_edited_by=row["last_edited_by"],
        last_content_change=row["last_content_change"],
        properties=dict(row["properties"] or {}),
        markdown=row["markdown"],
        content_hash=_bytes(row["content_hash"]),
        token_count=int(row["token_count"] or 0),
        lang=row["lang"],
        contributors=list(row["contributors"] or []),
        is_stub=bool(row["is_stub"]),
        status=row["status"],
        discovered_via=row["discovered_via"],
        first_seen_at=row["first_seen_at"],
        last_synced_at=row["last_synced_at"],
        chunks=[_chunk_from_row(item) for item in chunks],
        blocks=[
            BlockRecord(
                id=item["id"],
                type=item["type"] or "",
                start_char=int(item["start_char"] or 0),
                end_char=int(item["end_char"] or 0),
                created_time=item["created_time"],
                created_by=item["created_by"],
                last_edited_time=item["last_edited_time"],
                last_edited_by=item["last_edited_by"],
                is_synced=bool(item["is_synced"]),
            )
            for item in blocks
        ],
        links=list(links),
    )


def _chunk_from_row(row: dict[str, Any]) -> ChunkDraft:
    span_ids = list(row["span_ids"] or [])
    span_tokens = list(row["span_tokens"] or [])
    return ChunkDraft(
        ord=int(row["ord"]),
        heading_path=list(row["heading_path"] or []),
        text=row["text"] or "",
        embed_text=row["embed_text"] or "",
        start_char=int(row["start_char"] or 0),
        end_char=int(row["end_char"] or 0),
        block_ids=list(row["block_ids"] or []),
        spans=[
            TokenSpan(span_id=str(span_id), block_id=None, tokens=int(tokens))
            for span_id, tokens in zip(span_ids, span_tokens, strict=False)
        ],
        token_count=int(row["token_count"] or 0),
        text_hash=bytes(row["text_hash"]),
        embed_hash=bytes(row["embed_hash"]),
        minhash=bytes(row["minhash"] or b""),
        lang=row["lang"] or "",
        weight=float(1 if row["weight"] is None else row["weight"]),
        is_boilerplate=bool(row["is_boilerplate"]),
        boilerplate_reason=row["boilerplate_reason"],
        is_synced=bool(row["is_synced"]),
        earliest_block_time=row["earliest_block_time"],
        chunk_id=int(row["id"]),
    )


def _half(vector: np.ndarray) -> Any:
    from pgvector import HalfVector

    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return HalfVector(array)


def _vec(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    return bytes(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes | memoryview):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
