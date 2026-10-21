"""Process-local store for tests and for ``memory://`` development runs.

Postgres is the supported store. This one exists so the pipeline can be
tested without a server. A path after the scheme persists with pickle between
CLI invocations in the same working tree.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from ndup.analysis.stable_keys import review_after
from ndup.analysis.types import AnalysisResult, ClusterRecord, PairRecord, PrevCluster
from ndup.db.queue import choose_task, complete_task, fail_task, lease_task
from ndup.model import (
    DataSourceRow,
    DocumentDraft,
    IgnoreRule,
    ModelRow,
    ReviewRow,
    RootRow,
    SyncTask,
    UserRow,
)


class MemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.documents: dict[UUID, DocumentDraft] = {}
        self.tasks: dict[UUID, SyncTask] = {}
        self.roots: dict[UUID, RootRow] = {}
        self.data_sources: dict[UUID, DataSourceRow] = {}
        self.users: dict[UUID, UserRow] = {}
        self.state: dict[str, dict[str, Any]] = {}
        self.models: list[ModelRow] = []
        self.embeddings: dict[tuple[int, bytes], np.ndarray] = {}
        self.doc_source_hash: dict[tuple[int, UUID], bytes] = {}
        self.pairs: list[PairRecord] = []
        self.clusters: dict[UUID, ClusterRecord] = {}
        self.events: list[dict[str, Any]] = []
        self.reviews: dict[UUID, ReviewRow] = {}
        self.ignores: list[IgnoreRule] = []
        self.audit: list[dict[str, Any]] = []
        self._next_model = 1

    @classmethod
    def open(cls, url: str) -> MemoryStore:
        raw = url.removeprefix("memory://")
        path = Path(raw) if raw else None
        if path is not None and path.is_file():
            with path.open("rb") as handle:
                loaded = pickle.load(handle)
            if isinstance(loaded, MemoryStore):
                loaded.path = path
                return loaded
        return cls(path)

    def close(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as handle:
            pickle.dump(self, handle)

    def migrate(self) -> None:
        return None

    def get_state(self, key: str) -> dict[str, Any] | None:
        value = self.state.get(key)
        return dict(value) if value is not None else None

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        self.state[key] = dict(value)

    def upsert_root(self, root: RootRow) -> None:
        current = self.roots.get(root.id)
        if current is not None and current.kind == "configured":
            return
        self.roots[root.id] = root

    def list_roots(self) -> list[RootRow]:
        return list(self.roots.values())

    def upsert_data_source(self, source: DataSourceRow) -> None:
        self.data_sources[source.id] = source

    def list_data_sources(self) -> list[DataSourceRow]:
        return list(self.data_sources.values())

    def upsert_task(self, task: SyncTask) -> None:
        current = self.tasks.get(task.doc_id)
        if current is None:
            self.tasks[task.doc_id] = task
            return
        if task.priority < current.priority:
            current.priority = task.priority
        if current.state == "done" and task.discovered_via == "webhook":
            current.state = "pending"
            current.priority = task.priority
        current.remote_last_edited = task.remote_last_edited or current.remote_last_edited
        current.discovered_via = task.discovered_via or current.discovered_via

    def list_tasks(self) -> list[SyncTask]:
        return list(self.tasks.values())

    def known_ids(self) -> set[UUID]:
        return set(self.tasks) | set(self.documents)

    def claim_task(self, now: datetime, lease_for: timedelta) -> SyncTask | None:
        chosen = choose_task(list(self.tasks.values()), now)
        if chosen is None:
            return None
        lease_task(chosen, now, lease_for)
        return chosen

    def complete_task(self, doc_id: UUID) -> None:
        task = self.tasks.get(doc_id)
        if task is not None:
            complete_task(task)

    def fail_task(self, doc_id: UUID, error: str, now: datetime) -> None:
        task = self.tasks.get(doc_id)
        if task is not None:
            fail_task(task, error, now)

    def exclude_document(self, doc_id: UUID, discovered_via: str, now: datetime) -> None:
        self.documents[doc_id] = DocumentDraft(
            id=doc_id,
            kind="page",
            status="excluded",
            title=None,
            title_hash=None,
            markdown=None,
            properties={},
            discovered_via=discovered_via,
            first_seen_at=now,
            last_synced_at=now,
        )
        task = self.tasks.get(doc_id)
        if task is not None:
            task.state = "done"

    def commit_document(self, doc: DocumentDraft, now: datetime) -> None:
        previous = self.documents.get(doc.id)
        if (
            previous is not None
            and previous.content_hash is not None
            and previous.content_hash == doc.content_hash
            and previous.status == "active"
        ):
            previous.last_edited_time = doc.last_edited_time
            previous.last_edited_by = doc.last_edited_by
            previous.last_synced_at = now
            previous.area = doc.area or previous.area
            return
        if previous is not None and previous.content_hash not in {None, doc.content_hash}:
            doc.last_content_change = now
        elif previous is not None:
            doc.last_content_change = previous.last_content_change
        doc.first_seen_at = previous.first_seen_at if previous is not None else now
        doc.last_synced_at = now
        self.documents[doc.id] = doc

    def get_document(self, doc_id: UUID) -> DocumentDraft | None:
        return self.documents.get(doc_id)

    def list_documents(self) -> list[DocumentDraft]:
        return list(self.documents.values())

    def upsert_user(self, user: UserRow) -> None:
        self.users[user.id] = user

    def register_model(
        self,
        model: str,
        dims: int,
        runtime: str,
        thresholds: dict[str, Any],
    ) -> int:
        for existing in self.models:
            if existing.model == model and existing.dims == dims and existing.runtime == runtime:
                existing.thresholds = thresholds
                existing.is_active = True
                return existing.id
        row = ModelRow(
            id=self._next_model,
            model=model,
            dims=dims,
            runtime=runtime,
            thresholds=thresholds,
            is_active=True,
        )
        self._next_model += 1
        self.models.append(row)
        return row.id

    def get_model(self, model: str, runtime: str) -> ModelRow | None:
        for existing in self.models:
            if existing.model == model and existing.runtime == runtime:
                return existing
        return None

    def active_model(self) -> ModelRow | None:
        for existing in self.models:
            if existing.is_active:
                return existing
        return None

    def list_models(self) -> list[ModelRow]:
        return list(self.models)

    def put_embeddings(self, model_id: int, vectors: dict[bytes, np.ndarray]) -> None:
        for digest, vector in vectors.items():
            self.embeddings[(model_id, digest)] = np.asarray(vector, dtype=np.float32)

    def embedding_cache(self, model_id: int) -> dict[bytes, np.ndarray]:
        return {
            digest: vector for (stored_model, digest), vector in self.embeddings.items() if stored_model == model_id
        }

    def remember_source_hash(
        self,
        model_id: int,
        doc_id: UUID,
        digest: bytes,
        vector: np.ndarray | None = None,
    ) -> None:
        del vector
        self.doc_source_hash[(model_id, doc_id)] = digest

    def source_hash(self, model_id: int, doc_id: UUID) -> bytes | None:
        return self.doc_source_hash.get((model_id, doc_id))

    def gc_embeddings(self, model_id: int) -> int:
        live = {chunk.embed_hash for doc in self.documents.values() for chunk in doc.chunks}
        stale = [key for key in self.embeddings if key[0] == model_id and key[1] not in live]
        for key in stale:
            del self.embeddings[key]
        return len(stale)

    def load_previous_clusters(self) -> list[PrevCluster]:
        previous: list[PrevCluster] = []
        for cluster in self.clusters.values():
            review = self.reviews.get(cluster.stable_key)
            previous.append(
                PrevCluster(
                    stable_key=cluster.stable_key,
                    members={member.doc_id for member in cluster.members if not member.is_satellite},
                    created_at=cluster.created_at or datetime.now().astimezone(),
                    note=None if review is None else review.note,
                )
            )
        return previous

    def save_analysis(self, result: AnalysisResult, now: datetime) -> None:
        self.pairs = list(result.pairs)
        self.clusters = {cluster.stable_key: cluster for cluster in result.clusters}
        for event in result.events:
            self.events.append(
                {
                    "stable_key": event.stable_key,
                    "doc_id": event.doc_id,
                    "event": event.event,
                    "other_key": event.other_key,
                    "at": now,
                }
            )
        for key, note in result.review_notes.items():
            review = self.reviews.get(key)
            if review is None:
                self.reviews[key] = ReviewRow(
                    stable_key=key,
                    status="open",
                    reviewed_member_hash=None,
                    note=note,
                    reviewed_by=None,
                    reviewed_at=None,
                )
            else:
                review.note = f"{review.note}\n{note}".strip() if review.note else note
        for cluster in result.clusters:
            review = self.reviews.get(cluster.stable_key)
            if review is None:
                continue
            review.status = review_after(review.status, review.reviewed_member_hash, cluster.member_hash)

    def list_pairs(self) -> list[PairRecord]:
        return list(self.pairs)

    def list_clusters(self) -> list[ClusterRecord]:
        return list(self.clusters.values())

    def get_cluster(self, key: UUID) -> ClusterRecord | None:
        return self.clusters.get(key)

    def review(
        self,
        key: UUID,
        status: str,
        note: str | None,
        by: str | None,
        now: datetime,
    ) -> None:
        cluster = self.clusters.get(key)
        reviewed_hash = None if cluster is None else cluster.member_hash
        self.reviews[key] = ReviewRow(
            stable_key=key,
            status=status,
            reviewed_member_hash=reviewed_hash,
            note=note,
            reviewed_by=by,
            reviewed_at=now,
        )

    def ignore(self, kind: str, a: UUID, b: UUID | None, by: str | None, now: datetime) -> None:
        self.ignores.append(IgnoreRule(kind=kind, a=a, b=b, created_by=by, created_at=now))
        if kind == "doc":
            doc = self.documents.get(a)
            if doc is not None:
                doc.status = "ignored"
        if kind == "root":
            root = self.roots.get(a)
            if root is not None:
                root.excluded = True

    def list_ignores(self) -> list[IgnoreRule]:
        return list(self.ignores)

    def add_label(
        self,
        doc_a: UUID,
        doc_b: UUID,
        label: str,
        by: str | None,
        now: datetime,
    ) -> None:
        self.state.setdefault("labels", {})
        self.state["labels"][f"{doc_a}:{doc_b}"] = {"label": label, "by": by, "at": now.isoformat()}

    def list_labels(self) -> list[tuple[UUID, UUID, str]]:
        raw = self.state.get("labels", {})
        found: list[tuple[UUID, UUID, str]] = []
        if not isinstance(raw, dict):
            return found
        for key, value in raw.items():
            left, right = key.split(":", 1)
            text = value.get("label") if isinstance(value, dict) else None
            if isinstance(text, str):
                found.append((UUID(left), UUID(right), text))
        return found

    def purge(self, doc_id: UUID, now: datetime) -> None:
        doc = self.documents.get(doc_id)
        if doc is None:
            return
        hashes = [chunk.embed_hash for chunk in doc.chunks]
        touched = [
            key for key, cluster in self.clusters.items() if any(member.doc_id == doc_id for member in cluster.members)
        ]
        doc.markdown = None
        doc.properties = {}
        doc.title = None
        doc.title_hash = doc.title_hash
        doc.breadcrumb = []
        doc.contributors = []
        doc.chunks = []
        doc.blocks = []
        doc.links = []
        doc.status = "gone"
        doc.last_synced_at = now
        self.pairs = [pair for pair in self.pairs if doc_id not in {pair.doc_a, pair.doc_b}]
        for key in touched:
            cluster = self.clusters[key]
            cluster.members = [member for member in cluster.members if member.doc_id != doc_id]
            self.events.append({"stable_key": key, "doc_id": doc_id, "event": "left", "other_key": None, "at": now})
            if not cluster.members:
                del self.clusters[key]
                continue
            from ndup.analysis.stable_keys import member_hash

            cluster.member_hash = member_hash([member.doc_id for member in cluster.members])
            cluster.size = len(cluster.members)
            review = self.reviews.get(key)
            if review is not None:
                review.status = review_after(review.status, review.reviewed_member_hash, cluster.member_hash)
        active = self.active_model()
        self.gc_embeddings(active.id if active is not None else 0)
        del hashes
        self.audit.append(
            {
                "at": now,
                "user_email": None,
                "action": "purge",
                "target_kind": "document",
                "target_id": doc_id,
            }
        )

    def add_audit(
        self,
        action: str,
        target_kind: str,
        target_id: UUID,
        user: str | None,
        now: datetime,
    ) -> None:
        self.audit.append(
            {
                "at": now,
                "user_email": user,
                "action": action,
                "target_kind": target_kind,
                "target_id": target_id,
            }
        )

    def task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks.values():
            counts[task.state] = counts.get(task.state, 0) + 1
        return counts

    def dump(self, *, state_only: bool) -> dict[str, Any]:
        state: dict[str, Any] = {
            "roots": [row.__dict__ for row in self.roots.values()],
            "reviews": {str(key): row.__dict__ for key, row in self.reviews.items()},
            "ignores": [row.__dict__ for row in self.ignores],
            "audit": self.audit,
            "sync_state": self.state,
        }
        if state_only:
            return state
        state["documents"] = len(self.documents)
        state["pairs"] = len(self.pairs)
        state["clusters"] = len(self.clusters)
        return state
