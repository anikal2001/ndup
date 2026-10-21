"""Density-based row template detector.

After chunk kNN, a chunk whose in-source neighbors cover at least 30% of that
data source's rows is boilerplate even when the hash does not match.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

import numpy as np

from ndup.model import DocumentDraft


def apply_density_templates(docs: list[DocumentDraft], tau_chunk: float) -> int:
    """Mark chunks and return how many were newly zeroed."""
    rows_in_source: dict[UUID, set[UUID]] = defaultdict(set)
    for doc in docs:
        if doc.kind == "db_row" and doc.data_source_id is not None and doc.status == "active":
            rows_in_source[doc.data_source_id].add(doc.id)
    grouped: dict[UUID, list[tuple[UUID, int]]] = defaultdict(list)
    vectors: dict[UUID, list[np.ndarray]] = defaultdict(list)
    owners: dict[UUID, list[tuple[DocumentDraft, int]]] = defaultdict(list)
    for doc in docs:
        if doc.data_source_id is None or doc.kind != "db_row":
            continue
        for index, chunk in enumerate(doc.chunks):
            if chunk.vector is None or chunk.weight <= 0 or chunk.is_boilerplate or chunk.is_synced:
                continue
            grouped[doc.data_source_id].append((doc.id, index))
            vectors[doc.data_source_id].append(np.asarray(chunk.vector, dtype=np.float32))
            owners[doc.data_source_id].append((doc, index))
    marked = 0
    for source, items in owners.items():
        population = len(rows_in_source.get(source, ()))
        if population == 0 or len(items) < 2:
            continue
        matrix = np.stack(vectors[source])
        sims = matrix @ matrix.T
        for row_index, (doc, chunk_index) in enumerate(items):
            neighbor_docs = {doc.id}
            for other_index, (other, _chunk_index) in enumerate(items):
                if other.id != doc.id and float(sims[row_index, other_index]) >= tau_chunk:
                    neighbor_docs.add(other.id)
            if len(neighbor_docs) / population >= 0.30:
                chunk = doc.chunks[chunk_index]
                if not chunk.is_boilerplate:
                    marked += 1
                chunk.is_boilerplate = True
                chunk.boilerplate_reason = "density"
                chunk.weight = 0.0
    return marked
