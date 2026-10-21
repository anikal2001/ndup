"""Core clusters on undirected near-copy and duplicate edges.

Subsets attach as satellites and never merge two clusters.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ndup.analysis.relations import EvidenceView, subset_direction
from ndup.analysis.types import MemberRecord, PairRecord


def build_clusters(
    pairs: list[PairRecord],
) -> list[tuple[set[UUID], list[MemberRecord]]]:
    """Return ``(core_members, member records including satellites)`` per cluster."""
    graph: nx.Graph[UUID] = nx.Graph()
    for pair in pairs:
        if pair.relation in {"near_copy", "duplicate"}:
            graph.add_edge(pair.doc_a, pair.doc_b, score=pair.score)
    components = [set(nodes) for nodes in nx.connected_components(graph) if len(nodes) >= 2]
    cores: list[set[UUID]] = []
    for component in components:
        cores.extend(_density_split(component, graph))

    core_of: dict[UUID, list[int]] = defaultdict(list)
    for index, core in enumerate(cores):
        for doc_id in core:
            core_of[doc_id].append(index)

    satellites: dict[int, dict[UUID, str]] = defaultdict(dict)
    for pair in pairs:
        if pair.relation != "subset":
            continue
        direction = subset_direction(
            EvidenceView(
                cov_ab=pair.cov_ab,
                cov_ba=pair.cov_ba,
                lex_jaccard=pair.lex_jaccard,
                lex_cont_ab=pair.lex_cont_ab,
                lex_cont_ba=pair.lex_cont_ba,
                doc_sim=pair.doc_sim,
                median_match=pair.median_match,
                shared_hash=False,
                weighted_chunks_a=0,
                weighted_chunks_b=0,
            )
        )
        if direction is None:
            continue
        subset_id, superset_id = (pair.doc_a, pair.doc_b) if direction == "a" else (pair.doc_b, pair.doc_a)
        if subset_id in core_of:
            continue
        for cluster_index in core_of.get(superset_id, []):
            satellites[cluster_index][subset_id] = "subset"

    grouped: list[tuple[set[UUID], list[MemberRecord]]] = []
    for index, core in enumerate(cores):
        members: list[MemberRecord] = []
        for doc_id in sorted(core, key=lambda item: item.int):
            members.append(
                MemberRecord(
                    doc_id=doc_id,
                    relation_to_cluster=_core_relation(doc_id, core, pairs),
                    is_satellite=False,
                )
            )
        for doc_id, relation in sorted(satellites[index].items(), key=lambda item: item[0].int):
            members.append(MemberRecord(doc_id=doc_id, relation_to_cluster=relation, is_satellite=True))
        grouped.append((core, members))
    return grouped


def _core_relation(doc_id: UUID, core: set[UUID], pairs: list[PairRecord]) -> str:
    relations: set[str] = set()
    for pair in pairs:
        if pair.relation not in {"near_copy", "duplicate"}:
            continue
        other: UUID | None = None
        if pair.doc_a == doc_id and pair.doc_b in core:
            other = pair.doc_b
        elif pair.doc_b == doc_id and pair.doc_a in core:
            other = pair.doc_a
        if other is not None:
            relations.add(pair.relation)
    if "near_copy" in relations:
        return "near_copy"
    if "duplicate" in relations:
        return "duplicate"
    return "duplicate"


def _density_split(nodes: set[UUID], graph: nx.Graph[UUID]) -> list[set[UUID]]:
    if len(nodes) <= 4 or _density(nodes, graph) >= 0.5:
        return [nodes]
    ids = sorted(nodes, key=lambda item: item.int)
    size = len(ids)
    distance = np.ones((size, size), dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    index_of = {doc_id: index for index, doc_id in enumerate(ids)}
    subgraph = graph.subgraph(ids)
    for left, right, data in subgraph.edges(data=True):
        i = index_of[left]
        j = index_of[right]
        score = float(data.get("score", 0.0))
        distance[i, j] = distance[j, i] = max(0.0, 1.0 - score)
    if size < 2:
        return [nodes]
    labels = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)
    left_set = {ids[i] for i, label in enumerate(labels) if int(label) == 0}
    right_set = {ids[i] for i, label in enumerate(labels) if int(label) == 1}
    if not left_set or not right_set or left_set == nodes or right_set == nodes:
        return [nodes]
    return _density_split(left_set, graph) + _density_split(right_set, graph)


def _density(nodes: set[UUID], graph: nx.Graph[UUID]) -> float:
    size = len(nodes)
    if size < 2:
        return 1.0
    possible = size * (size - 1) / 2
    present = int(graph.subgraph(nodes).number_of_edges())
    return present / possible
