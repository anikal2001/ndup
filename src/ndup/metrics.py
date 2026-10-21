"""Prometheus counters shared by the workers and ``ndup doctor``."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

queue_depth = Gauge(
    "ndup_queue_depth",
    "Sync tasks by state",
    ["state"],
    registry=REGISTRY,
)
notion_requests = Counter(
    "ndup_notion_requests_total",
    "Notion API requests",
    registry=REGISTRY,
)
rate_limited = Counter(
    "ndup_notion_rate_limited_total",
    "429 and 529 responses",
    ["status"],
    registry=REGISTRY,
)
bucket_pause_seconds = Counter(
    "ndup_bucket_pause_seconds_total",
    "Seconds the token bucket was paused",
    registry=REGISTRY,
)
chunks_embedded = Counter(
    "ndup_chunks_embedded_total",
    "Chunk texts sent to the local embedder",
    registry=REGISTRY,
)
embed_cache_hits = Counter(
    "ndup_embed_cache_hits_total",
    "Chunks served from the embedding cache",
    registry=REGISTRY,
)
analysis_seconds = Histogram(
    "ndup_analysis_seconds",
    "Analysis stage duration",
    ["stage"],
    registry=REGISTRY,
)
pairs_current = Gauge(
    "ndup_pairs",
    "Current doc pairs",
    registry=REGISTRY,
)
clusters_current = Gauge(
    "ndup_clusters",
    "Current clusters",
    registry=REGISTRY,
)
false_neighbor_rate = Gauge(
    "ndup_false_neighbor_rate",
    "Mean kNN neighbors above tau that fail evidence",
    registry=REGISTRY,
)
backup_age_seconds = Gauge(
    "ndup_backup_age_seconds",
    "Age of the last successful backup",
    registry=REGISTRY,
)
lease_age_seconds = Gauge(
    "ndup_lease_age_seconds",
    "Age of the pod volume lease heartbeat",
    registry=REGISTRY,
)


def render_metrics() -> str:
    from prometheus_client import generate_latest

    payload = generate_latest(REGISTRY)
    return payload.decode("utf-8")
