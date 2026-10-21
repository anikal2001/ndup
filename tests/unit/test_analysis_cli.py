"""Relations, clustering, calibration, doctor, purge, and the CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import numpy as np
import pytest
import respx
from typer.testing import CliRunner

from ndup.analysis.calibrate import LabeledExample, fit_thresholds, p999_floor
from ndup.analysis.clustering import build_clusters
from ndup.analysis.relations import EvidenceView, classify
from ndup.analysis.stable_keys import member_hash
from ndup.analysis.thresholds import Thresholds, apply_floor
from ndup.analysis.types import ClusterRecord, MemberRecord, PairRecord
from ndup.cli.main import app, compose_argv
from ndup.db.memory import MemoryStore
from ndup.db.purge import embedding_rows, purge_document
from ndup.doctor import DoctorContext, evaluate, failed
from ndup.embed.fake_provider import FAKE_MODEL
from ndup.model import ChunkDraft, DocumentDraft, ReviewRow, order_pair
from ndup.notion.discovery.webhooks import verify_signature
from ndup.pod_lease import LeaseError, acquire_lease

NOW = datetime(2024, 6, 1, tzinfo=UTC)
RUNNER = CliRunner()


def _pair(
    left: UUID,
    right: UUID,
    relation: str,
    *,
    cov_ab: float,
    cov_ba: float,
    lex_ab: float = 0.0,
    lex_ba: float = 0.0,
) -> PairRecord:
    doc_a, doc_b = order_pair(left, right)
    if doc_a != left:
        cov_ab, cov_ba = cov_ba, cov_ab
        lex_ab, lex_ba = lex_ba, lex_ab
    return PairRecord(
        doc_a=doc_a,
        doc_b=doc_b,
        doc_sim=0.9,
        cov_ab=cov_ab,
        cov_ba=cov_ba,
        lex_jaccard=0.9 if relation == "near_copy" else 0.2,
        lex_cont_ab=lex_ab,
        lex_cont_ba=lex_ba,
        score=0.9,
        relation=relation,
        median_match=0.9,
    )


@pytest.mark.unit
def test_short_doc_rule_and_satellites() -> None:
    short = EvidenceView(0.95, 0.95, 0.1, 0.1, 0.1, 0.9, 0.7, False, 1, 4)
    thresholds = Thresholds(0.70, 0.66, short_doc_delta=0.05)
    assert classify(short, thresholds) == "related"
    lexical = EvidenceView(0.95, 0.95, 0.2, 0.8, 0.8, 0.9, 0.9, False, 1, 1)
    assert classify(lexical, thresholds) == "near_copy"
    a, b, c, d, satellite = (UUID(int=index) for index in range(1, 6))
    groups = build_clusters(
        [
            _pair(a, b, "near_copy", cov_ab=0.95, cov_ba=0.95),
            _pair(c, d, "near_copy", cov_ab=0.95, cov_ba=0.95),
            _pair(satellite, a, "subset", cov_ab=0.9, cov_ba=0.1, lex_ab=0.9, lex_ba=0.1),
            _pair(satellite, c, "subset", cov_ab=0.9, cov_ba=0.1, lex_ab=0.9, lex_ba=0.1),
        ]
    )
    cores = [core for core, _members in groups]
    assert len(cores) == 2
    assert all(not ({a, b} <= core and {c, d} <= core) for core in cores)
    satellites = [member.doc_id for _core, members in groups for member in members if member.is_satellite]
    assert satellite in satellites


@pytest.mark.unit
def test_fit_is_stable_and_floor_applies() -> None:
    positive = LabeledExample(
        EvidenceView(0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.9, False, 4, 4),
        "near_copy",
        0.9,
        True,
        False,
    )
    negative = LabeledExample(
        EvidenceView(0.95, 0.95, 0.1, 0.2, 0.2, 0.5, 0.6, False, 4, 4),
        "partial",
        0.5,
        True,
        False,
    )
    examples = [positive] * 30 + [negative] * 30
    fitted = [fit_thresholds(examples, seed=seed, subsample=0.9) for seed in (0, 1, 2)]
    assert max(item.tau_chunk for item in fitted) - min(item.tau_chunk for item in fitted) <= 0.005
    raised = apply_floor(Thresholds(0.70, 0.66), 0.85)
    assert raised.tau_chunk == 0.85
    assert raised.p999_floor == 0.85
    untouched = apply_floor(Thresholds(0.70, 0.66), 0.40)
    assert untouched.tau_chunk == 0.70


@pytest.mark.unit
def test_p999_floor_on_orthogonal_vectors() -> None:
    docs = []
    for index in range(8):
        vector = np.zeros(8, dtype=np.float32)
        vector[index] = 1.0
        chunk = ChunkDraft(
            ord=0,
            heading_path=[],
            text="x",
            embed_text="x",
            start_char=0,
            end_char=1,
            block_ids=[],
            spans=[],
            token_count=1,
            text_hash=b"t",
            embed_hash=b"e" + bytes([index]),
            minhash=b"",
            lang="en",
            weight=1,
            is_boilerplate=False,
            boilerplate_reason=None,
            is_synced=False,
            earliest_block_time=None,
            vector=vector,
        )
        docs.append(DocumentDraft(id=UUID(int=index + 1), kind="page", lang="en", chunks=[chunk]))
    assert p999_floor(docs, sample_pairs=20, seed=0) == 0.0


@pytest.mark.unit
def test_doctor_faults_and_lease(tmp_path: Path) -> None:
    healthy = evaluate(DoctorContext())
    assert failed(healthy) == []
    faults = [
        DoctorContext(database_ok=False),
        DoctorContext(pgvector_version="0.7.0"),
        DoctorContext(migrations_ok=False),
        DoctorContext(model_registered=False),
        DoctorContext(chunks_missing_hash=2),
        DoctorContext(token_present=False),
        DoctorContext(false_neighbor_rate=0.5),
        DoctorContext(
            profile="pod",
            lease_held=False,
            volume_writable=True,
            tunnel_connected=True,
            postgres_tcp=False,
            backup_age_hours=1,
        ),
    ]
    for context in faults:
        assert failed(evaluate(context))
    lease = tmp_path / "ndup.lease"
    acquire_lease(lease, "pod-a", NOW)
    with pytest.raises(LeaseError):
        acquire_lease(lease, "pod-b", NOW + timedelta(seconds=10))
    acquire_lease(lease, "pod-b", NOW + timedelta(minutes=5))


@pytest.mark.unit
def test_webhook_signature_and_compose_files() -> None:
    import hashlib
    import hmac

    body = b'{"type":"page.updated"}'
    secret = "hook-secret"
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_signature(body, digest, secret)
    assert verify_signature(body, "sha256=" + digest, secret)
    assert not verify_signature(body, digest, "other")
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    solo = (root / "docker-compose.solo.yml").read_text(encoding="utf-8")
    assert "internal: true" in compose
    assert "ports:" in compose
    web = compose.split("web:", 1)[1].split("proxy:", 1)[0]
    db = compose.split("db:", 1)[1].split("egress:", 1)[0]
    assert "ports:" not in web
    assert "ports:" not in db
    assert "notion_token" in compose.split("worker-sync:", 1)[1].split("worker-embed:", 1)[0]
    assert "notion_token" not in web
    assert "pgvector/pgvector:pg17" in solo
    assert compose_argv("up", solo=True)[2] == "-f"
    roles = (root / "src/ndup/db/roles.sql").read_text(encoding="utf-8")
    assert "ndup_web" in roles
    assert "GRANT DELETE" not in roles.upper()


@pytest.mark.unit
def test_purge_recomputes_cluster_and_drops_orphan_vectors() -> None:
    store = MemoryStore()
    model_id = store.register_model(FAKE_MODEL, 4, "onnx-int8", {})
    kept = UUID(int=1)
    gone = UUID(int=2)
    shared = b"shared"
    only = b"only-gone"
    store.documents[kept] = _stored(kept, shared)
    store.documents[gone] = _stored(gone, only)
    store.put_embeddings(model_id, {shared: np.ones(4), only: np.ones(4)})
    key = uuid4()
    original = member_hash([kept, gone])
    store.clusters[key] = ClusterRecord(
        stable_key=key,
        label="pair",
        size=2,
        member_hash=original,
        oldest_doc=kept,
        newest_doc=gone,
        active_copies=2,
        areas=["eng"],
        members=[
            MemberRecord(kept, "near_copy", False),
            MemberRecord(gone, "near_copy", False),
        ],
    )
    store.reviews[key] = ReviewRow(key, "reviewed", original, None, None, NOW)
    purge_document(store, gone, NOW)
    assert embedding_rows(store, model_id) == 1
    assert only not in {digest for (_model, digest) in store.embeddings}
    assert store.clusters[key].member_hash == member_hash([kept])
    assert store.clusters[key].member_hash != original
    assert store.reviews[key].status == "changed"
    assert store.documents[gone].markdown is None
    assert store.documents[gone].title is None


def _stored(doc_id: UUID, digest: bytes) -> DocumentDraft:
    return DocumentDraft(
        id=doc_id,
        kind="page",
        title="Title",
        markdown="secret body",
        status="active",
        chunks=[
            ChunkDraft(
                ord=0,
                heading_path=[],
                text="secret body",
                embed_text="secret body",
                start_char=0,
                end_char=11,
                block_ids=[],
                spans=[],
                token_count=2,
                text_hash=b"text",
                embed_hash=digest,
                minhash=b"",
                lang="en",
                weight=1,
                is_boilerplate=False,
                boilerplate_reason=None,
                is_synced=False,
                earliest_block_time=None,
            )
        ],
    )


@pytest.mark.unit
def test_cli_config_phase2_and_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pod = RUNNER.invoke(app, ["--profile", "pod", "config", "show", "--redacted"])
    assert pod.exit_code == 0
    assert "cloudflare-access" in pod.stdout
    assert "ndup:ndup@" not in pod.stdout
    team = RUNNER.invoke(app, ["--profile", "team", "config", "show"])
    assert "oauth2-proxy" in team.stdout
    solo = RUNNER.invoke(app, ["--profile", "solo", "config", "show"])
    assert "shared-secret" in solo.stdout
    assert RUNNER.invoke(app, ["web"]).exit_code == 2
    assert RUNNER.invoke(app, ["worker", "--role", "sync"]).exit_code == 2
    database = tmp_path / "state.pkl"
    config = tmp_path / "config.toml"
    monkeypatch.setenv("NDUP_CONFIG", str(config))
    init = RUNNER.invoke(
        app,
        ["init", "--database-url", f"memory://{database}", "--model", FAKE_MODEL, "--config-out", str(config)],
    )
    assert init.exit_code == 0, init.stdout + init.stderr
    assert "ETA" in init.stdout
    monkeypatch.setenv("NDUP_NOTION_TOKEN", "notion-secret-value")
    left, right = UUID(int=70), UUID(int=71)
    body = " ".join(f"token{index}" for index in range(80))

    def search(request: httpx.Request) -> httpx.Response:
        raw = request.read()
        if b"database" in raw:
            return httpx.Response(200, json={"results": [], "has_more": False})
        return httpx.Response(
            200,
            json={"results": [_notion_page(left), _notion_page(right)], "has_more": False},
        )

    def blocks(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if str(left) in url or str(right) in url:
            block_id = UUID(int=80 if str(left) in url else 81)
            payload = [
                {
                    "object": "block",
                    "id": str(block_id),
                    "type": "paragraph",
                    "has_children": False,
                    "paragraph": {"rich_text": [{"plain_text": body, "annotations": {}}]},
                }
            ]
        else:
            payload = []
        return httpx.Response(200, json={"results": payload, "has_more": False})

    def pages(request: httpx.Request) -> httpx.Response:
        doc_id = UUID(request.url.path.split("/")[-1])
        return httpx.Response(200, json=_notion_page(doc_id))

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.notion.com/v1/search").mock(side_effect=search)
        router.get(url__regex=r".*/blocks/.*/children").mock(side_effect=blocks)
        router.get(url__regex=r".*/pages/.*").mock(side_effect=pages)
        synced = RUNNER.invoke(app, ["--config", str(config), "sync"])
        assert synced.exit_code == 0, synced.stdout + synced.stderr
        embedded = RUNNER.invoke(app, ["--config", str(config), "embed"])
        assert embedded.exit_code == 0, embedded.stdout + embedded.stderr
        analyzed = RUNNER.invoke(app, ["--config", str(config), "analyze", "--full"])
        assert analyzed.exit_code == 0, analyzed.stdout + analyzed.stderr
        shown = RUNNER.invoke(app, ["--config", str(config), "show", "clusters", "--json"])
    assert shown.exit_code == 0, shown.stdout + shown.stderr
    payload = json.loads(shown.stdout)
    assert set(payload) >= {"clusters", "other_language_docs", "stub_docs", "stub_title_duplicates"}
    assert payload["clusters"]
    assert payload["clusters"][0]["size"] >= 2
    exported = RUNNER.invoke(app, ["--config", str(config), "export", "--metadata-only"])
    assert "markdown" not in exported.stdout


def _notion_page(doc_id: UUID) -> dict[str, object]:
    return {
        "object": "page",
        "id": str(doc_id),
        "url": f"https://www.notion.so/{doc_id.hex}",
        "created_time": "2024-01-01T00:00:00.000Z",
        "last_edited_time": "2024-06-01T00:00:00.000Z",
        "parent": {"type": "workspace", "workspace": True},
        "properties": {"title": {"type": "title", "title": [{"plain_text": "Status report", "annotations": {}}]}},
    }
