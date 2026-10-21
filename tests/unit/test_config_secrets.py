"""WP0.2 config profiles and secret redaction."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ndup.config import load_settings, redacted_dump, write_settings
from ndup.secrets import install_redaction, resolve_secret


@pytest.mark.unit
def test_three_profiles_redacted() -> None:
    solo = load_settings(profile="solo")
    team = load_settings(profile="team")
    pod = load_settings(profile="pod")
    assert solo.web.auth.mode == "shared-secret"
    assert solo.storage.database_url.startswith("postgresql://")
    assert team.web.auth.mode == "oauth2-proxy"
    assert team.deploy.target == "compose"
    assert pod.web.auth.mode == "cloudflare-access"
    assert pod.deploy.target == "pod"
    assert "/var/run/postgresql" in pod.storage.database_url
    assert pod.embedding.runtime == "cuda-fp16"
    solo.storage.database_url = "postgresql://ndup:s3cret-db@localhost:5432/ndup"
    dumped = json.dumps(redacted_dump(solo, ["s3cret-db"]))
    assert "s3cret-db" not in dumped
    assert "***" in dumped


@pytest.mark.unit
def test_env_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    write_settings(path, load_settings(profile="solo"))
    monkeypatch.setenv("NDUP_WEB__CONTENT_MODE", "metadata")
    loaded = load_settings(path, profile="solo")
    assert loaded.web.content_mode == "metadata"
    assert loaded.profile == "solo"


@pytest.mark.unit
def test_secret_absent_from_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NDUP_NOTION_TOKEN", raising=False)
    assert resolve_secret("notion_token", secrets_dir=tmp_path) is None
    monkeypatch.setenv("NDUP_NOTION_TOKEN", "notion-secret-value")
    assert resolve_secret("notion_token", secrets_dir=tmp_path) == "notion-secret-value"
    install_redaction(["notion-secret-value"])
    logger = logging.getLogger("ndup")
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class _List(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _List()
    logger.addHandler(handler)
    logger.info("token is %s", "notion-secret-value")
    logger.removeHandler(handler)
    assert records
    assert "notion-secret-value" not in records[0].getMessage()
    assert "***" in records[0].getMessage()
