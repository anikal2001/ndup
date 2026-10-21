"""Resolved configuration for solo, team, and pod profiles.

Every key in architecture v3 §13 is represented. Secrets never live here.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ndup.errors import NdUpError

Profile = Literal["solo", "team", "pod"]
ContentMode = Literal["full", "snippets", "metadata"]
AuthMode = Literal["oauth2-proxy", "cloudflare-access", "tailscale", "shared-secret"]
Device = Literal["auto", "cpu", "cuda", "mps"]
Preset = Literal["strict", "default", "loose"]
DeployTarget = Literal["pod", "compose"]
Ingress = Literal["cloudflare-tunnel", "runpod-proxy"]
KnnMode = Literal["exact", "hnsw"]


class RootConfig(BaseModel):
    url: str
    label: str


class NotionSettings(BaseModel):
    api_version: str = "2025-09-03"
    roots: list[RootConfig] = Field(default_factory=list)
    exclude_roots: list[str] = Field(default_factory=list)
    webhooks: bool = False


class SyncSettings(BaseModel):
    incremental_every: str = "1h"
    full_reconcile: str = "weekly"
    search_lag_buffer: str = "15m"
    hot_source_window: str = "7d"
    cold_source_poll: str = "1d"
    max_in_flight: int = 6


class EmbeddingSettings(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    device: Device = "auto"
    batch_size: int = 128
    languages: list[str] = Field(default_factory=lambda: ["en"])
    runtime: str = "onnx-int8"


class AnalysisSettings(BaseModel):
    preset: Preset = "default"
    full_every: str = "nightly"
    min_doc_tokens: int = 40
    short_doc_chunks: int = 2
    active_window_days: int = 90
    knn: KnnMode = "exact"


class AuthSettings(BaseModel):
    mode: AuthMode = "shared-secret"
    issuer: str = ""
    audience: str = "ndup"


class WebSettings(BaseModel):
    content_mode: ContentMode = "snippets"
    admin_groups: list[str] = Field(default_factory=lambda: ["ndup-admins"])
    auth: AuthSettings = Field(default_factory=AuthSettings)


class RetentionSettings(BaseModel):
    version_days: int = 180
    cluster_event_days: int = 400


DEFAULT_POD_IMAGE = "ghcr.io/anikal2001/ndup-pod:0.1.0"


class PodSettings(BaseModel):
    """Non-secret pod settings. The RunPod API key never lives here."""

    ingress: Ingress = "cloudflare-tunnel"
    backup_url: str = ""
    volume_name: str = "ndup"
    volume_id: str = ""
    data_center: str = "US-KS-2"
    volume_gb: int = 100
    gpu_id: str = "NVIDIA GeForce RTX 4090"
    cpu_id: str = "cpu5c"
    vcpu_count: int = 4
    image: str = DEFAULT_POD_IMAGE
    pod_id: str = ""
    disk_gb: int = 40
    web_port: int = 8080


class DeploySettings(BaseModel):
    target: DeployTarget = "compose"
    pod: PodSettings = Field(default_factory=PodSettings)


class StorageSettings(BaseModel):
    encrypt_text: bool = False
    database_url: str = "postgresql://ndup:ndup@localhost:5432/ndup"


class Settings(BaseSettings):
    """File and environment configuration. Environment wins over the file."""

    model_config = SettingsConfigDict(
        env_prefix="NDUP_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        del settings_cls, dotenv_settings, file_secret_settings
        return env_settings, init_settings

    profile: Profile = "solo"
    notion: NotionSettings = Field(default_factory=NotionSettings)
    sync: SyncSettings = Field(default_factory=SyncSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    deploy: DeploySettings = Field(default_factory=DeploySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "solo": {
        "profile": "solo",
        "deploy": {"target": "compose", "pod": {"ingress": "cloudflare-tunnel", "backup_url": ""}},
        "web": {"content_mode": "snippets", "auth": {"mode": "shared-secret", "audience": "ndup"}},
        "storage": {"database_url": "postgresql://ndup:ndup@localhost:5432/ndup"},
        "embedding": {"device": "auto", "runtime": "onnx-int8"},
    },
    "team": {
        "profile": "team",
        "deploy": {"target": "compose", "pod": {"ingress": "cloudflare-tunnel", "backup_url": ""}},
        "web": {
            "content_mode": "snippets",
            "auth": {
                "mode": "oauth2-proxy",
                "issuer": "https://login.example.com",
                "audience": "ndup",
            },
        },
        "storage": {"database_url": "postgresql://ndup_worker@db:5432/ndup"},
        "embedding": {"device": "cpu", "runtime": "onnx-int8"},
    },
    "pod": {
        "profile": "pod",
        "deploy": {
            "target": "pod",
            "pod": {"ingress": "cloudflare-tunnel", "backup_url": "s3://org-ndup-backups/"},
        },
        "web": {
            "content_mode": "snippets",
            "auth": {"mode": "cloudflare-access", "issuer": "", "audience": "ndup"},
        },
        "storage": {"database_url": "postgresql:///ndup?host=/var/run/postgresql"},
        "embedding": {"device": "auto", "runtime": "cuda-fp16"},
    },
}


def config_path_from_env() -> Path | None:
    raw = os.environ.get("NDUP_CONFIG")
    if raw:
        return Path(raw)
    team = Path("/config/ndup.toml")
    if team.is_file():
        return team
    solo = Path.home() / ".config" / "ndup" / "config.toml"
    if solo.is_file():
        return solo
    return None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_settings(
    path: Path | None = None,
    profile: Profile | None = None,
) -> Settings:
    """Load defaults for the profile, then the TOML file, then the environment."""
    chosen = path if path is not None else config_path_from_env()
    file_data: dict[str, Any] = {}
    if chosen is not None and chosen.is_file():
        file_data = tomllib.loads(chosen.read_text(encoding="utf-8"))
        if not isinstance(file_data, dict):
            raise NdUpError(f"Config file {chosen} must be a TOML table")
    file_profile = file_data.get("profile")
    name = profile or (file_profile if isinstance(file_profile, str) else None) or "solo"
    if name not in PROFILE_DEFAULTS:
        raise NdUpError(f"Unknown profile {name!r}")
    merged = _deep_merge(PROFILE_DEFAULTS[name], file_data)
    if profile is not None:
        merged["profile"] = profile
        merged = _deep_merge(PROFILE_DEFAULTS[profile], merged)
        merged["profile"] = profile
    settings = Settings(**merged)
    database_url = os.environ.get("NDUP_DATABASE_URL")
    if database_url:
        settings.storage.database_url = database_url
    return settings


def write_settings(path: Path, settings: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(settings.model_dump(mode="json")), encoding="utf-8")


def redacted_dump(settings: Settings, secrets: list[str]) -> dict[str, Any]:
    import json

    raw = json.dumps(settings.model_dump(mode="json"))
    url = settings.storage.database_url
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            creds, host = rest.split("@", 1)
            if ":" in creds:
                user = creds.split(":", 1)[0]
                raw = raw.replace(url, f"{scheme}://{user}:***@{host}")
    for secret in secrets:
        if secret:
            raw = raw.replace(secret, "***")
    parsed: dict[str, Any] = json.loads(raw)
    return parsed
