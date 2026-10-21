"""RunPod CLI against a mocked REST API. No live key and no Docker."""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ndup.cli.main import app
from ndup.config import load_settings, write_settings
from ndup.runpod import RunPodClient, RunPodError, create_pod_body
from ndup.secrets import install_redaction, known_secret_values, resolve_secret

RUNNER = CliRunner()
API = "https://api.runpod.io/v2"
KEY = "rp-test-key"


def _pod_config(tmp_path: Path, *, ingress: str = "runpod-proxy") -> Path:
    settings = load_settings(profile="pod")
    settings.deploy.pod.ingress = ingress  # type: ignore[assignment]
    path = tmp_path / "config.toml"
    write_settings(path, settings)
    return path


def _volume(volume_id: str = "vol1") -> dict[str, object]:
    return {"id": volume_id, "name": "ndup", "size": 100, "dataCenter": "US-KS-2"}


def _pod(
    pod_id: str = "pod1",
    *,
    status: str = "RUNNING",
    kind: str = "gpu",
    volume_id: str = "vol1",
) -> dict[str, object]:
    body: dict[str, object] = {
        "id": pod_id,
        "name": "ndup",
        "status": status,
        "cloud": "SECURE",
        "dataCenterId": "US-KS-2",
        "mounts": {"network": [{"volumeId": volume_id, "path": "/workspace"}]},
    }
    if kind == "gpu":
        body["gpu"] = {"id": "NVIDIA GeForce RTX 4090", "count": 1}
    else:
        body["cpu"] = {"id": "cpu5c", "vcpuCount": 4}
    return body


def _list(key: str, items: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(200, json={key: items, "pagination": {"hasNextPage": False}})


@pytest.mark.unit
def test_create_body_is_secure_and_exclusive() -> None:
    settings = load_settings(profile="pod")
    gpu = create_pod_body(settings, "vol1", cpu=False)
    cpu = create_pod_body(settings, "vol1", cpu=True)
    assert gpu["cloud"] == "SECURE"
    assert "gpu" in gpu and "cpu" not in gpu
    assert "cpu" in cpu and "gpu" not in cpu
    assert gpu["mounts"]["network"] == [{"volumeId": "vol1", "path": "/workspace"}]
    assert "ports" not in gpu
    assert gpu["env"] == {"HF_HUB_OFFLINE": "1", "NDUP_PROFILE": "pod"}
    assert gpu["image"] == "ghcr.io/anikal2001/ndup-pod:0.1.0"
    settings.deploy.pod.ingress = "runpod-proxy"
    proxied = create_pod_body(settings, "vol1", cpu=False)
    assert proxied["ports"] == ["8080/http"]


@pytest.mark.unit
def test_list_pods_follows_cursor_and_hides_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {KEY}"
        if request.url.params.get("cursor") == "next":
            return httpx.Response(200, json={"pods": [{"id": "b"}], "pagination": {"hasNextPage": False}})
        return httpx.Response(
            200,
            json={"pods": [{"id": "a"}], "pagination": {"hasNextPage": True, "nextCursor": "next"}},
        )

    client = RunPodClient(KEY, transport=httpx.MockTransport(handler))
    assert [pod["id"] for pod in client.list_pods()] == ["a", "b"]
    client.close()

    def denied(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": KEY})

    rejected = RunPodClient(KEY, transport=httpx.MockTransport(denied))
    with pytest.raises(RunPodError) as exc:
        rejected.list_pods()
    assert KEY not in str(exc.value)
    rejected.close()


@pytest.mark.unit
def test_runpod_key_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NDUP_NOTION_TOKEN", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    assert resolve_secret("runpod_api_key", secrets_dir=Path("/no/such/secrets")) == KEY
    install_redaction(known_secret_values(secrets_dir=Path("/no/such/secrets")))
    logger = logging.getLogger("ndup.tests.runpod")
    logger.setLevel(logging.INFO)
    records: list[logging.LogRecord] = []

    class _List(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _List()
    logger.addHandler(handler)
    logger.addFilter(logging.getLogger("ndup").filters[-1])
    logger.info("key %s", KEY)
    logger.removeHandler(handler)
    assert records
    assert KEY not in records[0].getMessage()


@pytest.mark.unit
def test_connect_stores_key_and_checks_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setattr("ndup.secrets._write_keyring", lambda _name, _value: False)
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    with respx.mock() as router:
        router.get(f"{API}/pods").mock(return_value=_list("pods", []))
        result = RUNNER.invoke(app, ["runpod", "connect"])
    assert result.exit_code == 0, result.output
    assert KEY not in result.output
    stored = tmp_path / ".config" / "ndup" / "secrets" / "runpod_api_key"
    assert stored.read_text(encoding="utf-8").strip() == KEY
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600
    assert "ndup" not in stored.parts or tmp_path in stored.parents


@pytest.mark.unit
def test_connect_rejects_unauthorized_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    written: list[str] = []
    monkeypatch.setattr("ndup.secrets._write_keyring", lambda name, _value: written.append(name) or True)
    with respx.mock() as router:
        router.get(f"{API}/pods").mock(return_value=httpx.Response(401, json={"error": "nope"}))
        result = RUNNER.invoke(app, ["runpod", "connect"])
    assert result.exit_code == 1
    assert written == []
    assert KEY not in result.output
    assert not (tmp_path / ".config" / "ndup" / "secrets" / "runpod_api_key").exists()


@pytest.mark.unit
def test_connect_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    result = RUNNER.invoke(app, ["runpod", "connect"])
    assert result.exit_code == 1
    assert "RUNPOD_API_KEY" in result.output


@pytest.mark.unit
def test_up_creates_volume_and_gpu_pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("NDUP_NOTION_TOKEN", "notion-secret-value")
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    config = _pod_config(tmp_path, ingress="runpod-proxy")
    with respx.mock() as router:
        router.get(f"{API}/network-volumes").mock(return_value=_list("networkVolumes", []))
        router.post(f"{API}/network-volumes").mock(return_value=httpx.Response(201, json=_volume()))
        router.get(f"{API}/pods").mock(return_value=_list("pods", []))
        created = router.post(f"{API}/pods").mock(
            return_value=httpx.Response(201, json=_pod(status="PROVISIONING"))
        )
        result = RUNNER.invoke(app, ["--config", str(config), "runpod", "up"])
    assert result.exit_code == 0, result.output
    assert "https://pod1-8080.proxy.runpod.net" in result.output
    assert "id: pod1" in result.output
    assert "cloud: SECURE" in result.output
    assert KEY not in result.output
    assert "notion-secret-value" not in result.output
    body = json.loads(created.calls.last.request.content)
    assert body["cloud"] == "SECURE"
    assert "gpu" in body and "cpu" not in body
    assert body["mounts"]["network"][0]["path"] == "/workspace"
    assert body["ports"] == ["8080/http"]
    assert KEY not in json.dumps(body)
    assert "notion-secret-value" not in json.dumps(body)
    saved = config.read_text(encoding="utf-8")
    assert KEY not in saved
    loaded = load_settings(config)
    assert loaded.deploy.target == "pod"
    assert loaded.deploy.pod.pod_id == "pod1"
    assert loaded.deploy.pod.volume_id == "vol1"


@pytest.mark.unit
def test_up_reuses_volume_and_running_pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    config = _pod_config(tmp_path)
    posted: list[str] = []

    def reject(request: httpx.Request) -> httpx.Response:
        posted.append(str(request.url))
        return httpx.Response(500, json={})

    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API}/network-volumes").mock(return_value=_list("networkVolumes", [_volume()]))
        router.get(f"{API}/pods").mock(return_value=_list("pods", [_pod()]))
        router.post(url__regex=r".*").mock(side_effect=reject)
        result = RUNNER.invoke(app, ["--config", str(config), "runpod", "up"])
    assert result.exit_code == 0, result.output
    assert posted == []
    assert "https://pod1-8080.proxy.runpod.net" in result.output
    assert KEY not in result.output


@pytest.mark.unit
def test_up_starts_exited_pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    config = _pod_config(tmp_path)
    with respx.mock() as router:
        router.get(f"{API}/network-volumes").mock(return_value=_list("networkVolumes", [_volume()]))
        router.get(f"{API}/pods").mock(return_value=_list("pods", [_pod(status="EXITED")]))
        action = router.post(f"{API}/pods/pod1/action").mock(
            return_value=httpx.Response(200, json=_pod(status="STARTING"))
        )
        result = RUNNER.invoke(app, ["--config", str(config), "runpod", "up"])
    assert result.exit_code == 0, result.output
    assert json.loads(action.calls.last.request.content) == {"action": "start"}
    assert "status: STARTING" in result.output


@pytest.mark.unit
def test_up_cpu_stops_gpu_pod_and_omits_ports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    config = _pod_config(tmp_path, ingress="cloudflare-tunnel")
    with respx.mock() as router:
        router.get(f"{API}/network-volumes").mock(return_value=_list("networkVolumes", [_volume()]))
        router.get(f"{API}/pods").mock(return_value=_list("pods", [_pod()]))
        stopped = router.post(f"{API}/pods/pod1/action").mock(
            return_value=httpx.Response(200, json=_pod(status="EXITED"))
        )
        created = router.post(f"{API}/pods").mock(
            return_value=httpx.Response(201, json=_pod(pod_id="pod2", status="PROVISIONING", kind="cpu"))
        )
        result = RUNNER.invoke(app, ["--config", str(config), "runpod", "up", "--cpu"])
    assert result.exit_code == 0, result.output
    assert json.loads(stopped.calls.last.request.content) == {"action": "stop"}
    body = json.loads(created.calls.last.request.content)
    assert "cpu" in body and "gpu" not in body
    assert body["cloud"] == "SECURE"
    assert "ports" not in body
    assert body["cpu"] == {"id": "cpu5c", "vcpuCount": 4}
    assert "url: cloudflare-tunnel" in result.output
    assert "proxy.runpod.net" not in result.output


@pytest.mark.unit
def test_status_and_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setattr("ndup.secrets._read_keyring", lambda _name: None)
    config = _pod_config(tmp_path)
    settings = load_settings(config)
    settings.deploy.pod.pod_id = "pod1"
    settings.deploy.pod.volume_id = "vol1"
    write_settings(config, settings)
    with respx.mock() as router:
        router.get(f"{API}/pods/pod1").mock(return_value=httpx.Response(200, json=_pod()))
        status = RUNNER.invoke(app, ["--config", str(config), "runpod", "status"])
        action = router.post(f"{API}/pods/pod1/action").mock(
            return_value=httpx.Response(200, json=_pod(status="EXITED"))
        )
        down = RUNNER.invoke(app, ["--config", str(config), "runpod", "down"])
    assert status.exit_code == 0, status.output
    assert "id: pod1" in status.output
    assert "status: RUNNING" in status.output
    assert "cloud: SECURE" in status.output
    assert "https://pod1-8080.proxy.runpod.net" in status.output
    assert KEY not in status.output
    assert down.exit_code == 0, down.output
    assert json.loads(action.calls.last.request.content) == {"action": "stop"}
    assert "status: EXITED" in down.output
    assert KEY not in down.output
