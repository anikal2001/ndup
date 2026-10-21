"""RunPod REST API v2 client.

The laptop calls the API. It does not build images. GitHub Actions publishes
``ghcr.io/<org>/ndup-pod:<tag>`` and the pod runs that image. Page text is
never sent to this API. Embeddings stay on the pod or the local machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from ndup.config import DEFAULT_POD_IMAGE, Settings
from ndup.errors import NdUpError
from ndup.secrets import store_secret

API_BASE = "https://api.runpod.io/v2"
VOLUME_PATH = "/workspace"
SECURE_CLOUD = "SECURE"


class RunPodError(NdUpError):
    """The RunPod API refused a call, or the local pod config is incomplete."""


@dataclass(frozen=True)
class PodReport:
    pod_id: str
    status: str
    cloud: str
    url: str
    volume_id: str


def format_report(report: PodReport) -> str:
    return "\n".join(
        [
            f"id: {report.pod_id}",
            f"status: {report.status}",
            f"cloud: {report.cloud}",
            f"url: {report.url}",
        ]
    )


def pod_url(pod_id: str, ingress: str, port: int) -> str:
    if ingress == "runpod-proxy":
        return f"https://{pod_id}-{port}.proxy.runpod.net"
    return "cloudflare-tunnel"


def pod_environment() -> dict[str, str]:
    """Container env. Secrets stay in the keychain or RunPod's own secret store."""
    return {"HF_HUB_OFFLINE": "1", "NDUP_PROFILE": "pod"}


def create_pod_body(settings: Settings, volume_id: str, *, cpu: bool) -> dict[str, Any]:
    """Build a Secure Cloud create-pod body. Exactly one of ``gpu`` or ``cpu``."""
    pod = settings.deploy.pod
    if not volume_id:
        raise RunPodError("A network volume id is required.")
    if not pod.data_center:
        raise RunPodError("Set deploy.pod.data_center to a Secure Cloud data center.")
    if not pod.image:
        raise RunPodError("Set deploy.pod.image to the published pod image.")
    if pod.volume_gb < 10 or pod.volume_gb > 4096:
        raise RunPodError("Network volume size must be between 10 and 4096 GB.")
    body: dict[str, Any] = {
        "name": pod.volume_name or "ndup",
        "image": pod.image or DEFAULT_POD_IMAGE,
        "cloud": SECURE_CLOUD,
        "disk": pod.disk_gb,
        "dataCenterIds": [pod.data_center],
        "env": pod_environment(),
        "mounts": {"network": [{"volumeId": volume_id, "path": VOLUME_PATH}]},
    }
    if cpu:
        if not pod.cpu_id or pod.vcpu_count < 2:
            raise RunPodError("Set deploy.pod.cpu_id and deploy.pod.vcpu_count for a CPU pod.")
        body["cpu"] = {"id": pod.cpu_id, "vcpuCount": pod.vcpu_count}
    else:
        if not pod.gpu_id:
            raise RunPodError("Set deploy.pod.gpu_id for a GPU pod.")
        body["gpu"] = {"id": pod.gpu_id, "count": 1}
    if pod.ingress == "runpod-proxy":
        body["ports"] = [f"{pod.web_port}/http"]
    return body


class RunPodClient:
    """Thin REST client. The API key is sent as a bearer token and never logged."""

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        base_url: str = API_BASE,
    ) -> None:
        cleaned = api_key.strip()
        if not cleaned:
            raise RunPodError("Set RUNPOD_API_KEY or run ndup runpod connect.")
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cleaned}"},
            transport=transport,
            timeout=60.0,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> RunPodClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def list_pods(self) -> list[dict[str, Any]]:
        return self._collect("/pods", "pods")

    def list_volumes(self) -> list[dict[str, Any]]:
        return self._collect("/network-volumes", "networkVolumes", "volumes")

    def create_volume(self, name: str, data_center: str, size: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/network-volumes",
            json_body={"name": name, "dataCenter": data_center, "size": size},
        )

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/pods", json_body=body)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")

    def pod_action(self, pod_id: str, action: str) -> dict[str, Any]:
        return self._request("POST", f"/pods/{pod_id}/action", json_body={"action": action})

    def _collect(self, path: str, *keys: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(20):
            params = {"cursor": cursor} if cursor else None
            payload = self._request("GET", path, params=params)
            batch: list[Any] | None = None
            for key in keys:
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    batch = candidate
                    break
            if batch is None:
                batch = []
            items.extend(item for item in batch if isinstance(item, dict))
            page = payload.get("pagination")
            if not isinstance(page, dict) or not page.get("hasNextPage"):
                break
            next_cursor = page.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor
        return items

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        response = self._http.request(method, path, json=json_body, params=params)
        if response.status_code == 401:
            raise RunPodError("RunPod rejected the API key (401).")
        if response.status_code >= 400:
            raise RunPodError(f"RunPod {method} {path} returned {response.status_code}.")
        if not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise RunPodError(f"RunPod {method} {path} returned a response that was not JSON.") from exc
        if not isinstance(payload, dict):
            raise RunPodError(f"RunPod {method} {path} returned an unexpected payload.")
        return payload


def connect_account(api_key: str, *, client: RunPodClient | None = None) -> str:
    """Check the key against the account, then store it. A rejected key is not stored."""
    cleaned = api_key.strip()
    if not cleaned:
        raise RunPodError("Set RUNPOD_API_KEY. It is stored in the OS keychain.")
    owns_client = client is None
    api = client or RunPodClient(cleaned)
    try:
        api.list_pods()
    finally:
        if owns_client:
            api.close()
    return store_secret("runpod_api_key", cleaned)


def bring_up(client: RunPodClient, settings: Settings, *, cpu: bool) -> PodReport:
    """Create or reuse one Secure Cloud volume and one pod, then start it."""
    pod_cfg = settings.deploy.pod
    volume = _reuse_volume(client, settings)
    if volume is None:
        volume = client.create_volume(pod_cfg.volume_name or "ndup", pod_cfg.data_center, pod_cfg.volume_gb)
    volume_id = _required_id(volume, "volume")
    kind = "cpu" if cpu else "gpu"
    owned = [
        pod
        for pod in client.list_pods()
        if _attached_volume(pod) == volume_id and pod.get("status") != "TERMINATED"
    ]
    wanted = [pod for pod in owned if _compute_kind(pod) == kind]
    chosen = wanted[0] if wanted else None
    for existing in owned:
        if chosen is not None and existing.get("id") == chosen.get("id"):
            continue
        existing_id = existing.get("id")
        if isinstance(existing_id, str) and existing.get("status") != "TERMINATED":
            client.pod_action(existing_id, "stop")
    if chosen is None:
        chosen = client.create_pod(create_pod_body(settings, volume_id, cpu=cpu))
    elif chosen.get("status") == "EXITED":
        chosen_id = chosen.get("id")
        if not isinstance(chosen_id, str):
            raise RunPodError("RunPod returned a pod without an id.")
        chosen = client.pod_action(chosen_id, "start")
    return _report(chosen, settings, volume_id)


def pod_status(client: RunPodClient, settings: Settings) -> PodReport:
    pod, volume_id = _load_pod(client, settings)
    return _report(pod, settings, volume_id)


def stop_pod(client: RunPodClient, settings: Settings) -> PodReport:
    pod, volume_id = _load_pod(client, settings)
    pod_id = pod.get("id")
    if not isinstance(pod_id, str):
        raise RunPodError("RunPod returned a pod without an id.")
    updated = client.pod_action(pod_id, "stop")
    if not updated.get("id"):
        updated = dict(pod)
        updated["status"] = "EXITED"
    return _report(updated, settings, volume_id)


def _reuse_volume(client: RunPodClient, settings: Settings) -> dict[str, Any] | None:
    pod_cfg = settings.deploy.pod
    volumes = client.list_volumes()
    if pod_cfg.volume_id:
        for volume in volumes:
            if volume.get("id") == pod_cfg.volume_id:
                return volume
    for volume in volumes:
        if volume.get("name") == pod_cfg.volume_name and volume.get("dataCenter") == pod_cfg.data_center:
            return volume
    return None


def _load_pod(client: RunPodClient, settings: Settings) -> tuple[dict[str, Any], str]:
    pod_cfg = settings.deploy.pod
    if pod_cfg.pod_id:
        return client.get_pod(pod_cfg.pod_id), pod_cfg.volume_id
    if pod_cfg.volume_id:
        for pod in client.list_pods():
            if _attached_volume(pod) == pod_cfg.volume_id and pod.get("status") != "TERMINATED":
                return pod, pod_cfg.volume_id
    raise RunPodError("No pod is recorded. Run ndup runpod up.")


def _report(pod: dict[str, Any], settings: Settings, volume_id: str) -> PodReport:
    pod_id = pod.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise RunPodError("RunPod returned a pod without an id.")
    status = pod.get("status")
    cloud = pod.get("cloud")
    return PodReport(
        pod_id=pod_id,
        status=status if isinstance(status, str) else "UNKNOWN",
        cloud=cloud if isinstance(cloud, str) else SECURE_CLOUD,
        url=pod_url(pod_id, settings.deploy.pod.ingress, settings.deploy.pod.web_port),
        volume_id=volume_id,
    )


def _attached_volume(pod: dict[str, Any]) -> str | None:
    mounts = pod.get("mounts")
    if not isinstance(mounts, dict):
        return None
    network = mounts.get("network")
    if isinstance(network, dict):
        volume_id = network.get("volumeId")
        return volume_id if isinstance(volume_id, str) else None
    if isinstance(network, list) and network:
        first = network[0]
        if isinstance(first, dict):
            volume_id = first.get("volumeId")
            return volume_id if isinstance(volume_id, str) else None
    return None


def _compute_kind(pod: dict[str, Any]) -> str:
    if pod.get("gpu"):
        return "gpu"
    if pod.get("cpu"):
        return "cpu"
    return "unknown"


def _required_id(payload: dict[str, Any], kind: str) -> str:
    value = payload.get("id")
    if not isinstance(value, str) or not value:
        raise RunPodError(f"RunPod returned a {kind} without an id.")
    return value
