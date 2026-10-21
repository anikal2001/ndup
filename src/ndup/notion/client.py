"""Notion REST client pinned to API version 2025-09-03.

Read-only. 429 and 529 honor Retry-After and pause the shared bucket.
Other 5xx responses use jittered exponential backoff.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ndup.errors import NdUpError
from ndup.metrics import notion_requests, rate_limited
from ndup.notion.ratelimit import Clock, TokenBucket

NOTION_VERSION = "2025-09-03"


class NotionError(NdUpError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class NotionClient:
    def __init__(
        self,
        token: str,
        *,
        api_version: str = NOTION_VERSION,
        bucket: TokenBucket | None = None,
        clock: Clock | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 5,
        timeout: float = 30.0,
    ) -> None:
        self.api_version = api_version
        self.bucket = bucket or TokenBucket(clock=clock)
        self.max_retries = max_retries
        self.request_count = 0
        self._rng = random.Random(0)
        self._client = httpx.AsyncClient(
            base_url="https://api.notion.com/v1",
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": api_version,
                "Content-Type": "application/json",
            },
            transport=transport,
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> NotionClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        last_error: NotionError | None = None
        for attempt in range(self.max_retries + 1):
            await self.bucket.acquire()
            try:
                response = await self._client.request(method, path, json=json, params=params)
            except httpx.TimeoutException as exc:
                self.bucket.release()
                last_error = NotionError(f"Timeout calling {path}")
                await self._backoff(attempt)
                del exc
                continue
            self.request_count += 1
            notion_requests.inc()
            self.bucket.release()
            if response.status_code in {429, 529}:
                rate_limited.labels(status=str(response.status_code)).inc()
                self.bucket.pause(_retry_after(response))
                last_error = NotionError(
                    f"Notion {response.status_code} on {path}",
                    status=response.status_code,
                )
                continue
            if response.status_code >= 500:
                last_error = NotionError(
                    f"Notion {response.status_code} on {path}",
                    status=response.status_code,
                )
                await self._backoff(attempt)
                continue
            if response.status_code == 404:
                raise NotionError(f"Not found: {path}", status=404)
            if response.status_code >= 400:
                raise NotionError(
                    f"Notion {response.status_code} on {path}: {response.text[:200]}",
                    status=response.status_code,
                )
            payload = response.json()
            if not isinstance(payload, dict):
                raise NotionError(f"Expected a JSON object from {path}")
            return payload
        raise last_error or NotionError(f"Failed calling {path}")

    async def paginate(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            body = dict(json or {})
            if cursor:
                body["start_cursor"] = cursor
            if method.upper() == "GET":
                query = dict(params or {})
                if cursor:
                    query["start_cursor"] = cursor
                payload = await self.request("GET", path, params=query)
            else:
                payload = await self.request(method, path, json=body, params=params)
            results = payload.get("results")
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        yield item
            if not payload.get("has_more"):
                return
            next_cursor = payload.get("next_cursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                return
            cursor = next_cursor

    async def me(self) -> dict[str, Any]:
        return await self.request("GET", "/users/me")

    async def search_pages(self) -> AsyncIterator[dict[str, Any]]:
        """Search hint channel. ``filter_properties=title`` keeps payloads small."""
        async for item in self.paginate(
            "POST",
            "/search",
            json={
                "filter": {"property": "object", "value": "page"},
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": 100,
            },
            params={"filter_properties": "title"},
        ):
            yield item

    async def search_databases(self) -> AsyncIterator[dict[str, Any]]:
        async for item in self.paginate(
            "POST",
            "/search",
            json={
                "filter": {"property": "object", "value": "database"},
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": 100,
            },
            params={"filter_properties": "title"},
        ):
            yield item

    async def page(self, page_id: str) -> dict[str, Any]:
        return await self.request("GET", f"/pages/{page_id}")

    async def block_children(self, block_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        async for item in self.paginate("GET", f"/blocks/{block_id}/children", params={"page_size": "100"}):
            items.append(item)
        return items

    async def database(self, database_id: str) -> dict[str, Any]:
        return await self.request("GET", f"/databases/{database_id}")

    async def query_data_source(
        self,
        data_source_id: str,
        *,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"page_size": 100}
        if since:
            body["filter"] = {
                "timestamp": "last_edited_time",
                "last_edited_time": {"after": since},
            }
        items: list[dict[str, Any]] = []
        async for item in self.paginate("POST", f"/data_sources/{data_source_id}/query", json=body):
            items.append(item)
        return items

    async def list_users(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        async for item in self.paginate("GET", "/users", params={"page_size": "100"}):
            items.append(item)
        return items

    async def _backoff(self, attempt: int) -> None:
        delay = min(30.0, (2**attempt) * 0.1) + self._rng.random() * 0.05
        await self.bucket.clock.sleep(delay)


def _retry_after(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After", "1")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0
