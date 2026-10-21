"""WP1.1 rate bucket and Notion client."""

from __future__ import annotations

import httpx
import pytest
import respx

from ndup.notion.client import NOTION_VERSION, NotionClient
from ndup.notion.ratelimit import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


@pytest.mark.unit
async def test_bucket_average_and_burst() -> None:
    clock = FakeClock()
    bucket = TokenBucket(clock=clock)
    for _ in range(3):
        await bucket.acquire()
        bucket.release()
    assert clock.t == 0.0
    await bucket.acquire()
    bucket.release()
    assert clock.t >= 1 / 3
    for _ in range(12):
        await bucket.acquire()
        bucket.release()
    elapsed = clock.t
    assert elapsed > 0
    assert (15 - 3) / elapsed <= 3.05


@pytest.mark.unit
async def test_pagination_retry_and_version() -> None:
    clock = FakeClock()
    page = {"object": "list", "results": [{"id": "a"}], "has_more": True, "next_cursor": "c2"}
    done = {"object": "list", "results": [{"id": "b"}], "has_more": False, "next_cursor": None}
    calls = {"search": 0, "flaky": 0}

    def search(request: httpx.Request) -> httpx.Response:
        calls["search"] += 1
        assert request.headers["Notion-Version"] == NOTION_VERSION
        assert request.url.params["filter_properties"] == "title"
        return httpx.Response(200, json=page if calls["search"] == 1 else done)

    def flaky(request: httpx.Request) -> httpx.Response:
        del request
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"object": "error"})
        if calls["flaky"] == 2:
            return httpx.Response(529, headers={"Retry-After": "1"}, json={"object": "error"})
        if calls["flaky"] == 3:
            return httpx.Response(500, json={"object": "error"})
        return httpx.Response(200, json={"object": "user", "id": "me"})

    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.notion.com/v1/search").mock(side_effect=search)
        router.get("https://api.notion.com/v1/users/me").mock(side_effect=flaky)
        async with NotionClient("token", clock=clock) as client:
            found = [item["id"] async for item in client.search_pages()]
            me = await client.me()
    assert found == ["a", "b"]
    assert me["id"] == "me"
    assert clock.t >= 3.0
