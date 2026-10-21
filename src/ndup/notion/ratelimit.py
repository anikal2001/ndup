"""Single-process async token bucket. Average 3 requests/second, burst 3."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from ndup.metrics import bucket_pause_seconds


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class TokenBucket:
    """All Notion calls share one bucket. 429/529 pause the whole bucket."""

    def __init__(
        self,
        *,
        rate: float = 3.0,
        burst: float = 3.0,
        max_in_flight: int = 6,
        clock: Clock | None = None,
    ) -> None:
        self.rate = rate
        self.capacity = burst
        self.max_in_flight = max_in_flight
        self.clock = clock or SystemClock()
        self._tokens = float(burst)
        self._updated = self.clock.now()
        self._pause_until = 0.0
        self._in_flight = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = self.clock.now()
                self._refill(now)
                if now < self._pause_until:
                    wait = self._pause_until - now
                elif self._in_flight >= self.max_in_flight:
                    wait = 0.01
                elif self._tokens >= 1.0 - 1e-9:
                    self._tokens = max(0.0, self._tokens - 1.0)
                    self._in_flight += 1
                    return
                else:
                    wait = max((1.0 - self._tokens) / self.rate, 1e-6)
            await self.clock.sleep(max(wait, 0.0))

    def release(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    def pause(self, seconds: float) -> None:
        """Honor Retry-After by stopping the bucket, not just one request."""
        if seconds < 0:
            seconds = 0
        now = self.clock.now()
        self._pause_until = max(self._pause_until, now + seconds)
        self._tokens = 0.0
        bucket_pause_seconds.inc(seconds)

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        if now < self._pause_until:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
