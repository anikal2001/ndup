"""Sync-task state machine shared by the memory store and the Postgres queue."""

from __future__ import annotations

from datetime import datetime, timedelta

from ndup.model import SyncTask


def choose_task(tasks: list[SyncTask], now: datetime) -> SyncTask | None:
    """Pending tasks, or in-progress tasks whose lease has expired."""
    eligible: list[SyncTask] = []
    for task in tasks:
        if task.state == "pending" and (task.next_attempt_at is None or task.next_attempt_at <= now):
            eligible.append(task)
        elif task.state == "in_progress" and task.leased_until is not None and task.leased_until <= now:
            eligible.append(task)
    if not eligible:
        return None
    eligible.sort(
        key=lambda task: (
            task.priority,
            -(task.remote_last_edited.timestamp() if task.remote_last_edited else float("-inf")),
            str(task.doc_id),
        )
    )
    return eligible[0]


def lease_task(task: SyncTask, now: datetime, lease_for: timedelta) -> None:
    task.state = "in_progress"
    task.attempts += 1
    task.leased_until = now + lease_for
    task.last_error = None


def complete_task(task: SyncTask) -> None:
    task.state = "done"
    task.leased_until = None
    task.last_error = None
    task.next_attempt_at = None


def fail_task(task: SyncTask, error: str, now: datetime, *, max_attempts: int = 6) -> None:
    task.last_error = error[:500]
    task.leased_until = None
    if task.attempts >= max_attempts:
        task.state = "failed"
        task.next_attempt_at = None
        return
    task.state = "pending"
    delay = min(3600, 2 ** max(task.attempts, 1))
    task.next_attempt_at = now + timedelta(seconds=delay)


def priority_for(*, kind: str, discovered_via: str | None, substantial: bool = False) -> int:
    if discovered_via == "webhook":
        return 0
    if kind == "page":
        return 1
    if kind == "db_row" and substantial:
        return 2
    return 3


def cursor_fresh(started_at: datetime | None, now: datetime, ttl: timedelta) -> bool:
    if started_at is None:
        return False
    return now - started_at < ttl
