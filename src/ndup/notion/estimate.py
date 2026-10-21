"""Sampling-based request estimate. The real workspace number is measured, not assumed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Estimate:
    pages: int
    rows: int
    sample_pages: int
    sample_rows: int
    requests_per_page: float
    requests_per_row: float
    empty_body_share: float
    tokens_per_doc: float
    eta_seconds_low: float
    eta_seconds_high: float

    def to_json(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "rows": self.rows,
            "sample_pages": self.sample_pages,
            "sample_rows": self.sample_rows,
            "requests_per_page": self.requests_per_page,
            "requests_per_row": self.requests_per_row,
            "empty_body_share": self.empty_body_share,
            "tokens_per_doc": self.tokens_per_doc,
            "eta_seconds_low": self.eta_seconds_low,
            "eta_seconds_high": self.eta_seconds_high,
        }


def build_estimate(
    *,
    page_requests: list[int],
    row_requests: list[int],
    empty_flags: list[bool],
    token_counts: list[int],
    pages: int,
    rows: int,
    rate_per_second: float = 3.0,
) -> Estimate:
    page_mean, page_low, page_high = _spread(page_requests, default=4.5)
    row_mean, row_low, row_high = _spread(row_requests, default=1.2)
    total_low = page_low * pages + row_low * rows
    total_high = page_high * pages + row_high * rows
    rate = rate_per_second if rate_per_second > 0 else 3.0
    empty = (sum(1 for flag in empty_flags if flag) / len(empty_flags)) if empty_flags else 0.0
    tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0
    return Estimate(
        pages=pages,
        rows=rows,
        sample_pages=len(page_requests),
        sample_rows=len(row_requests),
        requests_per_page=page_mean,
        requests_per_row=row_mean,
        empty_body_share=empty,
        tokens_per_doc=tokens,
        eta_seconds_low=total_low / rate,
        eta_seconds_high=total_high / rate,
    )


def format_estimate(estimate: Estimate) -> str:
    low = _hours(estimate.eta_seconds_low)
    high = _hours(estimate.eta_seconds_high)
    return (
        f"Sampled {estimate.sample_pages} pages and {estimate.sample_rows} rows. "
        f"{estimate.requests_per_page:.2f} requests/page, "
        f"{estimate.requests_per_row:.2f} requests/row, "
        f"empty body {estimate.empty_body_share:.0%}, "
        f"~{estimate.tokens_per_doc:.0f} tokens/doc. "
        f"ETA {low}–{high} at 3 req/s "
        f"for {estimate.pages} pages and {estimate.rows} rows. "
        "Reference at 3–6 req/page: 150k docs is about 25–40 h."
    )


def _spread(values: list[int], default: float) -> tuple[float, float, float]:
    if not values:
        return default, default * 0.8, default * 1.2
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean * 0.8, mean * 1.2
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    deviation = math.sqrt(variance)
    return mean, max(0.1, mean - deviation), mean + deviation


def _hours(seconds: float) -> str:
    hours = seconds / 3600
    if hours < 1:
        return f"{seconds / 60:.0f} min"
    return f"{hours:.1f} h"
