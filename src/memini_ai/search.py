"""Pure helpers for recall: reciprocal rank fusion, since parsing, SQL filter building."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

_REL_RE = re.compile(r"^(\d+)([dhm])$")


def rrf(ranked: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Score = sum over lists of 1/(k + rank), rank 1-based."""
    scores: dict[str, float] = {}
    for lst in ranked:
        for rank, item in enumerate(lst, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def parse_since(value: str, now: datetime | None = None) -> datetime:
    """Accept '7d', '24h', '30m', or an ISO date/datetime. Always returns an aware datetime."""
    now = now or datetime.now(UTC)
    m = _REL_RE.match(value.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "m": timedelta(minutes=n)}[unit]
        return now - delta
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError as e:
        raise ValueError(f"since must be like '7d', '24h', or an ISO date; got {value!r}") from e
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def build_filters(
    kind: str | None,
    project: str | None,
    since: datetime | None,
    include_superseded: bool,
    first_param: int,
) -> tuple[str, list[Any]]:
    """Return (sql_fragment, params). Placeholders start at $first_param."""
    parts: list[str] = []
    params: list[Any] = []
    n = first_param
    if kind is not None:
        parts.append(f"kind = ${n}")
        params.append(kind)
        n += 1
    if project is not None:
        parts.append(f"project = ${n}")
        params.append(project)
        n += 1
    if since is not None:
        parts.append(f"created_at >= ${n}")
        params.append(since)
        n += 1
    if not include_superseded:
        parts.append("superseded_by IS NULL")
    return ("".join(" AND " + p for p in parts), params)
