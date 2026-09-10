"""Tiny parquet cache so demos work offline and API limits are not burned twice."""

import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from copilot.timeutil import ts

log = logging.getLogger(__name__)

RECENT_TTL = timedelta(hours=1)
RECENT_WINDOW = timedelta(days=3)


def cache_key(*parts: object) -> str:
    """Build a filesystem-safe key from parts like ("fingrid", 181, start, end)."""
    text = "_".join(_fmt(p) for p in parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text)


def _fmt(part: object) -> str:
    if isinstance(part, datetime):
        return ts(part).tz_convert("UTC").strftime("%Y%m%dT%H%M")
    return str(part)


def ttl_for(end: pd.Timestamp, now: pd.Timestamp | None = None) -> timedelta | None:
    """Recent data may still be revised, so it expires. Old data is cached forever."""
    now = now or pd.Timestamp.now(tz=UTC)
    return RECENT_TTL if end > now - RECENT_WINDOW else None


def cached_frame(
    key: str,
    fetch: Callable[[], pd.DataFrame],
    *,
    cache_dir: Path,
    ttl: timedelta | None = None,
) -> pd.DataFrame:
    """Return the cached parquet for `key` if fresh, else call `fetch`, store, return."""
    path = cache_dir / f"{key}.parquet"
    if path.exists() and _fresh(path, ttl):
        log.debug("cache hit %s", path.name)
        return pd.read_parquet(path)
    frame = fetch()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    log.debug("cache store %s (%d rows)", path.name, len(frame))
    return frame


def _fresh(path: Path, ttl: timedelta | None) -> bool:
    if ttl is None:
        return True
    age = time.time() - path.stat().st_mtime
    return age < ttl.total_seconds()
