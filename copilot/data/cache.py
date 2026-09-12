"""Tiny parquet cache so demos work offline and API limits are not burned twice."""

import logging
import re
import time
from collections.abc import Callable, Iterator
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
    complete_after: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return the cached parquet for `key` if fresh, else call `fetch`, store, return.

    `complete_after` is when the data behind `key` stopped changing (the end of its month). A
    file written before that was fetched while the month was still in progress, so it holds a
    partial month and is refetched once, however old it is.
    """
    path = cache_dir / f"{key}.parquet"
    if path.exists() and _fresh(path, ttl, complete_after):
        log.debug("cache hit %s", path.name)
        return pd.read_parquet(path)
    frame = fetch()
    if frame.empty:
        # One empty-but-200 answer would otherwise be stored, and `ttl_for` returns None for
        # any month older than RECENT_WINDOW, so `_fresh` would call it fresh forever. The
        # only recovery would be deleting the file by hand.
        log.warning("empty result for %s: not cached", key)
        return frame
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    log.debug("cache store %s (%d rows)", path.name, len(frame))
    return frame


def _fresh(path: Path, ttl: timedelta | None, complete_after: pd.Timestamp | None = None) -> bool:
    mtime = path.stat().st_mtime
    if complete_after is not None and mtime < complete_after.timestamp():
        log.info("cache file %s predates the end of its month: refetching", path.name)
        return False
    if ttl is None:
        return True
    return time.time() - mtime < ttl.total_seconds()


RangeFetch = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


def month_chunks(
    start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Whole UTC calendar months covering [start, end)."""
    start, end = ts(start).tz_convert("UTC"), ts(end).tz_convert("UTC")
    cursor = ts(start.tz_localize(None).to_period("M").start_time)  # ts() localizes naive to UTC
    while cursor < end:
        nxt = ts((cursor + pd.offsets.MonthBegin(1)).normalize())
        yield cursor, nxt
        cursor = nxt


def cached_range(
    prefix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fetch: RangeFetch,
    *,
    cache_dir: Path,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch and cache one parquet per calendar month, then return the clipped [start, end).

    Month granularity means any later window that overlaps a fetched month is free, and only
    the month containing "now" ever expires (see `ttl_for`). A month file written before that
    month ended is partial and is refetched once the month is over.
    """
    start, end = ts(start).tz_convert("UTC"), ts(end).tz_convert("UTC")
    now = now or pd.Timestamp.now(tz=UTC)
    parts: list[pd.DataFrame] = []
    for chunk_start, chunk_end in month_chunks(start, end):
        key = cache_key(prefix, chunk_start.strftime("%Y%m"))
        part = cached_frame(
            key,
            lambda s=chunk_start, e=chunk_end: fetch(s, e),
            cache_dir=cache_dir,
            ttl=ttl_for(chunk_end, now),
            complete_after=chunk_end if chunk_end <= now else None,
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[(frame.index >= start) & (frame.index < end)]
