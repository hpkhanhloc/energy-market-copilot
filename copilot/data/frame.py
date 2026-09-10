"""Build one hourly market table for Finland from all sources, fetched in parallel."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from copilot.data import fingrid_source
from copilot.data.entsoe import FI, NEIGHBOURS
from copilot.timeutil import to_utc

log = logging.getLogger(__name__)

Fetcher = Callable[[], pd.Series | pd.DataFrame]


class EntsoeLike(Protocol):
    """What the frame needs from `EntsoeSource` (structural, so tests can fake it)."""

    def day_ahead_price(self, area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series: ...
    def load(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series: ...
    def load_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series: ...
    def generation_by_type(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...
    def net_import(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...
    def wind_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series: ...


class FingridLike(Protocol):
    def series(self, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketFrame:
    """Hourly UTC-indexed table plus the list of series that could not be loaded."""

    data: pd.DataFrame
    missing: tuple[str, ...] = field(default=())

    def has(self, *columns: str) -> bool:
        return all(c in self.data.columns and self.data[c].notna().any() for c in columns)


def build_market_frame(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    entsoe: EntsoeLike | None,
    fingrid: FingridLike | None,
    max_workers: int = 8,
) -> MarketFrame:
    """Fetch every series for [start, end) and outer-join them on an hourly UTC index.

    A source that is missing (no key) or fails leaves its columns out and is listed in `missing`,
    so drivers can answer "not enough data" instead of crashing.
    """
    start, end = to_utc(start), to_utc(end)
    jobs = _jobs(start, end, entsoe=entsoe, fingrid=fingrid)
    index = pd.date_range(start, end, freq="1h", inclusive="left", name="time")
    parts: list[pd.Series | pd.DataFrame] = []
    missing: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = pool.map(_safe, jobs.items())
    for name, result in zip(jobs, results, strict=True):
        if result is None or len(result) == 0:
            missing.append(name)
        else:
            parts.append(result)

    data = pd.DataFrame(index=index)
    for part in parts:
        data = data.join(part, how="left")
    return MarketFrame(data=data, missing=tuple(missing))


def _jobs(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    entsoe: EntsoeLike | None,
    fingrid: FingridLike | None,
) -> dict[str, Fetcher]:
    jobs: dict[str, Fetcher] = {}
    if entsoe is not None:
        jobs["price_fi"] = lambda: entsoe.day_ahead_price(FI, start, end)
        for area, suffix in NEIGHBOURS.items():
            jobs[f"price_{suffix}"] = lambda a=area: entsoe.day_ahead_price(a, start, end)
        jobs["load"] = lambda: entsoe.load(start, end)
        jobs["load_fc"] = lambda: entsoe.load_forecast(start, end)
        jobs["generation"] = lambda: entsoe.generation_by_type(start, end)
        jobs["net_import"] = lambda: entsoe.net_import(start, end)
        jobs["wind_fc"] = lambda: entsoe.wind_forecast(start, end)
    if fingrid is not None:
        for column in fingrid_source.SERIES:
            jobs[column] = lambda c=column: fingrid.series(c, start, end)
    return jobs


def _safe(item: tuple[str, Fetcher]) -> pd.Series | pd.DataFrame | None:
    name, fetch = item
    try:
        return fetch()
    except Exception as exc:
        log.warning("could not load %s: %s: %s", name, type(exc).__name__, str(exc)[:200])
        return None
