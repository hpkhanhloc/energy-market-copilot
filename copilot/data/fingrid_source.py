"""Cached, hourly view over the Fingrid client for the series the drivers need."""

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from copilot.data.cache import cached_range
from copilot.data.fingrid import Dataset
from copilot.timeutil import to_utc

log = logging.getLogger(__name__)

SERIES: dict[str, Dataset] = {
    "wind_rt": Dataset.WIND,
    "wind_fc_fingrid": Dataset.WIND_FORECAST,
    "nuclear_rt": Dataset.NUCLEAR,
    "consumption_rt": Dataset.CONSUMPTION,
}
"""Column name -> Fingrid dataset, only the series a driver check or a chart reads. Each costs a
throttled request per month (1 req / 2 s). `_rt` = real-time measurement averaged to the hour.
The imbalance price (Dataset.IMBALANCE_PRICE) joins here once a driver uses it."""


class HourlyFetcher(Protocol):
    """What we need from `FingridClient` (structural, so tests can fake it)."""

    def fetch_hourly(
        self, dataset: Dataset | int, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series: ...


class FingridSource:
    def __init__(self, client: HourlyFetcher, *, cache_dir: Path) -> None:
        self._client = client
        self._cache_dir = cache_dir

    def series(self, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """One named column from `SERIES`, hourly mean, UTC index, cached."""
        dataset = SERIES[column]
        start, end = to_utc(start), to_utc(end)

        def fetch(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
            log.info("fingrid fetch %s (%d) %s..%s", column, int(dataset), s, e)
            return self._client.fetch_hourly(dataset, s, e).to_frame(column)

        frame = cached_range(
            f"fingrid_{int(dataset)}", start, end, fetch, cache_dir=self._cache_dir
        )
        return frame[column]
