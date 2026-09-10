"""Cached, hourly view over the Fingrid client for the series the drivers need."""

import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from copilot.data.cache import cache_key, cached_frame, ttl_for
from copilot.data.fingrid import Dataset
from copilot.timeutil import to_utc

log = logging.getLogger(__name__)

SERIES: dict[str, Dataset] = {
    "wind_rt": Dataset.WIND,
    "wind_fc_fingrid": Dataset.WIND_FORECAST,
    "nuclear_rt": Dataset.NUCLEAR,
    "hydro_rt": Dataset.HYDRO,
    "production_rt": Dataset.PRODUCTION,
    "consumption_rt": Dataset.CONSUMPTION,
    "consumption_fc": Dataset.CONSUMPTION_FORECAST,
    "imbalance_price": Dataset.IMBALANCE_PRICE,
}
"""Column name -> Fingrid dataset. `_rt` = real-time measurement averaged to the hour."""


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
        key = cache_key("fingrid", int(dataset), start, end)

        def fetch() -> pd.DataFrame:
            log.info("fingrid fetch %s (%d) %s..%s", column, int(dataset), start, end)
            return self._client.fetch_hourly(dataset, start, end).to_frame(column)

        frame = cached_frame(key, fetch, cache_dir=self._cache_dir, ttl=ttl_for(end))
        return frame[column]
