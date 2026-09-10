"""ENTSO-E Transparency wrapper over `entsoe-py`: prices, load, generation, flows, forecasts.

The public API is slow (20-40 s per call in Sept 2026 tests), so everything is cached to parquet
and callers should fetch independent series in parallel (see `frame.py`).
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

from copilot.data.cache import cache_key, cached_frame, ttl_for
from copilot.timeutil import datetime_index, to_utc

log = logging.getLogger(__name__)

FI = "FI"
NEIGHBOURS: dict[str, str] = {"SE_1": "se1", "SE_3": "se3", "EE": "ee", "NO_4": "no4"}
"""ENTSO-E area code -> short column suffix. Order = order shown to the user."""

# entsoe-py generation column name -> our short column. Missing types are simply absent.
GENERATION_COLUMNS: dict[str, str] = {
    "Nuclear": "nuclear",
    "Hydro Run-of-river and pondage": "hydro",  # ENTSO-E spelling
    "Hydro Run-of-river and poundage": "hydro",  # spelling used by some entsoe-py versions
    "Hydro Water Reservoir": "hydro",
    "Hydro Pumped Storage": "hydro",
    "Wind Onshore": "wind",
    "Wind Offshore": "wind",
    "Solar": "solar",
    "Biomass": "thermal",
    "Fossil Gas": "thermal",
    "Fossil Hard coal": "thermal",
    "Fossil Oil": "thermal",
    "Fossil Peat": "thermal",
    "Other": "thermal",
    "Other renewable": "thermal",
    "Waste": "thermal",
}
REQUEST_TIMEOUT_S = 180


class EntsoeClient(Protocol):
    """The subset of `entsoe.EntsoePandasClient` we use, so tests can inject a fake."""

    def query_day_ahead_prices(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series: ...
    def query_load(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...
    def query_load_forecast(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...
    def query_generation(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...
    def query_crossborder_flows(
        self, country_code_from: str, country_code_to: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series: ...
    def query_wind_and_solar_forecast(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...


def real_client(api_key: str) -> EntsoeClient:
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(
        api_key=api_key, retry_count=2, retry_delay=2, timeout=REQUEST_TIMEOUT_S
    )


class EntsoeSource:
    """Hourly, UTC-indexed series for Finland and its neighbours, cached on disk."""

    def __init__(self, client: EntsoeClient, *, cache_dir: Path) -> None:
        self._client = client
        self._cache_dir = cache_dir

    def day_ahead_price(self, area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Day-ahead price in EUR/MWh for `area` (e.g. "FI", "SE_3"), hourly mean."""
        name = f"price_{'fi' if area == FI else NEIGHBOURS.get(area, area.lower())}"
        frame = self._cached(
            "price",
            area,
            start,
            end,
            lambda: _hourly(
                self._client.query_day_ahead_prices(area, start=start, end=end)
            ).to_frame(name),
        )
        return frame[name]

    def load(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Actual total load in Finland, MW, hourly mean."""
        frame = self._cached(
            "load",
            FI,
            start,
            end,
            lambda: (
                _hourly(self._client.query_load(FI, start=start, end=end))
                .iloc[:, 0]
                .to_frame("load")
            ),
        )
        return frame["load"]

    def load_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Day-ahead load forecast for Finland, MW, hourly mean."""
        frame = self._cached(
            "load_fc",
            FI,
            start,
            end,
            lambda: (
                _hourly(self._client.query_load_forecast(FI, start=start, end=end))
                .iloc[:, 0]
                .to_frame("load_fc")
            ),
        )
        return frame["load_fc"]

    def generation_by_type(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Finnish generation in MW, hourly, columns nuclear/hydro/wind/solar/thermal (if present)."""
        return self._cached(
            "generation",
            FI,
            start,
            end,
            lambda: _group_generation(
                _hourly(_actual_only(self._client.query_generation(FI, start=start, end=end)))
            ),
        )

    def net_import(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Net physical import into Finland per border (MW, positive = into FI) plus `import_total`."""
        frame = self._cached(
            "net_import", FI, start, end, self._fetch_net_import_factory(start, end)
        )
        return frame

    def wind_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Day-ahead wind onshore forecast for Finland, MW, hourly mean."""

        def fetch() -> pd.DataFrame:
            raw = _hourly(self._client.query_wind_and_solar_forecast(FI, start=start, end=end))
            wind = [c for c in raw.columns if str(c).startswith("Wind")]
            return (
                raw[wind].sum(axis=1).to_frame("wind_fc")
                if wind
                else pd.DataFrame(columns=["wind_fc"], index=raw.index)
            )

        return self._cached("wind_fc", FI, start, end, fetch)["wind_fc"]

    def _fetch_net_import_factory(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> Callable[[], pd.DataFrame]:
        def fetch() -> pd.DataFrame:
            columns: dict[str, pd.Series] = {}
            for area, suffix in NEIGHBOURS.items():
                inbound = _hourly(
                    self._client.query_crossborder_flows(area, FI, start=start, end=end)
                )
                outbound = _hourly(
                    self._client.query_crossborder_flows(FI, area, start=start, end=end)
                )
                columns[f"import_{suffix}"] = inbound.sub(outbound, fill_value=0.0)
            frame = pd.DataFrame(columns)
            frame["import_total"] = frame.sum(axis=1, min_count=1)
            return frame

        return fetch

    def _cached(
        self,
        series: str,
        area: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fetch: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        start, end = to_utc(start), to_utc(end)
        key = cache_key("entsoe", series, area, start, end)

        def fetch_and_clip() -> pd.DataFrame:
            log.info("entsoe fetch %s %s %s..%s", series, area, start, end)
            frame = fetch()
            return frame[(datetime_index(frame) >= start) & (datetime_index(frame) < end)]

        return cached_frame(key, fetch_and_clip, cache_dir=self._cache_dir, ttl=ttl_for(end))


def _hourly[T: (pd.Series, pd.DataFrame)](obj: T) -> T:
    """Convert to UTC and resample to hourly means (left-labelled)."""
    index = datetime_index(obj)
    localized = obj.tz_convert("UTC") if index.tz is not None else obj.tz_localize("UTC")
    return localized.resample("1h", label="left", closed="left").mean()


def _actual_only(frame: pd.DataFrame) -> pd.DataFrame:
    """entsoe-py may return a 2-level column index (type, Actual Aggregated|Actual Consumption)."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
    return frame


def _group_generation(frame: pd.DataFrame) -> pd.DataFrame:
    grouped: dict[str, pd.Series] = {}
    for column in frame.columns:
        short = GENERATION_COLUMNS.get(str(column))
        if short is None:
            log.debug("ignoring generation type %r", column)
            continue
        grouped[short] = (
            grouped[short].add(frame[column], fill_value=0.0) if short in grouped else frame[column]
        )
    out = pd.DataFrame(grouped, index=frame.index)
    out["generation_total"] = frame.sum(axis=1, min_count=1)
    return out
