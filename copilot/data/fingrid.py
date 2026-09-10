"""Fingrid Open Data client: https://data.fingrid.fi/api (needs FINGRID_API_KEY)."""

import logging
import time
from collections.abc import Callable
from enum import IntEnum
from typing import Any, Protocol

import pandas as pd

from copilot.timeutil import to_utc

log = logging.getLogger(__name__)

BASE_URL = "https://data.fingrid.fi/api"
MIN_INTERVAL_S = 2.0  # Fingrid allows 1 request per 2 seconds
PAGE_SIZE = 20_000


class Dataset(IntEnum):
    """Fingrid dataset ids used by the copilot. Units in comments."""

    IMBALANCE_PRICE = 319  # EUR/MWh, 15 min
    MFRR_UP_PRICE = 244  # EUR/MWh, 15 min
    MFRR_DOWN_PRICE = 106  # EUR/MWh, 15 min
    WIND = 181  # MW, 3 min
    WIND_FORECAST = 245  # MWh/h, 15 min, updated every 15 min
    NUCLEAR = 188  # MW, 3 min
    HYDRO = 191  # MW, 3 min
    PRODUCTION = 192  # MW, 3 min
    CONSUMPTION = 193  # MW, 3 min
    CONSUMPTION_FORECAST = 165  # MW, 15 min
    PRODUCTION_FORECAST = 241  # MW, 15 min


class _Response(Protocol):
    status_code: int
    text: str

    def json(self) -> dict: ...


class _Session(Protocol):
    def get(self, url: str, **kwargs: Any) -> _Response: ...


class FingridError(RuntimeError):
    """Non-2xx answer from Fingrid."""


class FingridClient:
    """Minimal client: paging, throttling, tz-aware output. Inject `session`/`sleep` in tests."""

    def __init__(
        self,
        api_key: str,
        *,
        session: _Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session: _Session = session if session is not None else _requests_session()
        self._headers = {"x-api-key": api_key}
        self._sleep = sleep
        self._clock = clock
        self._last_call = float("-inf")

    def fetch(self, dataset: Dataset | int, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Raw values for [start, end) as a float Series with a UTC DatetimeIndex."""
        start, end = to_utc(start), to_utc(end)
        params = {
            "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json",
            "pageSize": PAGE_SIZE,
            "sortBy": "startTime",
            "sortOrder": "asc",
            "page": 1,
        }
        rows: list[dict] = []
        while True:
            body = self._get(f"{BASE_URL}/datasets/{int(dataset)}/data", params)
            rows.extend(body.get("data", []))
            pagination = body.get("pagination") or {}
            next_page = pagination.get("nextPage")
            if not next_page:
                break
            params["page"] = next_page
        return _to_series(
            rows, name=Dataset(dataset).name.lower() if dataset in Dataset else str(dataset)
        )

    def fetch_hourly(
        self, dataset: Dataset | int, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Hourly mean of `fetch` (MW averages / price averages), left-labelled, UTC."""
        raw = self.fetch(dataset, start, end)
        if raw.empty:
            return raw
        return raw.resample("1h", label="left", closed="left").mean()

    def _get(self, url: str, params: dict) -> dict:
        wait = MIN_INTERVAL_S - (self._clock() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        resp = self._session.get(url, params=params, headers=self._headers, timeout=60)
        self._last_call = self._clock()
        if resp.status_code == 429:
            log.warning("fingrid rate limited, retrying once")
            self._sleep(MIN_INTERVAL_S)
            resp = self._session.get(url, params=params, headers=self._headers, timeout=60)
            self._last_call = self._clock()
        if resp.status_code != 200:
            raise FingridError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")
        return resp.json()


def _requests_session() -> Any:
    import requests

    return requests.Session()


def _to_series(rows: list[dict], *, name: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype="float64", name=name, index=pd.DatetimeIndex([], tz="UTC"))
    frame = pd.DataFrame(rows)
    index = pd.DatetimeIndex(pd.to_datetime(frame["startTime"], utc=True), name="time")
    series = pd.Series(frame["value"].astype("float64").to_numpy(), index=index, name=name)
    return series[~series.index.duplicated(keep="last")].sort_index()
