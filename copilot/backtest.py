"""Check the detector against years of cached price data. Plain code, no LLM.

A backtest runs `find_events` over history we already have and asks four questions an
analyst would ask before trusting it: how many events per month, are the known events at the
top, do the thresholds sit on a plateau, and where does it disagree with the textbook
mean/std method. `scripts/backtest.py` prints the answers; this module holds the logic.
"""

import logging
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import pandas as pd

from copilot.config import TZ
from copilot.data.cache import cache_key, month_chunks
from copilot.detect import DetectConfig, Event, EventKind, find_events, score_prices
from copilot.timeutil import datetime_index, to_utc

log = logging.getLogger(__name__)

PRICE_PREFIX = "entsoe_price_FI"
PRICE_COLUMN = "price_fi"


@dataclass(frozen=True, slots=True, kw_only=True)
class CachedPrice:
    """Hourly FI day-ahead price read from disk, plus the months that were not on disk."""

    price: pd.Series
    missing_months: tuple[str, ...] = ()


def load_cached_price(cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> CachedPrice:
    """Read `entsoe_price_FI_YYYYMM.parquet` files for [start, end) without any API key.

    Months missing on disk are reported, never fetched: a backtest must be repeatable offline.
    """
    start, end = to_utc(start), to_utc(end)
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for month_start, _ in month_chunks(start, end):
        stamp = month_start.strftime("%Y%m")
        path = cache_dir / f"{cache_key(PRICE_PREFIX, stamp)}.parquet"
        if not path.exists():
            missing.append(stamp)
            continue
        parts.append(pd.read_parquet(path))
    if not parts:
        empty = pd.Series(dtype="float64", name=PRICE_COLUMN, index=pd.DatetimeIndex([], tz="UTC"))
        return CachedPrice(price=empty, missing_months=tuple(missing))
    frame = pd.concat(parts).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    # Reindex to every hour so a gap inside a month becomes NaN rows. Otherwise `diff()` in
    # the detector would read a jump across the gap as a one-hour ramp.
    hours = pd.date_range(start, end, freq="1h", inclusive="left", name=frame.index.name)
    price = frame[PRICE_COLUMN].astype("float64").reindex(hours)
    return CachedPrice(price=price, missing_months=tuple(missing))


def naive_flags(price: pd.Series, *, days: int = 30, sigma: float = 2.0) -> pd.Series:
    """The textbook rule: same UTC hour, trailing `days` mean and std, flag beyond `sigma` std.

    Deliberately naive (mean/std, UTC hour, no weekday split): it is the comparison baseline,
    not a recommendation. False until `days` prior values exist for that hour.
    """
    values = price.astype("float64")
    hour = datetime_index(values).to_series(index=values.index).dt.hour
    prior = values.groupby(hour).shift(1)  # today's value never sits in its own window
    rolling = prior.groupby(hour).rolling(days, min_periods=days)
    mean = rolling.mean().reset_index(level=0, drop=True).sort_index()
    std = rolling.std().reset_index(level=0, drop=True).sort_index()
    flagged = (values - mean).abs() > sigma * std
    return flagged.fillna(False).astype(bool).rename("naive")


def our_flags(price: pd.Series, config: DetectConfig | None = None) -> pd.Series:
    """True for every hour the detector marks abnormal (any kind)."""
    return score_prices(price, config)["kind"].notna().rename("ours")


def events_per_month(events: list[Event]) -> pd.DataFrame:
    """Rows `YYYY-MM` (Helsinki), one column per event kind, values are episode counts."""
    kinds = [k.value for k in EventKind]
    if not events:
        return pd.DataFrame(columns=kinds, dtype="int64")
    rows = pd.DataFrame(
        {
            "month": [e.start.tz_convert(TZ).strftime("%Y-%m") for e in events],
            "kind": [e.kind.value for e in events],
        }
    )
    table = rows.pivot_table(index="month", columns="kind", aggfunc="size", fill_value=0)
    return table.reindex(columns=kinds, fill_value=0).astype("int64").sort_index()


def sweep(
    price: pd.Series,
    *,
    z_values: tuple[float, ...] = (3.0, 4.0, 5.0),
    abs_values: tuple[float, ...] = (30.0, 50.0, 80.0),
    ramp_values: tuple[float, ...] = (80.0, 100.0, 150.0),
) -> pd.DataFrame:
    """Episode count for every threshold combination. Stable counts mean stable thresholds."""
    rows = []
    for z, abs_dev, ramp in product(z_values, abs_values, ramp_values):
        config = DetectConfig(z_threshold=z, min_abs_deviation=abs_dev, min_ramp=ramp)
        events = find_events(price, config, top_n=None)
        rows.append(
            {
                "z": z,
                "min_abs_deviation": abs_dev,
                "min_ramp": ramp,
                "events": len(events),
                "hours": sum(e.hours for e in events),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True, kw_only=True)
class Comparison:
    """Hour-level agreement between the detector and the naive rule."""

    both: int
    only_ours: int
    only_naive: int
    neither: int
    only_ours_sample: pd.DataFrame = field(default_factory=pd.DataFrame)
    only_naive_sample: pd.DataFrame = field(default_factory=pd.DataFrame)


def compare(
    price: pd.Series, config: DetectConfig | None = None, *, sample: int = 10
) -> Comparison:
    """Where the detector and the naive rule disagree, with the most extreme hours of each side."""
    scored = score_prices(price, config)
    ours = scored["kind"].notna()
    naive = naive_flags(price)
    detail = scored[["value", "median", "z", "ramp"]].copy()
    detail["naive"] = naive
    only_ours = detail[ours & ~naive]
    only_naive = detail[~ours & naive]
    return Comparison(
        both=int((ours & naive).sum()),
        only_ours=len(only_ours),
        only_naive=len(only_naive),
        neither=int((~ours & ~naive).sum()),
        only_ours_sample=_most_extreme(only_ours, sample),
        only_naive_sample=_most_extreme(only_naive, sample),
    )


def _most_extreme(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    order = rows["z"].abs().fillna(0.0).sort_values(ascending=False)
    return rows.loc[order.index[:n]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpectedHit:
    when: pd.Timestamp
    label: str
    rank: int | None
    """1-based position in the ranked event list, None when no episode contains `when`."""
    in_data: bool
    """False when the hour is outside the price series, so nothing can be said."""


def expected_hits(
    events: list[Event], expected: list[tuple[pd.Timestamp, str]], price: pd.Series
) -> list[ExpectedHit]:
    """For each known event, the rank of the episode that contains it (None = missed)."""
    index = datetime_index(price)
    hits = []
    for when, label in expected:
        stamp = to_utc(when).floor("1h")
        rank = next((i for i, e in enumerate(events, 1) if e.start <= stamp <= e.end), None)
        hits.append(ExpectedHit(when=when, label=label, rank=rank, in_data=stamp in index))
    return hits
