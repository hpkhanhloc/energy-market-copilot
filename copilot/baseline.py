"""Robust 'what is normal for this hour' baseline shared by detection and every driver check.

For each hour we compare against the same *local* hour-of-day on previous days of the same
type (weekday vs weekend), using median + MAD. Median/MAD instead of mean/std so one earlier
spike does not hide the next one.

Two things matter for correctness:

* **Local hour, not UTC hour.** The daily price shape follows Europe/Helsinki wall-clock time
  (morning ramp, evening peak). Bucketing on UTC hours makes the whole shape jump by one hour
  at each DST switch, which would show up as a month of fake spikes and fake crashes twice a
  year.
* **Weekday and weekend kept apart.** Sunday midday demand is nothing like Tuesday midday
  demand, so pooling them pushes every weekend hour toward "crash".
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from copilot.config import HELSINKI
from copilot.timeutil import datetime_index

MAD_TO_SIGMA = 1.4826  # scales MAD to a std-dev equivalent for normal data
MIN_SAMPLES = 7
MIN_WEEKEND_SAMPLES = 3
WEEKDAY_SHARE = 5 / 7  # Mon-Fri
WEEKEND_SHARE = 2 / 7  # Sat-Sun
SIGMA_FLOOR_SHARE = 0.25  # a bucket may not be more than 4x as sensitive as a typical one


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineConfig:
    days: int = 28
    """Calendar days of history to look back over."""
    min_samples: int = MIN_SAMPLES
    """Same-hour, same-day-type values needed before the baseline is trusted."""

    def window(self, *, weekend: bool) -> int:
        """How many same-day-type days fit inside a `days`-long calendar lookback."""
        share = WEEKEND_SHARE if weekend else WEEKDAY_SHARE
        return max(1, round(self.days * share))

    def samples(self, *, weekend: bool) -> int:
        """Weekends only supply 2 days in 7, so they need a lower bar than weekdays."""
        if not weekend:
            return self.min_samples
        scaled = round(self.min_samples * WEEKEND_SHARE / WEEKDAY_SHARE)
        return max(MIN_WEEKEND_SAMPLES, scaled)


def baseline_frame(series: pd.Series, config: BaselineConfig | None = None) -> pd.DataFrame:
    """Return columns value, median, mad, sigma, z, n for every hour of `series`.

    Buckets are (local hour-of-day, weekday or weekend). `z` is NaN where fewer than
    `config.samples(...)` prior values share the bucket.
    Works on any hourly series with a tz-aware index (price, wind, load...).
    """
    config = config or BaselineConfig()
    index = datetime_index(series)
    if index.tz is None:
        raise ValueError("series must be tz-aware")
    hourly = series.astype("float64")
    hourly.index = index
    table = hourly.to_frame("value")
    local = index.tz_convert(HELSINKI).to_series(index=table.index)
    table["date"] = local.dt.normalize()
    table["hour"] = local.dt.hour
    table["weekend"] = local.dt.dayofweek >= 5

    out = table[["value"]].copy()
    for column, fill in (("median", np.nan), ("mad", np.nan), ("n", 0.0)):
        out[column] = fill
    for weekend, part in table.groupby("weekend", sort=False):
        stats = _bucket_stats(part, config, weekend=bool(weekend))
        out.loc[part.index, ["median", "mad", "n"]] = stats

    out["sigma"] = out["mad"] * MAD_TO_SIGMA
    out["z"] = (out["value"] - out["median"]) / out["sigma"].clip(lower=_sigma_floor(out["sigma"]))
    out["n"] = out["n"].fillna(0).astype(int)
    return out


def _bucket_stats(part: pd.DataFrame, config: BaselineConfig, *, weekend: bool) -> pd.DataFrame:
    """median/mad/n for one day type, from prior days of that same type only."""
    wide = part.pivot_table(index="date", columns="hour", values="value", aggfunc="mean")
    prior = wide.shift(1)  # today's value is never part of its own baseline
    days = config.window(weekend=weekend)
    window = prior.rolling(days, min_periods=1)
    count = _lookup(window.count(), part)
    stats = pd.DataFrame(
        {
            "median": _lookup(window.median(), part),
            "mad": _lookup(_rolling_mad(prior, days), part),
            "n": count.fillna(0),
        }
    )
    stats.loc[count < config.samples(weekend=weekend), ["median", "mad"]] = np.nan
    return stats


def _rolling_mad(wide: pd.DataFrame, days: int) -> pd.DataFrame:
    """Rolling median absolute deviation over `days` rows, per column.

    `rolling().apply()` would call back into Python once per window per column. An
    investigation runs this over the price plus every driver series, ~15 passes, and that
    callback was the slowest thing in it.
    """
    values = wide.to_numpy(dtype="float64")
    padded = np.full((len(values) + days - 1, values.shape[1]), np.nan)
    padded[days - 1 :] = values
    windows = np.lib.stride_tricks.sliding_window_view(padded, days, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are expected
        medians = np.nanmedian(windows, axis=-1)
        mad = np.nanmedian(np.abs(windows - medians[..., None]), axis=-1)
    return pd.DataFrame(mad, index=wide.index, columns=wide.columns)


def _lookup(wide: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    stacked = wide.stack(future_stack=True)
    keys = pd.MultiIndex.from_arrays([table["date"], table["hour"]])
    return pd.Series(stacked.reindex(keys).to_numpy(), index=table.index)


def _sigma_floor(sigma: pd.Series) -> float:
    """Avoid a huge z when one bucket happens to have a near-zero MAD.

    Floored against the series' *own* typical sigma, not against the price level. A fixed
    small floor makes the flattest bucket wildly more sensitive than a normal one: on the
    demo month typical sigma is ~43 EUR/MWh, so a 1 EUR/MWh floor turned an ordinary
    60 EUR/MWh drop into z = -61.
    """
    typical = float(np.nanmedian(sigma.to_numpy())) if sigma.notna().any() else 0.0
    return max(1.0, SIGMA_FLOOR_SHARE * typical)
