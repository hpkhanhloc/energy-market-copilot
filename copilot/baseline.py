"""Robust 'what is normal for this hour' baseline shared by detection and every driver check.

For each hour we compare against the same hour-of-day on the previous `days` days
(median + MAD). Median/MAD instead of mean/std so one earlier spike does not hide the next one.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from copilot.timeutil import datetime_index

MAD_TO_SIGMA = 1.4826  # scales MAD to a std-dev equivalent for normal data
MIN_SAMPLES = 7


@dataclass(frozen=True, slots=True, kw_only=True)
class BaselineConfig:
    days: int = 28
    min_samples: int = MIN_SAMPLES


def baseline_frame(series: pd.Series, config: BaselineConfig | None = None) -> pd.DataFrame:
    """Return columns value, median, mad, sigma, z, n for every hour of `series`.

    `z` is NaN where fewer than `min_samples` prior same-hour values exist.
    Works on any hourly UTC series (price, wind, load...).
    """
    config = config or BaselineConfig()
    index = datetime_index(series)
    if index.tz is None:
        raise ValueError("series must be tz-aware")
    hourly = series.astype("float64")
    hourly.index = index
    table = hourly.to_frame("value")
    utc = index.tz_convert("UTC").to_series(index=table.index)
    table["date"] = utc.dt.normalize()
    table["hour"] = utc.dt.hour
    wide = table.pivot_table(index="date", columns="hour", values="value", aggfunc="mean")

    # shift(1): today's value is never part of its own baseline
    window = wide.shift(1).rolling(config.days, min_periods=config.min_samples)
    median = window.median()
    mad = window.apply(_mad, raw=True)
    count = window.count()

    out = table[["value"]].copy()
    out["median"] = _lookup(median, table)
    out["mad"] = _lookup(mad, table)
    out["n"] = _lookup(count, table).fillna(0).astype(int)
    out["sigma"] = out["mad"] * MAD_TO_SIGMA
    floor = _sigma_floor(hourly)
    out["z"] = (out["value"] - out["median"]) / out["sigma"].clip(lower=floor)
    out.loc[out["n"] < config.min_samples, ["median", "mad", "sigma", "z"]] = np.nan
    return out


def _mad(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    return float(np.median(np.abs(values - np.median(values))))


def _lookup(wide: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    stacked = wide.stack(future_stack=True)
    keys = pd.MultiIndex.from_arrays([table["date"], table["hour"]])
    return pd.Series(stacked.reindex(keys).to_numpy(), index=table.index)


def _sigma_floor(series: pd.Series) -> float:
    """Avoid infinite z when a flat baseline has MAD 0: use 1% of the typical level, min 1."""
    typical = float(np.nanmedian(np.abs(series.to_numpy()))) if series.notna().any() else 0.0
    return max(1.0, 0.01 * typical)
