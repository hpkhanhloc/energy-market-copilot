import numpy as np
import pandas as pd
import pytest

from copilot.baseline import BaselineConfig, baseline_frame
from copilot.timeutil import ts


def _daily_pattern(days: int, *, level: float = 50.0, noise: float = 0.0) -> pd.Series:
    idx = pd.date_range(ts("2024-01-01"), periods=days * 24, freq="1h", name="time")
    hours = np.arange(len(idx)) % 24
    rng = np.random.default_rng(0)
    values = level + 10 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, noise, len(idx))
    return pd.Series(values, index=idx, name="price")


def test_baseline_is_same_hour_median_of_previous_days() -> None:
    price = _daily_pattern(10)
    frame = baseline_frame(price, BaselineConfig(days=28, min_samples=7))
    # first 7 days: not enough history
    assert frame["z"].iloc[: 7 * 24].isna().all()
    # day 8 onward: perfectly repeating pattern gives median == value, z == 0
    later = frame.iloc[7 * 24 :]
    assert np.allclose(later["median"], later["value"])
    assert np.allclose(later["z"], 0.0)
    assert (later["n"] >= 7).all()


def test_today_is_excluded_from_its_own_baseline() -> None:
    price = _daily_pattern(10)
    price.iloc[-1] = 1000.0  # last hour of day 10 spikes
    frame = baseline_frame(price, BaselineConfig(days=28, min_samples=7))
    assert frame["median"].iloc[-1] == pytest.approx(price.iloc[-25])
    assert frame["z"].iloc[-1] > 100


def test_mad_floor_prevents_infinite_z() -> None:
    price = _daily_pattern(10, noise=0.0)
    price.iloc[-1] += 5.0
    frame = baseline_frame(price)
    assert np.isfinite(frame["z"].iloc[-1])


def test_rejects_naive_index() -> None:
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="1h"))
    with pytest.raises(ValueError, match="tz-aware"):
        baseline_frame(naive)
