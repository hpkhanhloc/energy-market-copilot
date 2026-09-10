import numpy as np
import pandas as pd
import pytest

from copilot.baseline import BaselineConfig, _rolling_mad, baseline_frame
from copilot.timeutil import datetime_index, ts

TZ = "Europe/Helsinki"


def _series(
    start: str,
    days: int,
    *,
    weekday: float = 100.0,
    weekend: float | None = None,
    peak_hour: int | None = None,
    peak: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    """Hourly UTC series whose shape is defined in *local* time, so it survives a DST switch."""
    idx = pd.date_range(ts(start), periods=days * 24, freq="1h", name="time")
    local = idx.tz_convert(TZ).to_series(index=idx)
    saturday_or_sunday = (local.dt.dayofweek >= 5).to_numpy()
    weekend_level = weekday if weekend is None else weekend
    values = np.where(saturday_or_sunday, weekend_level, weekday).astype("float64")
    if peak_hour is not None:
        values += np.where((local.dt.hour == peak_hour).to_numpy(), peak, 0.0)
    if noise:
        values += np.random.default_rng(seed).normal(0, noise, len(idx))
    return pd.Series(values, index=idx, name="price")


def test_baseline_is_same_local_hour_median_of_previous_days() -> None:
    price = _series("2024-01-01", 56, peak_hour=19, peak=150.0)
    frame = baseline_frame(price, BaselineConfig(days=28, min_samples=7))
    assert frame["z"].iloc[:24].isna().all()  # day one has no history at all
    later = frame.iloc[24 * 25 :]  # every bucket has enough history by then
    assert later["median"].notna().all()
    assert np.allclose(later["median"], later["value"])
    assert np.allclose(later["z"], 0.0)


def test_today_is_excluded_from_its_own_baseline() -> None:
    price = _series("2024-01-01", 56)
    price.iloc[-1] = 1000.0
    frame = baseline_frame(price)
    assert frame["median"].iloc[-1] == pytest.approx(100.0)
    assert frame["z"].iloc[-1] > 100


def test_dst_switch_does_not_create_fake_spikes() -> None:
    """The whole point of bucketing on local hours: no fake events around a clock change."""
    price = _series("2024-02-15", 90, peak_hour=19, peak=150.0)  # spans 2024-03-31
    frame = baseline_frame(price)
    after = frame.loc["2024-03-31":"2024-04-28"].dropna(subset=["z"])
    assert not after.empty
    assert after["z"].abs().max() == pytest.approx(0.0)


def test_weekend_is_not_compared_against_weekdays() -> None:
    """Weekends are cheaper all day; pooling them with weekdays would read as a crash."""
    price = _series("2024-01-01", 56, weekday=100.0, weekend=40.0)
    frame = baseline_frame(price)
    settled = frame.iloc[24 * 25 :]
    local = datetime_index(settled).tz_convert(TZ).to_series(index=settled.index)
    weekend = settled[(local.dt.dayofweek >= 5).to_numpy()]
    assert not weekend.empty
    assert np.allclose(weekend["median"], 40.0)
    assert np.allclose(weekend["z"], 0.0)


def test_weekend_needs_a_smaller_window_and_fewer_samples() -> None:
    config = BaselineConfig(days=28, min_samples=7)
    assert config.window(weekend=False) == 20  # 5 of every 7 days
    assert config.window(weekend=True) == 8  # 2 of every 7 days
    assert config.samples(weekend=False) == 7
    assert config.samples(weekend=True) < 7


def test_thin_history_leaves_median_and_z_missing() -> None:
    frame = baseline_frame(_series("2024-01-01", 3))
    assert frame["median"].isna().all()
    assert frame["z"].isna().all()
    assert (frame["n"] < 7).all()


def test_mad_floor_prevents_infinite_z() -> None:
    price = _series("2024-01-01", 56)
    price.iloc[-1] += 5.0
    frame = baseline_frame(price)
    assert np.isfinite(frame["z"].iloc[-1])


def test_sigma_floor_scales_with_the_series_own_spread() -> None:
    """One flat bucket must not become far more sensitive than a normal one."""
    price = _series("2024-01-01", 56, noise=20.0)
    local = datetime_index(price).tz_convert(TZ).to_series(index=price.index)
    price[(local.dt.hour == 3).to_numpy()] = 100.0  # this bucket alone has zero spread
    price.iloc[-21] = 200.0  # a 03:00 hour, 100 EUR/MWh above its flat baseline
    frame = baseline_frame(price)
    z = float(frame["z"].iloc[-21])
    # a 1 EUR/MWh floor would make this z ~= 100; the floor now tracks typical sigma (~25)
    assert 2.0 < z < 25.0


def test_rejects_naive_index() -> None:
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="1h"))
    with pytest.raises(ValueError, match="tz-aware"):
        baseline_frame(naive)


def test_rolling_mad_matches_the_plain_python_version() -> None:
    """The vectorised MAD replaced a rolling().apply() callback; it must agree exactly."""
    rng = np.random.default_rng(7)
    wide = pd.DataFrame(rng.normal(0, 10, (40, 5)))
    wide.iloc[3, 2] = np.nan  # gaps happen: a missing hour in the source
    wide.iloc[:6, 4] = np.nan

    def naive(values: np.ndarray) -> float:
        values = values[~np.isnan(values)]
        if values.size == 0:
            return float("nan")
        return float(np.median(np.abs(values - np.median(values))))

    expected = wide.rolling(8, min_periods=1).apply(naive, raw=True)
    pd.testing.assert_frame_equal(_rolling_mad(wide, 8), expected, check_dtype=False)
