import numpy as np
import pandas as pd
import pytest

from copilot.detect import DetectConfig, EventKind, event_at, find_events
from copilot.timeutil import helsinki, ts


def _series(days: int = 40) -> pd.Series:
    idx = pd.date_range(ts("2023-12-01"), periods=days * 24, freq="1h", name="time")
    hours = np.arange(len(idx)) % 24
    rng = np.random.default_rng(1)
    values = 60 + 20 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 3, len(idx))
    return pd.Series(values, index=idx, name="price_fi")


def test_planted_spike_is_found_and_merged() -> None:
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0 : t0 + pd.Timedelta(hours=2)] = [900.0, 1896.0, 700.0]

    events = find_events(price)

    assert len(events) == 1
    event = events[0]
    assert event.kind is EventKind.SPIKE
    assert event.start == t0
    assert event.end == t0 + pd.Timedelta(hours=2)
    assert event.peak_time == t0 + pd.Timedelta(hours=1)
    assert event.peak_price == 1896.0
    assert event.hours == 3
    assert event.z > 50
    assert 40 < event.baseline_median < 90
    assert event.flagged


def test_negative_price_is_always_an_event() -> None:
    price = _series()
    t0 = ts("2024-01-03 02:00")
    price.loc[t0] = -5.0
    events = find_events(price)
    assert [e.kind for e in events] == [EventKind.NEGATIVE]
    assert events[0].peak_price == -5.0


def test_crash_detected() -> None:
    price = _series()
    price.loc[ts("2024-01-04 12:00")] = 5.0  # baseline ~60 -> deviation -55, big z
    events = find_events(price)
    assert len(events) == 1
    assert events[0].kind is EventKind.CRASH


def test_small_moves_ignored() -> None:
    price = _series()
    price.loc[ts("2024-01-04 12:00")] += 30.0  # statistically odd, but < min_abs_deviation
    assert find_events(price) == []


def test_separate_episodes_stay_separate_and_sorted() -> None:
    price = _series()
    price.loc[ts("2024-01-02 10:00")] = 400.0
    price.loc[ts("2024-01-06 10:00")] = 1500.0
    events = find_events(price)
    assert [e.peak_price for e in events] == [1500.0, 400.0]


def test_event_at_returns_containing_episode() -> None:
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0 : t0 + pd.Timedelta(hours=2)] = [900.0, 1896.0, 700.0]
    event = event_at(price, helsinki("2024-01-05 19:30"))  # 17:30 UTC, inside the episode
    assert event.flagged
    assert event.peak_price == 1896.0


def test_event_at_quiet_hour_is_unflagged_but_analysable() -> None:
    price = _series()
    event = event_at(price, ts("2024-01-05 12:00"))
    assert not event.flagged
    assert event.hours == 1
    assert abs(event.z) < 4


def test_event_at_outside_series_raises() -> None:
    with pytest.raises(KeyError):
        event_at(_series(), ts("2030-01-01"))


def test_custom_threshold() -> None:
    price = _series()
    price.loc[ts("2024-01-04 12:00")] += 30.0
    lenient = DetectConfig(z_threshold=3.0, min_abs_deviation=20.0)
    assert len(find_events(price, lenient)) >= 1


def test_event_at_nan_hour_raises_clear_error() -> None:
    price = _series()
    price.loc[ts("2024-01-05 12:00")] = float("nan")
    with pytest.raises(ValueError, match="no price data"):
        event_at(price, ts("2024-01-05 12:00"))


def test_one_normal_hour_between_abnormal_hours_is_bridged() -> None:
    """max_gap_hours=1 means one normal hour in the middle does not split the episode."""
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0 : t0 + pd.Timedelta(hours=2)] = [900.0, 60.0, 700.0]  # normal hour in between
    events = find_events(price)
    assert len(events) == 1
    assert events[0].hours == 3
    assert events[0].start == t0
    assert events[0].end == t0 + pd.Timedelta(hours=2)


def test_a_wider_gap_still_splits_the_episode() -> None:
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0] = 900.0
    price.loc[t0 + pd.Timedelta(hours=3)] = 700.0  # two normal hours in between
    events = find_events(price)
    assert len(events) == 2
    assert all(e.hours == 1 for e in events)


def test_a_long_episode_outranks_a_sharper_one_hour_blip() -> None:
    price = _series()
    blip = ts("2024-01-02 10:00")
    price.loc[blip] = 5.0  # one very deep hour
    long_start = ts("2024-01-06 00:00")
    price.loc[long_start : long_start + pd.Timedelta(hours=11)] = 8.0  # 12 milder hours
    events = find_events(price)
    assert events[0].hours == 12
    assert abs(events[0].z) < abs(events[1].z)  # shallower per hour, but a far bigger episode
    assert events[0].severity > events[1].severity


def test_event_without_a_baseline_reports_it_and_ranks_last() -> None:
    """A negative price is an event even with no history behind it, but it must say so."""
    price = _series(days=3)  # too short for any baseline
    price.loc[ts("2023-12-02 02:00")] = -20.0
    price.loc[ts("2023-12-03 02:00")] = -5.0
    events = find_events(price)
    assert [e.kind for e in events] == [EventKind.NEGATIVE, EventKind.NEGATIVE]
    assert not any(e.has_baseline for e in events)
    assert all(e.severity == 0.0 for e in events)
    assert [e.peak_price for e in events] == [-20.0, -5.0]  # deepest first


def test_crash_in_a_low_price_regime_is_caught() -> None:
    """A 93% collapse moves less than 50 EUR/MWh, so an absolute-only gate would miss it."""
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    rng = np.random.default_rng(2)
    price = pd.Series(40 + rng.normal(0, 2, len(idx)), index=idx, name="price_fi")
    price.loc[ts("2024-01-04 12:00")] = 3.0  # -37 EUR/MWh: under min_abs_deviation, but -93%
    events = find_events(price)
    assert [e.kind for e in events] == [EventKind.CRASH]
    assert events[0].peak_price == 3.0


def test_a_spike_still_needs_the_absolute_gate() -> None:
    """The relative gate is for crashes only; a 50% ramp off a normal baseline is not news."""
    price = _series()
    price.loc[ts("2024-01-04 12:00")] += 30.0  # ~50% of baseline, but under 50 EUR/MWh
    assert find_events(price) == []


def test_peak_is_never_taken_from_a_bridged_normal_hour() -> None:
    """A bridged normal hour can be cheaper than the abnormal hours it sits between."""
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    hours = idx.to_series().dt.hour.to_numpy()
    rng = np.random.default_rng(3)
    # 13:00 is naturally a cheap hour, the hours either side are expensive
    level = np.where(hours == 13, 20.0, 200.0)
    price = pd.Series(level + rng.normal(0, 2, len(idx)), index=idx, name="price_fi")
    noon = ts("2024-01-04 12:00")
    price.loc[noon] = 100.0  # -100 EUR/MWh off a 200 baseline: a crash
    price.loc[noon + pd.Timedelta(hours=1)] = 18.0  # normal for 13:00, but the lowest price
    price.loc[noon + pd.Timedelta(hours=2)] = 100.0  # a crash again

    events = find_events(price)

    assert len(events) == 1
    assert events[0].hours == 3  # the normal 13:00 is inside the span
    assert events[0].peak_price == 100.0  # ...but it is not the peak
    assert events[0].peak_time in (noon, noon + pd.Timedelta(hours=2))
    assert events[0].z <= -4


def test_a_tiny_drop_off_a_tiny_baseline_is_not_a_crash() -> None:
    """Half of a 2 EUR/MWh baseline is 1 EUR/MWh: the relative gate must not vanish with it."""
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    rng = np.random.default_rng(4)
    price = pd.Series(2.0 + rng.normal(0, 0.05, len(idx)), index=idx, name="price_fi")
    price.loc[ts("2024-01-04 12:00")] = 0.5  # -75%, extreme z, but only -1.5 EUR/MWh
    assert find_events(price) == []


def test_ramp_inside_baseline_spread_is_flagged_with_its_size() -> None:
    """Both hours sit within their own spread, so z never fires; the move itself does."""
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0] = price.loc[t0 - pd.Timedelta(hours=1)] + 120.0
    config = DetectConfig(z_threshold=1e9, min_abs_deviation=1e9)  # z gate off

    events = find_events(price, config)

    assert len(events) == 1
    assert events[0].kind is EventKind.SPIKE
    assert events[0].start == events[0].end == t0
    assert events[0].max_ramp == pytest.approx(120.0, abs=5)  # baseline shape moves a little
    assert events[0].max_ramp_time == t0


def test_ramp_below_min_ramp_is_not_an_event() -> None:
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0] = price.loc[t0 - pd.Timedelta(hours=1)] + 60.0
    config = DetectConfig(z_threshold=1e9, min_abs_deviation=1e9, min_ramp=100.0)
    assert find_events(price, config) == []


def test_ordinary_daily_step_is_not_a_ramp() -> None:
    """A 180 EUR/MWh step that happens every day at the same hour is the baseline, not news."""
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    hours = idx.to_series().dt.hour.to_numpy()
    level = np.where(hours == 13, 20.0, 200.0)
    price = pd.Series(level, index=idx, name="price_fi")
    assert find_events(price) == []


def test_return_to_normal_after_a_spike_is_not_a_second_event() -> None:
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0 : t0 + pd.Timedelta(hours=2)] = [900.0, 1896.0, 700.0]

    events = find_events(price)

    assert len(events) == 1
    assert events[0].end == t0 + pd.Timedelta(hours=2)  # the 700 -> 60 drop is not flagged
    assert events[0].max_ramp == pytest.approx(996.0)  # the 900 -> 1896 run-up
    assert events[0].max_ramp_time == t0 + pd.Timedelta(hours=1)


def test_crash_reports_the_drop_not_the_recovery() -> None:
    price = _series()
    t0 = ts("2024-01-05 15:00")
    price.loc[t0 : t0 + pd.Timedelta(hours=1)] = [-40.0, -80.0]

    events = find_events(price)

    assert events[0].kind is EventKind.NEGATIVE
    assert events[0].max_ramp == pytest.approx(-89, abs=5)  # ~49 -> -40
    assert events[0].max_ramp_time == t0


def test_first_hour_of_series_has_no_ramp() -> None:
    price = _series()
    event = event_at(price, ts(price.index[0]))
    assert not event.flagged
    assert np.isnan(event.max_ramp)
    assert event.max_ramp_time is None
