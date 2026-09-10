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
