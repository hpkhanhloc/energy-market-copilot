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
