"""Rendering helpers in views.py. No Streamlit runtime is needed for these."""

import math

import pandas as pd
import pytest

from copilot.detect import Event, EventKind
from copilot.timeutil import ts
from views import episode_label, events_table, number


def _event(*, median: float, z: float, price: float = -12.0) -> Event:
    return Event(
        start=ts("2023-12-17 00:00"),
        end=ts("2023-12-17 07:00"),
        peak_time=ts("2023-12-17 02:00"),
        peak_price=price,
        baseline_median=median,
        z=z,
        kind=EventKind.NEGATIVE,
        hours=8,
    )


NO_HISTORY = _event(median=math.nan, z=math.nan)
WITH_HISTORY = _event(median=60.0, z=-5.5)


def test_number_falls_back_to_a_dash_when_there_is_no_baseline() -> None:
    assert number(60.0, "{:,.0f}") == "60"
    assert number(-5.5, "{:+.1f}") == "-5.5"
    assert number(math.nan, "{:,.0f} EUR/MWh") == "—"


def test_events_table_survives_an_event_with_no_baseline() -> None:
    """A negative price is an event even with no history, so the table must not blow up."""
    table = events_table([NO_HISTORY, WITH_HISTORY])
    assert list(table["Usual EUR/MWh"]) == ["—", "60"]
    assert list(table["Difference EUR/MWh"]) == ["—", "-72"]
    assert list(table["Rarity (z)"]) == ["—", "-5.5"]
    assert list(table["Peak EUR/MWh"]) == [-12, -12]
    assert not any("nan" in str(v) for v in table.to_numpy().ravel())


def test_events_table_matches_the_number_of_events() -> None:
    assert len(events_table([NO_HISTORY, WITH_HISTORY])) == 2
    assert events_table([]).empty


def test_episode_label_says_no_baseline_instead_of_printing_nan() -> None:
    assert "no baseline yet" in episode_label(NO_HISTORY)
    assert "nan" not in episode_label(NO_HISTORY)
    assert "usual 60 EUR/MWh" in episode_label(WITH_HISTORY)


@pytest.mark.parametrize("event", [NO_HISTORY, WITH_HISTORY])
def test_episode_label_is_rendered_in_helsinki_time(event: Event) -> None:
    assert "Sun 17 Dec 2023 02:00" in episode_label(event)  # 00:00 UTC


def test_has_baseline_reflects_missing_history() -> None:
    assert WITH_HISTORY.has_baseline
    assert not NO_HISTORY.has_baseline
    assert pd.isna(NO_HISTORY.deviation)
