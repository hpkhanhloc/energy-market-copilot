"""Rendering and turn handling in views.py. No Streamlit runtime is needed for these."""

import math

import pandas as pd
import pytest
import streamlit as st

from copilot.detect import Event, EventKind
from copilot.report import format_number
from copilot.timeutil import ts
from views import (
    EVENT_COLUMNS,
    GLOSSARY,
    MAX_TURNS,
    add_turn,
    episode_label,
    events_table,
    table_help,
    turn_by_id,
    turns,
)


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


def test_format_number_falls_back_to_a_dash_when_there_is_no_baseline() -> None:
    assert format_number(60.0, "{:,.0f}") == "60"
    assert format_number(-5.5, "{:+.1f}") == "-5.5"
    assert format_number(math.nan, "{:,.0f} EUR/MWh") == "—"


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


@pytest.fixture(autouse=True)
def _clean_session() -> None:
    st.session_state["turns"] = []


def test_a_turn_keeps_its_id_when_older_turns_are_trimmed_away() -> None:
    """Turns used to be addressed by list position, which shifts once trimming starts."""
    add_turn({"role": "assistant", "kind": "scan", "events": ["first"]})
    wanted = turns()[0]["id"]
    for _ in range(MAX_TURNS):
        add_turn({"role": "user", "kind": "text", "text": "filler"})

    assert len(turns()) == MAX_TURNS
    assert turn_by_id(wanted) is None  # trimmed away, and it says so instead of mis-pointing
    newest = turns()[-1]
    assert turn_by_id(newest["id"]) is newest


def test_every_turn_gets_its_own_id() -> None:
    for _ in range(5):
        add_turn({"role": "user", "kind": "text", "text": "hi"})
    ids = [t["id"] for t in turns()]
    assert len(set(ids)) == 5
    assert all(turn_by_id(i) is not None for i in ids)


def test_table_help_covers_only_real_columns() -> None:
    cfg = table_help()
    assert set(cfg) <= set(EVENT_COLUMNS)
    assert "Rarity (z)" in cfg
    assert set(events_table([WITH_HISTORY]).columns) == set(EVENT_COLUMNS)
    assert all(GLOSSARY[k] for k in cfg)
