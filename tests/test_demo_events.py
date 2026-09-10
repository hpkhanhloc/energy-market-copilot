"""Regression tests on a real-data snapshot (README 'concrete test cases')."""

from pathlib import Path

import pandas as pd
import pytest

from copilot.data.frame import MarketFrame
from copilot.detect import EventKind
from copilot.drivers.base import Verdict
from copilot.investigate import Investigation, investigate_at, scan
from copilot.timeutil import helsinki

FIXTURE = Path(__file__).parent / "fixtures" / "market_2023-12-08_2024-01-08.parquet"


@pytest.fixture(scope="module")
def frame() -> MarketFrame:
    return MarketFrame(data=pd.read_parquet(FIXTURE).astype("float64"))


def _verdicts(inv: Investigation) -> dict[str, Verdict]:
    return {r.name: r.verdict for r in inv.results}


def test_cold_snap_is_the_top_event(frame: MarketFrame) -> None:
    events = scan(frame, top_n=3)
    top = events[0]
    assert top.kind is EventKind.SPIKE
    assert top.peak_time == helsinki("2024-01-05 19:00")
    assert top.peak_price == pytest.approx(1896, abs=1)
    assert top.z > 20


def test_cold_snap_drivers(frame: MarketFrame) -> None:
    inv = investigate_at(frame, helsinki("2024-01-05 19:00"))
    v = _verdicts(inv)
    assert v["load"] is Verdict.SUPPORTS
    assert v["imports"] is Verdict.SUPPORTS
    assert v["neighbours"] is Verdict.SUPPORTS
    assert v["nuclear"] is Verdict.DOES_NOT_SUPPORT
    assert v["wind_forecast"] is Verdict.DOES_NOT_SUPPORT
    assert [r.name for r in inv.results[:1]] == ["neighbours"]  # strongest first


def test_windy_night_is_negative_price_episode(frame: MarketFrame) -> None:
    inv = investigate_at(frame, helsinki("2023-12-17 02:00"))
    assert inv.event.kind is EventKind.NEGATIVE
    assert inv.event.flagged
    assert inv.event.hours >= 10
    v = _verdicts(inv)
    assert v["wind_forecast"] is Verdict.SUPPORTS
    assert v["wind_actual"] is Verdict.SUPPORTS
    assert v["residual_load"] is Verdict.SUPPORTS
    assert v["nuclear"] is Verdict.DOES_NOT_SUPPORT


def test_quiet_hour_is_unflagged_and_says_so(frame: MarketFrame) -> None:
    from copilot.report import render_facts

    inv = investigate_at(frame, helsinki("2023-12-29 03:00"))
    assert not inv.event.flagged
    assert abs(inv.event.z) < 4
    assert "NOT abnormal" in render_facts(inv)
    assert inv.supporting == []  # this particular hour: nothing moved either


def test_scan_since_keeps_negative_night_in_range(frame: MarketFrame) -> None:
    events = scan(frame, top_n=5, since=helsinki("2023-12-15"))
    kinds = {(e.kind, e.start.tz_convert("Europe/Helsinki").strftime("%m-%d")) for e in events}
    assert (EventKind.NEGATIVE, "12-16") in kinds
    assert all(e.start >= helsinki("2023-12-15") for e in events)
