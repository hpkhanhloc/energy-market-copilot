"""Orchestrate one investigation: fetch window, find/anchor the event, run driver checks."""

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from copilot.config import Settings
from copilot.data.frame import MarketFrame, build_market_frame
from copilot.data.sources import entsoe_source, fingrid_source
from copilot.detect import DetectConfig, Event, event_at, find_events
from copilot.drivers import DriverResult, Verdict, run_all
from copilot.timeutil import helsinki, to_utc, ts

log = logging.getLogger(__name__)

HISTORY_DAYS = 30
"""Days fetched before the event.

The baseline looks back 28 days (`BaselineConfig.days`: 20 weekdays and 8 weekend days, pooled
separately). Fetching 30 leaves two days of slack so the event day itself has a full window.
"""
AFTER_DAYS = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class Investigation:
    """Everything the report and the charts need. Facts only; no prose yet."""

    event: Event
    frame: MarketFrame
    results: list[DriverResult]

    @property
    def supporting(self) -> list[DriverResult]:
        return [r for r in self.results if r.verdict is Verdict.SUPPORTS]

    @property
    def missing_data(self) -> tuple[str, ...]:
        return self.frame.missing


def load_window(settings: Settings, start: pd.Timestamp, end: pd.Timestamp) -> MarketFrame:
    """Fetch (or read from cache) the hourly market frame for [start, end)."""
    return build_market_frame(
        to_utc(start), to_utc(end), entsoe=entsoe_source(settings), fingrid=fingrid_source(settings)
    )


def window_for(when: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fetch window for one hour: HISTORY_DAYS before its UTC day, AFTER_DAYS after."""
    day = to_utc(when).floor("D")
    return ts(day - pd.Timedelta(days=HISTORY_DAYS)), ts(day + pd.Timedelta(days=AFTER_DAYS))


def scan_window(start: date, end: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fetch window for a scan over Helsinki calendar days [start, end], inclusive.

    HISTORY_DAYS before `start` so the baseline exists on day one; `end` + 1 day because
    `helsinki(end)` is midnight at the start of that day and the window is half-open.
    """
    return (
        ts(helsinki(str(start)) - pd.Timedelta(days=HISTORY_DAYS)),
        helsinki(str(end + timedelta(days=1))),
    )


def investigate_at(
    frame: MarketFrame, when: pd.Timestamp, config: DetectConfig | None = None
) -> Investigation:
    """Investigate the hour `when` (any hour; unflagged hours are still analysed)."""
    _require_price(frame)
    event = event_at(frame.data["price_fi"], when, config)
    return _build(frame, event)


def scan(
    frame: MarketFrame,
    config: DetectConfig | None = None,
    *,
    top_n: int = 5,
    since: pd.Timestamp | None = None,
) -> list[Event]:
    """Strongest price events in the frame, optionally only those touching hours at/after `since`.

    The frame usually carries extra history for the baseline; `since` keeps that history out of
    the ranking so events in the requested range are not crowded out by earlier ones. An episode
    that starts before `since` and runs into the range still counts: its hours inside the range
    are abnormal, and saying "no abnormal hours" about them would be false.
    """
    _require_price(frame)
    events = find_events(frame.data["price_fi"], config, top_n=None)
    if since is not None:
        cutoff = to_utc(since)
        events = [e for e in events if e.end >= cutoff]
    return events[:top_n]


def _build(frame: MarketFrame, event: Event) -> Investigation:
    return Investigation(event=event, frame=frame, results=run_all(frame, event))


def _require_price(frame: MarketFrame) -> None:
    if not frame.has("price_fi"):
        raise ValueError(
            "no Finnish day-ahead price in the frame (ENTSO-E key missing or fetch failed)"
        )
