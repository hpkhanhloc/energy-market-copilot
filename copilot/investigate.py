"""Orchestrate one investigation: fetch window, find/anchor the event, run driver checks."""

import logging
from dataclasses import dataclass

import pandas as pd

from copilot.config import Settings
from copilot.data.frame import MarketFrame, build_market_frame
from copilot.data.sources import entsoe_source, fingrid_source
from copilot.detect import DetectConfig, Event, event_at, find_events
from copilot.drivers import DriverResult, run_all
from copilot.timeutil import to_utc, ts

log = logging.getLogger(__name__)

HISTORY_DAYS = 30
"""Days fetched before the event so the 28-day same-hour baseline has enough samples."""
AFTER_DAYS = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class Investigation:
    """Everything the report and the charts need. Facts only; no prose yet."""

    event: Event
    frame: MarketFrame
    results: list[DriverResult]
    window_start: pd.Timestamp
    window_end: pd.Timestamp

    @property
    def supporting(self) -> list[DriverResult]:
        return [r for r in self.results if r.verdict == "supports"]

    @property
    def missing_data(self) -> tuple[str, ...]:
        return self.frame.missing


def load_window(settings: Settings, start: pd.Timestamp, end: pd.Timestamp) -> MarketFrame:
    """Fetch (or read from cache) the hourly market frame for [start, end)."""
    return build_market_frame(
        to_utc(start), to_utc(end), entsoe=entsoe_source(settings), fingrid=fingrid_source(settings)
    )


def window_for(when: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = to_utc(when).floor("D")
    return ts(day - pd.Timedelta(days=HISTORY_DAYS)), ts(day + pd.Timedelta(days=AFTER_DAYS))


def investigate_at(
    frame: MarketFrame, when: pd.Timestamp, config: DetectConfig | None = None
) -> Investigation:
    """Investigate the hour `when` (any hour; unflagged hours are still analysed)."""
    _require_price(frame)
    event = event_at(frame.data["price_fi"], when, config)
    return _build(frame, event)


def investigate_event(frame: MarketFrame, event: Event) -> Investigation:
    return _build(frame, event)


def scan(frame: MarketFrame, config: DetectConfig | None = None, *, top_n: int = 5) -> list[Event]:
    """List the strongest price events in the frame (baseline needs ~a week of lead-in)."""
    _require_price(frame)
    return find_events(frame.data["price_fi"], config, top_n=top_n)


def _build(frame: MarketFrame, event: Event) -> Investigation:
    results = run_all(frame, event)
    index = frame.data.index
    return Investigation(
        event=event,
        frame=frame,
        results=results,
        window_start=ts(index[0]),
        window_end=ts(index[-1]),
    )


def _require_price(frame: MarketFrame) -> None:
    if not frame.has("price_fi"):
        raise ValueError(
            "no Finnish day-ahead price in the frame (ENTSO-E key missing or fetch failed)"
        )
