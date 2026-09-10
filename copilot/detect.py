"""Find hours where the Finnish day-ahead price is abnormal. Plain statistics, no LLM."""

import math
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from copilot.baseline import BaselineConfig, baseline_frame
from copilot.timeutil import datetime_index, to_utc, ts


class EventKind(StrEnum):
    SPIKE = "spike"
    CRASH = "crash"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectConfig:
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    z_threshold: float = 4.0
    """Robust z beyond which an hour is abnormal."""
    min_abs_deviation: float = 50.0
    """EUR/MWh: ignore tiny moves even if statistically odd (calm summer nights)."""
    negative_price: float = 0.0
    """Price at or below this is always an event of kind NEGATIVE."""
    max_gap_hours: int = 1
    """Up to this many normal hours between two abnormal hours are bridged into one episode."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """One abnormal price episode. Times are UTC, prices EUR/MWh."""

    start: pd.Timestamp
    end: pd.Timestamp  # inclusive last abnormal hour
    peak_time: pd.Timestamp
    peak_price: float
    baseline_median: float
    z: float
    kind: EventKind
    hours: int
    flagged: bool = True
    """False when the user asked about an hour that our rules do not consider abnormal."""

    @property
    def deviation(self) -> float:
        return self.peak_price - self.baseline_median

    @property
    def has_baseline(self) -> bool:
        """False when too little history backs this hour, so `baseline_median` and `z` are NaN.

        A negative-price hour is always an event, even with no history behind it, so callers
        that print numbers must check this first.
        """
        return not (math.isnan(self.baseline_median) or math.isnan(self.z))

    @property
    def severity(self) -> float:
        """Rank key: peak |z| grown by episode length, so 13 abnormal hours outrank a blip.

        sqrt so length helps without letting a long, mild episode bury a violent short one.
        0.0 when there is no baseline; `_rank` sorts those separately.
        """
        return abs(self.z) * math.sqrt(self.hours) if self.has_baseline else 0.0


def score_prices(price: pd.Series, config: DetectConfig | None = None) -> pd.DataFrame:
    """Baseline frame plus `kind` per hour (None where normal)."""
    config = config or DetectConfig()
    frame = baseline_frame(price, config.baseline)
    deviation = frame["value"] - frame["median"]
    spike = (frame["z"] >= config.z_threshold) & (deviation >= config.min_abs_deviation)
    crash = (frame["z"] <= -config.z_threshold) & (deviation <= -config.min_abs_deviation)
    negative = frame["value"] <= config.negative_price
    kind = pd.Series([None] * len(frame), index=frame.index, dtype="object")
    kind[crash] = EventKind.CRASH
    kind[spike] = EventKind.SPIKE
    kind[negative] = EventKind.NEGATIVE  # negative wins: it is always worth explaining
    frame["kind"] = kind
    return frame


def find_events(
    price: pd.Series, config: DetectConfig | None = None, *, top_n: int | None = 5
) -> list[Event]:
    """Abnormal episodes in `price`, strongest first (see `_rank`)."""
    config = config or DetectConfig()
    scored = score_prices(price, config)
    events = [_make_event(scored.loc[group], flagged=True) for group in _groups(scored, config)]
    events.sort(key=_rank, reverse=True)
    return events if top_n is None else events[:top_n]


def _rank(event: Event) -> tuple[bool, float, float]:
    """Strongest first: hours with a real baseline beat hours without one, then severity,
    then depth below zero so the deepest negative episode wins a tie."""
    return (event.has_baseline, event.severity, -event.peak_price)


def event_at(price: pd.Series, when: pd.Timestamp, config: DetectConfig | None = None) -> Event:
    """The event containing `when`, or a one-hour unflagged Event so it can still be analysed."""
    config = config or DetectConfig()
    when = to_utc(when).floor("1h")
    scored = score_prices(price, config)
    if when not in scored.index:
        raise KeyError(f"{when} not in price series")
    for group in _groups(scored, config):
        if when in group:
            return _make_event(scored.loc[group], flagged=True)
    return _make_event(scored.loc[[when]], flagged=False)


def _groups(scored: pd.DataFrame, config: DetectConfig) -> list[pd.DatetimeIndex]:
    """Episodes of abnormal hours, bridging up to `max_gap_hours` normal hours in between.

    Each span runs from its first to its last abnormal hour and *includes* the bridged
    normal hours, so `Event.hours` is the length a reader would count off a clock.
    """
    flagged = scored[scored["kind"].notna()]
    if flagged.empty:
        return []
    index = datetime_index(flagged)
    # +1: `max_gap_hours` normal hours between two abnormal hours means their stamps are
    # max_gap_hours + 1 apart, and that still counts as one episode.
    gap = pd.Timedelta(hours=config.max_gap_hours + 1)
    group_id = (index.to_series().diff() > gap).cumsum().to_numpy()
    hours = datetime_index(scored)
    spans = []
    for group in pd.unique(group_id):
        abnormal = index[group_id == group]
        spans.append(hours[(hours >= abnormal[0]) & (hours <= abnormal[-1])])
    return spans


def _kind_of(rows: pd.DataFrame) -> EventKind:
    kinds = set(rows["kind"].dropna())
    if EventKind.NEGATIVE in kinds:
        return EventKind.NEGATIVE
    if kinds:
        return EventKind.CRASH if kinds == {EventKind.CRASH} else EventKind.SPIKE
    # unflagged hour: call it a crash only if below its own baseline
    below = rows["value"].iloc[0] < rows["median"].fillna(rows["value"]).iloc[0]
    return EventKind.CRASH if below else EventKind.SPIKE


def _make_event(rows: pd.DataFrame, *, flagged: bool) -> Event:
    if rows["value"].isna().all():
        raise ValueError(f"no price data at {rows.index[0]} (gap in the source)")
    kind = _kind_of(rows)
    peak = rows["value"].idxmax() if kind is EventKind.SPIKE else rows["value"].idxmin()
    index = datetime_index(rows)
    return Event(
        start=index[0],
        end=index[-1],
        peak_time=ts(peak),
        peak_price=float(rows.loc[peak, "value"]),
        baseline_median=float(rows.loc[peak, "median"]),
        z=float(rows.loc[peak, "z"]),
        kind=kind,
        hours=len(rows),
        flagged=flagged,
    )
