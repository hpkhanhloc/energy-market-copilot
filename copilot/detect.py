"""Find hours where the Finnish day-ahead price is abnormal. Plain statistics, no LLM."""

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
    """Abnormal hours this close together are merged into one event."""


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
    price: pd.Series, config: DetectConfig | None = None, *, top_n: int = 5
) -> list[Event]:
    """Abnormal episodes in `price`, strongest first (by |z|, negative prices by depth)."""
    config = config or DetectConfig()
    scored = score_prices(price, config)
    flagged = scored[scored["kind"].notna()]
    events = [_make_event(scored.loc[group], flagged=True) for group in _groups(flagged, config)]
    events.sort(key=lambda e: (abs(e.z) if pd.notna(e.z) else 0.0, -e.peak_price), reverse=True)
    return events[:top_n]


def event_at(price: pd.Series, when: pd.Timestamp, config: DetectConfig | None = None) -> Event:
    """The event containing `when`, or a one-hour unflagged Event so it can still be analysed."""
    config = config or DetectConfig()
    when = to_utc(when).floor("1h")
    scored = score_prices(price, config)
    if when not in scored.index:
        raise KeyError(f"{when} not in price series")
    flagged = scored[scored["kind"].notna()]
    for group in _groups(flagged, config):
        if when in group:
            return _make_event(scored.loc[group], flagged=True)
    return _make_event(scored.loc[[when]], flagged=False)


def _groups(flagged: pd.DataFrame, config: DetectConfig) -> list[pd.DatetimeIndex]:
    if flagged.empty:
        return []
    index = datetime_index(flagged)
    gap = pd.Timedelta(hours=config.max_gap_hours)
    breaks = index.to_series().diff() > gap
    group_id = breaks.cumsum()
    return [pd.DatetimeIndex(index[group_id.to_numpy() == g]) for g in group_id.unique()]


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
