"""Shared shape of every driver check. Plain code, numbers first, honest verdicts."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from copilot.baseline import BaselineConfig, baseline_frame
from copilot.data.frame import MarketFrame
from copilot.detect import Event, EventKind

Z_SUPPORT = 2.0
"""|robust z| at or above this counts as a real move for a driver."""
RELATIVE_SUPPORT = 0.20
"""...or a move of at least this share of the baseline (for noisy series where MAD is wide)."""
Z_NOT_ORDINARY = 1.0
"""The relative branch still needs this much z: a big-looking share of a wide baseline can be
an entirely ordinary hour, and calling that "supports" states a hypothesis the data does not
back. Wind 24% below normal at z = +0.3 is exactly that case."""
MIN_COVERAGE = 0.5
"""Share of the event hours a series must have data (and a baseline) for before it is judged.
One hour out of twenty-one is not "during the event"; below this the verdict is insufficient."""


class Verdict(StrEnum):
    SUPPORTS = "supports"
    DOES_NOT_SUPPORT = "does_not_support"
    INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True, slots=True, kw_only=True)
class DriverResult:
    """One hypothesis, one number, one verdict. `detail` is deterministic text with the numbers."""

    name: str
    title: str
    hypothesis: str
    verdict: Verdict
    unit: str
    value: float | None = None
    baseline: float | None = None
    z: float | None = None
    detail: str = ""
    columns: tuple[str, ...] = ()
    """Frame columns to plot for this driver."""

    @property
    def deviation(self) -> float | None:
        if self.value is None or self.baseline is None:
            return None
        return self.value - self.baseline


DriverCheck = Callable[[MarketFrame, Event], DriverResult]


def event_window(frame: MarketFrame, event: Event) -> pd.DataFrame:
    return frame.data.loc[event.start : event.end]


def price_up(event: Event) -> bool:
    """True when the event is a high price; False for crashes and negative prices."""
    return event.kind is EventKind.SPIKE


def pick_column(frame: MarketFrame, primary: str, backup: str) -> str:
    """The twin column (ENTSO-E first, Fingrid as backup) with more hours of data.

    `frame.has` is true for a single non-NaN value, so an ENTSO-E series with a gap over the
    event would otherwise win against a complete Fingrid series measuring the same thing.
    """
    counts = {
        c: int(frame.data[c].notna().sum()) if c in frame.data.columns else 0
        for c in (primary, backup)
    }
    return backup if counts[backup] > counts[primary] else primary


def compare_to_baseline(
    frame: MarketFrame,
    event: Event,
    *,
    column: str,
    name: str,
    title: str,
    hypothesis_up: str,
    hypothesis_down: str,
    unit: str,
    bullish_when: str,
    config: BaselineConfig | None = None,
    columns: tuple[str, ...] | None = None,
) -> DriverResult:
    """Compare `column` during the event with its same-hour baseline.

    `bullish_when` is "lower" or "higher": the direction of the driver that pushes price UP.
    For a spike the driver supports if it moved in the bullish direction by |z| >= Z_SUPPORT,
    or by RELATIVE_SUPPORT of its baseline while still being at least Z_NOT_ORDINARY away from
    normal; for a crash/negative event the opposite direction is required. `hypothesis_up` is
    the story tested for high prices, `hypothesis_down` for crashes and negative prices.
    """
    hypothesis = hypothesis_up if price_up(event) else hypothesis_down
    if not frame.has(column):
        return DriverResult(
            name=name,
            title=title,
            hypothesis=hypothesis,
            verdict=Verdict.INSUFFICIENT,
            unit=unit,
            detail=f"{title}: no data for '{column}'.",
            columns=columns or (column,),
        )
    stats = baseline_frame(frame.data[column], config).loc[event.start : event.end]
    valid = stats.dropna(subset=["value", "median", "z"])
    if valid.empty:
        return DriverResult(
            name=name,
            title=title,
            hypothesis=hypothesis,
            verdict=Verdict.INSUFFICIENT,
            unit=unit,
            detail=f"{title}: not enough history to build a baseline for the event hours.",
            columns=columns or (column,),
        )
    covered = len(valid)
    if covered < event.hours * MIN_COVERAGE:
        return DriverResult(
            name=name,
            title=title,
            hypothesis=hypothesis,
            verdict=Verdict.INSUFFICIENT,
            unit=unit,
            detail=(
                f"{title}: data and a baseline for only {covered} of {event.hours} event hours, "
                "too few to judge."
            ),
            columns=columns or (column,),
        )
    partial = f" Based on {covered} of {event.hours} event hours." if covered < event.hours else ""
    value = float(valid["value"].mean())
    baseline = float(valid["median"].mean())
    z_signed = float(valid["z"].mean())
    wants_lower = (bullish_when == "lower") == price_up(event)
    verdict = (
        Verdict.SUPPORTS
        if _moved(value, baseline, z_signed, lower=wants_lower)
        else Verdict.DOES_NOT_SUPPORT
    )
    direction = "below" if value < baseline else "above"
    pct = _pct(value, baseline)
    detail = (
        f"{title}: {value:,.0f} {unit} during the event vs a same-hour baseline of "
        f"{baseline:,.0f} {unit} ({direction} normal by {abs(value - baseline):,.0f} {unit}"
        f"{pct}, robust z = {z_signed:+.1f}, baseline from {int(valid['n'].min())} prior days)."
        f"{partial}"
    )
    return DriverResult(
        name=name,
        title=title,
        hypothesis=hypothesis,
        verdict=verdict,
        unit=unit,
        value=value,
        baseline=baseline,
        z=z_signed,
        detail=detail,
        columns=columns or (column,),
    )


def _moved(value: float, baseline: float, z: float, *, lower: bool) -> bool:
    """Big enough move in the required direction: by robust z or by share of baseline."""
    diff = value - baseline
    if lower:
        diff, z = -diff, -z
    if diff <= 0:
        return False
    relative = diff / abs(baseline) if baseline else 0.0
    return z >= Z_SUPPORT or (relative >= RELATIVE_SUPPORT and z >= Z_NOT_ORDINARY)


def _pct(value: float, baseline: float) -> str:
    if baseline == 0 or not np.isfinite(baseline):
        return ""
    return f", {abs(value - baseline) / abs(baseline):.0%}"
