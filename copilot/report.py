"""Turn an Investigation into text. Facts are rendered by code; prose comes from the LLM.

The LLM never sees raw time series. It gets the deterministic facts below and must return
a typed `Narrative` whose numbers all exist in those facts (checked by `unknown_numbers`).
"""

import logging
import math
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from copilot.detect import EventKind
from copilot.drivers.base import Verdict
from copilot.investigate import Investigation

log = logging.getLogger(__name__)

TZ = "Europe/Helsinki"

KIND_TEXT = {
    EventKind.SPIKE: "price spike",
    EventKind.CRASH: "price crash",
    EventKind.NEGATIVE: "negative / zero price",
}


class Narrative(BaseModel):
    """What the LLM must return. Lists keep facts and hypotheses physically apart."""

    summary: str = Field(
        description="Two or three plain sentences: what happened and the leading explanation, hedged."
    )
    facts: list[str] = Field(description="Observed numbers only, one per item, each with its unit.")
    hypotheses: list[str] = Field(
        description="Plausible drivers the facts are consistent with. Never claim causation."
    )
    insufficient: list[str] = Field(
        default_factory=list, description="What could not be checked and why."
    )


def render_facts(inv: Investigation) -> str:
    """Deterministic markdown with every number. Works with no LLM at all."""
    e = inv.event
    start = e.start.tz_convert(TZ)
    end_ts = e.end.tz_convert(TZ)
    end = None
    if e.hours > 1:
        end = (
            end_ts.strftime("%H:%M")
            if end_ts.date() == start.date()
            else end_ts.strftime("%a %d %b %H:%M")
        )
    peak = e.peak_time.tz_convert(TZ)
    lines = [
        f"## {KIND_TEXT[e.kind].capitalize()} on {start:%a %d %b %Y}",
        "",
        f"- Window: {start:%H:%M}" + (f" to {end}" if end else "") + f" ({e.hours} h, {TZ})",
        f"- Peak: {e.peak_price:,.0f} EUR/MWh at {peak:%H:%M}",
    ]
    if math.isnan(e.baseline_median) or math.isnan(e.z):
        lines.append(
            "- Same-hour baseline: not enough history (fewer than 7 prior days for this hour)"
        )
    else:
        lines += [
            f"- Same-hour baseline (28-day median): {e.baseline_median:,.0f} EUR/MWh",
            f"- Deviation: {e.deviation:+,.0f} EUR/MWh, robust z = {e.z:+.1f}",
        ]
    if not e.flagged:
        lines.append(
            "- Note: this hour is NOT abnormal by the copilot's rules; analysed on request."
        )
    for verdict, heading in (
        (Verdict.SUPPORTS, "Drivers the evidence supports"),
        (Verdict.DOES_NOT_SUPPORT, "Drivers the evidence does not support"),
        (Verdict.INSUFFICIENT, "Not enough data to judge"),
    ):
        group = [r for r in inv.results if r.verdict is verdict]
        if not group:
            continue
        lines += ["", f"### {heading}", ""]
        lines += [f"- **{r.title}**: {r.detail}" for r in group]
    if inv.missing_data:
        lines += ["", f"Series that could not be loaded: {', '.join(inv.missing_data)}."]
    return "\n".join(lines)


def fallback_narrative(inv: Investigation) -> Narrative:
    """No LLM: build the narrative from the driver results directly."""
    e = inv.event
    supporting = [r for r in inv.results if r.verdict is Verdict.SUPPORTS]
    lead = ", ".join(r.title.lower() for r in supporting[:3]) or "none of the checked drivers"
    baseline = (
        ""
        if math.isnan(e.baseline_median)
        else f" against a baseline of {e.baseline_median:,.0f} EUR/MWh"
    )
    summary = (
        f"A {KIND_TEXT[e.kind]} of {e.peak_price:,.0f} EUR/MWh{baseline}. "
        f"Evidence is consistent with: {lead}. "
        "This is a correlation-based read, not proof of cause."
    )
    return Narrative(
        summary=summary,
        facts=[r.detail for r in inv.results if r.verdict is not Verdict.INSUFFICIENT],
        hypotheses=[f"Consistent with: {r.hypothesis}" for r in supporting],
        insufficient=[r.detail for r in inv.results if r.verdict is Verdict.INSUFFICIENT],
    )


NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s?%?")
COUNT_CONTEXT = re.compile(
    r"(?:\bn\s*=\s*|\b(?:of|top)\s+)?(?P<num>\b\d{1,2}\b)(?:\s*(?:h\b|hours?\b|days?\b|drivers?\b|checks?\b|of\b))?"
)
SMALL_COUNT_LIMIT = (
    31  # hours, days, counts of drivers: allowed without a matching fact when used as a count
)


def unknown_numbers(narrative: Narrative, facts: str) -> list[str]:
    """Numbers in the narrative that do not appear in the facts text (possible hallucinations)."""
    allowed = {_norm(n) for n in NUMBER.findall(facts)}
    found: list[str] = []
    for text in _texts(narrative):
        counts = _count_numbers(text)
        for raw in NUMBER.findall(text):
            value = _norm(raw)
            if value in allowed:
                continue
            if not value.endswith("%") and _small_count(value) and value in counts:
                continue
            found.append(raw.strip())
    return found


def _count_numbers(text: str) -> set[str]:
    """Small integers that read as counts ("19 h", "n = 28", "3 of 7 drivers"), not values."""
    counts: set[str] = set()
    for match in COUNT_CONTEXT.finditer(text):
        whole = match.group(0)
        if whole != match.group("num"):  # some count context matched around the number
            counts.add(match.group("num"))
    return counts


def _texts(narrative: Narrative) -> Iterable[str]:
    yield narrative.summary
    yield from narrative.facts
    yield from narrative.hypotheses
    yield from narrative.insufficient


def _small_count(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return number.is_integer() and abs(number) <= SMALL_COUNT_LIMIT


def _norm(number: str) -> str:
    return number.replace(",", "").replace(" ", "").lstrip("+")
