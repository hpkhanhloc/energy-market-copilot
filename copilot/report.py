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
    if not e.has_baseline:
        lines.append(
            "- Same-hour baseline: not enough history "
            "(too few prior days of the same hour and day type)"
        )
    else:
        lines += [
            f"- Same-hour baseline (median of recent same-day-type days): "
            f"{e.baseline_median:,.0f} EUR/MWh",
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
        "" if not e.has_baseline else f" against a baseline of {e.baseline_median:,.0f} EUR/MWh"
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


NO_VALUE = "—"
"""Printed where an hour has too little history for a baseline, so the number is NaN."""


def format_number(value: float, spec: str) -> str:
    """`spec.format(value)`, or `NO_VALUE` when there is no baseline behind the number."""
    return NO_VALUE if math.isnan(value) else spec.format(value)


NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s?%?")
COUNT_CONTEXT = re.compile(
    r"(?:\bn\s*=\s*|\b(?:of|top)\s+)?(?P<num>\b\d{1,2}\b)(?:\s*(?:h\b|hours?\b|days?\b|drivers?\b|checks?\b|of\b))?"
)
SMALL_COUNT_LIMIT = (
    31  # hours, days, counts of drivers: allowed without a matching fact when used as a count
)


BANNED = ("caused", "because", "due to", "led to", "resulted in")
BANNED_RE = re.compile(r"\b(?:" + "|".join(re.escape(b) for b in BANNED) + r")\b", re.IGNORECASE)


def unknown_numbers(narrative: Narrative, facts: str) -> list[str]:
    """Numbers in the narrative that do not appear in the facts text (possible hallucinations)."""
    found: list[str] = []
    for text in _texts(narrative):
        found.extend(unknown_numbers_in_text(text, facts))
    return found


def unknown_numbers_in_text(text: str, facts: str) -> list[str]:
    """Numbers in `text` that do not appear in the facts text (possible hallucinations)."""
    pools = _sign_pools(facts)
    anywhere = pools["-"] | pools["+"] | pools[""]
    counts = _count_numbers(text)
    found: list[str] = []
    for raw in NUMBER.findall(text):
        value = _norm(raw)
        sign = raw.strip()[0] if _has_sign(raw) else ""
        # An unsigned number only has to exist in the facts ("05 Jan" is "5 Jan"). A number the
        # model signed itself must match a fact carrying that same sign, or one carrying none:
        # facts that only ever write "-1,816" must never be quoted as "+1,816", because a
        # flipped sign is the worst numeric error this tool can make.
        if value in (anywhere if not sign else pools[sign] | pools[""]):
            continue
        if not value.endswith("%") and _small_count(value) and value in counts:
            continue
        found.append(raw.strip())
    return found


def banned_phrases(text: str) -> list[str]:
    """Causal words the report must not use ("caused", "because", "due to"), lowercased."""
    return [m.group(0).lower() for m in BANNED_RE.finditer(text)]


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


def _rounded(values: set[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        if "." in value and not value.endswith("%"):
            try:
                out.add(str(round(float(value))))
            except ValueError:
                continue
    return out


def _small_count(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return False
    return number.is_integer() and abs(number) <= SMALL_COUNT_LIMIT


def _norm(number: str) -> str:
    """Comparable form: no separators, whitespace, sign or leading zeros (05 Jan == 5 Jan)."""
    core = "".join(number.split()).replace(",", "").lstrip("+-")
    return core.lstrip("0") or "0" if not core.startswith("0.") else core


def _has_sign(number: str) -> bool:
    return number.strip().startswith(("+", "-"))


def _sign_pools(facts: str) -> dict[str, set[str]]:
    """Magnitudes in the facts, grouped by the sign they were written with ("-", "+", or none).

    A magnitude the facts write bare carries its direction in words ("above normal by 2,383 MW"),
    so prose may sign it either way. One the facts only ever write signed ("-1,816", "z = +27.9")
    has to keep that sign.
    """
    pools: dict[str, set[str]] = {"-": set(), "+": set(), "": set()}
    for raw in NUMBER.findall(facts):
        pools[raw.strip()[0] if _has_sign(raw) else ""].add(_norm(raw))
    for pool in pools.values():
        pool |= _rounded(pool)  # "z = 30.3" may be quoted as "30"
    return pools
