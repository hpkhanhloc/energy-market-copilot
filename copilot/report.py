"""Turn an Investigation into text. Facts are rendered by code; prose comes from the LLM.

The LLM never sees raw time series. It gets the deterministic facts below and must return
a typed `Narrative` whose numbers all exist in those facts (checked by `unknown_numbers`).
"""

import logging
import math
import re
from collections.abc import Iterable

import pandas as pd
from pydantic import BaseModel, Field

from copilot.config import TZ
from copilot.detect import EventKind
from copilot.drivers.base import Verdict
from copilot.investigate import Investigation

log = logging.getLogger(__name__)

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
    if e.flagged and e.max_ramp_time is not None and not math.isnan(e.max_ramp):
        top = e.max_ramp_time.tz_convert(TZ)
        before = top - pd.Timedelta(hours=1)
        lines.append(
            f"- Steepest hour-to-hour move: {e.max_ramp:+,.0f} EUR/MWh "
            f"({before:%H:%M} to {top:%H:%M})"
        )
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
    supporting = inv.supporting
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
        hypotheses=[r.hypothesis for r in supporting],
        insufficient=[r.detail for r in inv.results if r.verdict is Verdict.INSUFFICIENT],
    )


NO_VALUE = "—"
"""Printed where an hour has too little history for a baseline, so the number is NaN."""


def format_number(value: float, spec: str) -> str:
    """`spec.format(value)`, or `NO_VALUE` when there is no baseline behind the number."""
    return NO_VALUE if math.isnan(value) else spec.format(value)


NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s?%?")
MINUS_SIGNS = str.maketrans({"\u2212": "-", "\u2012": "-"})
"""Unicode minus and figure dash read as a sign, so a flipped sign cannot hide behind them."""
# Clock times and dates are checked as whole tokens, not as loose numbers: otherwise "19:00" and
# "05 Jan 2024" would put 19, 5 and 2024 into the pool of allowed values for every report.
CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})\b")
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DAY_MONTH = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)
MONTH_DAY = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
# Small integers that read as counts, never as values: "3 of 7 drivers", "n = 28", "top 5".
# Durations ("13 h", "28 prior days") are facts with numbers and must match the facts text.
COUNT_CONTEXT = re.compile(
    r"\bn\s*=\s*(?P<n>\d{1,2})\b"
    r"|\btop\s+(?P<top>\d{1,2})\b"
    r"|\b(?P<of_a>\d{1,2})\s+of\s+(?P<of_b>\d{1,2})\b"
    r"|\b(?P<noun>\d{1,2})\s+(?:drivers?|checks?|neighbou?rs?|borders?|series|sources?)\b",
    re.IGNORECASE,
)
SMALL_COUNT_LIMIT = 31


# Causal wording, with the verb forms the model actually produces. Matched case-insensitively.
BANNED = (
    r"caus(?:e|es|ed|ing)",
    r"because",
    r"due to",
    r"owing to",
    r"thanks to",
    r"led to",
    r"lead(?:s|ing)? to",
    r"result(?:ed|s|ing)? (?:in|from)",
    r"as a result",
    r"drove",
    r"driven by",
    r"drives? (?:up|down|prices?)",
    r"trigger(?:s|ed|ing)?",
    r"explain(?:s|ed) (?:by|the|this|that|why|it)",
    r"explained by",
    r"attributable to",
    r"stem(?:s|med|ming) from",
    r"responsible for",
    r"pushed (?:up|down|prices?)",
)
BANNED_RE = re.compile(r"\b(?:" + "|".join(BANNED) + r")\b", re.IGNORECASE)


def unknown_numbers(narrative: Narrative, facts: str) -> list[str]:
    """Numbers in the narrative that do not appear in the facts text (possible hallucinations)."""
    found: list[str] = []
    for text in narrative_texts(narrative):
        found.extend(unknown_numbers_in_text(text, facts))
    return found


def unknown_numbers_in_text(text: str, facts: str) -> list[str]:
    """Numbers, clock times and dates in `text` that the facts text does not contain."""
    text, text_times = _split_times(text.translate(MINUS_SIGNS))
    facts, fact_times = _split_times(facts.translate(MINUS_SIGNS))
    found: list[str] = [t for t in dict.fromkeys(text_times) if t not in set(fact_times)]
    pools = _sign_pools(facts)
    anywhere = pools["-"] | pools["+"] | pools[""]
    counts = _count_numbers(text)
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


def _split_times(text: str) -> tuple[str, list[str]]:
    """Pull clock times and dates out of `text` as normalised tokens ("19:00", "5 jan", "2024")."""
    tokens: list[str] = []

    def clock(m: re.Match[str]) -> str:
        tokens.append(f"{int(m.group(1)):02d}:{m.group(2)}")
        return " "

    def iso(m: re.Match[str]) -> str:
        tokens.extend((m.group(1), f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]}"))
        return " "

    def day_month(m: re.Match[str]) -> str:
        tokens.append(f"{int(m.group(1))} {m.group(2).lower()}")
        return " "

    def month_day(m: re.Match[str]) -> str:
        tokens.append(f"{int(m.group(2))} {m.group(1).lower()}")
        return " "

    def year(m: re.Match[str]) -> str:
        tokens.append(m.group(0))
        return " "

    text = CLOCK.sub(clock, text)
    text = ISO_DATE.sub(iso, text)
    text = DAY_MONTH.sub(day_month, text)
    text = MONTH_DAY.sub(month_day, text)
    text = YEAR.sub(year, text)
    return text, tokens


def banned_phrases(text: str) -> list[str]:
    """Causal wording the report must not use ("caused", "because", "driven by"), lowercased."""
    return [m.group(0).lower() for m in BANNED_RE.finditer(text)]


def _count_numbers(text: str) -> set[str]:
    """Small integers that read as counts ("n = 28", "3 of 7 drivers", "top 5"), not values."""
    return {
        value for m in COUNT_CONTEXT.finditer(text) for value in m.groupdict().values() if value
    }


def narrative_texts(narrative: Narrative) -> Iterable[str]:
    """Every string in a Narrative, for guards that check all of them the same way."""
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
