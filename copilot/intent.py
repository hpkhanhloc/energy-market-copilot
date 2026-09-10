"""The routing LLM call: user text in, one typed intent out. Plain code runs whatever it names.

The model never sees market data here. It maps words to a Scan / Investigate / Ask / Reply and
`guard_intent` re-checks every date it returns before any data is loaded.
"""

import calendar
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from copilot.report import banned_phrases, unknown_numbers_in_text
from copilot.timeutil import helsinki
from copilot.trace import Guard, LlmCall, now_iso, record, timed

log = logging.getLogger(__name__)

MAX_SCAN_DAYS = 60
HISTORY_DAYS = 30  # same-hour baseline needs this much history before any requested day
TRANSCRIPT_LINES = 6
FALLBACK_TEXT = (
    "I could not read that. Try: 'what happened on 5 Jan 2024 at 19:00' or "
    "'find odd hours from 2023-12-08 to 2024-01-08'."
)
SCOPE_TEXT = (
    "I only look at the Finnish day-ahead power price: find abnormal hours in a date range, "
    "explain one hour, or answer questions about the current report."
)


class Scan(BaseModel):
    """Find abnormal hours between two dates (inclusive, Helsinki calendar days)."""

    start: date
    end: date


class Investigate(BaseModel):
    """Explain one specific hour. `when` is Helsinki local time, full date and hour required."""

    when: datetime

    @field_validator("when")
    @classmethod
    def _helsinki(cls, value: datetime) -> datetime:
        return helsinki(value).to_pydatetime()


class Ask(BaseModel):
    """A question about the current report or a market term; answered later from the facts."""

    question: str


class Reply(BaseModel):
    """Plain text back to the user: out of scope, unclear, or a question to pin down a date."""

    text: str = Field(description="Short, friendly, no market numbers or claims about data.")


# Union order matters: tests pick a member with TestModel(seed=index).
type Intent = Scan | Investigate | Ask | Reply
INTENT_TYPES: Sequence[type[Intent]] = (Scan, Investigate, Ask, Reply)


@dataclass(frozen=True, slots=True, kw_only=True)
class Context:
    """What the model is told so it can resolve relative dates; all computed by code."""

    today: date
    reach_start: date  # first day with cached price data (baseline needs 30 more days)
    reach_end: date
    last_hour: datetime | None = None
    last_range: tuple[date, date] | None = None
    transcript: tuple[tuple[str, str], ...] = ()  # (role, text), most recent last


INSTRUCTIONS = """You route messages for an energy-market copilot that explains abnormal hours
of the Finnish day-ahead electricity price. Return exactly one of:

- Scan: the user wants to find abnormal / odd / interesting hours in a date range.
- Investigate: the user names one specific hour to explain (date and hour both known).
  Times are Europe/Helsinki. If only a day is given, use 19:00 of that day.
- Ask: a question about the report already on screen, about a number in it, or about a market
  term (residual load, mFRR, day-ahead, z-score ...). Also "what about 21:00" style follow-ups.
- Reply: anything else. Out of scope (other countries, imbalance or intraday prices, gas, weather
  on its own, poems) gets a one-line note that you only handle the Finnish day-ahead price.
  Dates that cannot be pinned down (a day number with no month, a month with no year) get a
  short question back. Never put market numbers or claims in Reply.

Use CONTEXT to resolve relative dates: "the hour before" refers to the last investigated hour,
"widen that" refers to the last range, "yesterday" is relative to today. Only use dates inside
the available data range; if the user asks outside it, Reply with the available range."""


def build_intent_agent(model: str) -> Agent[None, Intent]:
    return Agent[None, Intent](
        model, output_type=INTENT_TYPES, instructions=INSTRUCTIONS, retries=2
    )


def data_reach(cache_dir: Path, today: date) -> tuple[date, date]:
    """First and last calendar day with cached FI price data, from the month files on disk."""
    months = sorted(p.stem.rsplit("_", 1)[-1] for p in cache_dir.glob("entsoe_price_FI_*.parquet"))
    if not months:
        return today, today
    first = date(int(months[0][:4]), int(months[0][4:]), 1)
    year, month = int(months[-1][:4]), int(months[-1][4:])
    last = min(date(year, month, calendar.monthrange(year, month)[1]), today)
    return first, last


def render_context(ctx: Context) -> str:
    """The CONTEXT block sent with every routing call."""
    lines = [
        "CONTEXT:",
        f"today: {ctx.today.isoformat()}",
        f"data available: {ctx.reach_start.isoformat()} to {ctx.reach_end.isoformat()}",
        f"last investigated hour: {ctx.last_hour:%Y-%m-%d %H:%M}" if ctx.last_hour else "",
        f"last scanned range: {ctx.last_range[0]} to {ctx.last_range[1]}" if ctx.last_range else "",
    ]
    if ctx.transcript:
        lines.append("recent conversation:")
        lines.extend(f"  {role}: {text}" for role, text in ctx.transcript[-TRANSCRIPT_LINES:])
    return "\n".join(line for line in lines if line)


def parse_intent(
    text: str, ctx: Context, *, model: str, agent: Agent[None, Intent] | None = None
) -> Intent:
    """Map user text to an intent; any model failure becomes a Reply, never an exception."""
    agent = agent or build_intent_agent(model)
    prompt = f"{render_context(ctx)}\n\nUSER: {text.strip()}"
    result, latency = timed(lambda: agent.run_sync(prompt))
    if isinstance(result, Exception):
        log.warning("intent parsing failed (%s: %s)", type(result).__name__, result)
        _trace(model, prompt, repr(result), latency, guard="exception")
        return Reply(text=FALLBACK_TEXT)
    intent: Intent = result.output
    guard: Guard | None = None
    if isinstance(intent, Reply):
        # Reply text reaches the screen as-is: numbers must come from the context or the
        # user's own words, and causal wording is out.
        if unknown_numbers_in_text(intent.text, prompt):
            guard = "unknown_numbers"
        elif banned_phrases(intent.text):
            guard = "banned_phrase"
        if guard:
            log.warning("routing reply failed guard %s: %r", guard, intent.text)
            _trace(model, prompt, intent.model_dump_json(), latency, guard=guard)
            return Reply(text=SCOPE_TEXT)
    _trace(model, prompt, intent.model_dump_json(), latency, guard=None)
    return intent


def guard_intent(intent: Intent, ctx: Context, *, max_days: int = MAX_SCAN_DAYS) -> Intent:
    """Deterministic checks on model output: range size, data reach, future dates."""
    reach = f"{ctx.reach_start:%d %b %Y} to {ctx.reach_end:%d %b %Y}"
    earliest = ctx.reach_start + timedelta(days=HISTORY_DAYS)
    match intent:
        case Scan(start=start, end=end):
            if end < start:
                start, end = end, start
            if (end - start).days > max_days:
                return Reply(
                    text=f"I can scan at most {max_days} days at a time. Narrow the range."
                )
            if start < earliest or end > ctx.reach_end:
                return Reply(
                    text=f"I have data from {earliest:%d %b %Y} to {ctx.reach_end:%d %b %Y}."
                )
            return Scan(start=start, end=end)
        case Investigate(when=when):
            day = when.date()
            if day > ctx.today:
                return Reply(
                    text="That hour is in the future. Day-ahead prices only exist for published days."
                )
            if day < earliest or day > ctx.reach_end:
                return Reply(
                    text=f"I have data from {earliest:%d %b %Y} to {ctx.reach_end:%d %b %Y} "
                    f"(cached from {reach}, the first 30 days are baseline only)."
                )
            return intent
        case _:
            return intent


def _trace(model: str, prompt: str, output: str, latency: int, *, guard: Guard | None) -> None:
    record(
        LlmCall(
            kind="intent",
            model=model,
            input=prompt,
            output=output,
            ok=guard is None,
            guard=guard,
            fallback=guard is not None,
            latency_ms=latency,
            ts=now_iso(),
        )
    )
