"""The routing LLM call: user text in, one typed intent out. Plain code runs whatever it names.

The model never sees market data here. It maps words to a Scan / Investigate / Ask / Reply and
`guard_intent` re-checks every date it returns before any data is loaded.
"""

import calendar
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from copilot.investigate import HISTORY_DAYS
from copilot.report import banned_phrases, unknown_numbers_in_text
from copilot.timeutil import helsinki
from copilot.trace import Guard, record_call, timed

log = logging.getLogger(__name__)

KIND = "intent"

MAX_SCAN_DAYS = 60  # for ranges that need a network fetch
MAX_CACHED_SCAN_DAYS = 366  # every month already on disk: reading parquet is cheap
TRANSCRIPT_LINES = 6
FALLBACK_TEXT = (
    "I could not read that. Try: 'what happened on 5 Jan 2024 at 19:00' or "
    "'find odd hours from 2023-12-08 to 2024-01-08'."
)
FUTURE_TEXT = "That date is in the future. Day-ahead prices only exist for published days."
SCOPE_TEXT = (
    "I only look at the Finnish day-ahead power price: find abnormal hours in a date range, "
    "explain one hour, or answer questions about the current report."
)


class Scan(BaseModel):
    """Find abnormal hours between two dates (inclusive, Helsinki calendar days)."""

    start: date
    end: date


TRAILING_ZONE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


class Investigate(BaseModel):
    """Explain one specific hour. `when` is Helsinki local time, full date and hour required."""

    when: datetime

    @field_validator("when", mode="before")
    @classmethod
    def _model_text_is_helsinki(cls, value: object) -> object:
        # The model is told times are Helsinki. When it still appends "Z" or an offset, that is
        # decoration, not a conversion request: "2024-01-05T19:00Z" means 19:00 Helsinki.
        # Datetime objects built by code keep their zone and are converted.
        if isinstance(value, str):
            return TRAILING_ZONE.sub("", value.strip())
        return value

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
    cached_months: frozenset[str] = frozenset()  # "YYYYMM" with a price file on disk


INSTRUCTIONS = """You route messages for an energy-market copilot that explains abnormal hours
of the Finnish day-ahead electricity price. Return exactly one of:

- Scan: the user wants to find abnormal / odd / interesting hours in a date range.
- Investigate: the user names one specific hour to explain (date and hour both given).
  Times are Europe/Helsinki; write them without a zone suffix.
  If no clock hour is given ("tell me about 4 Jan 2024", "16 Dec 2023 evening", "last Friday
  night"), return Scan with start = end = that day instead: code finds the abnormal hours of
  that day. "Evening", "morning" or "night" is not an hour. Never choose an hour yourself.
- Ask: a question about the report already on screen, about a number in it, or about a market
  term (residual load, mFRR, day-ahead, z-score ...). Also "what about 21:00" style follow-ups.
- Reply: anything else. Out of scope (other countries, imbalance or intraday prices, gas, weather
  on its own, poems) gets a one-line note that you only handle the Finnish day-ahead price.
  Dates that cannot be pinned down (a day number with no month, a month with no year) get a
  short question back: "explain the 5th" is a Reply asking which month, not an Ask.
  Never put market numbers or claims in Reply.

Use CONTEXT to resolve relative dates: "the hour before" refers to the last investigated hour,
"widen that" refers to the last range, "yesterday" is relative to today. Any past date is fine:
months not cached yet are fetched. Dates after today get a Reply saying day-ahead prices do not
exist yet for them."""


def build_intent_agent(model: str) -> Agent[None, Intent]:
    return Agent[None, Intent](
        model, output_type=INTENT_TYPES, instructions=INSTRUCTIONS, retries=2
    )


def cached_months(cache_dir: Path) -> frozenset[str]:
    """Months ("YYYYMM") that have a cached FI price file."""
    return frozenset(p.stem.rsplit("_", 1)[-1] for p in cache_dir.glob("entsoe_price_FI_*.parquet"))


def months_between(start: date, end: date) -> list[str]:
    """Every "YYYYMM" from start to end inclusive."""
    out: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append(f"{year:04d}{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def data_reach(cache_dir: Path, today: date) -> tuple[date, date]:
    """First and last calendar day with cached FI price data, from the month files on disk."""
    months = sorted(cached_months(cache_dir))
    if not months:
        return today, today
    first = date(int(months[0][:4]), int(months[0][4:]), 1)
    year, month = int(months[-1][:4]), int(months[-1][4:])
    last = min(date(year, month, calendar.monthrange(year, month)[1]), today)
    return first, last


def render_context(ctx: Context) -> str:
    """The CONTEXT block sent with every routing call."""
    cached = (
        f"{ctx.reach_start.isoformat()} to {ctx.reach_end.isoformat()}"
        if ctx.cached_months
        else "none yet"
    )
    lines = [
        "CONTEXT:",
        f"today: {ctx.today.isoformat()}",
        f"cached data: {cached} (other past dates are fetched live, up to {MAX_SCAN_DAYS} days "
        "at a time)",
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
    prompt = f"{render_context(ctx)}\n\nUSER: {text.strip()}"
    result, latency = timed(lambda: (agent or build_intent_agent(model)).run_sync(prompt))
    if isinstance(result, Exception):
        log.warning("intent parsing failed (%s: %s)", type(result).__name__, result)
        record_call(KIND, model, prompt, repr(result), latency, guard="exception")
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
            record_call(KIND, model, prompt, intent.model_dump_json(), latency, guard=guard)
            return Reply(text=SCOPE_TEXT)
    record_call(KIND, model, prompt, intent.model_dump_json(), latency, guard=None)
    return intent


def guard_intent(intent: Intent, ctx: Context, *, max_days: int = MAX_SCAN_DAYS) -> Intent:
    """Deterministic checks on model output: range size, cache-aware fetch cap, future dates.

    Any past date is allowed: months not on disk are fetched live, which is why a range that
    needs uncached months is capped at `max_days`. Only the future is refused.
    """
    match intent:
        case Scan(start=start, end=end):
            if end < start:
                start, end = end, start
            if start > ctx.today:
                return Reply(text=FUTURE_TEXT)
            end = min(end, ctx.today)
            span = (end - start).days
            needed = months_between(start - timedelta(days=HISTORY_DAYS), end)
            missing = [m for m in needed if m not in ctx.cached_months]
            if span > MAX_CACHED_SCAN_DAYS:
                return Reply(text="I can scan at most one year at a time. Narrow the range.")
            if missing and span > max_days:
                return Reply(
                    text=f"That range needs {len(missing)} month(s) not cached yet, so I can "
                    f"scan at most {max_days} days of it at a time. Narrow the range, or warm "
                    "the cache first: uv run python scripts/warm_cache.py <start> <end>."
                )
            return Scan(start=start, end=end)
        case Investigate(when=when):
            if when.date() > ctx.today:
                return Reply(text=FUTURE_TEXT)
            return intent
        case _:
            return intent
