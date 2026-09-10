"""Streamlit glue for the chat: turn handling and rendering. Logic lives in `copilot/`.

A turn is a dict: {"role": "user"|"assistant", "kind": "text"|"scan"|"investigation"|"error", ...}.
Every rerun re-renders the stored turns; nothing is recomputed.
"""

from datetime import date, datetime, timedelta
from functools import partial
from typing import Any

import pandas as pd
import streamlit as st

from copilot.chat import answer_question
from copilot.config import Settings
from copilot.data.frame import MarketFrame
from copilot.detect import Event, EventKind
from copilot.drivers.base import Verdict
from copilot.intent import (
    Ask,
    Context,
    Intent,
    Investigate,
    Reply,
    Scan,
    cached_months,
    data_reach,
    guard_intent,
    parse_intent,
)
from copilot.investigate import Investigation, investigate_at, load_window, scan, window_for
from copilot.llm import narrate
from copilot.plots import all_figures
from copilot.report import Narrative, fallback_narrative, render_facts
from copilot.timeutil import helsinki

TZ = "Europe/Helsinki"
HISTORY_DAYS = 30  # fetched before a scan range so the 28-day baseline exists on day one
MAX_TURNS = 30
KIND_WORD = {
    EventKind.SPIKE: "Price spike",
    EventKind.CRASH: "Price crash",
    EventKind.NEGATIVE: "Negative price",
}
VERDICT_ICON = {Verdict.SUPPORTS: "🟠", Verdict.DOES_NOT_SUPPORT: "⚪", Verdict.INSUFFICIENT: "❔"}
VERDICT_GROUPS = (
    (
        Verdict.SUPPORTS,
        "Evidence supports",
        "These moved in the direction consistent with the price move.",
    ),
    (
        Verdict.DOES_NOT_SUPPORT,
        "Evidence does not support",
        "Checked, but normal or moved the other way.",
    ),
    (Verdict.INSUFFICIENT, "Could not check", "Data missing for these hours."),
)
# Starter chips map straight to intents: no LLM call, so the demo works with no key.
STARTERS: dict[str, Intent] = {
    "Spike on 5 Jan 2024 19:00": Investigate(when=datetime(2024, 1, 5, 19)),  # noqa: DTZ001
    "Negative night 16 Dec 2023": Investigate(when=datetime(2023, 12, 16, 19)),  # noqa: DTZ001
    "Odd hours 8 Dec 2023 to 8 Jan 2024": Scan(start=date(2023, 12, 8), end=date(2024, 1, 8)),
}
NOTHING_YET = "Nothing investigated yet. Ask me to explain an hour first, or pick a starter above."


# ---------- data + narrative (cached) ----------


@st.cache_data(show_spinner="Loading market data. ENTSO-E is slow the first time, about a minute.")
def cached_window(start: str, end: str) -> pd.DataFrame:
    settings: Settings = st.session_state["settings"]
    return load_window(settings, helsinki(start), helsinki(end)).data


@st.cache_data(show_spinner="Writing the summary...")
def cached_narrative(facts: str, model: str, _inv: Investigation) -> dict:
    return narrate(_inv, model=model).model_dump()


def narrative_for(inv: Investigation, *, model: str, ai: bool) -> Narrative:
    if not ai:
        return fallback_narrative(inv)
    return Narrative(**cached_narrative(render_facts(inv), model, inv))


# ---------- turns ----------


def turns() -> list[dict[str, Any]]:
    return st.session_state.setdefault("turns", [])


def add_turn(turn: dict[str, Any]) -> None:
    items = turns()
    items.append(turn)
    del items[:-MAX_TURNS]


def user_text(text: str) -> None:
    add_turn({"role": "user", "kind": "text", "text": text})


def assistant_text(text: str, label: str | None = None) -> None:
    add_turn({"role": "assistant", "kind": "text", "text": text, "label": label})


def current_investigation() -> dict[str, Any] | None:
    return next((t for t in reversed(turns()) if t["kind"] == "investigation"), None)


def last_scan() -> dict[str, Any] | None:
    return next((t for t in reversed(turns()) if t["kind"] == "scan"), None)


def transcript() -> tuple[tuple[str, str], ...]:
    return tuple((t["role"], t["text"]) for t in turns() if t["kind"] == "text")


def context(settings: Settings) -> Context:
    today = pd.Timestamp.now(tz=TZ).date()
    start, end = data_reach(settings.cache_dir, today)
    inv, sc = current_investigation(), last_scan()
    return Context(
        today=today,
        reach_start=start,
        reach_end=end,
        last_hour=inv["when"].to_pydatetime() if inv else None,
        last_range=(sc["start"], sc["end"]) if sc else None,
        transcript=transcript(),
        cached_months=cached_months(settings.cache_dir),
    )


# ---------- handlers ----------


def handle_text(text: str, settings: Settings, *, ai: bool) -> None:
    """Free text: one LLM routing call, then plain code."""
    ctx = context(settings)
    user_text(text)
    intent = guard_intent(parse_intent(text, ctx, model=settings.copilot_model), ctx)
    handle_intent(intent, settings, ai=ai)


def handle_intent(intent: Intent, settings: Settings, *, ai: bool) -> None:
    try:
        match intent:
            case Scan(start=start, end=end):
                add_turn(run_scan(start, end))
            case Investigate(when=when):
                add_turn(run_investigate(helsinki(when), settings, ai=ai))
            case Ask(question=question):
                inv = current_investigation()
                if inv is None:
                    assistant_text(NOTHING_YET)
                    return
                answer = answer_question(
                    question, inv["inv"], transcript()[:-1], model=settings.copilot_model
                )
                label = None
                if answer.source == "general":
                    label = "General knowledge, not from your data."
                elif answer.hour_not_in_report:
                    label = "Not in the current report. Ask me to investigate that hour."
                assistant_text(answer.text, label)
            case Reply(text=text):
                assistant_text(text)
    except Exception as exc:  # data loading, ENTSO-E, missing key: show it, keep chatting
        add_turn({"role": "assistant", "kind": "error", "text": f"{type(exc).__name__}: {exc}"})


def run_scan(start: date, end: date) -> dict[str, Any]:
    since = helsinki(str(start))
    data = cached_window(str(since - pd.Timedelta(days=HISTORY_DAYS)), str(end + timedelta(days=1)))
    events = scan(MarketFrame(data=data), top_n=10, since=since)
    return {"role": "assistant", "kind": "scan", "start": start, "end": end, "events": events}


def run_investigate(when: pd.Timestamp, settings: Settings, *, ai: bool) -> dict[str, Any]:
    start, end = window_for(when)
    inv = investigate_at(MarketFrame(data=cached_window(str(start), str(end))), when)
    return {
        "role": "assistant",
        "kind": "investigation",
        "when": when.tz_convert(TZ),
        "inv": inv,
        "narrative": narrative_for(inv, model=settings.copilot_model, ai=ai),
        "figures": all_figures(inv),
    }


# ---------- widget callbacks (queue work; the main script runs it) ----------


def chip_picked() -> None:
    label = st.session_state.get("chip")
    if label:
        st.session_state["pending"] = (label, STARTERS[label])
    st.session_state["chip"] = None


def row_picked(turn_index: int) -> None:
    label = st.session_state.get(f"pick_{turn_index}")
    if not label:
        return
    events: list[Event] = turns()[turn_index]["events"]
    event = events[[episode_label(e) for e in events].index(label)]
    when = event.peak_time.tz_convert(TZ)
    st.session_state["pending"] = (f"Explain {when:%a %d %b %Y %H:%M}", Investigate(when=when))


def clear_chat() -> None:
    st.session_state["turns"] = []
    st.session_state["pending"] = None
    for key in [k for k in st.session_state if str(k).startswith("pick_")]:
        del st.session_state[key]


# ---------- rendering ----------


def show_turn(i: int, turn: dict[str, Any], *, latest_investigation: bool) -> None:
    match turn["kind"]:
        case "text":
            st.write(turn["text"])
            if turn.get("label"):
                st.caption(turn["label"])
        case "error":
            st.error(turn["text"])
        case "scan":
            show_scan(i, turn)
        case "investigation":
            if latest_investigation:
                show_investigation(turn["inv"], turn["narrative"], turn["figures"])
            else:
                with st.expander(investigation_title(turn["inv"])):
                    show_investigation(turn["inv"], turn["narrative"], turn["figures"])


def show_scan(i: int, turn: dict[str, Any]) -> None:
    events: list[Event] = turn["events"]
    if not events:
        st.write(
            f"No abnormal hours between {turn['start']} and {turn['end']}. "
            "Try a wider range, or name one hour."
        )
        return
    st.write(
        f"{len(events)} abnormal episodes between {turn['start']:%d %b %Y} and "
        f"{turn['end']:%d %b %Y}, most unusual first."
    )
    st.dataframe(events_table(events), hide_index=True, use_container_width=True)
    st.caption(
        "Usual = median price for the same hour over the previous 28 days. "
        "Rarity (z) = how many robust standard deviations the peak is from usual; "
        "above 4 counts as abnormal."
    )
    st.selectbox(
        "Pick an episode to investigate",
        [episode_label(e) for e in events],
        index=None,
        placeholder="Pick an episode to investigate",
        key=f"pick_{i}",
        on_change=partial(row_picked, i),
        label_visibility="collapsed",
    )


def episode_label(e: Event) -> str:
    start = e.start.tz_convert(TZ)
    return (
        f"{start:%a %d %b %Y %H:%M}, {KIND_WORD[e.kind].lower()}, "
        f"peak {e.peak_price:,.0f} vs usual {e.baseline_median:,.0f} EUR/MWh, {e.hours} h"
    )


def events_table(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "When (Helsinki)": [
                e.start.tz_convert(TZ).strftime("%a %d %b %Y %H:%M") for e in events
            ],
            "Lasted": [f"{e.hours} h" for e in events],
            "What": [KIND_WORD[e.kind] for e in events],
            "Peak EUR/MWh": [round(e.peak_price) for e in events],
            "Usual EUR/MWh": [round(e.baseline_median) for e in events],
            "Difference EUR/MWh": [f"{e.deviation:+,.0f}" for e in events],
            "Rarity (z)": [round(float(e.z), 1) for e in events],
        }
    )


def investigation_title(inv: Investigation) -> str:
    e = inv.event
    when = e.peak_time.tz_convert(TZ)
    head = f"{e.peak_price:,.0f} EUR/MWh on {when:%a %d %b %Y} at {when:%H:%M}"
    return f"{KIND_WORD[e.kind]}: {head}" if e.flagged else head


def show_investigation(inv: Investigation, narrative: Narrative, figures: dict) -> None:
    e = inv.event
    st.subheader(investigation_title(inv))
    if not e.flagged:
        st.caption("This hour is within its normal range. Analysed anyway.")
    m = st.columns(4)
    m[0].metric("Peak", f"{e.peak_price:,.0f} EUR/MWh")
    m[1].metric("Normal for this hour", f"{e.baseline_median:,.0f} EUR/MWh", f"{e.deviation:+,.0f}")
    m[2].metric("How unusual (z)", f"{e.z:+.1f}")
    m[3].metric("Lasted", f"{e.hours} h")

    st.write(narrative.summary)
    facts, hyps = st.columns(2)
    with facts:
        st.markdown("**What the data shows**")
        for item in narrative.facts:
            st.markdown(f"- {item}")
    with hyps:
        st.markdown("**What it is consistent with** (not proven)")
        for item in narrative.hypotheses:
            st.markdown(f"- {item}")
        if narrative.insufficient:
            st.markdown("**Could not check**")
            for item in narrative.insufficient:
                st.markdown(f"- {item}")

    st.plotly_chart(figures["price"], use_container_width=True)

    st.markdown("**Driver checks**")
    for verdict, heading, note in VERDICT_GROUPS:
        group = [r for r in inv.results if r.verdict is verdict]
        if not group:
            continue
        st.markdown(f"**{VERDICT_ICON[verdict]} {heading}** ({len(group)})")
        st.caption(note)
        for r in group:
            with st.expander(r.title, expanded=verdict is Verdict.SUPPORTS):
                st.markdown(f"Hypothesis tested: {r.hypothesis}")
                st.markdown(r.detail)
                fig = figures.get(r.name)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Full report as text"):
        st.markdown(render_facts(inv))
