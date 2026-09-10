"""Streamlit demo. Run: uv run streamlit run app.py

Two ways in: find abnormal hours in a date range, or check one specific hour.
Either way ends in the same investigation view.
"""

import logging
from datetime import date

import pandas as pd
import streamlit as st

from copilot.config import load_settings
from copilot.data.frame import MarketFrame
from copilot.detect import Event, EventKind
from copilot.drivers.base import Verdict
from copilot.investigate import Investigation, investigate_at, investigate_event, load_window, scan
from copilot.llm import narrate
from copilot.plots import all_figures
from copilot.report import Narrative, fallback_narrative, render_facts
from copilot.timeutil import helsinki

logging.basicConfig(level=logging.WARNING)
TZ = "Europe/Helsinki"
HISTORY_DAYS = 30  # fetched before the range so the 28-day baseline exists on day one
KIND_WORD = {
    EventKind.SPIKE: "Price spike",
    EventKind.CRASH: "Price crash",
    EventKind.NEGATIVE: "Negative price",
}
VERDICT_WORD = {
    Verdict.SUPPORTS: "Supports",
    Verdict.DOES_NOT_SUPPORT: "Does not support",
    Verdict.INSUFFICIENT: "Not enough data",
}
VERDICT_ICON = {Verdict.SUPPORTS: "🟠", Verdict.DOES_NOT_SUPPORT: "⚪", Verdict.INSUFFICIENT: "❔"}

st.set_page_config(page_title="Energy Market Copilot", page_icon="⚡", layout="wide")
settings = load_settings()


@st.cache_data(show_spinner="Loading market data. ENTSO-E is slow the first time, about a minute.")
def cached_window(start: str, end: str) -> pd.DataFrame:
    return load_window(settings, helsinki(start), helsinki(end)).data


@st.cache_data(show_spinner="Writing the summary...")
def cached_narrative(facts: str, model: str, _inv: Investigation) -> dict:
    return narrate(_inv, model=model).model_dump()


def narrative_for(inv: Investigation, ai: bool) -> Narrative:
    if not ai:
        return fallback_narrative(inv)
    return Narrative(**cached_narrative(render_facts(inv), settings.copilot_model, inv))


def events_table(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "What": [KIND_WORD[e.kind] for e in events],
            "Starts (Helsinki)": [
                e.start.tz_convert(TZ).strftime("%a %d %b %H:%M") for e in events
            ],
            "Hours": [e.hours for e in events],
            "Peak EUR/MWh": [round(e.peak_price) for e in events],
            "Normal EUR/MWh": [round(e.baseline_median) for e in events],
            "How unusual (z)": [round(float(e.z), 1) for e in events],
        }
    )


def show_investigation(inv: Investigation, narrative: Narrative) -> None:
    e = inv.event
    when = e.peak_time.tz_convert(TZ)
    if e.flagged:
        st.header(
            f"{KIND_WORD[e.kind]}: {e.peak_price:,.0f} EUR/MWh on {when:%a %d %b %Y} at {when:%H:%M}"
        )
    else:
        st.header(f"{e.peak_price:,.0f} EUR/MWh on {when:%a %d %b %Y} at {when:%H:%M}")
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

    figures = all_figures(inv)
    st.plotly_chart(figures["price"], use_container_width=True)

    st.subheader("Driver checks")
    groups = (
        (
            Verdict.SUPPORTS,
            "Evidence supports",
            "These moved the way that pushes the price where it went.",
        ),
        (
            Verdict.DOES_NOT_SUPPORT,
            "Evidence does not support",
            "Checked, but they were normal or moved the other way.",
        ),
        (Verdict.INSUFFICIENT, "Could not check", "Data missing for these hours."),
    )
    for verdict, heading, note in groups:
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


st.title("Energy Market Copilot")
st.caption(
    "Finnish day-ahead price. What happened, the numbers behind it, and which drivers the evidence supports."
)

find_tab, check_tab = st.tabs(["Find abnormal hours", "Check one hour"])
with find_tab:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
    start = c1.date_input("From", value=date(2023, 12, 8))
    end = c2.date_input("To", value=date(2024, 1, 8))
    if c3.button("Find", type="primary", use_container_width=True):
        st.session_state["mode"] = ("range", str(start), str(end))
    ai = c4.toggle(
        "AI summary",
        value=True,
        help=f"On: {settings.copilot_model} writes the summary from the facts. Off: a fixed template. Numbers and charts do not change.",
    )
with check_tab:
    c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    day = c1.date_input("Day", value=date(2024, 1, 5))
    hour = c2.slider("Hour (Helsinki)", 0, 23, 19)
    if c3.button("Explain this hour", type="primary", use_container_width=True):
        st.session_state["mode"] = ("hour", f"{day}T{hour:02d}:00")

mode = st.session_state.get("mode")
if mode is None:
    st.info(
        "Pick a date range and press Find, or pick one hour and press Explain. December 2023 and January 2024 are ready offline."
    )
elif mode[0] == "hour":
    when = helsinki(mode[1])
    day0 = when.floor("D")
    data = cached_window(
        str(day0 - pd.Timedelta(days=HISTORY_DAYS)), str(day0 + pd.Timedelta(days=2))
    )
    inv = investigate_at(MarketFrame(data=data), when)
    show_investigation(inv, narrative_for(inv, ai))
else:
    _, start_s, end_s = mode
    frame = MarketFrame(
        data=cached_window(str(helsinki(start_s) - pd.Timedelta(days=HISTORY_DAYS)), end_s)
    )
    events = scan(frame, top_n=10, since=helsinki(start_s))
    if not events:
        st.info(
            f"No abnormal hours between {start_s} and {end_s}. Try a wider range, or check one hour."
        )
    else:
        st.subheader(f"{len(events)} abnormal episodes, strongest first. Click one.")
        picked = st.dataframe(
            events_table(events),
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = picked.selection.rows if picked and picked.selection else []
        inv = investigate_event(frame, events[rows[0] if rows else 0])
        show_investigation(inv, narrative_for(inv, ai))
