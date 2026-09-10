"""Streamlit demo. Run: uv run streamlit run app.py

Flow: pick a date range and Scan -> table of abnormal episodes -> pick one -> investigation.
Or open "Any hour" to investigate a specific hour even if it is not abnormal.
"""

import logging
from datetime import date

import pandas as pd
import streamlit as st

from copilot.config import load_settings
from copilot.data.frame import MarketFrame
from copilot.detect import Event
from copilot.drivers.base import Verdict
from copilot.investigate import Investigation, investigate_at, investigate_event, load_window, scan
from copilot.llm import narrate
from copilot.plots import all_figures
from copilot.report import Narrative, fallback_narrative, render_facts
from copilot.timeutil import helsinki

logging.basicConfig(level=logging.WARNING)
TZ = "Europe/Helsinki"
BADGE = {
    Verdict.SUPPORTS: "🟠 supports",
    Verdict.DOES_NOT_SUPPORT: "⚪ does not support",
    Verdict.INSUFFICIENT: "❔ not enough data",
}
HISTORY_DAYS = 30  # fetched before the range so the 28-day baseline exists on day one

st.set_page_config(page_title="Energy Market Copilot", layout="wide")
st.title("Energy Market Copilot")
st.caption(
    "Finland, day-ahead price. Facts from data, hypotheses clearly labelled, nothing proven."
)
settings = load_settings()


@st.cache_data(show_spinner="Fetching market data (ENTSO-E can take a minute the first time)...")
def cached_window(start: str, end: str) -> pd.DataFrame:
    return load_window(settings, helsinki(start), helsinki(end)).data


@st.cache_data(show_spinner="Writing the narrative...")
def cached_narrative(facts: str, model: str, _inv: Investigation) -> dict:
    return narrate(_inv, model=model).model_dump()


def narrative_for(inv: Investigation, use_llm: bool) -> Narrative:
    if not use_llm:
        return fallback_narrative(inv)
    return Narrative(**cached_narrative(render_facts(inv), settings.copilot_model, inv))


with st.sidebar:
    st.header("1. Scan a date range")
    start = st.date_input("From", value=date(2023, 12, 8))
    end = st.date_input("To", value=date(2024, 1, 8))
    if st.button("Scan for abnormal hours", type="primary"):
        st.session_state["range"] = (str(start), str(end))
        st.session_state.pop("any_hour", None)
    st.header("Or: any hour")
    day = st.date_input("Day", value=date(2024, 1, 5))
    hour = st.slider("Hour (Helsinki)", 0, 23, 19)
    if st.button("Investigate this hour"):
        st.session_state["any_hour"] = f"{day}T{hour:02d}:00"
    st.divider()
    use_llm = st.toggle(
        "Write narrative with LLM", value=True, help=f"Model: {settings.copilot_model}"
    )


def show_investigation(inv: Investigation, narrative: Narrative) -> None:
    e = inv.event
    kind = "not flagged as abnormal" if not e.flagged else e.kind
    st.subheader(
        f"{e.peak_time.tz_convert(TZ):%a %d %b %Y %H:%M}: {e.peak_price:,.0f} EUR/MWh ({kind})"
    )
    cols = st.columns(4)
    cols[0].metric("Peak price", f"{e.peak_price:,.0f} EUR/MWh")
    cols[1].metric(
        "Same-hour baseline", f"{e.baseline_median:,.0f} EUR/MWh", f"{e.deviation:+,.0f}"
    )
    cols[2].metric("Robust z", f"{e.z:+.1f}")
    cols[3].metric("Event length", f"{e.hours} h")
    st.markdown("### What happened")
    st.write(narrative.summary)
    left, right = st.columns(2)
    with left:
        st.markdown("**Facts (from data)**")
        for item in narrative.facts:
            st.markdown(f"- {item}")
    with right:
        st.markdown("**Hypotheses (consistent with the evidence, not proven)**")
        for item in narrative.hypotheses:
            st.markdown(f"- {item}")
        if narrative.insufficient:
            st.markdown("**Not enough data**")
            for item in narrative.insufficient:
                st.markdown(f"- {item}")
    figures = all_figures(inv)
    st.plotly_chart(figures["price"], use_container_width=True)
    st.markdown("### Driver checks")
    for result in inv.results:
        expanded = result.verdict is Verdict.SUPPORTS
        with st.expander(f"{BADGE[result.verdict]} · {result.title}", expanded=expanded):
            st.markdown(f"*Hypothesis tested:* {result.hypothesis}")
            st.markdown(result.detail)
            fig = figures.get(result.name)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw facts (deterministic report)"):
        st.markdown(render_facts(inv))


def events_table(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": [str(e.kind) for e in events],
            "start (Helsinki)": [e.start.tz_convert(TZ).strftime("%a %d %b %H:%M") for e in events],
            "hours": [e.hours for e in events],
            "peak EUR/MWh": [round(e.peak_price) for e in events],
            "baseline": [round(e.baseline_median) for e in events],
            "z": [round(float(e.z), 1) for e in events],
        }
    )


if "any_hour" in st.session_state:
    when = helsinki(st.session_state["any_hour"])
    window_start = when.floor("D") - pd.Timedelta(days=HISTORY_DAYS)
    data = cached_window(str(window_start), str(when.floor("D") + pd.Timedelta(days=2)))
    inv = investigate_at(MarketFrame(data=data), when)
    show_investigation(inv, narrative_for(inv, use_llm))
elif "range" in st.session_state:
    start_s, end_s = st.session_state["range"]
    fetch_start = helsinki(start_s) - pd.Timedelta(days=HISTORY_DAYS)
    frame = MarketFrame(data=cached_window(str(fetch_start), end_s))
    events = scan(frame, top_n=10, since=helsinki(start_s))
    st.markdown(f"### Abnormal episodes, {start_s} to {end_s}")
    if not events:
        st.info("No abnormal hours in that range.")
    else:
        table = events_table(events)
        st.dataframe(table, hide_index=True, use_container_width=True)
        pick = st.selectbox(
            "2. Pick an episode to investigate",
            options=range(len(events)),
            format_func=lambda i: (
                f"{table['start (Helsinki)'][i]} · {table['kind'][i]} · {table['peak EUR/MWh'][i]} EUR/MWh"
            ),
        )
        inv = investigate_event(frame, events[pick])
        show_investigation(inv, narrative_for(inv, use_llm))
else:
    st.info(
        "Scan a date range on the left (Dec 2023 to Jan 2024 is cached), then pick an episode. Or investigate any hour."
    )
