"""Streamlit demo. Run: uv run streamlit run app.py"""

import logging
from datetime import date

import pandas as pd
import streamlit as st

from copilot.config import load_settings
from copilot.data.frame import MarketFrame
from copilot.drivers.base import Verdict
from copilot.investigate import Investigation, investigate_at, load_window, scan, window_for
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

st.set_page_config(page_title="Energy Market Copilot", layout="wide")
st.title("Energy Market Copilot")
st.caption(
    "Finland, day-ahead price. Facts from data, hypotheses clearly labelled, nothing proven."
)

settings = load_settings()


@st.cache_data(show_spinner="Fetching market data (ENTSO-E can take a minute the first time)...")
def cached_window(start: str, end: str) -> pd.DataFrame:
    return load_window(settings, helsinki(start), helsinki(end)).data


with st.sidebar:
    st.header("What to look at")
    mode = st.radio("Mode", ["Investigate an hour", "Scan a date range"])
    use_llm = st.toggle(
        "Write narrative with LLM", value=True, help=f"Model: {settings.copilot_model}"
    )
    when = helsinki("2024-01-05T19:00")
    start, end = date(2023, 12, 8), date(2024, 1, 8)
    if mode == "Investigate an hour":
        day = st.date_input("Day", value=date(2024, 1, 5))
        hour = st.slider("Hour (Helsinki)", 0, 23, 19)
        when = helsinki(f"{day}T{hour:02d}:00")
    else:
        start = st.date_input("From", value=start)
        end = st.date_input("To", value=end)
    run = st.button("Run", type="primary")


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
        with st.expander(
            f"{BADGE[result.verdict]} · {result.title}", expanded=result.verdict is Verdict.SUPPORTS
        ):
            st.markdown(f"*Hypothesis tested:* {result.hypothesis}")
            st.markdown(result.detail)
            fig = figures.get(result.name)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw facts (deterministic report)"):
        st.markdown(render_facts(inv))


if run and mode == "Investigate an hour":
    start_ts, end_ts = window_for(when)
    data = cached_window(str(start_ts), str(end_ts))
    inv = investigate_at(MarketFrame(data=data), when)
    narrative = narrate(inv, model=settings.copilot_model) if use_llm else fallback_narrative(inv)
    show_investigation(inv, narrative)
elif run:
    data = cached_window(str(helsinki(str(start))), str(helsinki(str(end))))
    frame = MarketFrame(data=data)
    events = scan(frame, top_n=8)
    if not events:
        st.info("No abnormal hours in that range.")
    else:
        table = pd.DataFrame(
            {
                "kind": [e.kind for e in events],
                "start (Helsinki)": [
                    e.start.tz_convert(TZ).strftime("%Y-%m-%d %H:%M") for e in events
                ],
                "hours": [e.hours for e in events],
                "peak EUR/MWh": [round(e.peak_price) for e in events],
                "baseline": [round(e.baseline_median) for e in events],
                "z": [round(e.z, 1) for e in events],
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        pick = st.selectbox(
            "Investigate",
            options=range(len(events)),
            format_func=lambda i: table["start (Helsinki)"][i],
        )
        inv = investigate_at(frame, events[pick].peak_time)
        narrative = (
            narrate(inv, model=settings.copilot_model) if use_llm else fallback_narrative(inv)
        )
        show_investigation(inv, narrative)
else:
    st.info(
        "Pick an hour or a range on the left and press Run. Try 5 Jan 2024 19:00 or scan Dec 2023."
    )
