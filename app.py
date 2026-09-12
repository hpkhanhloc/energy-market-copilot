"""Streamlit demo. Run: uv run streamlit run app.py

A chat. One LLM call turns your words into a typed intent (scan a range, explain one hour,
answer a question, or reply). Plain code runs the scan and the driver checks. Starter chips and
table clicks never call the LLM. See views.py for the turn handling.
"""

import logging

import streamlit as st

from copilot.config import load_settings
from copilot.trace import last_call
from views import (
    STARTERS,
    chip_picked,
    clear_chat,
    handle_intent,
    handle_text,
    is_clear_command,
    show_turn,
    turns,
    user_text,
)

logging.basicConfig(level=logging.WARNING)
st.set_page_config(
    page_title="Energy Market Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
settings = load_settings()
st.session_state["settings"] = settings

with st.sidebar:
    st.header("Settings")
    ai = st.toggle(
        "AI-written summary",
        value=True,
        help=f"On: {settings.copilot_model} writes the summary from the facts. "
        "Off: a fixed template. Numbers, verdicts and charts do not change either way.",
    )
    st.caption(f"Model: `{settings.copilot_model}` (COPILOT_MODEL in .env)")
    st.button("Clear chat", on_click=clear_chat, width="stretch")
    with st.expander("Developer: last LLM call"):
        call = last_call()
        if call is None:
            st.caption("No calls yet.")
        else:
            st.json(
                {
                    "kind": call.kind,
                    "ok": call.ok,
                    "guard": call.guard,
                    "fallback": call.fallback,
                    "latency_ms": call.latency_ms,
                    "output": call.output,
                },
                expanded=False,
            )
            st.text_area("Prompt", call.input, height=200, disabled=True)
        st.caption("Every call is appended to data/logs/llm.jsonl.")

st.title("Energy Market Copilot")
st.caption(
    "Finnish day-ahead price. Ask what happened, get the numbers, the charts, and which drivers "
    "the evidence supports."
)
st.pills("Try one", list(STARTERS), key="chip", on_change=chip_picked, label_visibility="collapsed")

history = turns()
last_inv = max((i for i, t in enumerate(history) if t["kind"] == "investigation"), default=-1)
for i, turn in enumerate(history):
    with st.chat_message(turn["role"]):
        show_turn(turn, latest_investigation=i == last_inv)

if history:
    st.button("Start a new chat", on_click=clear_chat, icon=":material/delete_sweep:")

pending = st.session_state.pop("pending", None)
if pending is not None:
    label, intent = pending
    user_text(label)
    handle_intent(intent, settings, ai=ai)
    st.rerun()

if text := st.chat_input("Ask me, e.g. what happened on 5 Jan 2024 at 19:00?"):
    if is_clear_command(text):
        clear_chat()
    else:
        handle_text(text, settings, ai=ai)
    st.rerun()
