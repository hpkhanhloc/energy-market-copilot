import numpy as np
import pandas as pd
import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from copilot.chat import FALLBACK_ANSWER, Answer, answer_question, build_answer_agent
from copilot.data.frame import MarketFrame
from copilot.investigate import Investigation, investigate_at
from copilot.timeutil import ts
from copilot.trace import last_call

models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def investigation() -> Investigation:
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "price_fi": 60 + rng.normal(0, 3, len(idx)),
            "load": 10_000 + rng.normal(0, 100, len(idx)),
        },
        index=idx,
    )
    data.loc[ts("2024-01-05 17:00"), "price_fi"] = 1896.0
    return investigate_at(MarketFrame(data=data), ts("2024-01-05 17:00"))


def answer(investigation: Investigation, output: dict, history=()) -> Answer:
    agent = build_answer_agent("test")
    with agent.override(model=TestModel(custom_output_args=output)):
        return answer_question("why?", investigation, history, model="test", agent=agent)


def test_answer_from_report_passes_through(investigation: Investigation) -> None:
    out = answer(
        investigation, {"text": "The peak was 1,896 EUR/MWh, consistent with tight supply."}
    )
    assert out.text.startswith("The peak was 1,896")
    assert out.source == "report"
    call = last_call()
    assert call is not None
    assert call.kind == "answer"
    assert call.ok is True
    assert "QUESTION: why?" in call.input


def test_general_and_not_in_report_flags_survive(investigation: Investigation) -> None:
    general = answer(
        investigation,
        {"text": "Residual load is load minus wind and nuclear.", "source": "general"},
    )
    assert general.source == "general"
    missing = answer(
        investigation,
        {"text": "That hour is not in this report.", "hour_not_in_report": True},
    )
    assert missing.hour_not_in_report is True


def test_invented_number_falls_back(investigation: Investigation) -> None:
    out = answer(investigation, {"text": "Wind was 2,500 MW."})
    assert out.text == FALLBACK_ANSWER
    call = last_call()
    assert call is not None
    assert call.guard == "unknown_numbers"


def test_causal_wording_falls_back(investigation: Investigation) -> None:
    out = answer(investigation, {"text": "The spike happened because of low wind."})
    assert out.text == FALLBACK_ANSWER
    call = last_call()
    assert call is not None
    assert call.guard == "banned_phrase"


def test_exception_falls_back(investigation: Investigation, monkeypatch) -> None:
    agent = build_answer_agent("test")

    def boom(*_a, **_k):
        raise RuntimeError("down")

    monkeypatch.setattr(agent, "run_sync", boom)
    out = answer_question("why?", investigation, model="test", agent=agent)
    assert out.text == FALLBACK_ANSWER
    call = last_call()
    assert call is not None
    assert call.guard == "exception"


def test_history_is_capped_to_recent_lines(investigation: Investigation) -> None:
    history = [("user", f"msg {i}") for i in range(12)]
    answer(investigation, {"text": "ok"}, history)
    call = last_call()
    assert call is not None
    assert "msg 3" not in call.input
    assert "msg 4" in call.input
    assert "msg 11" in call.input
