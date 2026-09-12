import numpy as np
import pandas as pd
import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from copilot.data.frame import MarketFrame
from copilot.investigate import Investigation, investigate_at
from copilot.llm import build_agent, narrate
from copilot.report import (
    Narrative,
    banned_phrases,
    fallback_narrative,
    render_facts,
    unknown_numbers,
    unknown_numbers_in_text,
)
from copilot.timeutil import ts

models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def investigation() -> Investigation:
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    hours = np.arange(len(idx)) % 24
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "price_fi": 60 + 20 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 3, len(idx)),
            "price_se3": 45 + rng.normal(0, 2, len(idx)),
            "load": 10_000 + rng.normal(0, 100, len(idx)),
            "wind_fc": 2_000 + rng.normal(0, 100, len(idx)),
            "nuclear": 4_300 + rng.normal(0, 5, len(idx)),
        },
        index=idx,
    )
    t0 = ts("2024-01-05 15:00")
    data.loc[t0 : t0 + pd.Timedelta(hours=2), "price_fi"] = [900.0, 1896.0, 700.0]
    data.loc[t0 : t0 + pd.Timedelta(hours=2), "wind_fc"] = 300.0
    return investigate_at(MarketFrame(data=data, missing=("import_se1",)), ts("2024-01-05 16:00"))


def test_render_facts_has_every_section(investigation: Investigation) -> None:
    text = render_facts(investigation)
    assert text.startswith("## Price spike on Fri 05 Jan 2024")
    assert "- Window: 17:00 to 19:00 (3 h, Europe/Helsinki)" in text
    assert "- Peak: 1,896 EUR/MWh at 18:00" in text
    assert "### Drivers the evidence supports" in text
    assert "**Wind forecast (day-ahead)**" in text
    assert "### Not enough data to judge" in text
    assert "could not be loaded: import_se1" in text


def test_fallback_narrative_separates_lists(investigation: Investigation) -> None:
    narrative = fallback_narrative(investigation)
    assert "1,896 EUR/MWh" in narrative.summary
    assert "not proof of cause" in narrative.summary
    assert narrative.hypotheses == [
        "Low forecast wind for these hours (less cheap supply in the day-ahead auction)."
    ]
    assert any("no data" in item or "no cross-border" in item for item in narrative.insufficient)
    assert unknown_numbers(narrative, render_facts(investigation)) == []


def test_unknown_numbers_accepts_sign_newline_and_year_variants() -> None:
    facts = "## Spike on Sat 16 Dec 2023\n- Deviation: -84 EUR/MWh, robust z = -9.0\n"
    text = Narrative(
        summary="On 16 Dec 2023, about 84 EUR/MWh below (z=-9.0).", facts=[], hypotheses=[]
    )
    assert unknown_numbers(text, facts) == []


def test_unknown_numbers_flags_invented_values() -> None:
    facts = "Peak: 1,896 EUR/MWh. Load 13,784 MW (21%)."
    good = Narrative(
        summary="Peak 1896 EUR/MWh, load 13,784 MW, 21% above, 3 hours.", facts=[], hypotheses=[]
    )
    bad = Narrative(summary="Peak 1900 EUR/MWh.", facts=["load up 25%"], hypotheses=[])
    assert unknown_numbers(good, facts) == []
    assert unknown_numbers(bad, facts) == ["1900", "25%"]


def test_narrate_uses_model_output_when_numbers_check_out(investigation: Investigation) -> None:
    agent = build_agent("test")
    output = {
        "summary": "A spike to 1,896 EUR/MWh against a baseline of about 70 EUR/MWh.",
        "facts": ["Wind forecast 300 MW during the event."],
        "hypotheses": ["Consistent with low forecast wind."],
        "insufficient": [],
    }
    facts = render_facts(investigation)
    # keep the test honest: the baseline number must really be in the facts
    output["summary"] = output["summary"].replace(
        "about 70", f"{investigation.event.baseline_median:,.0f}"
    )
    with agent.override(model=TestModel(custom_output_args=output)):
        narrative = narrate(investigation, model="test", agent=agent)
    assert narrative.summary == output["summary"]
    assert unknown_numbers(narrative, facts) == []


def test_narrate_falls_back_when_model_invents_numbers(investigation: Investigation) -> None:
    agent = build_agent("test")
    output = {
        "summary": "Price hit 2,500 EUR/MWh.",
        "facts": [],
        "hypotheses": [],
        "insufficient": [],
    }
    with agent.override(model=TestModel(custom_output_args=output)):
        narrative = narrate(investigation, model="test", agent=agent)
    assert "2,500" not in narrative.summary
    assert "not proof of cause" in narrative.summary


def test_narrate_falls_back_on_error(
    investigation: Investigation, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = build_agent("test")

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("provider down")

    monkeypatch.setattr(agent, "run_sync", boom)
    narrative = narrate(investigation, model="test", agent=agent)
    assert "not proof of cause" in narrative.summary


def test_render_facts_midnight_crossing_and_missing_baseline() -> None:
    from copilot.detect import Event, EventKind

    event = Event(
        start=ts("2023-12-16 17:00"),
        end=ts("2023-12-17 05:00"),
        peak_time=ts("2023-12-17 00:00"),
        peak_price=-1.0,
        baseline_median=float("nan"),
        z=float("nan"),
        kind=EventKind.NEGATIVE,
        hours=13,
    )
    frame = MarketFrame(data=pd.DataFrame(index=pd.date_range(event.start, event.end, freq="1h")))
    inv = Investigation(
        event=event, frame=frame, results=[], window_start=event.start, window_end=event.end
    )
    text = render_facts(inv)
    assert "Window: 19:00 to Sun 17 Dec 07:00 (13 h" in text
    assert "not enough history" in text
    assert "nan" not in text
    assert "nan" not in fallback_narrative(inv).summary


def test_narrate_drops_invented_insufficient_items(investigation: Investigation) -> None:
    from copilot.drivers.base import Verdict

    full = Investigation(
        event=investigation.event,
        frame=investigation.frame,
        results=[r for r in investigation.results if r.verdict is not Verdict.INSUFFICIENT],
        window_start=investigation.window_start,
        window_end=investigation.window_end,
    )
    agent = build_agent("test")
    output = {
        "summary": "ok",
        "facts": [],
        "hypotheses": [],
        "insufficient": ["No fuel price data."],
    }
    with agent.override(model=TestModel(custom_output_args=output)):
        narrative = narrate(full, model="test", agent=agent)
    assert narrative.insufficient == []


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        ("Price rose because wind fell", ["because"]),
        ("Because of low wind, prices rose", ["because"]),
        ("High load due to cold weather led to a spike", ["due to", "led to"]),
        ("This is consistent with low wind", []),
        ("becauseless word", []),
    ],
)
def test_banned_phrases(text: str, hits: list[str]) -> None:
    assert banned_phrases(text) == hits


def test_unknown_numbers_in_text(investigation: Investigation) -> None:
    facts = render_facts(investigation)
    assert unknown_numbers_in_text("Peak 1,896 EUR/MWh over 3 h", facts) == []
    assert unknown_numbers_in_text("Peak 2,500 EUR/MWh", facts) == ["2,500"]


def test_unknown_numbers_in_text_allows_count_context_numbers() -> None:
    facts = "Peak: 1,896 EUR/MWh."
    text = "3 of 7 drivers support this reading, with a peak of 1,896 EUR/MWh."
    assert unknown_numbers_in_text(text, facts) == []


def test_unknown_numbers_in_text_still_flags_values_that_look_like_counts() -> None:
    # a small integer with no count context (no "h"/"days"/"of"/"drivers" ...) is not exempt
    facts = "Peak: 1,896 EUR/MWh."
    assert unknown_numbers_in_text("Wind was 25 MW below normal.", facts) == ["25"]


def test_banned_phrases_is_case_insensitive() -> None:
    assert banned_phrases("Prices rose BECAUSE of low wind.") == ["because"]
    assert banned_phrases("This was Due To a cold snap.") == ["due to"]


def test_narrate_falls_back_on_causal_wording(investigation: Investigation) -> None:
    agent = build_agent("test")
    output = {
        "summary": "The spike was caused by low wind.",
        "facts": [],
        "hypotheses": [],
        "insufficient": [],
    }
    with agent.override(model=TestModel(custom_output_args=output)):
        narrative = narrate(investigation, model="test", agent=agent)
    assert "not proof of cause" in narrative.summary


def test_narrate_guard_order_unknown_number_checked_before_causal_wording(
    investigation: Investigation,
) -> None:
    """When a model output has both problems, the number guard must fire, not the phrase guard."""
    from copilot.trace import last_call

    agent = build_agent("test")
    output = {
        "summary": "Price hit 2,500 EUR/MWh because of low wind.",
        "facts": [],
        "hypotheses": [],
        "insufficient": [],
    }
    with agent.override(model=TestModel(custom_output_args=output)):
        narrative = narrate(investigation, model="test", agent=agent)
    assert "not proof of cause" in narrative.summary
    call = last_call()
    assert call is not None
    assert call.guard == "unknown_numbers"


def test_narrate_writes_trace(investigation: Investigation, tmp_path) -> None:
    from copilot.trace import last_call

    agent = build_agent("test")
    output = {"summary": "x 9999 EUR/MWh", "facts": [], "hypotheses": [], "insufficient": []}
    with agent.override(model=TestModel(custom_output_args=output)):
        narrate(investigation, model="test", agent=agent)
    call = last_call()
    assert call is not None
    assert call.kind == "narrative"
    assert call.guard == "unknown_numbers"
    assert call.fallback is True


def test_number_guard_catches_a_flipped_sign() -> None:
    """Reporting a crash's deviation as a rise is the worst numeric error this tool can make."""
    facts = "- Deviation: -1,816 EUR/MWh, robust z = -27.9"
    assert unknown_numbers_in_text("the price moved -1,816 EUR/MWh", facts) == []
    assert unknown_numbers_in_text("the price moved +1,816 EUR/MWh", facts) == ["+1,816"]
    assert unknown_numbers_in_text("robust z = +27.9", facts) == ["+27.9"]


def test_number_guard_stays_lenient_about_an_unsigned_number() -> None:
    """Facts write "-55"; prose writes "fell 55 below normal". Both mean the same thing."""
    facts = "- Deviation: -55 EUR/MWh on 05 Jan"
    assert unknown_numbers_in_text("the price fell 55 EUR/MWh on 5 Jan", facts) == []


def test_number_guard_rounds_a_signed_number_the_same_way() -> None:
    facts = "- Deviation: -54.6 EUR/MWh"
    assert unknown_numbers_in_text("about -55 EUR/MWh", facts) == []
    assert unknown_numbers_in_text("about +55 EUR/MWh", facts) == ["+55"]


def test_number_guard_lets_prose_sign_a_magnitude_the_facts_left_bare() -> None:
    """Driver facts put the direction in words: "above normal by 2,383 MW". Either sign is fine."""
    facts = "Consumption: 13,710 MW vs a baseline of 11,327 MW (above normal by 2,383 MW, 21%)."
    assert unknown_numbers_in_text("consumption ran +2,383 MW over normal", facts) == []
    assert unknown_numbers_in_text("residual demand was -2,383 MW off normal", facts) == []


def test_render_facts_reports_the_steepest_move(investigation: Investigation) -> None:
    text = render_facts(investigation)
    assert "- Steepest hour-to-hour move: +996 EUR/MWh (17:00 to 18:00)" in text


def test_render_facts_skips_the_move_when_there_is_none() -> None:
    from copilot.detect import Event, EventKind

    event = Event(
        start=ts("2023-12-16 17:00"),
        end=ts("2023-12-16 17:00"),
        peak_time=ts("2023-12-16 17:00"),
        peak_price=-1.0,
        baseline_median=float("nan"),
        z=float("nan"),
        kind=EventKind.NEGATIVE,
        hours=1,
    )
    frame = MarketFrame(data=pd.DataFrame(index=pd.date_range(event.start, event.end, freq="1h")))
    inv = Investigation(
        event=event, frame=frame, results=[], window_start=event.start, window_end=event.end
    )
    assert "Steepest" not in render_facts(inv)
