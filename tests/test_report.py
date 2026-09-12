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
from copilot.trace import last_call

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
    facts = "Peak: 1,896 EUR/MWh over 3 h. Load 13,784 MW (21%)."
    good = Narrative(
        summary="Peak 1896 EUR/MWh, load 13,784 MW, 21% above, 3 hours.", facts=[], hypotheses=[]
    )
    bad = Narrative(summary="Peak 1900 EUR/MWh for 12 hours.", facts=["load up 25%"], hypotheses=[])
    assert unknown_numbers(good, facts) == []
    assert unknown_numbers(bad, facts) == ["1900", "12", "25%"]


def test_number_guard_does_not_whitelist_a_value_after_of_or_before_hours() -> None:
    """'baseline of 30' and '12 hours' are values, not counts; both must match the facts."""
    facts = "- Window: 19:00 (1 h, Europe/Helsinki)\n- Baseline 52 EUR/MWh"
    assert unknown_numbers_in_text("The baseline of 30 EUR/MWh", facts) == ["30"]
    assert unknown_numbers_in_text("Prices stayed high for 12 hours", facts) == ["12"]
    assert unknown_numbers_in_text("Prices stayed high for 1 hour", facts) == []


def test_number_guard_checks_dates_and_times_as_tokens() -> None:
    """Dates and clock times never feed the pool of allowed values, and must match as a whole."""
    facts = "## Spike on Fri 05 Jan 2024\n- Peak: 1,896 EUR/MWh at 19:00"
    assert unknown_numbers_in_text("On 5 January 2024 at 19:00 the peak was 1,896", facts) == []
    assert unknown_numbers_in_text("On 2024-01-05 the peak was 1,896", facts) == []
    assert unknown_numbers_in_text("load was 19 GW", facts) == ["19"]
    assert unknown_numbers_in_text("a baseline of 2,024 EUR/MWh", facts) == ["2,024"]
    assert unknown_numbers_in_text("at 21:00 the peak was 1,896", facts) == ["21:00"]
    assert unknown_numbers_in_text("on 6 Jan the peak was 1,896", facts) == ["6 jan"]


def test_number_guard_reads_a_unicode_minus_as_a_sign() -> None:
    facts = "- Deviation: +1,845 EUR/MWh"
    assert unknown_numbers_in_text("the deviation was \u22121,845 EUR/MWh", facts) == ["-1,845"]


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


def test_narrate_insufficient_list_comes_from_code(investigation: Investigation) -> None:
    """The model cannot add 'could not check' items: code knows what it could not check."""
    from copilot.drivers.base import Verdict

    agent = build_agent("test")
    output = {"summary": "ok", "facts": [], "hypotheses": []}
    with agent.override(model=TestModel(custom_output_args=output)):
        narrative = narrate(investigation, model="test", agent=agent)
    expected = [r.detail for r in investigation.results if r.verdict is Verdict.INSUFFICIENT]
    assert expected
    assert narrative.insufficient == expected


def test_narrate_drops_hypotheses_when_no_driver_supports(investigation: Investigation) -> None:
    from copilot.drivers.base import Verdict

    none = Investigation(
        event=investigation.event,
        frame=investigation.frame,
        results=[r for r in investigation.results if r.verdict is not Verdict.SUPPORTS],
        window_start=investigation.window_start,
        window_end=investigation.window_end,
    )
    agent = build_agent("test")
    output = {"summary": "ok", "facts": [], "hypotheses": ["Gas prices were high."]}
    with agent.override(model=TestModel(custom_output_args=output)):
        narrative = narrate(none, model="test", agent=agent)
    assert narrative.hypotheses == []
    call = last_call()
    assert call is not None
    assert call.guard == "invented_hypotheses"
    assert call.fallback is False


def test_narrate_falls_back_when_no_provider_key(
    investigation: Investigation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building the agent is inside the guard: a missing key gives the code-written text."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    narrative = narrate(investigation, model="anthropic:claude-sonnet-5")
    assert narrative == fallback_narrative(investigation)
    call = last_call()
    assert call is not None
    assert call.guard == "exception"


@pytest.mark.parametrize(
    ("text", "hits"),
    [
        ("Price rose because wind fell", ["because"]),
        ("Because of low wind, prices rose", ["because"]),
        ("High load due to cold weather led to a spike", ["due to", "led to"]),
        ("This is consistent with low wind", []),
        ("becauseless word", []),
        ("Low wind causes spikes", ["causes"]),
        ("High demand drove prices up; driven by cold", ["drove", "driven by"]),
        ("Triggered by an outage, as a result of the cold", ["triggered", "as a result"]),
        ("Thanks to low wind, the gap was explained by imports", ["thanks to", "explained by"]),
        ("Drivers the evidence supports; I can explain one hour", []),
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
