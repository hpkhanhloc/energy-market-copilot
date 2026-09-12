"""The narrative LLM call: facts in, typed Narrative out. Provider chosen by COPILOT_MODEL."""

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from copilot.drivers.base import Verdict
from copilot.investigate import Investigation
from copilot.report import (
    Narrative,
    banned_phrases,
    fallback_narrative,
    render_facts,
    unknown_numbers,
)
from copilot.trace import Guard, LlmCall, now_iso, record, timed

log = logging.getLogger(__name__)

INSTRUCTIONS = """You write short investigation notes for an energy market analyst.
You are given the FACTS section produced by deterministic code about one abnormal hour
of the Finnish day-ahead electricity price, and the verdict of each driver check.

Rules:
- Use only numbers that appear in the FACTS. Never invent, round differently, or compute new ones.
- `facts`: the 4 to 6 most decisive observations with their numbers and units, one per item.
  Drivers that did not move, or moved the wrong way, are facts: state them here with numbers.
- `hypotheses`: possible reasons, one per driver whose verdict is "supports". Say "consistent
  with" or "supports". Never say "caused", "because" or "due to". Do not put "does not
  support" items here; they belong in `facts`.
  If no driver supports, return an empty list.
- `summary`: two or three plain sentences for a busy reader, hedged the same way. Items the
  FACTS mark as not enough data are listed by code; mention them only in passing.
- Keep it under 200 words in total. No headers, no markdown."""


class Draft(BaseModel):
    """What the model returns. `insufficient` is not here: code knows what it could not check."""

    summary: str = Field(
        description="Two or three plain sentences: what happened and the leading explanation, hedged."
    )
    facts: list[str] = Field(description="Observed numbers only, one per item, each with its unit.")
    hypotheses: list[str] = Field(
        description="Plausible drivers the facts are consistent with. Never claim causation."
    )


def build_agent(model: str) -> Agent[None, Draft]:
    return Agent(model, output_type=Draft, instructions=INSTRUCTIONS, retries=3)


def narrate(
    inv: Investigation, *, model: str, agent: Agent[None, Draft] | None = None
) -> Narrative:
    """Ask the model for a Narrative; fall back to the deterministic one on any failure.

    If the model returns a number that is not in the facts, or causal wording, the deterministic
    narrative is used instead and the incident is logged: an honest report beats a fluent one.
    Building the agent is inside the guarded call too, so a missing provider key falls back
    rather than raising.
    """
    facts = render_facts(inv)
    prompt = f"FACTS:\n{facts}"
    result, latency = timed(lambda: (agent or build_agent(model)).run_sync(prompt))
    if isinstance(result, Exception):
        log.warning(
            "LLM narrative failed (%s: %s); using deterministic narrative",
            type(result).__name__,
            result,
        )
        _trace(model, prompt, repr(result), latency, guard="exception")
        return fallback_narrative(inv)
    draft: Draft = result.output
    output = draft.model_dump_json()
    narrative = Narrative(
        summary=draft.summary,
        facts=draft.facts,
        hypotheses=draft.hypotheses,
        insufficient=[r.detail for r in inv.results if r.verdict is Verdict.INSUFFICIENT],
    )
    bad = unknown_numbers(narrative, facts)
    if bad:
        log.warning(
            "LLM narrative used numbers not in the facts %s; using deterministic narrative", bad
        )
        _trace(model, prompt, output, latency, guard="unknown_numbers")
        return fallback_narrative(inv)
    causal = [p for text in _narrative_texts(narrative) for p in banned_phrases(text)]
    if causal:
        log.warning("LLM narrative used causal wording %s; using deterministic narrative", causal)
        _trace(model, prompt, output, latency, guard="banned_phrase")
        return fallback_narrative(inv)
    guard: Guard | None = None
    if narrative.hypotheses and not any(r.verdict is Verdict.SUPPORTS for r in inv.results):
        # Hypotheses are one per supporting driver. With none, the model has nothing to
        # hypothesise about, so whatever it wrote is its own idea, not the data's.
        log.info("dropping %d hypotheses with no supporting driver", len(narrative.hypotheses))
        narrative = narrative.model_copy(update={"hypotheses": []})
        guard = "invented_hypotheses"
    _trace(model, prompt, output, latency, guard=guard, fallback=False)
    return narrative


def _narrative_texts(n: Narrative) -> list[str]:
    return [n.summary, *n.facts, *n.hypotheses, *n.insufficient]


def _trace(
    model: str,
    prompt: str,
    output: str,
    latency: int,
    *,
    guard: Guard | None,
    fallback: bool = True,
) -> None:
    record(
        LlmCall(
            kind="narrative",
            model=model,
            input=prompt,
            output=output,
            ok=guard is None,
            guard=guard,
            fallback=fallback,
            latency_ms=latency,
            ts=now_iso(),
        )
    )
