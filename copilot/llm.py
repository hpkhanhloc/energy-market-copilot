"""The narrative LLM call: facts in, typed Narrative out. Provider chosen by COPILOT_MODEL."""

import logging

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
- `insufficient`: only items the FACTS mark as not enough data, one line each. If there are
  none, return an empty list. Do not list data the system does not have.
- `summary`: two or three plain sentences for a busy reader, hedged the same way.
- Keep it under 200 words in total. No headers, no markdown."""


def build_agent(model: str) -> Agent[None, Narrative]:
    return Agent(model, output_type=Narrative, instructions=INSTRUCTIONS, retries=3)


def narrate(
    inv: Investigation, *, model: str, agent: Agent[None, Narrative] | None = None
) -> Narrative:
    """Ask the model for a Narrative; fall back to the deterministic one on any failure.

    If the model returns a number that is not in the facts, or causal wording, the deterministic
    narrative is used instead and the incident is logged: an honest report beats a fluent one.
    """
    facts = render_facts(inv)
    agent = agent or build_agent(model)
    prompt = f"FACTS:\n{facts}"
    result, latency = timed(lambda: agent.run_sync(prompt))
    if isinstance(result, Exception):
        log.warning(
            "LLM narrative failed (%s: %s); using deterministic narrative",
            type(result).__name__,
            result,
        )
        _trace(model, prompt, repr(result), latency, guard="exception")
        return fallback_narrative(inv)
    narrative: Narrative = result.output
    output = narrative.model_dump_json()
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
    if narrative.insufficient and not any(r.verdict is Verdict.INSUFFICIENT for r in inv.results):
        log.info(
            "dropping %d 'insufficient' items the model added on its own",
            len(narrative.insufficient),
        )
        narrative = narrative.model_copy(update={"insufficient": []})
        guard = "invented_insufficient"
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
