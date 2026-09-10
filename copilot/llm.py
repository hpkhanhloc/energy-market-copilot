"""The one LLM call: facts in, typed Narrative out. Provider chosen by COPILOT_MODEL."""

import logging

from pydantic_ai import Agent

from copilot.drivers.base import Verdict
from copilot.investigate import Investigation
from copilot.report import Narrative, fallback_narrative, render_facts, unknown_numbers

log = logging.getLogger(__name__)

INSTRUCTIONS = """You write short investigation notes for an energy market analyst.
You are given the FACTS section produced by deterministic code about one abnormal hour
of the Finnish day-ahead electricity price, and the verdict of each driver check.

Rules:
- Use only numbers that appear in the FACTS. Never invent, round differently, or compute new ones.
- `facts`: the 4 to 6 most decisive observations with their numbers and units, one per item.
- `hypotheses`: what the evidence is consistent with. Say "consistent with", "supports",
  "does not support". Never say "caused", "because" or "due to".
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

    If the model returns a number that is not in the facts, the deterministic narrative is used
    instead and the incident is logged: an honest report beats a fluent one.
    """
    facts = render_facts(inv)
    agent = agent or build_agent(model)
    try:
        result = agent.run_sync(f"FACTS:\n{facts}")
    except Exception as exc:
        log.warning(
            "LLM narrative failed (%s: %s); using deterministic narrative", type(exc).__name__, exc
        )
        return fallback_narrative(inv)
    narrative = result.output
    bad = unknown_numbers(narrative, facts)
    if bad:
        log.warning(
            "LLM narrative used numbers not in the facts %s; using deterministic narrative", bad
        )
        return fallback_narrative(inv)
    if narrative.insufficient and not any(r.verdict is Verdict.INSUFFICIENT for r in inv.results):
        log.info(
            "dropping %d 'insufficient' items the model added on its own",
            len(narrative.insufficient),
        )
        narrative = narrative.model_copy(update={"insufficient": []})
    return narrative
