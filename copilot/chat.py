"""The follow-up LLM call: a question about the report on screen, answered from the facts only."""

import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from copilot.investigate import Investigation
from copilot.report import banned_phrases, render_facts, unknown_numbers_in_text
from copilot.trace import record_call, timed

log = logging.getLogger(__name__)

KIND = "answer"

HISTORY_LINES = 8
FALLBACK_ANSWER = (
    "I could not answer that safely from the report. The facts are shown above; ask me to "
    "investigate a specific hour (HH:MM) if you need another one."
)


class Answer(BaseModel):
    text: str = Field(description="Two to four plain sentences. Numbers only from FACTS.")
    source: Literal["report", "general"] = Field(
        default="report",
        description="'report' when answered from FACTS; 'general' when explaining a market term "
        "from general knowledge.",
    )
    hour_not_in_report: bool = Field(
        default=False,
        description="True when the question is about an hour or day the FACTS do not cover.",
    )


INSTRUCTIONS = """You answer follow-up questions for an energy market analyst about one
investigation of the Finnish day-ahead electricity price. You get the FACTS produced by
deterministic code, the recent CONVERSATION, and the QUESTION.

Rules:
- Answer from the FACTS. Use only numbers that appear there, with their units. Never invent,
  round differently, or compute new numbers.
- If the question is about a market term (residual load, mFRR, day-ahead, z-score, baseline ...)
  answer from general knowledge and set source="general".
- If the question is about an hour or day the FACTS do not cover, say so, set
  hour_not_in_report=true, and suggest asking to investigate that hour. Do not guess.
- Say "consistent with", "supports", "does not support". Never say "caused", "because",
  "due to", "led to". Correlation is not proof of cause.
- Two to four plain sentences. No headers, no markdown."""


def build_answer_agent(model: str) -> Agent[None, Answer]:
    return Agent(model, output_type=Answer, instructions=INSTRUCTIONS, retries=2)


def answer_question(
    question: str,
    inv: Investigation,
    history: Sequence[tuple[str, str]] = (),
    *,
    model: str,
    agent: Agent[None, Answer] | None = None,
) -> Answer:
    """Answer `question` about `inv`; on any failure or guard hit return a fixed safe answer."""
    facts = render_facts(inv)
    prompt = _prompt(question, facts, history)
    result, latency = timed(lambda: (agent or build_answer_agent(model)).run_sync(prompt))
    if isinstance(result, Exception):
        log.warning("answer failed (%s: %s)", type(result).__name__, result)
        record_call(KIND, model, prompt, repr(result), latency, guard="exception")
        return Answer(text=FALLBACK_ANSWER, source="report")
    answer: Answer = result.output
    output = answer.model_dump_json()
    # Numbers the user typed ("what about 21:00?") may be echoed back, e.g. to say the report
    # does not cover that hour.
    bad = unknown_numbers_in_text(answer.text, f"{facts}\nQUESTION: {question}")
    if bad:
        log.warning("answer used numbers not in the facts %s", bad)
        record_call(KIND, model, prompt, output, latency, guard="unknown_numbers")
        return Answer(text=FALLBACK_ANSWER, source="report")
    causal = banned_phrases(answer.text)
    if causal:
        log.warning("answer used causal wording %s", causal)
        record_call(KIND, model, prompt, output, latency, guard="banned_phrase")
        return Answer(text=FALLBACK_ANSWER, source="report")
    record_call(KIND, model, prompt, output, latency, guard=None)
    return answer


def _prompt(question: str, facts: str, history: Sequence[tuple[str, str]]) -> str:
    lines = [f"FACTS:\n{facts}"]
    recent = list(history)[-HISTORY_LINES:]
    if recent:
        lines.append("CONVERSATION:\n" + "\n".join(f"{role}: {text}" for role, text in recent))
    lines.append(f"QUESTION: {question.strip()}")
    return "\n\n".join(lines)
