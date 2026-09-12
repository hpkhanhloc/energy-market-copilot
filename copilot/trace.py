"""One JSON line per LLM call, so guard hits and latency can be counted after the fact."""

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from copilot.config import ROOT

log = logging.getLogger(__name__)

type CallKind = Literal["narrative", "intent", "answer"]
type Guard = Literal["exception", "unknown_numbers", "banned_phrase", "invented_hypotheses"]


@dataclass(frozen=True, slots=True, kw_only=True)
class LlmCall:
    kind: CallKind
    model: str
    input: str
    output: str  # JSON of the typed output, or the exception text
    ok: bool  # model answered and every guard passed
    guard: Guard | None  # which guard tripped, if any
    fallback: bool  # deterministic text shown instead of the model's
    latency_ms: int
    ts: str  # ISO 8601, UTC


def trace_path() -> Path:
    """Where calls are appended: COPILOT_TRACE env, else data/logs/llm.jsonl under the repo."""
    override = os.environ.get("COPILOT_TRACE")
    return Path(override) if override else ROOT / "data" / "logs" / "llm.jsonl"


def record(call: LlmCall, path: Path | None = None) -> None:
    """Append one line. Never raises: a broken log must not break a report."""
    target = path or trace_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(call), ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("could not write LLM trace to %s: %s", target, exc)


def last_call(path: Path | None = None) -> LlmCall | None:
    """The most recent recorded call, or None when the log is empty or missing."""
    target = path or trace_path()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if line.strip():
            return LlmCall(**json.loads(line))
    return None


def timed[T](fn: Callable[[], T]) -> tuple[T | Exception, int]:
    """Run `fn`, return (result or the exception it raised, wall time in ms)."""
    start = time.perf_counter()
    try:
        out: T | Exception = fn()
    except Exception as exc:
        out = exc
    return out, round((time.perf_counter() - start) * 1000)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
