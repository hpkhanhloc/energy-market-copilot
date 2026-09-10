"""Score the intent parser on the golden set with the real model. Not part of CI.

Run: uv run python scripts/eval_intents.py [--min 0.85] [--fixture path]
Each line of the fixture: {"text", "expect": scan|investigate|ask|reply, optional "start", "end",
"when", "last_hour", "last_range"}. A case passes when the parsed type matches and every given
field matches after guard_intent.
"""

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from copilot.config import load_settings
from copilot.intent import (
    Context,
    Investigate,
    Scan,
    cached_months,
    data_reach,
    guard_intent,
    parse_intent,
)
from copilot.timeutil import helsinki
from copilot.trace import last_call

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "intents.jsonl"
TODAY = date(2026, 9, 10)  # fixed so relative-date cases are stable


def context(case: dict, reach: tuple[date, date], months: frozenset[str]) -> Context:
    last_hour = case.get("last_hour")
    last_range = case.get("last_range")
    return Context(
        today=TODAY,
        reach_start=reach[0],
        reach_end=reach[1],
        last_hour=datetime.fromisoformat(last_hour) if last_hour else None,
        last_range=(date.fromisoformat(last_range[0]), date.fromisoformat(last_range[1]))
        if last_range
        else None,
        cached_months=months,
    )


def check(case: dict, intent: object) -> tuple[bool, str]:
    got = type(intent).__name__.lower()
    if got != case["expect"]:
        return False, got
    if isinstance(intent, Scan):
        want = (case.get("start"), case.get("end"))
        have = (intent.start.isoformat(), intent.end.isoformat())
        if any(w and w != h for w, h in zip(want, have, strict=True)):
            return False, f"scan {have[0]}..{have[1]}"
    if isinstance(intent, Investigate) and case.get("when"):
        want = helsinki(case["when"])
        if intent.when != want.to_pydatetime():
            return False, f"investigate {intent.when:%Y-%m-%dT%H:%M}"
    return True, got


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", type=float, default=0.85, help="exit 1 below this accuracy")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)

    settings = load_settings()
    reach = data_reach(settings.cache_dir, TODAY)
    months = cached_months(settings.cache_dir)
    cases = [json.loads(line) for line in args.fixture.read_text().splitlines() if line.strip()]
    per_class: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    latencies: list[int] = []
    print(f"model: {settings.copilot_model}   cases: {len(cases)}\n")
    for case in cases:
        intent = parse_intent(
            case["text"], context(case, reach, months), model=settings.copilot_model
        )
        intent = guard_intent(intent, context(case, reach, months))
        ok, got = check(case, intent)
        call = last_call()
        if call is not None:
            latencies.append(call.latency_ms)
        per_class[case["expect"]] += 1
        hits[case["expect"]] += ok
        mark = "ok " if ok else "BAD"
        print(f"{mark}  {case['expect']:<11} {got:<28} {case['text']}")

    total = sum(hits.values()) / len(cases)
    print()
    for label in sorted(per_class):
        print(f"{label:<12} {hits[label]}/{per_class[label]}")
    mean_ms = sum(latencies) / len(latencies) if latencies else 0
    print(f"\naccuracy {total:.0%}   mean latency {mean_ms:.0f} ms")
    return 0 if total >= args.min else 1


if __name__ == "__main__":
    sys.exit(main())
