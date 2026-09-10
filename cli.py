"""Command line entry point.

uv run python cli.py investigate 2024-01-05T19:00      # one hour (Helsinki time)
uv run python cli.py scan 2023-12-08 2024-01-08         # list abnormal episodes in a range
uv run python cli.py investigate 2024-01-05T19:00 --no-llm --charts out/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from copilot.config import load_settings
from copilot.investigate import HISTORY_DAYS, investigate_at, load_window, scan, window_for
from copilot.llm import narrate
from copilot.plots import all_figures
from copilot.report import Narrative, fallback_narrative, format_number, render_facts
from copilot.timeutil import helsinki, ts

TZ = "Europe/Helsinki"


def cmd_investigate(when: str, *, use_llm: bool, charts: Path | None) -> None:
    settings = load_settings()
    at = helsinki(when)
    start, end = window_for(at)
    frame = load_window(settings, start, end)
    inv = investigate_at(frame, at)
    narrative = narrate(inv, model=settings.copilot_model) if use_llm else fallback_narrative(inv)
    print(render_facts(inv))
    print()
    print(format_narrative(narrative))
    if charts is not None:
        charts.mkdir(parents=True, exist_ok=True)
        for name, fig in all_figures(inv).items():
            path = charts / f"{name}.html"
            fig.write_html(path, include_plotlyjs="cdn")
        print(f"\nCharts written to {charts}/ (open the .html files in a browser).")


def cmd_scan(start: str, end: str, *, top_n: int) -> None:
    settings = load_settings()
    # +1 day: `helsinki(end)` is midnight at the *start* of the end day and the window is
    # half-open, so without this the last day of the range is never looked at.
    frame = load_window(
        settings,
        ts(helsinki(start) - pd.Timedelta(days=HISTORY_DAYS)),
        ts(helsinki(end) + pd.Timedelta(days=1)),
    )
    events = scan(frame, top_n=top_n, since=helsinki(start))
    if not events:
        print("No abnormal hours found in that range.")
        return
    print(f"{'kind':9} {'start (Helsinki)':17} {'hours':>5} {'peak':>8} {'baseline':>9} {'z':>6}")
    for e in events:
        print(
            f"{e.kind:9} {e.start.tz_convert(TZ):%Y-%m-%d %H:%M} {e.hours:>5} "
            f"{e.peak_price:>8,.0f} {format_number(e.baseline_median, '{:,.0f}'):>9} "
            f"{format_number(e.z, '{:+.1f}'):>6}"
        )
    print(
        f"\nInvestigate one with: uv run python cli.py investigate {events[0].peak_time.tz_convert(TZ):%Y-%m-%dT%H:%M}"
    )


def format_narrative(n: Narrative) -> str:
    lines = ["## Narrative", "", n.summary, "", "FACTS:"]
    lines += [f"  - {x}" for x in n.facts]
    lines += ["", "HYPOTHESES (consistent with the evidence, not proven):"]
    lines += [f"  - {x}" for x in n.hypotheses]
    if n.insufficient:
        lines += ["", "NOT ENOUGH DATA:"] + [f"  - {x}" for x in n.insufficient]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Energy Market Copilot (Finland, day-ahead price)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("investigate", help="explain one hour, e.g. 2024-01-05T19:00 (Helsinki)")
    inv.add_argument("when")
    inv.add_argument("--no-llm", action="store_true", help="skip the LLM narrative")
    inv.add_argument("--charts", type=Path, default=None, help="directory for HTML charts")
    sc = sub.add_parser("scan", help="list abnormal episodes between two dates")
    sc.add_argument("start")
    sc.add_argument("end")
    sc.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s %(message)s"
    )
    if args.command == "investigate":
        cmd_investigate(args.when, use_llm=not args.no_llm, charts=args.charts)
    else:
        cmd_scan(args.start, args.end, top_n=args.top)


if __name__ == "__main__":
    main()
