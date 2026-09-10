"""Run the detector over the cached price history and print how it behaves.

Usage: uv run python scripts/backtest.py [start] [end] [--top N]
Defaults cover the whole warmed cache. Reads parquet only; never calls an API.
"""

import argparse
import time

import pandas as pd

from copilot.backtest import (
    TZ,
    compare,
    events_per_month,
    expected_hits,
    load_cached_price,
    sweep,
)
from copilot.config import load_settings
from copilot.detect import DetectConfig, Event, find_events
from copilot.report import format_number
from copilot.timeutil import datetime_index, helsinki

EXPECTED = [
    (helsinki("2024-01-05 19:00"), "cold-snap spike, 1896 EUR/MWh"),
    (helsinki("2023-12-17 02:00"), "windy night, 13 h at or below 0 EUR/MWh"),
    (helsinki("2023-11-24 15:00"), "-500 EUR/MWh, published bid-error day"),
]


def print_events(events: list[Event]) -> None:
    print(
        f"{'#':>3} {'kind':9} {'start (Helsinki)':17} {'hours':>5} {'peak':>8} "
        f"{'baseline':>9} {'z':>6} {'ramp':>7}"
    )
    for i, e in enumerate(events, 1):
        print(
            f"{i:>3} {e.kind:9} {e.start.tz_convert(TZ):%Y-%m-%d %H:%M} {e.hours:>5} "
            f"{e.peak_price:>8,.0f} {format_number(e.baseline_median, '{:,.0f}'):>9} "
            f"{format_number(e.z, '{:+.1f}'):>6} {format_number(e.max_ramp, '{:+,.0f}'):>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", nargs="?", default="2023-10-01")
    parser.add_argument("end", nargs="?", default="2026-09-01")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    settings = load_settings()
    t0 = time.time()
    cached = load_cached_price(settings.cache_dir, helsinki(args.start), helsinki(args.end))
    price = cached.price
    if cached.missing_months:
        print("months not in cache (skipped):", ", ".join(cached.missing_months))
        print("warm them with: uv run python scripts/warm_cache.py <start> <end>")
    if price.empty:
        print("no cached price data in that range")
        return
    first, last = price.index[0].tz_convert(TZ), price.index[-1].tz_convert(TZ)
    print(f"{len(price)} hours, {first:%Y-%m-%d} .. {last:%Y-%m-%d}, {price.isna().sum()} NaN")

    config = DetectConfig()
    events = find_events(price, config, top_n=None)

    print("\n== Events per month (episodes) ==")
    monthly = events_per_month(events)
    print(monthly.to_string())
    print(f"total {len(events)} episodes, median {monthly.sum(axis=1).median():.0f} per month")

    print(f"\n== Top {args.top} events ==")
    print_events(events[: args.top])

    print("\n== Known events ==")
    for hit in expected_hits(events, EXPECTED, price):
        when = hit.when.strftime("%Y-%m-%d %H:%M")
        if not hit.in_data:
            status = "not in cache"
        elif hit.rank is None:
            status = "MISSED"
        else:
            status = f"rank {hit.rank}"
        print(f"  {when}  {status:14} {hit.label}")

    print("\n== Threshold sweep (episodes / abnormal hours) ==")
    table = sweep(price)
    wide = table.pivot_table(
        index=["z", "min_abs_deviation"], columns="min_ramp", values=["events", "hours"]
    )
    print(wide.to_string())

    print("\n== Versus naive same-UTC-hour mean +/- 2 std over 30 days ==")
    cmp = compare(price, config)
    print(
        f"hours flagged by both {cmp.both}, only ours {cmp.only_ours}, "
        f"only naive {cmp.only_naive}, neither {cmp.neither}"
    )
    with pd.option_context("display.float_format", "{:,.1f}".format):
        print("\nmost extreme hours only we flag:")
        print(_local(cmp.only_ours_sample).to_string())
        print("\nmost extreme hours only naive flags:")
        print(_local(cmp.only_naive_sample).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


def _local(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = datetime_index(out).tz_convert(TZ).strftime("%Y-%m-%d %H:%M")
    return out


if __name__ == "__main__":
    main()
