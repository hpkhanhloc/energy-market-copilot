"""The driver checks. Each is `check(frame, event) -> DriverResult`. Order = display order."""

from dataclasses import replace

import pandas as pd

from copilot.baseline import baseline_frame
from copilot.data.frame import MarketFrame
from copilot.detect import Event
from copilot.drivers.base import (
    Z_SUPPORT,
    DriverCheck,
    DriverResult,
    Verdict,
    compare_to_baseline,
    event_window,
    pick_column,
    price_up,
)

NEIGHBOUR_PRICES: dict[str, str] = {
    "price_se1": "SE1",
    "price_se3": "SE3",
    "price_ee": "EE",
    "price_no4": "NO4",
}
BORDERS: dict[str, str] = {
    "import_se1": "SE1",
    "import_se3": "SE3",
    "import_ee": "EE",
    "import_no4": "NO4",
}
SWEDISH_BORDERS = ("import_se1", "import_se3")
MIN_NEIGHBOURS = 2
"""One neighbour cannot tell a regional move from a local one."""


def with_derived(frame: MarketFrame) -> MarketFrame:
    """The frame plus the derived series the checks judge and the charts must therefore show.

    `import_sweden` is SE1+SE3 (NaN when either border is missing for the hour); `residual_load`
    is load minus wind minus nuclear, each from the twin column with more data.
    """
    data = frame.data
    extra: dict[str, pd.Series] = {}
    if "import_sweden" not in data.columns and frame.has(*SWEDISH_BORDERS):
        extra["import_sweden"] = data[list(SWEDISH_BORDERS)].sum(axis=1, min_count=2)
    if "residual_load" not in data.columns:
        wind = pick_column(frame, "wind", "wind_rt")
        nuclear = pick_column(frame, "nuclear", "nuclear_rt")
        if frame.has("load", wind, nuclear):
            extra["residual_load"] = data["load"] - data[wind] - data[nuclear]
    if not extra:
        return frame
    return MarketFrame(data=data.assign(**extra), missing=frame.missing)


def wind_forecast(frame: MarketFrame, event: Event) -> DriverResult:
    """Day-ahead prices are set on the day-ahead wind forecast, so the forecast is the driver."""
    return compare_to_baseline(
        frame,
        event,
        column="wind_fc",
        name="wind_forecast",
        title="Wind forecast (day-ahead)",
        unit="MW",
        bullish_when="lower",
        hypothesis_up="Low forecast wind for these hours (less cheap supply in the day-ahead auction).",
        hypothesis_down="High forecast wind for these hours (cheap supply flooding the day-ahead auction).",
        columns=("wind_fc", "wind_fc_fingrid"),
    )


def wind_actual(frame: MarketFrame, event: Event) -> DriverResult:
    column = pick_column(frame, "wind", "wind_rt")
    return compare_to_baseline(
        frame,
        event,
        column=column,
        name="wind_actual",
        title="Wind generation (actual)",
        unit="MW",
        bullish_when="lower",
        hypothesis_up="Actual wind was low (mainly an imbalance-market signal; cross-checks the forecast).",
        hypothesis_down="Actual wind was high (mainly an imbalance-market signal; cross-checks the forecast).",
        # Both sources of actual wind (Fingrid hidden until clicked) plus the forecast it is
        # cross-checked against. The Fingrid forecast lives on the forecast chart only.
        columns=("wind", "wind_rt", "wind_fc"),
    )


def nuclear(frame: MarketFrame, event: Event) -> DriverResult:
    column = pick_column(frame, "nuclear", "nuclear_rt")
    return compare_to_baseline(
        frame,
        event,
        column=column,
        name="nuclear",
        title="Nuclear generation",
        unit="MW",
        bullish_when="lower",
        hypothesis_up="A nuclear unit out or ramped down (less cheap baseload available).",
        hypothesis_down="More nuclear than usual (extra cheap baseload).",
        columns=("nuclear", "nuclear_rt"),
    )


def load(frame: MarketFrame, event: Event) -> DriverResult:
    return compare_to_baseline(
        frame,
        event,
        column="load",
        name="load",
        title="Consumption (actual load)",
        unit="MW",
        bullish_when="higher",
        hypothesis_up="Unusually high demand, e.g. a cold snap (more expensive plants needed).",
        hypothesis_down="Unusually low demand, e.g. a mild weekend or holiday (cheap plants suffice).",
        columns=("load", "consumption_rt", "load_fc"),
    )


def imports(frame: MarketFrame, event: Event) -> DriverResult:
    """Swedish borders carry most Finnish imports, so they are judged; every border is reported.

    Estonia often flips from export to import in tight hours, which can make the total look
    fine while the Nordic supply actually fell. Judging SE1+SE3 avoids that blind spot. Both
    Swedish borders must be loaded: a single border is not the Swedish total.
    """
    hypothesis = (
        "Less import from Sweden (capacity limits, or Sweden short too)."
        if price_up(event)
        else "More import from Sweden than usual (cheap Nordic power flowing in)."
    )
    derived = with_derived(frame)
    columns = tuple(c for c in ("import_sweden", *BORDERS) if derived.has(c))
    if not frame.has(*SWEDISH_BORDERS):
        missing = [BORDERS[c] for c in SWEDISH_BORDERS if not frame.has(c)]
        extra = _border_breakdown(frame, event)
        return DriverResult(
            name="imports",
            title="Imports from Sweden (SE1+SE3)",
            unit="MW",
            verdict=Verdict.INSUFFICIENT,
            hypothesis=hypothesis,
            detail=f"Imports from Sweden (SE1+SE3): no data for {' and '.join(missing)}, so the "
            "Swedish total cannot be judged." + (f" Per border: {extra}." if extra else ""),
            columns=columns,
        )
    result = compare_to_baseline(
        derived,
        event,
        column="import_sweden",
        name="imports",
        title="Imports from Sweden (SE1+SE3)",
        unit="MW",
        bullish_when="lower",
        hypothesis_up="Less import from Sweden (capacity limits, or Sweden short too).",
        hypothesis_down="More import from Sweden than usual (cheap Nordic power flowing in).",
        columns=columns,
    )
    if result.verdict is Verdict.INSUFFICIENT:
        return result
    extra = _border_breakdown(frame, event)
    if frame.has("import_total"):
        window = event_window(frame, event)
        base = (
            baseline_frame(frame.data["import_total"]).loc[event.start : event.end]["median"].mean()
        )
        normal = f" (normal {base:,.0f})" if pd.notna(base) else ""  # as in _border_breakdown
        total = f" Total net import {window['import_total'].mean():,.0f} MW{normal}."
    else:
        total = ""
    return replace(result, detail=f"{result.detail} Per border: {extra}.{total}")


def neighbour_prices(frame: MarketFrame, event: Event) -> DriverResult:
    """Were neighbours abnormal too? If yes the move is regional, if no it is Finland-specific."""
    rows: list[str] = []
    zs: list[float] = []
    for column, label in NEIGHBOUR_PRICES.items():
        if not frame.has(column):
            continue
        stats = baseline_frame(frame.data[column]).loc[event.start : event.end].dropna(subset=["z"])
        if stats.empty:
            continue
        z = float(stats["z"].mean())
        zs.append(z)
        rows.append(f"{label} {stats['value'].mean():,.0f} EUR/MWh (z {z:+.1f})")
    if len(zs) < MIN_NEIGHBOURS:
        detail = (
            "Neighbouring prices: no data."
            if not zs
            else f"Neighbouring prices: only {rows[0]} loaded; at least {MIN_NEIGHBOURS} "
            "neighbours are needed to tell a regional move from a Finnish one."
        )
        return DriverResult(
            name="neighbours",
            title="Neighbouring prices",
            unit="EUR/MWh",
            verdict=Verdict.INSUFFICIENT,
            hypothesis="A regional move across the Nordic/Baltic market rather than Finland alone.",
            detail=detail,
            columns=tuple(c for c in NEIGHBOUR_PRICES if frame.has(c)),
        )
    regional = [z for z in zs if (z >= Z_SUPPORT if price_up(event) else z <= -Z_SUPPORT)]
    # Strict majority: more than half, so 1 of 2 and 1 of 3 are not enough. `len(zs) // 2`
    # let a single abnormal neighbour print "a regional move" as a flat assertion, and
    # rounding half up still allowed it whenever only two neighbours had data.
    verdict = Verdict.SUPPORTS if len(regional) > len(zs) / 2 else Verdict.DOES_NOT_SUPPORT
    where = "also abnormal" if verdict is Verdict.SUPPORTS else "mostly normal"
    tally = f"{len(regional)} of {len(zs)} neighbouring prices moved with Finland"
    detail = (
        f"Neighbouring prices during the event were {where}: "
        + "; ".join(rows)
        + f". {tally}"
        + (
            ": a regional move."
            if verdict is Verdict.SUPPORTS
            else ", so Finland moved on its own."
        )
    )
    return DriverResult(
        name="neighbours",
        title="Neighbouring prices",
        unit="EUR/MWh",
        verdict=verdict,
        hypothesis="A regional move across the Nordic/Baltic market rather than Finland alone.",
        value=None,
        baseline=None,
        z=float(pd.Series(zs).mean()),
        detail=detail,
        columns=tuple(c for c in NEIGHBOUR_PRICES if frame.has(c)),
    )


def residual_load(frame: MarketFrame, event: Event) -> DriverResult:
    """Load minus wind minus nuclear: what the expensive, flexible plants must cover."""
    derived = with_derived(frame)
    if "residual_load" not in derived.data.columns:
        needed = (
            "load",
            pick_column(frame, "wind", "wind_rt"),
            pick_column(frame, "nuclear", "nuclear_rt"),
        )
        return DriverResult(
            name="residual_load",
            title="Residual load",
            unit="MW",
            verdict=Verdict.INSUFFICIENT,
            hypothesis="An unusually large gap for flexible plants and imports to fill.",
            detail="Residual load: needs load, wind and nuclear.",
            columns=needed,
        )
    return compare_to_baseline(
        derived,
        event,
        column="residual_load",
        name="residual_load",
        title="Residual load (demand minus wind minus nuclear)",
        unit="MW",
        bullish_when="higher",
        hypothesis_up="An unusually large gap for flexible plants and imports to fill.",
        hypothesis_down="Very little left for flexible plants to cover (oversupply from wind and baseload).",
        columns=("residual_load",),
    )


ALL_CHECKS: tuple[DriverCheck, ...] = (
    wind_forecast,
    wind_actual,
    nuclear,
    load,
    imports,
    neighbour_prices,
    residual_load,
)


def run_all(
    frame: MarketFrame, event: Event, checks: tuple[DriverCheck, ...] = ALL_CHECKS
) -> list[DriverResult]:
    """Run every check; supporting drivers first, strongest |z| first within a verdict."""
    results = [check(frame, event) for check in checks]
    order = {Verdict.SUPPORTS: 0, Verdict.DOES_NOT_SUPPORT: 1, Verdict.INSUFFICIENT: 2}
    return sorted(results, key=lambda r: (order[r.verdict], -abs(r.z or 0.0)))


def _border_breakdown(frame: MarketFrame, event: Event) -> str:
    parts: list[str] = []
    window = event_window(frame, event)
    for column, label in BORDERS.items():
        if frame.has(column):
            stats = baseline_frame(frame.data[column]).loc[event.start : event.end]
            base = stats["median"].mean()
            base_text = f" (normal {base:,.0f})" if pd.notna(base) else ""
            parts.append(f"{label} {window[column].mean():,.0f} MW{base_text}")
    return ", ".join(parts)
