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
    price_up,
)

NEIGHBOUR_PRICES: dict[str, str] = {
    "price_se1": "SE1",
    "price_se3": "SE3",
    "price_ee": "EE",
    "price_no4": "NO4",
}


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
        hypothesis="Low forecast wind for these hours (less cheap supply in the day-ahead auction).",
    )


def wind_actual(frame: MarketFrame, event: Event) -> DriverResult:
    column = "wind" if frame.has("wind") else "wind_rt"
    return compare_to_baseline(
        frame,
        event,
        column=column,
        name="wind_actual",
        title="Wind generation (actual)",
        unit="MW",
        bullish_when="lower",
        hypothesis="Actual wind was low (mainly an imbalance-market signal; cross-checks the forecast).",
        columns=(column, "wind_fc"),
    )


def nuclear(frame: MarketFrame, event: Event) -> DriverResult:
    column = "nuclear" if frame.has("nuclear") else "nuclear_rt"
    return compare_to_baseline(
        frame,
        event,
        column=column,
        name="nuclear",
        title="Nuclear generation",
        unit="MW",
        bullish_when="lower",
        hypothesis="A nuclear unit out or ramped down (less cheap baseload available).",
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
        hypothesis="Unusually high demand, e.g. a cold snap (more expensive plants needed).",
        columns=("load", "load_fc"),
    )


def imports(frame: MarketFrame, event: Event) -> DriverResult:
    """Swedish borders carry most Finnish imports, so they are judged; every border is reported.

    Estonia often flips from export to import in tight hours, which can make the total look
    fine while the Nordic supply actually fell. Judging SE1+SE3 avoids that blind spot.
    """
    swedish = [c for c in ("import_se1", "import_se3") if frame.has(c)]
    if not swedish:
        return DriverResult(
            name="imports",
            title="Imports from Sweden",
            unit="MW",
            verdict=Verdict.INSUFFICIENT,
            hypothesis="Less import from Sweden (capacity limits, or Sweden short too).",
            detail="Imports: no cross-border data.",
            columns=(),
        )
    data = frame.data.assign(import_sweden=frame.data[swedish].sum(axis=1, min_count=1))
    result = compare_to_baseline(
        MarketFrame(data=data, missing=frame.missing),
        event,
        column="import_sweden",
        name="imports",
        title="Imports from Sweden (SE1+SE3)",
        unit="MW",
        bullish_when="lower",
        hypothesis="Less import from Sweden (capacity limits, or Sweden short too).",
        columns=tuple(
            c for c in ("import_se1", "import_se3", "import_ee", "import_no4") if frame.has(c)
        ),
    )
    if result.verdict is Verdict.INSUFFICIENT:
        return result
    extra = _border_breakdown(frame, event)
    if frame.has("import_total"):
        window = event_window(frame, event)
        base = (
            baseline_frame(frame.data["import_total"]).loc[event.start : event.end]["median"].mean()
        )
        total = f" Total net import {window['import_total'].mean():,.0f} MW (normal {base:,.0f})."
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
    if not zs:
        return DriverResult(
            name="neighbours",
            title="Neighbouring prices",
            unit="EUR/MWh",
            verdict=Verdict.INSUFFICIENT,
            hypothesis="A regional move across the Nordic/Baltic market rather than Finland alone.",
            detail="Neighbouring prices: no data.",
            columns=tuple(NEIGHBOUR_PRICES),
        )
    regional = [z for z in zs if (z >= Z_SUPPORT if price_up(event) else z <= -Z_SUPPORT)]
    verdict = (
        Verdict.SUPPORTS if len(regional) >= max(1, len(zs) // 2) else Verdict.DOES_NOT_SUPPORT
    )
    where = "also abnormal" if verdict is Verdict.SUPPORTS else "roughly normal"
    detail = (
        f"Neighbouring prices during the event were {where}: "
        + "; ".join(rows)
        + ". "
        + ("A regional move." if verdict is Verdict.SUPPORTS else "Finland moved on its own.")
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
    needed = (
        "load",
        "wind" if frame.has("wind") else "wind_rt",
        "nuclear" if frame.has("nuclear") else "nuclear_rt",
    )
    if not frame.has(*needed):
        return DriverResult(
            name="residual_load",
            title="Residual load",
            unit="MW",
            verdict=Verdict.INSUFFICIENT,
            hypothesis="An unusually large gap for flexible plants and imports to fill.",
            detail="Residual load: needs load, wind and nuclear.",
            columns=needed,
        )
    data = frame.data
    if "residual_load" not in data.columns:
        data = data.assign(residual_load=data[needed[0]] - data[needed[1]] - data[needed[2]])
    return compare_to_baseline(
        MarketFrame(data=data, missing=frame.missing),
        event,
        column="residual_load",
        name="residual_load",
        title="Residual load (load - wind - nuclear)",
        unit="MW",
        bullish_when="higher",
        hypothesis="An unusually large gap for flexible plants and imports to fill.",
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
    for column, label in (
        ("import_se1", "SE1"),
        ("import_se3", "SE3"),
        ("import_ee", "EE"),
        ("import_no4", "NO4"),
    ):
        if frame.has(column):
            stats = baseline_frame(frame.data[column]).loc[event.start : event.end]
            base = stats["median"].mean()
            base_text = f" (normal {base:,.0f})" if pd.notna(base) else ""
            parts.append(f"{label} {window[column].mean():,.0f} MW{base_text}")
    return ", ".join(parts)
