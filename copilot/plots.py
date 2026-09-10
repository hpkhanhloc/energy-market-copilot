"""Plotly figures for an investigation. One unit per chart, fixed series colours, event shaded."""

import pandas as pd
import plotly.graph_objects as go

from copilot.detect import Event
from copilot.drivers.base import DriverResult
from copilot.investigate import Investigation

# Categorical palette in fixed slot order (validated colour-blind safe as adjacent pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BASELINE = "#8a8984"
EVENT_FILL = "rgba(235, 104, 52, 0.12)"
TZ = "Europe/Helsinki"
HOURS_BEFORE = 72
HOURS_AFTER = 24

LABELS: dict[str, str] = {
    "price_fi": "Finland",
    "price_se1": "SE1",
    "price_se3": "SE3",
    "price_ee": "Estonia",
    "price_no4": "NO4",
    "load": "Load",
    "load_fc": "Load forecast",
    "wind": "Wind",
    "wind_rt": "Wind (Fingrid real-time)",
    "wind_fc": "Wind forecast",
    "nuclear": "Nuclear",
    "nuclear_rt": "Nuclear (Fingrid)",
    "import_se1": "From SE1",
    "import_se3": "From SE3",
    "import_ee": "From Estonia",
    "import_no4": "From NO4",
    "import_total": "Total net import",
    "residual_load": "Residual load",
    "imbalance_price": "Imbalance price",
}


def price_figure(inv: Investigation) -> go.Figure:
    """Finnish day-ahead price around the event with its same-hour baseline and the event shaded."""
    from copilot.baseline import baseline_frame

    data = _clip(inv.frame.data, inv.event)
    stats = baseline_frame(inv.frame.data["price_fi"]).loc[data.index]
    fig = go.Figure()
    fig.add_trace(
        _line(
            stats.index, stats["median"], "Same-hour baseline (28-day median)", BASELINE, width=1.5
        )
    )
    fig.add_trace(_line(data.index, data["price_fi"], "Day-ahead price, Finland", SERIES[0]))
    _shade_event(fig, inv.event)
    peak = inv.event.peak_time.tz_convert(TZ)
    fig.add_annotation(
        x=peak,
        y=inv.event.peak_price,
        text=f"{inv.event.peak_price:,.0f} EUR/MWh",
        showarrow=True,
        arrowhead=0,
        ax=0,
        ay=-30,
    )
    _layout(fig, title="Day-ahead price (EUR/MWh)", unit="EUR/MWh")
    return fig


def neighbours_figure(inv: Investigation) -> go.Figure:
    columns = [
        c
        for c in ("price_fi", "price_se1", "price_se3", "price_ee", "price_no4")
        if inv.frame.has(c)
    ]
    return _multi_line(
        inv, columns, title="Day-ahead prices, Finland and neighbours (EUR/MWh)", unit="EUR/MWh"
    )


def driver_figure(inv: Investigation, result: DriverResult) -> go.Figure | None:
    """The series behind one driver check, or None when nothing is plottable."""
    data = inv.frame.data
    if result.name == "residual_load" and "residual_load" not in data.columns:
        wind = "wind" if inv.frame.has("wind") else "wind_rt"
        nuclear = "nuclear" if inv.frame.has("nuclear") else "nuclear_rt"
        if not inv.frame.has("load", wind, nuclear):
            return None
        data = data.assign(residual_load=data["load"] - data[wind] - data[nuclear])
    columns = [c for c in result.columns if c in data.columns and data[c].notna().any()]
    if not columns:
        return None
    frame_inv = Investigation(
        event=inv.event,
        frame=type(inv.frame)(data=data, missing=inv.frame.missing),
        results=inv.results,
        window_start=inv.window_start,
        window_end=inv.window_end,
    )
    return _multi_line(
        frame_inv, columns, title=f"{result.title} ({result.unit})", unit=result.unit
    )


def all_figures(inv: Investigation) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {"price": price_figure(inv)}
    for result in inv.results:
        if result.name == "neighbours":
            figures["neighbours"] = neighbours_figure(inv)
            continue
        fig = driver_figure(inv, result)
        if fig is not None:
            figures[result.name] = fig
    return figures


def _multi_line(inv: Investigation, columns: list[str], *, title: str, unit: str) -> go.Figure:
    data = _clip(inv.frame.data, inv.event)
    fig = go.Figure()
    for slot, column in enumerate(columns[: len(SERIES)]):
        fig.add_trace(_line(data.index, data[column], LABELS.get(column, column), SERIES[slot]))
    _shade_event(fig, inv.event)
    _layout(fig, title=title, unit=unit, legend=len(columns) > 1)
    return fig


def _clip(data: pd.DataFrame, event: Event) -> pd.DataFrame:
    start = event.start - pd.Timedelta(hours=HOURS_BEFORE)
    end = event.end + pd.Timedelta(hours=HOURS_AFTER)
    clipped = data.loc[start:end].copy()
    clipped.index = clipped.index.tz_convert(TZ)
    return clipped


def _line(x: pd.Index, y: pd.Series, name: str, colour: str, *, width: float = 2.0) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        mode="lines",
        line={"color": colour, "width": width},
        hovertemplate="%{x|%a %d %b %H:%M}<br>%{y:,.0f}<extra>" + name + "</extra>",
    )


def _shade_event(fig: go.Figure, event: Event) -> None:
    fig.add_vrect(
        x0=event.start.tz_convert(TZ),
        x1=(event.end + pd.Timedelta(hours=1)).tz_convert(TZ),
        fillcolor=EVENT_FILL,
        line_width=0,
        layer="below",
    )


def _layout(fig: go.Figure, *, title: str, unit: str, legend: bool = True) -> None:
    fig.update_layout(
        title=title,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        height=320,
        hovermode="x unified",
        showlegend=legend,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
        yaxis={"title": unit, "gridcolor": "#ebebe8", "zeroline": True, "zerolinecolor": "#c9c8c2"},
        xaxis={"showgrid": False, "title": f"Time ({TZ})"},
    )
