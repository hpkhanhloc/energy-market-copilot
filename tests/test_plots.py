import numpy as np
import pandas as pd
import pytest

from copilot.data.frame import MarketFrame
from copilot.investigate import Investigation, investigate_at, scan
from copilot.plots import all_figures, driver_figure, neighbours_figure, price_figure
from copilot.timeutil import ts


@pytest.fixture
def investigation() -> Investigation:
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    hours = np.arange(len(idx)) % 24
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "price_fi": 60 + 20 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 3, len(idx)),
            "price_se3": 45 + rng.normal(0, 2, len(idx)),
            "load": 10_000 + rng.normal(0, 100, len(idx)),
            "load_fc": 10_000 + rng.normal(0, 100, len(idx)),
            "consumption_rt": 10_000 + rng.normal(0, 100, len(idx)),
            "wind": 2_000 + rng.normal(0, 100, len(idx)),
            "wind_rt": 2_000 + rng.normal(0, 100, len(idx)),
            "wind_fc": 2_000 + rng.normal(0, 100, len(idx)),
            "wind_fc_fingrid": 2_000 + rng.normal(0, 100, len(idx)),
            "nuclear": 4_300 + rng.normal(0, 5, len(idx)),
            "import_se1": 1_000 + rng.normal(0, 50, len(idx)),
            "import_se3": 1_100 + rng.normal(0, 50, len(idx)),
        },
        index=idx,
    )
    data["import_total"] = data["import_se1"] + data["import_se3"]
    t0 = ts("2024-01-05 15:00")
    data.loc[t0 : t0 + pd.Timedelta(hours=2), "price_fi"] = [900.0, 1896.0, 700.0]
    data.loc[t0 : t0 + pd.Timedelta(hours=2), "wind_fc"] = 300.0
    return investigate_at(MarketFrame(data=data), ts("2024-01-05 16:00"))


def test_investigation_finds_event_and_runs_drivers(investigation: Investigation) -> None:
    assert investigation.event.peak_price == 1896.0
    assert investigation.event.flagged
    assert [r.name for r in investigation.supporting] == ["wind_forecast"]
    assert len(investigation.results) == 7


def test_scan_lists_events(investigation: Investigation) -> None:
    events = scan(investigation.frame)
    assert events[0].peak_price == 1896.0


def test_price_figure_has_baseline_price_and_annotation(investigation: Investigation) -> None:
    fig = price_figure(investigation)
    assert [t.name for t in fig.data] == [
        "Same-hour baseline (median)",
        "Day-ahead price, Finland",
    ]
    assert "1,896 EUR/MWh" in fig.layout.annotations[0].text
    assert str(fig.data[1].x[0].tzinfo) == "Europe/Helsinki"


def test_neighbours_figure_uses_available_columns_only(investigation: Investigation) -> None:
    fig = neighbours_figure(investigation)
    assert [t.name for t in fig.data] == ["Finland", "SE3"]


def test_driver_figures(investigation: Investigation) -> None:
    figures = all_figures(investigation)
    assert {"price", "neighbours", "wind_forecast", "residual_load", "imports"} <= set(figures)
    residual = figures["residual_load"]
    assert [t.name for t in residual.data] == [
        "Residual load same-hour baseline (median)",
        "Residual load",
    ]
    imports = figures["imports"]
    assert imports.data[0].name == "From Sweden (SE1+SE3) same-hour baseline (median)"
    assert imports.data[1].name == "From Sweden (SE1+SE3)"
    wind = figures["wind_actual"]
    assert [t.name for t in wind.data] == [
        "Wind (ENTSO-E) same-hour baseline (median)",
        "Wind (ENTSO-E)",
        "Wind (Fingrid real-time)",
        "Wind forecast (ENTSO-E)",
    ]
    assert [t.name for t in wind.data if t.visible == "legendonly"] == ["Wind (Fingrid real-time)"]
    forecast = figures["wind_forecast"]
    assert [t.name for t in forecast.data if t.visible == "legendonly"] == [
        "Wind forecast (Fingrid)"
    ]
    load = figures["load"]
    baseline, actual = load.data[0], load.data[1]
    assert baseline.name == "Consumption (ENTSO-E) same-hour baseline (median)"
    assert actual.name == "Consumption (ENTSO-E)"
    assert [t.name for t in load.data if t.visible == "legendonly"] == [
        "Consumption (Fingrid real-time)"
    ]
    assert len(baseline.x) == len(actual.x)


def test_driver_figure_none_when_no_columns(investigation: Investigation) -> None:
    result = next(r for r in investigation.results if r.name == "nuclear")
    frame = MarketFrame(data=investigation.frame.data.drop(columns=["nuclear"]))
    inv = Investigation(
        event=investigation.event,
        frame=frame,
        results=investigation.results,
        window_start=investigation.window_start,
        window_end=investigation.window_end,
    )
    assert driver_figure(inv, result) is None


def test_investigate_without_price_raises() -> None:
    frame = MarketFrame(
        data=pd.DataFrame(index=pd.date_range(ts("2024-01-01"), periods=3, freq="1h"))
    )
    with pytest.raises(ValueError, match="price"):
        investigate_at(frame, ts("2024-01-01 01:00"))


def test_every_figure_marks_the_peak_hour(investigation: Investigation) -> None:
    peak = investigation.event.peak_time.tz_convert("Europe/Helsinki")
    for fig in all_figures(investigation).values():
        lines = [sh for sh in fig.layout.shapes if sh.type == "line"]
        assert len(lines) == 1
        assert pd.Timestamp(lines[0].x0) == peak
