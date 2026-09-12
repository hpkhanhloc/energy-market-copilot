import numpy as np
import pandas as pd
import pytest

from copilot.data.frame import MarketFrame
from copilot.detect import Event, EventKind
from copilot.drivers import Verdict, run_all
from copilot.drivers.base import pick_column
from copilot.drivers.checks import (
    imports,
    load,
    neighbour_prices,
    nuclear,
    residual_load,
    wind_actual,
    wind_forecast,
    with_derived,
)
from copilot.timeutil import ts

DAYS = 40
T0 = ts("2024-01-05 15:00")


def _index() -> pd.DatetimeIndex:
    return pd.date_range(ts("2023-12-01"), periods=DAYS * 24, freq="1h", name="time")


def _flat(level: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level + rng.normal(0, noise, DAYS * 24)


def _frame(**overrides: np.ndarray) -> MarketFrame:
    idx = _index()
    hours = np.arange(len(idx)) % 24
    base = {
        "price_fi": 60 + 20 * np.sin(hours / 24 * 2 * np.pi),
        "price_se1": _flat(40, 2, 1),
        "price_se3": _flat(45, 2, 2),
        "price_ee": _flat(70, 2, 3),
        "price_no4": _flat(30, 2, 4),
        "load": _flat(10_000, 100, 5),
        "load_fc": _flat(10_000, 100, 6),
        "wind": _flat(2_000, 100, 7),
        "wind_fc": _flat(2_000, 100, 8),
        "nuclear": _flat(4_300, 5, 9),
        "import_se1": _flat(1_000, 50, 10),
        "import_se3": _flat(1_100, 50, 11),
        "import_ee": _flat(500, 50, 12),
        "import_no4": _flat(50, 10, 13),
    }
    base.update(overrides)
    data = pd.DataFrame(base, index=idx)
    data["import_total"] = data[[c for c in data if c.startswith("import_")]].sum(axis=1)
    return MarketFrame(data=data)


def _spike(hours: int = 3) -> Event:
    return Event(
        start=T0,
        end=ts(T0 + pd.Timedelta(hours=hours - 1)),
        peak_time=ts(T0 + pd.Timedelta(hours=1)),
        peak_price=1896.0,
        baseline_median=70.0,
        z=200.0,
        kind=EventKind.SPIKE,
        hours=hours,
    )


def _crash() -> Event:
    return Event(
        start=T0,
        end=T0,
        peak_time=T0,
        peak_price=-5.0,
        baseline_median=60.0,
        z=-30.0,
        kind=EventKind.NEGATIVE,
        hours=1,
    )


def _with_event_values(column: str, values: float) -> np.ndarray:
    arr = _frame().data[column].to_numpy().copy()
    idx = _index()
    mask = (idx >= T0) & (idx <= T0 + pd.Timedelta(hours=2))
    arr[mask] = values
    return arr


def test_low_wind_forecast_supports_spike() -> None:
    frame = _frame(wind_fc=_with_event_values("wind_fc", 300.0))
    result = wind_forecast(frame, _spike())
    assert result.verdict is Verdict.SUPPORTS
    assert result.value == pytest.approx(300.0)
    assert 1_900 < (result.baseline or 0) < 2_100
    assert "300 MW" in result.detail
    assert "below normal" in result.detail


def test_normal_wind_does_not_support_spike() -> None:
    result = wind_forecast(_frame(), _spike())
    assert result.verdict is Verdict.DOES_NOT_SUPPORT


def test_low_wind_does_not_support_negative_price() -> None:
    frame = _frame(wind_fc=_with_event_values("wind_fc", 300.0))
    assert wind_forecast(frame, _crash()).verdict is Verdict.DOES_NOT_SUPPORT


def test_high_wind_supports_negative_price() -> None:
    frame = _frame(wind_fc=_with_event_values("wind_fc", 6_000.0))
    assert wind_forecast(frame, _crash()).verdict is Verdict.SUPPORTS


def test_missing_column_is_insufficient() -> None:
    frame = _frame()
    frame = MarketFrame(data=frame.data.drop(columns=["nuclear"]))
    result = nuclear(frame, _spike())
    assert result.verdict is Verdict.INSUFFICIENT
    assert "no data" in result.detail


def test_no_history_is_insufficient() -> None:
    frame = _frame()
    short = MarketFrame(data=frame.data.loc[T0 - pd.Timedelta(days=2) :])
    assert load(short, _spike()).verdict is Verdict.INSUFFICIENT


def test_nuclear_drop_supports_spike() -> None:
    frame = _frame(nuclear=_with_event_values("nuclear", 2_700.0))
    result = nuclear(frame, _spike())
    assert result.verdict is Verdict.SUPPORTS
    assert result.deviation == pytest.approx(-1_600.0, abs=20)


def test_imports_swedish_shortfall_supports_even_if_total_is_up() -> None:
    frame = _frame(
        import_se3=_with_event_values("import_se3", 0.0),
        import_ee=_with_event_values("import_ee", 2_000.0),
    )
    result = imports(frame, _spike())
    assert result.verdict is Verdict.SUPPORTS
    assert result.value == pytest.approx(1_000.0, abs=100)  # SE1 only (noisy fixture)
    assert "SE3 0 MW" in result.detail
    assert "Per border" in result.detail
    assert "Total net import" in result.detail


def test_relative_threshold_supports_big_move_with_low_z() -> None:
    # noisy series: z stays small but the move is 40% of baseline
    frame = _frame(load=_flat(10_000, 1_500, 20))
    frame.data.loc[T0 : T0 + pd.Timedelta(hours=2), "load"] = 14_000.0
    assert load(frame, _spike()).verdict is Verdict.SUPPORTS


def test_neighbours_regional_vs_local() -> None:
    local = neighbour_prices(_frame(), _spike())
    assert local.verdict is Verdict.DOES_NOT_SUPPORT
    assert "on its own" in local.detail

    regional = neighbour_prices(
        _frame(
            price_se1=_with_event_values("price_se1", 500.0),
            price_se3=_with_event_values("price_se3", 500.0),
            price_ee=_with_event_values("price_ee", 500.0),
        ),
        _spike(),
    )
    assert regional.verdict is Verdict.SUPPORTS
    assert "SE3 500 EUR/MWh" in regional.detail


def test_residual_load_supports_when_load_up_and_wind_down() -> None:
    frame = _frame(
        load=_with_event_values("load", 13_000.0), wind=_with_event_values("wind", 300.0)
    )
    assert residual_load(frame, _spike()).verdict is Verdict.SUPPORTS


def test_run_all_orders_supporting_first() -> None:
    frame = _frame(
        wind_fc=_with_event_values("wind_fc", 300.0), nuclear=_with_event_values("nuclear", 2_700.0)
    )
    results = run_all(frame, _spike())
    verdicts = [r.verdict for r in results]
    assert verdicts[:2] == [Verdict.SUPPORTS, Verdict.SUPPORTS]
    assert verdicts == sorted(
        verdicts, key=[Verdict.SUPPORTS, Verdict.DOES_NOT_SUPPORT, Verdict.INSUFFICIENT].index
    )
    assert {r.name for r in results} == {
        "wind_forecast",
        "wind_actual",
        "nuclear",
        "load",
        "imports",
        "neighbours",
        "residual_load",
    }
    assert all(r.detail for r in results)


def test_neighbours_needs_more_than_one_of_three_to_call_it_regional() -> None:
    """`len(zs) // 2` let a single abnormal neighbour assert "a regional move"."""
    frame = _frame(price_se1=_with_event_values("price_se1", 500.0))
    frame.data.drop(columns=["price_no4"], inplace=True)  # three neighbours left
    result = neighbour_prices(frame, _spike())
    assert result.verdict is Verdict.DOES_NOT_SUPPORT
    assert "1 of 3 neighbouring prices moved with Finland" in result.detail
    assert "on its own" in result.detail


def test_neighbours_reports_how_many_of_how_many_moved() -> None:
    frame = _frame(
        price_se1=_with_event_values("price_se1", 500.0),
        price_se3=_with_event_values("price_se3", 500.0),
        price_ee=_with_event_values("price_ee", 500.0),
    )
    result = neighbour_prices(frame, _spike())
    assert result.verdict is Verdict.SUPPORTS
    assert "3 of 4 neighbouring prices moved with Finland: a regional move." in result.detail


def test_neighbours_half_is_not_a_majority() -> None:
    frame = _frame(
        price_se1=_with_event_values("price_se1", 500.0),
        price_se3=_with_event_values("price_se3", 500.0),
    )
    result = neighbour_prices(frame, _spike())
    assert result.verdict is Verdict.DOES_NOT_SUPPORT
    assert "2 of 4 neighbouring prices moved with Finland, so Finland moved on its own." in (
        result.detail
    )


def test_neighbours_one_of_two_is_not_a_regional_move() -> None:
    """Rounding half up would have called this regional; two neighbours is a reachable state."""
    frame = _frame(price_se1=_with_event_values("price_se1", 500.0))
    frame.data.drop(columns=["price_ee", "price_no4"], inplace=True)
    result = neighbour_prices(frame, _spike())
    assert result.verdict is Verdict.DOES_NOT_SUPPORT
    assert "1 of 2 neighbouring prices moved with Finland" in result.detail


def test_imports_omits_the_total_baseline_when_it_has_no_history() -> None:
    """The per-border breakdown already guarded this; the total did not and printed "nan"."""
    frame = _frame(import_se3=_with_event_values("import_se3", 0.0))
    frame.data.loc[: T0 - pd.Timedelta(hours=1), "import_total"] = np.nan
    result = imports(frame, _spike())
    assert "Total net import" in result.detail
    assert "nan" not in result.detail
    assert "normal" not in result.detail.split("Total net import")[1]


def test_a_big_relative_move_that_is_statistically_ordinary_does_not_support() -> None:
    """20% of a very wide baseline can still be an entirely normal hour."""
    frame = _frame(load=_flat(10_000, 4_000, 21))
    frame.data.loc[T0 : T0 + pd.Timedelta(hours=2), "load"] = 12_500.0  # +25%, z well under 1
    result = load(frame, _spike())
    assert abs(result.z or 0.0) < 1.0
    assert result.verdict is Verdict.DOES_NOT_SUPPORT


def test_imports_needs_both_swedish_borders() -> None:
    """SE1 alone is not the Swedish total: no verdict, but the borders we have are still listed."""
    frame = _frame()
    frame = MarketFrame(data=frame.data.drop(columns=["import_se3"]))
    result = imports(frame, _spike())
    assert result.verdict is Verdict.INSUFFICIENT
    assert "no data for SE3" in result.detail
    assert "Per border: SE1" in result.detail
    assert "import_sweden" not in result.columns


def test_imports_plots_the_judged_sum_first() -> None:
    result = imports(_frame(), _spike())
    assert result.columns[0] == "import_sweden"
    assert "import_sweden" in with_derived(_frame()).data.columns


def test_neighbours_single_neighbour_is_insufficient() -> None:
    frame = _frame(price_ee=_with_event_values("price_ee", 1_000.0))
    frame = MarketFrame(data=frame.data.drop(columns=["price_se1", "price_se3", "price_no4"]))
    result = neighbour_prices(frame, _spike())
    assert result.verdict is Verdict.INSUFFICIENT
    assert "only EE" in result.detail
    assert "regional move" not in result.detail.split("tell")[0]


def test_too_few_event_hours_with_data_is_insufficient() -> None:
    """One hour out of a long event is not 'during the event'."""
    frame = _frame()
    event = _spike(hours=12)
    data = frame.data.copy()
    covered = data.index[(data.index >= event.start) & (data.index <= event.end)]
    data.loc[covered[1:], "wind"] = np.nan
    result = wind_actual(MarketFrame(data=data), event)
    assert result.verdict is Verdict.INSUFFICIENT
    assert "only 1 of 12 event hours" in result.detail


def test_partial_coverage_is_stated_in_the_detail() -> None:
    frame = _frame()
    event = _spike(hours=4)
    data = frame.data.copy()
    covered = data.index[(data.index >= event.start) & (data.index <= event.end)]
    data.loc[covered[:1], "load"] = np.nan
    result = load(MarketFrame(data=data), event)
    assert result.verdict is not Verdict.INSUFFICIENT
    assert result.detail.endswith("Based on 3 of 4 event hours.")


def test_pick_column_prefers_the_twin_with_more_data() -> None:
    frame = _frame(wind_rt=_flat(2_000, 100, 7))
    data = frame.data.copy()
    data.loc[data.index[48:], "wind"] = np.nan  # ENTSO-E wind stops after two days
    sparse = MarketFrame(data=data)
    assert pick_column(sparse, "wind", "wind_rt") == "wind_rt"
    assert pick_column(frame, "wind", "wind_rt") == "wind"
    assert pick_column(_frame(), "wind", "wind_rt") == "wind"  # no backup column at all
    assert wind_actual(sparse, _spike()).verdict is not Verdict.INSUFFICIENT
