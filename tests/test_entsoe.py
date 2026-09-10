from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copilot.data.entsoe import EntsoeSource
from copilot.timeutil import helsinki, ts

START = helsinki("2024-01-05 00:00")
END = helsinki("2024-01-06 00:00")


def _quarter_hours(values: list[float]) -> pd.DatetimeIndex:
    # entsoe-py returns local-time indexes at 15-min or hourly resolution
    return pd.date_range(START, periods=len(values), freq="15min", tz="Europe/Helsinki")


@dataclass
class FakeClient:
    calls: list[tuple] = field(default_factory=list)

    def query_day_ahead_prices(self, country_code, start, end) -> pd.Series:
        self.calls.append(("price", country_code))
        values = [10.0, 20.0, 30.0, 40.0, 100.0, 100.0, 100.0, 100.0]
        return pd.Series(values, index=_quarter_hours(values), name="price")

    def query_load(self, country_code, start, end) -> pd.DataFrame:
        self.calls.append(("load", country_code))
        values = [1000.0, 1100.0, 1200.0, 1300.0]
        return pd.DataFrame({"Actual Load": values}, index=_quarter_hours(values))

    def query_load_forecast(self, country_code, start, end) -> pd.DataFrame:
        values = [900.0, 900.0, 900.0, 900.0]
        return pd.DataFrame({"Forecasted Load": values}, index=_quarter_hours(values))

    def query_generation(self, country_code, start, end) -> pd.DataFrame:
        self.calls.append(("generation", country_code))
        idx = _quarter_hours([0.0] * 4)
        columns = pd.MultiIndex.from_tuples(
            [
                ("Nuclear", "Actual Aggregated"),
                ("Hydro Water Reservoir", "Actual Aggregated"),
                ("Hydro Run-of-river and pondage", "Actual Aggregated"),
                ("Wind Onshore", "Actual Aggregated"),
                ("Hydro Pumped Storage", "Actual Consumption"),
                ("Mystery", "Actual Aggregated"),
            ]
        )
        data = np.array([[4000.0, 1000.0, 500.0, 800.0, 50.0, 7.0]] * 4)
        return pd.DataFrame(data, index=idx, columns=columns)

    def query_crossborder_flows(self, country_code_from, country_code_to, start, end) -> pd.Series:
        self.calls.append(("flow", country_code_from, country_code_to))
        into_fi = country_code_to == "FI"
        value = 1000.0 if into_fi else 200.0
        if country_code_from == "SE_3" or country_code_to == "SE_3":
            value = 1200.0 if into_fi else 0.0
        values = [value] * 4
        return pd.Series(values, index=_quarter_hours(values))

    def query_wind_and_solar_forecast(self, country_code, start, end) -> pd.DataFrame:
        values = [500.0] * 4
        return pd.DataFrame(
            {"Solar": [1.0] * 4, "Wind Onshore": values, "Wind Offshore": [5.0] * 4},
            index=_quarter_hours(values),
        )


@pytest.fixture
def source(tmp_path: Path) -> tuple[EntsoeSource, FakeClient]:
    client = FakeClient()
    return EntsoeSource(client, cache_dir=tmp_path), client


def test_price_is_hourly_mean_in_utc(source: tuple[EntsoeSource, FakeClient]) -> None:
    src, _ = source
    price = src.day_ahead_price("FI", START, END)
    assert price.name == "price_fi"
    assert price.index[0] == ts("2024-01-04 22:00")  # Helsinki midnight is 22:00 UTC
    assert price.tolist() == [25.0, 100.0]


def test_neighbour_price_column_name(source: tuple[EntsoeSource, FakeClient]) -> None:
    src, _ = source
    assert src.day_ahead_price("SE_3", START, END).name == "price_se3"


def test_price_is_cached(source: tuple[EntsoeSource, FakeClient]) -> None:
    src, client = source
    src.day_ahead_price("FI", START, END)
    src.day_ahead_price("FI", START, END)
    assert client.calls.count(("price", "FI")) == 1


def test_load_and_forecast(source: tuple[EntsoeSource, FakeClient]) -> None:
    src, _ = source
    assert src.load(START, END).tolist() == [1150.0]
    assert src.load_forecast(START, END).tolist() == [900.0]


def test_generation_groups_types_and_drops_consumption(
    source: tuple[EntsoeSource, FakeClient],
) -> None:
    src, _ = source
    gen = src.generation_by_type(START, END)
    assert set(gen.columns) == {"nuclear", "hydro", "wind", "generation_total"}
    assert gen["hydro"].iloc[0] == 1500.0
    assert gen["nuclear"].iloc[0] == 4000.0
    # total counts every "Actual Aggregated" type, even ones we do not name
    assert gen["generation_total"].iloc[0] == 4000.0 + 1000.0 + 500.0 + 800.0 + 7.0


def test_net_import_per_border(source: tuple[EntsoeSource, FakeClient]) -> None:
    src, client = source
    imports = src.net_import(START, END)
    assert imports["import_se1"].iloc[0] == 800.0
    assert imports["import_se3"].iloc[0] == 1200.0
    assert imports["import_total"].iloc[0] == 800.0 + 1200.0 + 800.0 + 800.0
    assert ("flow", "SE_1", "FI") in client.calls
    assert ("flow", "FI", "SE_1") in client.calls


def test_wind_forecast_sums_wind_columns_only(source: tuple[EntsoeSource, FakeClient]) -> None:
    src, _ = source
    assert src.wind_forecast(START, END).tolist() == [505.0]


def test_naive_timestamps_rejected(source: tuple[EntsoeSource, FakeClient]) -> None:
    src, _ = source
    with pytest.raises(ValueError, match="tz-aware"):
        src.load(pd.Timestamp("2024-01-05"), END)  # ty: ignore[invalid-argument-type]
