import numpy as np
import pandas as pd
import pytest

import cli
from copilot.data.frame import MarketFrame
from copilot.timeutil import helsinki, ts


def _frame() -> MarketFrame:
    idx = pd.date_range(ts("2023-12-01"), periods=40 * 24, freq="1h", name="time")
    hours = np.arange(len(idx)) % 24
    rng = np.random.default_rng(3)
    data = pd.DataFrame(
        {
            "price_fi": 60 + 20 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 3, len(idx)),
            "wind_fc": 2_000 + rng.normal(0, 100, len(idx)),
        },
        index=idx,
    )
    t0 = ts("2024-01-05 15:00")
    data.loc[t0 : t0 + pd.Timedelta(hours=2), "price_fi"] = [900.0, 1896.0, 700.0]
    data.loc[t0 : t0 + pd.Timedelta(hours=2), "wind_fc"] = 300.0
    return MarketFrame(data=data)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_window", lambda settings, start, end: _frame())


def test_investigate_no_llm(capsys: pytest.CaptureFixture[str], tmp_path) -> None:
    cli.main(["investigate", "2024-01-05T18:00", "--no-llm", "--charts", str(tmp_path)])
    out = capsys.readouterr().out
    assert "## Price spike on Fri 05 Jan 2024" in out
    assert "HYPOTHESES (consistent with the evidence, not proven):" in out
    assert "Low forecast wind" in out
    assert (tmp_path / "price.html").exists()


def test_scan(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["scan", "2023-12-08", "2024-01-08", "--top", "3"])
    out = capsys.readouterr().out
    assert "spike" in out
    assert "1,896" in out
    assert "investigate 2024-01-05T18:00" in out


def test_scan_asks_for_a_window_that_covers_the_last_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """`helsinki(end)` is midnight at the *start* of the end day and the window is half-open."""
    seen: list[pd.Timestamp] = []

    def capture(settings, start, end):
        seen.append(end)
        return _frame()

    monkeypatch.setattr(cli, "load_window", capture)
    cli.main(["scan", "2023-12-15", "2024-01-05"])
    assert seen[0] == helsinki("2024-01-06")  # not 2024-01-05, which would drop the last day


def test_scan_prints_a_dash_when_an_episode_has_no_baseline(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    short = MarketFrame(data=_frame().data.loc[: ts("2023-12-04")].copy())
    short.data.loc[ts("2023-12-03 02:00"), "price_fi"] = -20.0  # negative: an event with no history
    monkeypatch.setattr(cli, "load_window", lambda settings, start, end: short)
    cli.main(["scan", "2023-12-01", "2023-12-04"])
    out = capsys.readouterr().out
    assert "negative" in out
    assert "nan" not in out
    assert "—" in out
