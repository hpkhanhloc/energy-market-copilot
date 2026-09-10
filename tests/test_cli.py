import numpy as np
import pandas as pd
import pytest

import cli
from copilot.data.frame import MarketFrame
from copilot.timeutil import ts


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
    assert "Consistent with: Low forecast wind" in out
    assert (tmp_path / "price.html").exists()


def test_scan(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["scan", "2023-12-08", "2024-01-08", "--top", "3"])
    out = capsys.readouterr().out
    assert "spike" in out
    assert "1,896" in out
    assert "investigate 2024-01-05T18:00" in out
