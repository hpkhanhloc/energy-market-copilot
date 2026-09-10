# Implementation plan

Checklist for the 72h MVP. Tick boxes as steps land. Each step ends with `make check` green
and a commit. Keep scope: Finnish day-ahead spot price, two demo investigations.

Demo events to keep working at every step:

- 2024-01-05 19:00 Helsinki: day-ahead 1896 EUR/MWh (cold snap, tight imports)
- 2023-11-24: spot -500 EUR/MWh (bid error; "evidence not sufficient" case)

## Step 0: Setup (done)

- [x] Python 3.14, uv, ruff, ty, pytest, hooks, CI, CLAUDE.md
- [x] `.env` with `FINGRID_API_KEY`, ENTSO-E token requested

## Step 1: Data layer (`copilot/data/`)

Goal: one call gives a clean hourly DataFrame for any date range, cached, works offline.

- [ ] `copilot/config.py`: load `.env`, expose keys, cache dir, `Europe/Helsinki` tz constant
- [ ] `copilot/data/cache.py`: parquet cache keyed by (source, series, start, end); TTL for recent data
- [ ] `copilot/data/entsoe.py`: thin wrapper over `entsoe-py`: `day_ahead_price(area)`, `load()`,
      `generation_by_type()`, `net_import_by_border()`, `wind_solar_forecast()`. Hourly, UTC index
- [ ] `copilot/data/fingrid.py`: generic `dataset(id, start, end)` with paging + 2 s throttle;
      helpers for wind (181), wind forecast (245), nuclear (188), imbalance price (319)
- [ ] `copilot/data/frame.py`: `market_frame(start, end)` joins everything into one hourly table:
      `price_fi, price_se1, price_se3, price_ee, load, wind, wind_fc, nuclear, hydro, import_net, ...`
- [ ] Tests with small fixture parquet files; no network. Fixture builder script in `scripts/`
- [ ] Commit

## Step 2: Event detection (`copilot/detect.py`)

Goal: given a date range, list hours where the price is abnormal, with a score.

- [ ] Baseline: same hour-of-day, same weekday type, trailing 28 days (median + MAD)
- [ ] Flags: `z_score`, `jump_vs_prev_hour`, `abs_level` thresholds. `Event` dataclass
      (frozen): start, end, peak_price, baseline, z, kind ∈ {spike, crash, negative}
- [ ] Merge adjacent abnormal hours into one event window
- [ ] `find_events(frame, top_n)` and `event_at(frame, timestamp)` for the "I know the hour" path
- [ ] Tests: synthetic series with one planted spike; DST day; both demo events found
- [ ] Commit

## Step 3: Driver checks (`copilot/drivers/`)

Goal: each check = one function, one number, one verdict. No LLM.

`DriverResult(name, verdict, value, baseline, unit, detail)`; verdict ∈
`supports | does_not_support | insufficient_data`.

- [ ] `base.py`: result type, baseline helper (same window as detect)
- [ ] `wind.py`: wind vs forecast and vs baseline (shortfall in MW)
- [ ] `nuclear.py`: nuclear output drop vs prior 7 days (OL3 trip pattern)
- [ ] `load.py`: consumption vs baseline (cold snap)
- [ ] `imports.py`: net import vs baseline, per border; capacity if available
- [ ] `neighbours.py`: SE1/SE3/EE price in same hour. High everywhere = imported;
      only FI high = local
- [ ] `residual.py`: residual load (load minus wind minus nuclear minus hydro) vs baseline
- [ ] `run_all(frame, event) -> list[DriverResult]` ordered by strength
- [ ] Tests: each driver has supports / does_not_support / insufficient_data case
- [ ] Commit

## Step 4: Charts (`copilot/plots.py`)

- [ ] Price chart: event window ±48 h, baseline band, event shaded
- [ ] Driver chart: one small multiple per driver, same x-axis
- [ ] Neighbour price chart
- [ ] Plotly figures returned, not shown (UI decides)
- [ ] Commit

## Step 5: Report (`copilot/report.py`)

- [ ] `Investigation` dataclass: event + driver results + figures
- [ ] `render_facts(inv) -> str`: deterministic text, all numbers, no LLM
- [ ] `copilot/llm.py`: Pydantic AI `Agent(COPILOT_MODEL, output_type=Narrative)` where
      `Narrative` has `summary`, `facts: list[str]`, `hypotheses: list[str]`, `insufficient: list[str]`.
      Model never sees raw series. Fallback to `render_facts` if no provider key
- [ ] Test: `TestModel` + `ALLOW_MODEL_REQUESTS = False`; assert every number in narrative exists
      in facts (no invented numbers)
- [ ] Commit

## Step 6: Interfaces

- [ ] `cli.py`: `investigate <date|datetime>` and `scan <start> <end>`; prints report, saves PNGs
- [ ] `app.py` (Streamlit, <150 lines): date picker or "scan last 30 days", event list,
      charts, report, facts/hypotheses split visually
- [ ] Run both demo events end to end, screenshot for README
- [ ] Commit

## Step 7: README + eval

- [ ] `README.md` max 2 pages: first user, use case, left out, data + signals, plain code vs
      LLM, how to measure improvement, test cases, top 3 next steps
- [ ] `tests/test_cases.md` or pytest: the two demo events as regression tests
      (detect finds them, expected drivers rank top)
- [ ] Fresh-clone test: `make setup && make check && make app` on clean checkout
- [ ] Commit, tag `v0.1`

## Stretch (only if time)

- [ ] Fingrid imbalance price (319) as second use case
- [ ] ENTSO-E outages (A80) driver if token arrives
- [ ] Auto-scan "latest interesting day" button
