# Implementation plan

Checklist for the 72h MVP. Tick boxes as steps land. Each step ends with `make check` green
and a commit. Keep scope: Finnish day-ahead spot price, two demo investigations.

Demo events to keep working at every step:

- 2024-01-05 19:00 Helsinki: day-ahead 1896 EUR/MWh (cold snap, tight imports)
- 2023-12-16 19:00 to 2023-12-17 07:00 Helsinki: prices at or below 0 for 13 h (windy, low load)
- 2023-11-24: spot -500 EUR/MWh (bid error; "evidence not sufficient" case), stretch

## Step 0: Setup (done)

- [x] Python 3.14, uv, ruff, ty, pytest, hooks, CI, CLAUDE.md
- [x] `.env` with `FINGRID_API_KEY`, ENTSO-E token requested

## Step 1: Data layer (`copilot/data/`)

Goal: one call gives a clean hourly DataFrame for any date range, cached, works offline.

- [x] `copilot/config.py`: load `.env`, expose keys, cache dir, `Europe/Helsinki` tz constant
- [x] `copilot/data/cache.py`: parquet cache keyed by (source, series, start, end); TTL for recent data
- [x] `copilot/data/entsoe.py`: thin wrapper over `entsoe-py`: `day_ahead_price(area)`, `load()`,
      `generation_by_type()`, `net_import_by_border()`, `wind_solar_forecast()`. Hourly, UTC index
- [x] `copilot/data/fingrid.py`: generic `dataset(id, start, end)` with paging + 2 s throttle;
      helpers for wind (181), wind forecast (245), nuclear (188), imbalance price (319)
- [x] `copilot/data/frame.py`: `market_frame(start, end)` joins everything into one hourly table:
      `price_fi, price_se1, price_se3, price_ee, load, wind, wind_fc, nuclear, hydro, import_net, ...`
- [x] Tests use in-memory fakes; `scripts/warm_cache.py` pulls real windows into `data/cache/`
- [x] Commit

## Step 2: Event detection (`copilot/detect.py`)

Goal: given a date range, list hours where the price is abnormal, with a score.

- [x] Baseline: same *local* hour-of-day, same day type (weekday/weekend), trailing 28 days (median + MAD)
- [x] Flags: `z_score`, `jump_vs_prev_hour`, `abs_level` thresholds. `Event` dataclass
      (frozen): start, end, peak_price, baseline, z, kind ∈ {spike, crash, negative}
- [x] Merge adjacent abnormal hours into one event window
- [x] `find_events(frame, top_n)` and `event_at(frame, timestamp)` for the "I know the hour" path
- [x] Tests: synthetic series with one planted spike; DST day; both demo events found
- [x] Commit

## Step 3: Driver checks (`copilot/drivers/`)

Goal: each check = one function, one number, one verdict. No LLM.

`DriverResult(name, verdict, value, baseline, unit, detail)`; verdict ∈
`supports | does_not_support | insufficient_data`.

- [x] `base.py`: result type, baseline helper (same window as detect)
- [x] `wind.py`: wind vs forecast and vs baseline (shortfall in MW)
- [x] `nuclear.py`: nuclear output drop vs prior 7 days (OL3 trip pattern)
- [x] `load.py`: consumption vs baseline (cold snap)
- [x] `imports.py`: net import vs baseline, per border; capacity if available
- [x] `neighbours.py`: SE1/SE3/EE price in same hour. High everywhere = imported;
      only FI high = local
- [x] `residual.py`: residual load (load minus wind minus nuclear minus hydro) vs baseline
- [x] `run_all(frame, event) -> list[DriverResult]` ordered by strength
- [x] Tests: each driver has supports / does_not_support / insufficient_data case
- [x] Commit

## Step 4: Charts (`copilot/plots.py`)

- [x] Price chart: event window ±48 h, baseline band, event shaded
- [x] Driver chart: one small multiple per driver, same x-axis
- [x] Neighbour price chart
- [x] Plotly figures returned, not shown (UI decides)
- [x] Commit

## Step 5: Report (`copilot/report.py`)

- [x] `Investigation` dataclass: event + driver results + figures
- [x] `render_facts(inv) -> str`: deterministic text, all numbers, no LLM
- [x] `copilot/llm.py`: Pydantic AI `Agent(COPILOT_MODEL, output_type=Narrative)` where
      `Narrative` has `summary`, `facts: list[str]`, `hypotheses: list[str]`, `insufficient: list[str]`.
      Model never sees raw series. Fallback to `render_facts` if no provider key
- [x] Test: `TestModel` + `ALLOW_MODEL_REQUESTS = False`; assert every number in narrative exists
      in facts (no invented numbers)
- [x] Commit

## Step 6: Interfaces

- [x] `cli.py`: `investigate <date|datetime>` and `scan <start> <end>`; prints report, saves PNGs
- [x] `app.py` (Streamlit, <150 lines): date picker or "scan last 30 days", event list,
      charts, report, facts/hypotheses split visually
- [x] Run both demo events end to end (CLI + real LLM); screenshots skipped, README stays text
- [x] Commit

## Step 7: README + eval

- [x] `README.md` max 2 pages: first user, use case, left out, data + signals, plain code vs
      LLM, how to measure improvement, test cases, top 3 next steps
- [x] `tests/test_demo_events.py`: the two demo events as regression tests on a real-data fixture
      (detect finds them, expected drivers rank top)
- [x] Fresh-clone test: `make setup && make check` on clean checkout (84 tests pass)
- [x] Commit, tag `v0.1`

## Step 8: Chat UI (v0.2)

Goal: the user types "something odd happened"; the LLM only routes, plain code answers.

- [x] `copilot/report.py`: `unknown_numbers_in_text`, `banned_phrases` (caused / because / due to)
- [x] `copilot/trace.py`: one JSON line per LLM call (kind, guard, fallback, latency)
- [x] `copilot/intent.py`: `Scan | Investigate | Ask | Reply`, context block, `guard_intent`
- [x] `copilot/chat.py`: follow-up answers from the facts text, same guards
- [x] `views.py` + `app.py`: chat loop, starter chips (no LLM), clickable scan table (no LLM),
      collapsed older investigations, sidebar debug of the last LLM call
- [x] Manual run: both demo chips, row click, term question, off-topic, relative date
- [x] Commit

## Step 9: Eval + docs

- [x] `tests/fixtures/intents.jsonl` (28 cases) + `scripts/eval_intents.py` + `make eval`
      (real model, not in CI). First run 86%, after prompt fix 96%.
- [x] Offline tests for every guard path with `TestModel`
- [x] `CLAUDE.md`, `README.md`, this plan
- [x] Commit, tag `v0.2`

## Stretch (only if time)

- [ ] Fingrid imbalance price (319) as second use case
- [ ] ENTSO-E outages (A80) driver if token arrives
- [ ] Suggested next questions after each answer (chips, still no LLM tool calls)
- [ ] Auto-scan "latest interesting day" chip
