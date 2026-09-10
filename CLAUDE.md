# Energy Market Copilot

MVP for a 72h home assignment (Capalo AI). Assignment PDF: `~/Downloads/AI_Engineer_Assignment.pdf`.
Goal: user says "something odd happened in the Finnish power market" and the tool answers
"here is what happened, the evidence, and the most plausible drivers to investigate".

## Product decisions (do not re-open without asking)

- Market: Finland. Use case: **day-ahead spot price** anomalies. Imbalance price is a stretch goal.
- First user: an energy-market analyst who investigates price moves by hand today.
- Interface: `copilot/` library is the core. `app.py` (Streamlit) is the demo. `cli.py` is a thin backup.
- No cloud deploy, no styling, no auth. Local only.
- Must demo at least two investigations. Known good cases: 2024-01-05 17:00 (~2180 EUR/MWh spike),
  2023-11-24 (-500 EUR/MWh, bid error, "evidence not sufficient" case).

## Honesty rules (assignment grades this)

- Facts and hypotheses stay separate. Every claim carries the number that backs it.
- Never say "caused". Say "consistent with" / "supports" / "does not support" / "not enough data".
- If a driver check has missing data, report "not enough data", never guess.

## Where plain code vs LLM

- **Plain code, deterministic, tested:** data fetch + cache, anomaly detection, every driver check,
  charts. These must give the same answer every run.
- **LLM (Claude via `anthropic`):** only turns the structured check results into a short readable
  report. It never sees raw time series, never picks the event, never invents numbers.
  Model: `claude-sonnet-5`. Prompt must force `FACT:` / `HYPOTHESIS:` labels.

## Data sources (verified 2026-09-10)

- `https://api.energy-charts.info` — no key. `price?bzn=FI` (spot, 2015+), `public_power?country=fi`
  (generation by type + load, 15-min), `cbpf?country=fi` (flows by neighbour). Also SE1, SE3, EE, NO4
  prices. License: private/internal use only, say so in README. **Primary source for MVP.**
- Fingrid `https://data.fingrid.fi/api/datasets/{id}/data` — needs `FINGRID_API_KEY` header
  `x-api-key`. 1 req / 2 s, 10k/day. Key IDs: 319 imbalance price, 244/106 mFRR up/down,
  181 wind, 245 wind forecast, 188 nuclear, 192/193 real-time prod/cons. No spot price here.
- ENTSO-E via `entsoe-py` — needs `ENTSOE_API_KEY`, may not arrive in time. Optional.
- All fetches go through `copilot/data/` and cache to `data/cache/*.parquet` so demos work offline.

## Stack and layout

- Python 3.14.7 (`.python-version`), `uv` for env, `ruff` format+lint, `ty` type check,
  `pytest` + coverage (floor 80%), `pandas`, `plotly`, `streamlit`.
- Load skill `modern-python` before writing Python here (3.14 idioms, no `__future__` annotations).
- Run: `make setup` once, then `make check` (fmt, lint, type, test). `make app`, `make cli ARGS=2024-01-05`.
- Pre-commit runs the same checks; CI (`.github/workflows/ci.yml`) too. If CI would fail, do not commit.
- Layout (target):
  - `copilot/data/` fetch + cache per source
  - `copilot/detect.py` anomaly finder
  - `copilot/drivers/` one file per driver check, each returns a `DriverResult`
  - `copilot/report.py` LLM narrative
  - `copilot/plots.py` charts
  - `tests/` pytest, offline only, fixtures under `tests/fixtures/`

## Coding rules

- Every new function in `copilot/` gets a pytest. Tests never hit the network: use small fixture
  CSV/parquet files or monkeypatch the fetcher.
- Timestamps: always tz-aware, store UTC, show `Europe/Helsinki`.
- Type hints on public functions. Small functions. No classes unless state is real.
- Keep `app.py` under ~150 lines. Logic lives in `copilot/`, not in the UI.
- Secrets only in `.env` (git-ignored). Never hardcode keys.
- Hooks in `.claude/hooks/` auto-format, lint, type-check and run tests after every edit, and block
  ending a turn with red checks. If a hook fails, fix the code, do not disable the hook.
- Workflow for a feature: plan in plan mode (or `/feature-dev` for big ones), write code in the main
  thread, then run `test-writer` and `honesty-reviewer` agents, then `make check`, then commit.

## Agents

- `.claude/agents/test-writer.md` — writes offline pytest tests for a module.
- `.claude/agents/honesty-reviewer.md` — reviews diff for bugs and for fact/hypothesis leaks.
Use them before declaring a feature done.
