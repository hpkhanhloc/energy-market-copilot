# Energy Market Copilot

MVP for a 72h home assignment (Capalo AI). Assignment PDF: `~/Downloads/AI_Engineer_Assignment.pdf`.
Goal: user says "something odd happened in the Finnish power market" and the tool answers
"here is what happened, the evidence, and the most plausible drivers to investigate".

Step-by-step build plan with checkboxes: `docs/PLAN.md`. Tick steps there as they land.

## Product decisions (do not re-open without asking)

- Market: Finland. Use case: **day-ahead spot price** anomalies. Imbalance price is a stretch goal.
- First user: an energy-market analyst who investigates price moves by hand today.
- Interface: `copilot/` library is the core. `app.py` (Streamlit) is a chat demo: one LLM call maps
  the user's words to a typed intent (`Scan` / `Investigate` / `Ask` / `Reply`), plain code runs it,
  the same tables and charts render in the chat bubble. Starter chips and table-row clicks never
  call the LLM. `views.py` holds the Streamlit glue. `cli.py` is a thin backup.
- No cloud deploy, no styling, no auth. Local only.
- Must demo at least two investigations. Known good cases: 2024-01-05 19:00 Helsinki (1896 EUR/MWh day-ahead spike),
  2023-12-16 19:00 to 12-17 07:00 Helsinki (13 h at or below 0 EUR/MWh, windy night).
  `make warm` caches Dec 2023 + Jan 2024; `uv run python scripts/warm_cache.py <start> <end>` for more.

## Honesty rules (assignment grades this)

- Facts and hypotheses stay separate. Every claim carries the number that backs it.
- Never say "caused". Say "consistent with" / "supports" / "does not support" / "not enough data".
- If a driver check has missing data, report "not enough data", never guess.

## Where plain code vs LLM

- **Plain code, deterministic, tested:** data fetch + cache, anomaly detection, every driver check,
  charts. These must give the same answer every run.
- **LLM (via Pydantic AI, provider-agnostic):** three calls, all typed, all guarded, all traced.
  Model comes from `COPILOT_MODEL` env (default `anthropic:claude-sonnet-5`); swapping provider is
  an env change, not a code change. Tests use `TestModel` and set
  `models.ALLOW_MODEL_REQUESTS = False`. No agents / tool calling: the LLM never sees raw time
  series, never picks the event, never runs a check. Evidence is reproducible run to run.
  1. `copilot/intent.py` routing: user text + context (today, data reach, last hour, last range,
     short transcript) -> `Scan | Investigate | Ask | Reply`. `guard_intent` re-checks dates
     (60-day cap, data reach, future). Any failure becomes a `Reply`.
  2. `copilot/llm.py` narrative: facts text -> `Narrative(facts, hypotheses, insufficient,
     summary)`. Guards: unknown numbers, causal words (`caused`, `because`, `due to`, ...),
     invented `insufficient` items. Guard hit -> deterministic `fallback_narrative`.
  3. `copilot/chat.py` follow-up: question + facts text -> `Answer(text, source, hour_not_in_report)`.
     Same number and causal guards; guard hit -> fixed safe text. `source="general"` is shown
     with a "general knowledge, not from your data" label.
  Every call appends one line to `data/logs/llm.jsonl` (`copilot/trace.py`: kind, guard,
  fallback, latency). `make eval` scores routing on `tests/fixtures/intents.jsonl` with the real
  model (not in CI; last run 96% on 28 cases).

## Data sources (verified 2026-09-10)

Only the two sources named in the assignment. Both keys are in `.env` and verified working.
- ENTSO-E via `entsoe-py` (`ENTSOE_API_KEY`): day-ahead spot price FI + neighbours (SE_1, SE_3,
  EE, NO_4), load, generation by type, cross-border physical flows, wind/solar forecast, NTC.
  **Source for the price series and everything cross-border.** 400 req/min. Host is blocked in
  the Claude sandbox; test calls need the sandbox off.
- Fingrid `https://data.fingrid.fi/api/datasets/{id}/data` (`FINGRID_API_KEY`, header
  `x-api-key`, 1 req / 2 s, 10k/day): Finnish real-time detail. IDs: 319 imbalance price,
  244/106 mFRR up/down, 181 wind, 245 wind forecast, 188 nuclear, 191 hydro, 192/193 real-time
  prod/cons, 165 consumption forecast, 241 production forecast. No spot price here.
- Not used: energy-charts, porssisahko etc. Outside the assignment brief.
- All fetches go through `copilot/data/` and cache to `data/cache/*.parquet` so demos work offline.

## Stack and layout

- Python 3.14.7 (`.python-version`), `uv` for env, `ruff` format+lint, `ty` type check,
  `pytest` + coverage (floor 80%), `pandas`, `plotly`, `streamlit`, `pydantic-ai-slim`.
- Load skill `modern-python` before writing Python here (3.14 idioms, no `__future__` annotations).
- Run: `make setup` once, then `make check` (fmt, lint, type, test). `make app`, `make cli ARGS=2024-01-05`.
- Pre-commit runs the same checks; CI (`.github/workflows/ci.yml`) too. If CI would fail, do not commit.
- Layout (target):
  - `copilot/data/` fetch + cache per source
  - `copilot/detect.py` anomaly finder
  - `copilot/drivers/` one file per driver check, each returns a `DriverResult`
  - `copilot/report.py` facts text, fallback narrative, number + causal-word guards
  - `copilot/llm.py`, `copilot/intent.py`, `copilot/chat.py` the three LLM calls
  - `copilot/trace.py` JSONL log of every LLM call
  - `copilot/plots.py` charts
  - `tests/` pytest, offline only, fixtures under `tests/fixtures/`

## Coding rules

- Every new function in `copilot/` gets a pytest. Tests never hit the network: use small fixture
  CSV/parquet files or monkeypatch the fetcher.
- Timestamps: always tz-aware, store UTC, show `Europe/Helsinki`.
- Type hints on public functions. Small functions. No classes unless state is real.
- Keep `app.py` under ~150 lines. Streamlit rendering goes in `views.py`; logic lives in `copilot/`.
- Secrets only in `.env` (git-ignored). Never hardcode keys.
- Hooks in `.claude/hooks/` auto-format, lint, type-check and run tests after every edit, and block
  ending a turn with red checks. If a hook fails, fix the code, do not disable the hook.
- Workflow for a feature: plan in plan mode (or `/feature-dev` for big ones), write code in the main
  thread, then run `test-writer` and `honesty-reviewer` agents, then `make check`, then commit.

## Agents

- `.claude/agents/test-writer.md` — writes offline pytest tests for a module.
- `.claude/agents/honesty-reviewer.md` — reviews diff for bugs and for fact/hypothesis leaks.
Use them before declaring a feature done.
