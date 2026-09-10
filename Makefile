.PHONY: eval backtest setup fmt lint type test check app cli

setup:      ## install everything + git hooks
	uv sync
	uv run pre-commit install

fmt:        ## format code
	uv run ruff format .

lint:       ## lint + autofix
	uv run ruff check --fix .

type:       ## type check
	uv run ty check

test:       ## run tests with coverage
	uv run pytest --cov

check: fmt lint type test  ## everything CI runs

app:
	uv run streamlit run app.py

cli:
	uv run python cli.py $(ARGS)

eval:  ## score the intent parser on tests/fixtures/intents.jsonl with the real model (not in CI)
	uv run python scripts/eval_intents.py

backtest:  ## run the detector over the cached price history (offline, not in CI)
	uv run python scripts/backtest.py $(ARGS)
