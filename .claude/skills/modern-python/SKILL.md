---
name: modern-python
description: Python 3.14 + uv + ruff + ty conventions for this repo. Load before writing or reviewing any Python code here.
user-invocable: false
---

# Modern Python (3.14) for this repo

Target: Python 3.14.7, managed by `uv`. Format/lint `ruff`, types `ty`, tests `pytest`.
Run everything with `make check`. Never call `pip` or bare `python`; use `uv run` / `uv add`.

## Use 3.14 features (do)

- **Annotations are lazy by default** (PEP 649/749). Do NOT write `from __future__ import annotations`.
  Forward references just work: `def f(x: Foo) -> Bar:` before `Foo` is defined is fine.
- **Modern typing syntax**: `list[str]`, `X | None`, `type Alias = ...`, PEP 695 generics
  `def first[T](xs: list[T]) -> T:`. Never `Optional`, `List`, `Dict`, `TypeVar` boilerplate.
- **`except A, B:`** without parentheses is allowed (PEP 758) when there is no `as`.
- **Template strings** `t"..."` (PEP 750) exist. Use them only for safe structured interpolation
  (e.g. building LLM prompts where you want to inspect fields); f-strings stay the default.
- **`annotationlib.get_annotations()`** instead of `__annotations__` if introspecting.
- **`compression.zstd`** is stdlib; `uuid.uuid7()` exists; `concurrent.interpreters` exists (do not use here).
- Dataclasses: `@dataclass(frozen=True, slots=True, kw_only=True)` for result/value objects.
- `pathlib.Path` for every path (`PTH` rule enforces). `datetime` must be tz-aware (`DTZ` enforces).
- `match` statements for dispatch on result kinds. `enum.StrEnum` for closed label sets.

## Project style

- Small pure functions that take DataFrames/values in, return values out. IO at the edges
  (`copilot/data/`). No global state; no module-level network calls.
- Public functions: full type hints + one-line docstring saying what and units (EUR/MWh, MW).
- Errors: raise specific exceptions (`ValueError`, custom `DataUnavailable`), never return `None`
  to signal failure silently. Driver checks return a `DriverResult` with `verdict`
  in {`supports`, `does_not_support`, `insufficient_data`} and the numbers behind it.
- Logging via `logging.getLogger(__name__)`; `print` only in `cli.py`/`app.py` (`T20` enforces).
- Config via environment (`python-dotenv` loads `.env`). Read keys once in `copilot/config.py`.

## Tests

- `tests/test_<module>.py`, offline only, fixtures in `tests/fixtures/` (small CSV/parquet).
- Use `pytest` fixtures + `monkeypatch`, parametrize edge cases, assert exact numbers.
- Coverage floor 80% (`fail_under`). Do not test the LLM call for real; stub the client.

## Tooling cheatsheet

```bash
uv add <pkg>            # runtime dep
uv add --dev <pkg>      # dev dep
uv run pytest -x        # stop on first failure
uv run ty check         # type check
uv run ruff check --fix . && uv run ruff format .
make check              # all of the above, what CI runs
```

Ruff rule sets on: E W F I B UP SIM RUF PT PTH DTZ T20 N. If a rule blocks a legit case,
add a targeted `# noqa: RULE  reason` on that line, never disable globally.
