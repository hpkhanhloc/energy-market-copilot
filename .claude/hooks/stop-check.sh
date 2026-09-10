#!/usr/bin/env bash
# Stop hook: before Claude ends a turn, full lint + test suite must be green.
cd "$(dirname "$0")/../.." || exit 0
# Avoid infinite loop: if this hook already fired for this stop, let it through.
active=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("stop_hook_active", False))' 2>/dev/null)
[ "$active" = "True" ] && exit 0
# Only run when there are python changes in the working tree.
git status --porcelain 2>/dev/null | grep -q '\.py$' || exit 0

out=$(uv run ruff check . 2>&1) || { echo "ruff check failed:"; echo "$out"; exit 2; } >&2
out=$(uv run ty check 2>&1) || { echo "ty check failed:"; echo "$out"; exit 2; } >&2
out=$(uv run pytest -q 2>&1 | tail -30) || { echo "pytest failed:"; echo "$out"; exit 2; } >&2
exit 0
