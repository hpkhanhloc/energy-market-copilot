#!/usr/bin/env bash
# PostToolUse hook: after Claude edits a .py file, format + lint it, then run tests.
# Reads tool input JSON on stdin. Non-zero exit code 2 sends stderr back to Claude.
set -u
cd "$(dirname "$0")/../.." || exit 0

file=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

uv run ruff format "$file" >/dev/null 2>&1
lint=$(uv run ruff check --fix "$file" 2>&1)
if [ $? -ne 0 ]; then
  echo "ruff check failed for $file:" >&2
  echo "$lint" >&2
  exit 2
fi

types=$(uv run ty check "$file" 2>&1)
if [ $? -ne 0 ]; then
  echo "ty check failed for $file:" >&2
  echo "$types" >&2
  exit 2
fi

tests=$(uv run pytest -x -q --no-cov 2>&1 | tail -20)
if [ $? -ne 0 ] && ! echo "$tests" | grep -q "no tests ran"; then
  echo "pytest failed after editing $file:" >&2
  echo "$tests" >&2
  exit 2
fi
exit 0
