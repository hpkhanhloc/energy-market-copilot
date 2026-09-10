#!/usr/bin/env bash
# PreToolUse hook: block edits to secrets and lock files.
file=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("file_path",""))' 2>/dev/null)
case "$file" in
  */.env|.env|*/uv.lock|uv.lock)
    echo "Blocked: $file is a secret or lock file. Edit it by hand or via uv." >&2
    exit 2 ;;
esac
exit 0
