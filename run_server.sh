#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "hkp-python is managed with uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# The same environment run_tests.sh uses, so neither script removes what the
# other installed; see pyproject.toml's dev group.
exec uv run --locked python -m hkp "$@"
