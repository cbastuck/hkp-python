#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "hkp-python is managed with uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# `uv run` first makes .venv exactly what uv.lock pins — the runtime, its
# asr/tts/mp3 extras and the test tooling (the dev group in pyproject.toml) —
# and refuses a lockfile that no longer matches pyproject.toml.
exec uv run --locked pytest "$@"
