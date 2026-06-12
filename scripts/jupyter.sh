#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export JUPYTER_CONFIG_DIR="$ROOT/jupyter_config"

cd "$ROOT"
uv run python scripts/setup_pandoc.py
exec uv run jupyter "$@"
