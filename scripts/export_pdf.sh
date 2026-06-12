#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <notebook.ipynb> [autres options nbconvert]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export JUPYTER_CONFIG_DIR="$ROOT/jupyter_config"

cd "$ROOT"
uv run python scripts/setup_pandoc.py
exec uv run jupyter nbconvert --to webpdf "$@"
