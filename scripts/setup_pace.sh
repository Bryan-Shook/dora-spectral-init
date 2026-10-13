#!/usr/bin/env bash
# Install once before submitting jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
backend="${1:-cpu}"
case "$backend" in
  cpu|cuda) ;;
  *) echo 'Usage: bash scripts/setup_pace.sh [cpu|cuda]' >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo 'Install/load uv first. See docs/SETUP.md.' >&2; exit 1; }
if [[ "$backend" == cuda && "$(uname -s)" != Linux ]]; then
  echo 'The cuda extra is for Linux ICE compute nodes.' >&2
  exit 2
fi
if [[ "$backend" == cuda ]]; then
  echo 'Selected CUDA 12.6 build. Verify driver compatibility in an allocated GPU session.'
fi
uv sync --locked --extra "$backend" --extra models
echo 'Environment ready. Run CPU checks in an appropriate CPU allocation:'
echo '.venv/bin/python -m pytest'
