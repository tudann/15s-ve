#!/usr/bin/env bash
# Read configs/eval_run.yaml and score the listed text-to-video cases.
# Edit that file, then run this script. The same file is used for a subset or the full set.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec .venv/bin/python scripts/run_eval.py "$@"
