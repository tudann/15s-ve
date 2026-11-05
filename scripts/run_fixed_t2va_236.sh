#!/usr/bin/env bash
set -u
cd /pfs/pfs-spMCeu/eval_team/Video-eval/LT/long_eval || exit 1
output=outputs/fixed_t2va_236_20261105
mkdir -p "$output"
export TMPDIR="$PWD/$output/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
{
    date -Is
    .venv/bin/python -u scripts/run_eval.py --config configs/eval_fixed_t2va_236.yaml
    status=$?
    printf '\nexit_code=%s\n' "$status"
    date -Is
    printf '%s\n' "$status" > "$output/exit_code.txt"
} >> "$output/run.log" 2>&1
