#!/bin/bash
# HLE for one served model: generate all 2,500 answers, then judge with o3-mini-2025-01-31.
#   bash scripts/run_hle.sh <served name> <api base> <results group: base|experts|merged> [concurrency]
set -euo pipefail
NAME=$1; API=$2; GROUP=$3; C=${4:-48}
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/var/tmp/$USER/hf-cache} PYTHONPATH=src
PY=/var/tmp/$USER/venvs/train/bin/python
OUT=results/$GROUP/hle/$NAME/run1
mkdir -p "$OUT"
"$PY" -m takehome.eval_hle generate --name "$NAME" --api "$API" --out "$OUT" --concurrency "$C"
# second pass picks up anything that failed transiently (resumable by id)
"$PY" -m takehome.eval_hle generate --name "$NAME" --api "$API" --out "$OUT" --concurrency "$C"
"$PY" -m takehome.eval_hle judge --out "$OUT" --concurrency 32
"$PY" -m takehome.eval_hle judge --out "$OUT" --concurrency 32
