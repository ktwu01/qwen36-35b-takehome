#!/bin/bash
# Public local scoring of one model's ASI-Bench seed31415 run (asibench score) ->
# $BASE/asi-runs/<name>/local_score_seed31415.json (read by src/takehome/collect.py).
# Judge: the task contracts' openai/gpt-5.5. They request temperature 0.0, which the OpenAI API rejects
# for gpt-5.5 with reasoning ("only the default (1) value is supported"), so LITELLM_DROP_PARAMS drops it
# and the judge runs at the API default; same for every model (disclosed). The TSP task's validity gate
# re-runs the solver in Docker, which these hosts do not have: those instances are scorer-invalid for
# every model (excluded from the official mean, counted as 0 in the all-instances mean).
#   bash scripts/score_asi.sh <served name> [output json]
set -euo pipefail
NAME=$1
BASE=${BASE:-/var/tmp/$USER}
OUT=${2:-$BASE/asi-runs/$NAME/local_score_seed31415.json}
export HF_HOME=$BASE/hf-cache LITELLM_DROP_PARAMS=true
OPENAI_API_KEY=$(cat ~/.config/openai/api_key)
export OPENAI_API_KEY
cd "$BASE"
# asibench exits non-zero when any instance is scorer-invalid; the report is still written
"$BASE/venvs/asibench/bin/asibench" score --repo seed31415 --results-dir "asi-runs/$NAME" \
  --instances-dir asi-instances --tasks-dir asi-src/tasks --output "$OUT" --parallel 4 || true
test -s "$OUT"
