#!/bin/bash
# ASI-Bench seed31415 (asibench 0.2.0, tasks from apexin-ai/ASI-Bench@684ac2c), all final tasks,
# all four prompt levels, 3 h per instance, linux_ns sandbox, mini-swe-agent (scripts/asi_agent.sh).
#   bash scripts/run_asi.sh <served name> <api base> [parallel]
set -euo pipefail
NAME=$1; API=$2; PAR=${3:-8}
BASE=${BASE:-/var/tmp/$USER}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=$BASE/hf-cache
OUT=$BASE/asi-runs/$NAME
mkdir -p "$OUT"
cd "$BASE"
"$BASE/venvs/asibench/bin/asibench" run --instances-dir asi-instances --tasks-dir asi-src/tasks --tasks all \
  --prompt-levels b1,b2,b3,b4 --seed 31415 --timeout 10800 --sandbox linux_ns --parallel "$PAR" \
  --agent-cmd "bash $REPO/scripts/asi_agent.sh {workspace} $NAME $API 1 $OUT/trajectories $BASE" \
  --output-dir "$OUT" $([ -d "$OUT/instances" ] && echo --resume "$OUT")
