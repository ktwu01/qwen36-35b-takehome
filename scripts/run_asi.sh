#!/bin/bash
# ASI-Bench seed31415 (asibench 0.2.0, tasks from apexin-ai/ASI-Bench@684ac2c), all final tasks,
# all four prompt levels, linux_ns sandbox, mini-swe-agent (scripts/asi_agent.sh). Budget: 1,800 s per
# instance for every model (experiment setting fixed before any ASI result; the asibench default is
# 10,800 s, which at ~half the instances running to the cap would cost ~40 GPU-h per model here).
#   bash scripts/run_asi.sh <served name> <api base> [parallel]
set -euo pipefail
NAME=$1; API=$2; PAR=${3:-8}; TIMEOUT=${ASI_TIMEOUT:-1800}
BASE=${BASE:-/var/tmp/$USER}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export HF_HOME=$BASE/hf-cache
OUT=$BASE/asi-runs/$NAME
mkdir -p "$OUT"
MINI_CFG=$("$BASE/venvs/harbor/bin/python" -c "import minisweagent,os;print(os.path.dirname(minisweagent.__file__))" 2>/dev/null | tail -1)/config/mini_textbased.yaml
"$BASE/venvs/harbor/bin/python" "$REPO/scripts/make_swepro_config.py" "$MINI_CFG" "$NAME" "$API" "$OUT/mini_config.yaml"
cd "$BASE"
"$BASE/venvs/asibench/bin/asibench" run --instances-dir asi-instances --tasks-dir asi-src/tasks --tasks all \
  --prompt-levels b1,b2,b3,b4 --seed 31415 --timeout "$TIMEOUT" --sandbox linux_ns --parallel "$PAR" \
  --agent-cmd "bash $REPO/scripts/asi_agent.sh {workspace} $NAME $API 1 $OUT/trajectories $BASE" \
  --output-dir "$OUT" $([ -d "$OUT/instances" ] && echo --resume "$OUT")
