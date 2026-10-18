#!/bin/bash
# Score each model's ASI-Bench run as soon as its `asibench run` has exited (scripts/score_asi.sh).
# The score report also tells scripts/lane_hle.sh that the model's GPU no longer serves ASI agents.
#   bash scripts/lane_asi_score.sh >> logs/lane_asi_score.log 2>&1      (resumable: scored models skip)
set -u
cd "$(dirname "$0")/.."
B=/var/tmp/$USER
finish() {
  local m=$1 out=$B/asi-runs/$1/local_score_seed31415.json
  [ -s "$out" ] && return
  while pgrep -u "$USER" -f "[a]sibench run .*--output-dir $B/asi-runs/$m( |$)" > /dev/null; do sleep 120; done
  n=$(ls "$B/asi-runs/$m"/*/*__b[1-4].json 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] ASI $m run exited with $n/236 results; scoring"
  bash scripts/score_asi.sh "$m" >> "logs/asi_score_$m.log" 2>&1 && echo "[$(date '+%F %T')] ASI $m scored"
}
for m in base sci merged; do finish "$m" & done
wait
