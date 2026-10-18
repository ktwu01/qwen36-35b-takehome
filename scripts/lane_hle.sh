#!/bin/bash
# Start HLE automatically once the cais/hle access request is approved (poller writes HLE_GRANTED):
# lock HLE, add HLE to the decontamination texts, check the training sets, then generate + judge for
# base / sci / merged on the A100 serving slots (seismic18 ports 18107/18108/18109).
#   bash scripts/lane_hle.sh >> logs/lane_hle.log 2>&1
set -u
cd "$(dirname "$0")/.."
B=/var/tmp/$USER
export HF_HOME=$B/hf-cache PYTHONPATH=src
PY=$B/venvs/train/bin/python
until [ -f "$B/HLE_GRANTED" ]; do sleep 120; done
echo "[$(date '+%F %T')] HLE access granted"
"$PY" scripts/lock_benchmarks.py --only hle
"$PY" -m takehome.build_contam
"$PY" scripts/hle_contam_check.py
for spec in "base 18107 base" "sci 18108 experts" "merged 18109 merged"; do
  set -- $spec
  until curl -sf --max-time 5 "http://127.0.0.1:$2/v1/models" | grep -q "\"$1\""; do sleep 60; done
  bash scripts/run_hle.sh "$1" "http://127.0.0.1:$2/v1" "$3" 48 >> "logs/hle_$1.log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] HLE done"
