#!/bin/bash
# Start HLE automatically once the cais/hle access request is approved (poller writes HLE_GRANTED):
# lock HLE, add HLE to the decontamination texts, check the training sets, then generate + judge for
# base / sci / merged on the A100 serving endpoints (all A100, identical vLLM flags). Merged also uses
# the three HPC dev-window endpoints that serve it; questions are routed per endpoint by id hash.
#   bash scripts/lane_hle.sh >> logs/lane_hle.log 2>&1        (resumable: done steps are skipped)
set -u
cd "$(dirname "$0")/.."
B=/var/tmp/$USER
export HF_HOME=$B/hf-cache PYTHONPATH=src
PY=$B/venvs/train/bin/python
until [ -f "$B/HLE_GRANTED" ]; do sleep 120; done
echo "[$(date '+%F %T')] HLE access granted"
[ -f configs/eval/hle.lock.json ] || "$PY" scripts/lock_benchmarks.py --only hle
grep -q '"bench": "hle"' data/bench/contam_texts.jsonl 2>/dev/null || "$PY" -m takehome.build_contam
[ -f data/manifests/hle_contamination.json ] || "$PY" scripts/hle_contam_check.py
u() { local s=""; for p in "$@"; do s="$s${s:+,}http://127.0.0.1:$p/v1"; done; echo "$s"; }
for spec in "base 18107 base $(u 18107)" "sci 18108 experts $(u 18108)" \
            "merged 18109 merged $(u 18109 18101 18102 18103)"; do
  set -- $spec
  until curl -sf --max-time 5 "http://127.0.0.1:$2/v1/models" | grep -q "\"$1\""; do sleep 60; done
  echo "[$(date '+%F %T')] HLE $1 -> $4"
  bash scripts/run_hle.sh "$1" "$4" "$3" 48 >> "logs/hle_$1.log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] HLE done"
