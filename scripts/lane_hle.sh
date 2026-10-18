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
# In-flight requests per endpoint: 12 while the model's A100 also serves its ASI-Bench agents (their
# budgets are wall-clock, so HLE must not starve them), 24 once that model's ASI run is scored (vLLM runs
# ~32 sequences at once; the rest stays for TB2). Same rule for every model.
run() {  # run <name> <endpoints> <results group>
  local n_ep pid asi=$B/asi-runs/$1/local_score_seed31415.json
  n_ep=$(echo "$2" | tr "," "\n" | wc -l)
  if [ ! -s "$asi" ]; then
    bash scripts/run_hle.sh "$1" "$2" "$3" $((12 * n_ep)) >> "logs/hle_$1.log" 2>&1 &
    pid=$!
    until [ -s "$asi" ] || ! kill -0 "$pid" 2>/dev/null; do sleep 120; done
    kill -0 "$pid" 2>/dev/null || return 0
    echo "[$(date '+%F %T')] ASI $1 scored: HLE $1 -> 24 per endpoint"
    kill "$pid"; pkill -u "$USER" -f "takehome[.]eval_hle .*--name $1 "; wait "$pid"
  fi
  bash scripts/run_hle.sh "$1" "$2" "$3" $((24 * n_ep)) >> "logs/hle_$1.log" 2>&1
}
# each model is gated on its always-on A100 slot; eval_hle waits out gaps between HPC dev windows.
# $B/hle_endpoints (lines "<name> <gate port> <group> <port>...") overrides the endpoint lists, e.g. when
# a dev-window slot is retargeted to another model (scripts/rebalance_dev_slot.sh)
specs=("base 18107 base 18107" "merged 18109 merged 18109 18101 18102 18103" "sci 18108 experts 18108")
[ -s "$B/hle_endpoints" ] && mapfile -t specs < "$B/hle_endpoints"
for spec in "${specs[@]}"; do
  set -- $spec
  set -- "$1" "$2" "$3" "$(u "${@:4}")"
  until curl -sf --max-time 5 "http://127.0.0.1:$2/v1/models" | grep -q "\"$1\""; do sleep 60; done
  echo "[$(date '+%F %T')] HLE $1 -> $4"
  run "$1" "$4" "$3" &
done
wait
echo "[$(date '+%F %T')] HLE done"
