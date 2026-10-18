#!/bin/bash
# Sequential evaluation lane for the experts on seismic15 (run from seismic18, which hosts the
# agent containers). Required cells first:
#   term TB2 seed 1 (already running) -> swe on SWE-Pro -> sci on ASI-Bench -> term TB2 seed 2.
#   bash scripts/lane_s15.sh >> logs/lane_s15.log 2>&1
set -u
cd "$(dirname "$0")/.."
B=/var/tmp/$USER
API=http://seismic15.geo.utexas.edu:18000/v1
log() { echo "[$(date '+%F %T')] $*"; }

finished() {  # finished <job glob>: number of finished, non-cancelled trials
  /usr/bin/python3.12 - "$1" <<'PY'
import glob, json, sys
n = 0
for f in glob.glob(sys.argv[1] + "/*/result.json"):
    try:
        r = json.load(open(f))
    except ValueError:
        continue
    if r.get("finished_at") and (r.get("exception_info") or {}).get("exception_type") != "CancelledError":
        n += 1
print(n)
PY
}

serve() {  # serve <name> <model dir under ~/models>: (re)start vLLM on seismic15 and wait until healthy
  log "serving $1 on seismic15"
  ssh -o BatchMode=yes seismic15 "~/.local/bin/tmux kill-session -t vserve 2>/dev/null; ~/.local/bin/tmux kill-session -t vterm 2>/dev/null; \
    mkdir -p $B/models; [ -d $B/models/$2 ] || cp -r ~/models/$2 $B/models/; cd ~/projects/qwen36-takehome && \
    ~/.local/bin/tmux new-session -d -s vserve 'bash scripts/serve.sh $B/models/$2 $1 18000 > logs/vllm_$1_seismic15.log 2>&1'"
  until curl -sf "$API/models" | grep -q "\"$1\""; do sleep 15; done
  log "$1 ready"
}

log "waiting for term TB2 seed 1 (89 tasks)"
until [ "$(finished "$B/tb2-jobs/tb2-term-s1*")" -ge 89 ]; do sleep 120; done
~/.local/bin/tmux kill-session -t tb2term 2>/dev/null
pkill -u "$USER" -f "[t]b2-term-s2" ; sleep 5
[ -d "$B/tb2-jobs/tb2-term-s2" ] && mv "$B/tb2-jobs/tb2-term-s2" "$B/tb2-jobs/DISCARDED-preempted-tb2-term-s2"
rm -f "$B/tb2-jobs/tb2-term-s2.claim"
log "term seed 1 done"

serve swe expert-swe
bash scripts/run_swepro.sh swe "$API" 6 >> logs/sbp_swe.log 2>&1
log "swe SWE-Pro done"

serve sci expert-sci
bash scripts/run_asi.sh sci "$API" 6 >> logs/asi_sci.log 2>&1
log "sci ASI-Bench done"

serve term expert-term
/usr/bin/python3.12 scripts/run_tb2.py --name term --api "$API" --seeds 2 3 4 5 -n 6 >> logs/tb2/term-s15.log 2>&1
log "lane done"
