#!/bin/bash
# Run one model's TB2 seed shard 1/2 on chained HPC gpu-a100-dev serving windows (3 GPUs, 2 h each,
# tunneled to seismic18:18101-18103). Each window: one continuation part per GPU on sub-shard j/3 of
# the still-unfinished tasks; when the window ends (tunnels drop) the runners are stopped, cancelled
# trials are re-run in the next window's parts (never scored). Shard 0/2 runs on a seismic lane.
#   bash scripts/lane_dev_windows.sh <name> <seed> <first part number>
set -u
cd "$(dirname "$0")/.."
NAME=$1; SEED=$2; PART=$3; B=/var/tmp/$USER
up() { for p in 18101 18102 18103; do curl -sf --max-time 5 "http://127.0.0.1:$p/v1/models" | grep -q "\"$NAME\"" || return 1; done; }
left() {  # unfinished tasks in shard 1/2 of this seed
  /usr/bin/python3.12 - "$B/tb2-jobs" "$NAME" "$SEED" <<'PY'
import glob, json, sys, zlib
from pathlib import Path
jobs, name, seed = sys.argv[1:4]
done = set()
for f in glob.glob(f"{jobs}/tb2-{name}-s{seed}*/*/result.json"):
    try:
        r = json.load(open(f))
    except ValueError:
        continue
    if r.get("finished_at") and (r.get("exception_info") or {}).get("exception_type") != "CancelledError":
        done.add(Path(r["task_id"]["path"]).name)
tasks = json.load(open("configs/eval/terminal_bench_2.lock.json"))["task_ids"]
print(sum(1 for t in tasks if zlib.crc32(t.encode()) % 2 == 1 and t not in done))
PY
}
while [ "$(left)" -gt 0 ]; do
  until up; do sleep 60; done
  echo "[$(date '+%F %T')] window up; $(left) shard-1 tasks left; parts $PART..$((PART + 2))"
  for j in 0 1 2; do
    ~/.local/bin/tmux new-session -d -s "dev$j" "/usr/bin/python3.12 scripts/run_tb2.py --name $NAME \
      --api http://127.0.0.1:$((18101 + j))/v1 --seeds $SEED --reclaim --part $((PART + j)) --shard 1/2:$j/3 -n 8 \
      >> logs/tb2/$NAME-dev.log 2>&1"
  done
  while up && [ "$(left)" -gt 0 ]; do sleep 60; done
  for j in 0 1 2; do ~/.local/bin/tmux send-keys -t "dev$j" C-c 2>/dev/null; done
  sleep 30
  for j in 0 1 2; do ~/.local/bin/tmux kill-session -t "dev$j" 2>/dev/null; done
  pkill -u "$USER" -f "[t]b2-$NAME-s$SEED-p($PART|$((PART + 1))|$((PART + 2)))\.json"
  PART=$((PART + 3))
done
echo "[$(date '+%F %T')] shard 1 of $NAME seed $SEED complete"
