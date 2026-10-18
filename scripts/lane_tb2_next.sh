#!/bin/bash
# Continue a model's Terminal-Bench seeds after seed 1 (all parts) has all 89 tasks finished.
#   bash scripts/lane_tb2_next.sh <name> <api> <n_concurrent>
set -u
cd "$(dirname "$0")/.."
NAME=$1; API=$2; N=$3; B=/var/tmp/$USER
count() {
  /usr/bin/python3.12 - "$B/tb2-jobs" "$NAME" <<'PY'
import glob, json, sys
from pathlib import Path
done = set()
for f in glob.glob(f"{sys.argv[1]}/tb2-{sys.argv[2]}-s1*/*/result.json"):
    try:
        r = json.load(open(f))
    except ValueError:
        continue
    if r.get("finished_at") and (r.get("exception_info") or {}).get("exception_type") != "CancelledError":
        done.add(Path(r["task_id"]["path"]).name)
print(len(done))
PY
}
until [ "$(count)" -ge 89 ]; do sleep 120; done
echo "[$(date '+%F %T')] $NAME seed 1 complete; starting seeds 2-5"
/usr/bin/python3.12 scripts/run_tb2.py --name "$NAME" --api "$API" --seeds 2 3 4 5 -n "$N"
