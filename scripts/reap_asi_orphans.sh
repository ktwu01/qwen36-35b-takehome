#!/bin/bash
# asibench enforces its time budget by killing the `unshare` sandbox wrapper, but the mini-swe-agent
# inside survives (re-parented to init) and keeps calling the model endpoint, starving live agents.
# Such an orphan belongs to an instance asibench already recorded as timed out, so kill it.
# Same rule for every model. Runs as a loop:  bash scripts/reap_asi_orphans.sh
while true; do
  # pgrep -f only sees the first 4 KB of a command line and the task prompt comes first, so match the
  # model flag and check the full /proc cmdline for the ASI workspace
  for p in $(pgrep -u "$USER" -f "[h]osted_vllm/"); do
    [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = 1 ] || continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "environment.cwd=/tmp/ai4sci_ws" || continue
    echo "[$(date '+%F %T')] reap $p $(ps -o etime=,args= -p "$p" | cut -c1-80)"
    kill -9 "$p"
  done
  sleep 60
done
