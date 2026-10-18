#!/bin/bash
# Split the merged model's TB2 seed 1 across seismic17 and the A100 slot 8 once that slot is up,
# then continue seeds 2+ on seismic17. (Base seed 1 was split the same way by hand: parts 3/4.)
cd ~/projects/qwen36-takehome
until curl -sf --max-time 5 http://127.0.0.1:18109/v1/models | grep -q "\"merged\""; do sleep 60; done
echo "[$(date)] slot 8 (merged) up; splitting merged TB2 seed 1"
~/.local/bin/tmux send-keys -t tb2merged C-c; sleep 30; ~/.local/bin/tmux kill-session -t tb2merged 2>/dev/null
pkill -u $USER -f "[t]b2-merged-s1.json"; sleep 5
~/.local/bin/tmux new-session -d -s tb2mergedA "/usr/bin/python3.12 scripts/run_tb2.py --name merged --api http://seismic17.geo.utexas.edu:18000/v1 --seeds 1 --reclaim --part 2 --shard 0/2 -n 6 >> logs/tb2/merged-s17.log 2>&1"
~/.local/bin/tmux new-session -d -s tb2mergedB "/usr/bin/python3.12 scripts/run_tb2.py --name merged --api http://127.0.0.1:18109/v1 --seeds 1 --reclaim --part 3 --shard 1/2 -n 8 >> logs/tb2/merged-a100.log 2>&1"
bash scripts/lane_tb2_next.sh merged http://seismic17.geo.utexas.edu:18000/v1 6 >> logs/tb2/merged-s17.log 2>&1
