#!/bin/bash
# Run where both `ssh hpc` and `ssh seismic18` work. Once the merged TB2 seed-1 shard on the HPC dev
# windows is complete, the dev GPUs only serve merged HLE, whose answers are short (~15K tokens) while
# base's are long (~25K, the HLE critical path). Retarget dev slot 2 (seismic18:18103) to base and
# re-route HLE: base on 18107+18103, merged on 18109+18101+18102 (HLE is resumable by question id).
#   bash scripts/rebalance_dev_slot.sh >> logs/rebalance_dev_slot.log 2>&1
set -u
until ssh -o BatchMode=yes seismic18 'grep -q "shard 1 of merged seed 1 complete" ~/projects/qwen36-takehome/logs/lane_dev.log'; do
  sleep 300
done
echo "[$(date '+%F %T')] merged TB2 seed-1 dev shard complete; slot 2 -> base"
echo "base Qwen/Qwen3.5-9B" | ssh -o BatchMode=yes hpc 'tee $SCRATCH/qwen36/serve/gpu2.model'
printf '%s\n' "base 18107 base 18107 18103" "merged 18109 merged 18109 18101 18102" "sci 18108 experts 18108" |
  ssh -o BatchMode=yes seismic18 'tee /var/tmp/$USER/hle_endpoints'
ssh -o BatchMode=yes seismic18 'cd ~/projects/qwen36-takehome && ~/.local/bin/tmux kill-session -t hlelane; sleep 2
  P=$(pgrep -u "$USER" -f "takehome[.]eval_hle"); [ -n "$P" ] && kill $P; sleep 3
  ~/.local/bin/tmux new-session -d -s hlelane "bash scripts/lane_hle.sh >> logs/lane_hle.log 2>&1"'
echo "[$(date '+%F %T')] HLE re-routed"
