#!/bin/bash
# Run where `ssh hpc` works (the Mac), in the background, until the dev-window cutoff:
#   bash scripts/hpc_dev_keeper.sh >> /tmp/hpc_dev_keeper.log 2>&1
set -u
until [ "$(date +%s)" -ge "$(date -j -f '%Y-%m-%d %H:%M' '2026-10-19 19:00' +%s)" ]; do
  ssh -o BatchMode=yes hpc 'cd $SCRATCH/qwen36/repo && git pull -q && bash scripts/hpc_dev_chain.sh'
  sleep 600
done
