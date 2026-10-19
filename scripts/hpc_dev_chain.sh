#!/bin/bash
# Run on a HPC login node (sbatch is refused on compute nodes). Tops the chain of 2 h gpu-a100-dev
# serving windows (slots 0-2) back up to 3 submitted jobs (the dev QOS per-user submit limit), each
# afterany the previous one, while it is before the cutoff in $CTRL/RESUBMIT_DEV and the window budget
# in $CTRL/RESUBMIT_DEV_LEFT (6 the project allocation SU per window) is > 0.
#   cd $SCRATCH/qwen36/repo && bash scripts/hpc_dev_chain.sh
# scripts/hpc_dev_keeper.sh runs this every 10 min from a machine with `ssh hpc`.
set -u
CTRL=$SCRATCH/qwen36/serve
[ -f "$CTRL/RESUBMIT_DEV" ] && [ "$(date +%s)" -lt "$(date -d "$(cat "$CTRL/RESUBMIT_DEV")" +%s)" ] || exit 0
while true; do
  left=$(cat "$CTRL/RESUBMIT_DEV_LEFT" 2>/dev/null || echo 0)
  jobs=$(squeue -u "$USER" -h -n qwen36-serve -p gpu-a100-dev -o %i | sort -n)
  [ "$left" -gt 0 ] && [ "$(echo "$jobs" | grep -c .)" -lt 3 ] || exit 0
  prev=$(echo "$jobs" | tail -1)
  id=$(sbatch --parsable -p gpu-a100-dev -t 02:00:00 ${prev:+-d afterany:$prev} \
    --export=ALL,K0=0,NGPU=3 scripts/hpc_serve.slurm 2>/dev/null | grep -Eo '^[0-9]+' | tail -1)
  [ -n "$id" ] || { echo "[$(date '+%F %T')] sbatch failed"; exit 1; }
  echo $((left - 1)) > "$CTRL/RESUBMIT_DEV_LEFT"
  echo "[$(date '+%F %T')] queued $id (afterany ${prev:-none}), $((left - 1)) windows left"
done
