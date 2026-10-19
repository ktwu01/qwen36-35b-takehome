#!/bin/bash
# Run on a HPC login node (sbatch is refused on compute nodes). Queues N more 2 h gpu-a100-dev serving
# windows (slots 0-2), each afterany the previous one, after the last queued qwen36-serve dev job.
# the project allocation cost: 6 SU per window; check `/usr/local/etc/the allocation balance command` first.
#   cd $SCRATCH/qwen36/repo && bash scripts/hpc_dev_chain.sh 9
set -eu
n=${1:?number of windows}
prev=$(squeue -u "$USER" -h -n qwen36-serve -p gpu-a100-dev -o %i | sort -n | tail -1)
for _ in $(seq "$n"); do
  prev=$(sbatch --parsable -p gpu-a100-dev -t 02:00:00 ${prev:+-d afterany:$prev} \
    --export=ALL,K0=0,NGPU=3 scripts/hpc_serve.slurm | tail -1)
  echo "queued $prev"
done
