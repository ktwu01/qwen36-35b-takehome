#!/bin/bash
# Train the three experts in parallel on one an HPC supercomputer gpu-a100-dev node (3x A100 40GB),
# one expert per GPU. 2 h wall limit per job; resubmit/chain and the trainer resumes from
# the last checkpoint. Allocation: the project allocation (user cap 245 SU).
#   sbatch scripts/hpc_train_experts.slurm            (from $SCRATCH/qwen36/repo)
#SBATCH -J qwen36-experts
#SBATCH -p gpu-a100-dev
#SBATCH -N 1 -n 1
#SBATCH -t 02:00:00
#SBATCH -A the project allocation
#SBATCH -o logs/slurm-%x-%j.out
set -u
cd "$SCRATCH/qwen36/repo"
export HF_HOME=$SCRATCH/hf-cache HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/qwen36/venv/bin/python
mkdir -p logs outputs
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw --format=csv -l 30 \
  > "logs/gpu_${SLURM_JOB_ID}.csv" &
i=0
for e in ${EXPERTS:-swe term sci}; do
  if [ -f "outputs/$e/adapter/adapter_model.safetensors" ]; then echo "$e already done"; i=$((i+1)); continue; fi
  CUDA_VISIBLE_DEVICES=$i "$PY" -m takehome.sft --expert "$e" --data "data/processed/$e" --out "outputs/$e" \
    >> "logs/train_$e.log" 2>&1 &
  i=$((i+1))
done
wait
